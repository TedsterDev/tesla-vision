"""
natix_worker.py

Keeps the NATIX VX360 fed.

The car writes to the Jetson, not to the stick, so nothing reaches the stick
unless we put it there. This service is what puts it there: watch for the stick,
mount it, copy across every clip it doesn't have yet in Tesla's own directory
layout, unmount, sleep, repeat.

Why this runs on the host and not in Docker
-------------------------------------------
Every other service in this stack is a container. This one is not, and that is
deliberate: it mounts and unmounts a real block device, and a mount made inside
a container lands in that container's mount namespace, where nothing else can
see it - including the stick's own firmware, which is looking at the flash from
the other side. Bind-propagating /mnt as rshared into a privileged container
would work, but it trades a clear ownership boundary for a subtle one, on a
machine that loses power without warning. So: a plain systemd unit, host
Python, standard library only.

It shares the SQLite database with the containers. That is fine - WAL plus a
busy timeout is exactly the topology SQLite is built for, and it is already how
four containers coexist here.

Usage:
    python3 src/natix_worker.py --once        # one pass, then exit
    python3 src/natix_worker.py --status      # print what we can see, change nothing
    python3 src/natix_worker.py --dry-run     # plan the copies without doing them
    python3 src/natix_worker.py               # run forever (what systemd starts)
"""
import argparse
import json
import os
import signal
import sys
import time

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import natix                                    # noqa: E402
from src.common import LOGS_DIR, ensure_dirs, env_int    # noqa: E402
from src.db import connect, set_setting, write_with_retry  # noqa: E402

# How long to wait between passes. A dashcam stick is not a low-latency device:
# it only uploads when the car is parked on known WiFi, so a minute of lag
# costs nothing and a tight loop would spin the flash for no reason.
NATIX_POLL_SECONDS = env_int("NATIX_POLL_SECONDS", 60)

# Cap the work in one pass so a fresh 132-clip archive doesn't hold the stick
# mounted for ten minutes straight. The next pass picks up where this left off.
NATIX_BATCH = env_int("NATIX_BATCH", 24)

_should_stop = False


def _handle_signal(signum, _frame):
    global _should_stop
    _should_stop = True
    print(f"[📼 natix] signal {signum} - finishing current file then stopping")


# Where the dashboard reads this service's history from.
#
# The worker runs on the host as a systemd unit, so its natural home for logs
# is the journal - but the dashboard runs in a container, which has no access
# to journald and no reasonable way to get it. Rather than punch a hole for
# that, the worker writes the same lines to a file inside BASE_DIR, which the
# container already mounts as /data. One bind mount, no new privileges, and the
# file survives a container recreate.
NATIX_LOG_PATH = LOGS_DIR / "natix.log"
NATIX_LOG_MAX_BYTES = env_int("NATIX_LOG_MAX_BYTES", 2_000_000)


def _append_to_log_file(line: str) -> None:
    """
    Append one line, rotating at NATIX_LOG_MAX_BYTES.

    Rotation is one generation deep on purpose: this runs on a car appliance
    whose whole storage budget is a 119GB card shared with the clip archive,
    and nobody is ever going to read natix.log.7. Any failure here is swallowed
    - a service that dies because it could not write its own log would be a
    poor trade.
    """
    try:
        ensure_dirs()
        if NATIX_LOG_PATH.exists() and NATIX_LOG_PATH.stat().st_size > NATIX_LOG_MAX_BYTES:
            NATIX_LOG_PATH.replace(NATIX_LOG_PATH.with_suffix(".log.1"))
        with open(NATIX_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def log(message: str) -> None:
    # The journal stamps its own lines; the file has to carry its own, because
    # the dashboard reads it with no other source of time.
    stamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [📼 natix] {message}"
    print(f"[📼 natix] {message}", flush=True)
    _append_to_log_file(stamped)


def publish_status(connection, payload: dict) -> None:
    """
    Park the last pass's outcome in `settings` so the dashboard can show it
    without needing to touch USB itself.
    """
    payload = dict(payload, updated_ts=int(time.time()))
    write_with_retry(
        connection,
        lambda active: set_setting(active, "natix_last_pass", json.dumps(payload)),
    )


def run_once(connection, dry_run: bool = False, limit: int = NATIX_BATCH) -> dict:
    """One discover -> mount -> mirror -> unmount cycle."""
    candidate = natix.find_device()
    if candidate is None:
        seen = natix.discover(include_all=False)
        detail = (
            "; ".join(
                f"{c.volume.path} ({c.confidence}"
                + (f", {c.disqualifiers[0]}" if c.disqualifiers else "")
                + ")"
                for c in seen
            )
            or "nothing attached"
        )
        return {"state": "no_device", "detail": detail}

    volume = candidate.volume
    log(
        f"found {volume.path} - {volume.label or 'unlabelled'} "
        f"{volume.size_gb:.0f}GB {volume.fstype}, confidence {candidate.confidence}"
    )
    for reason in candidate.reasons:
        log(f"  because {reason}")

    mounted_here = not volume.mountpoint
    try:
        mountpoint = natix.mount(candidate)
    except natix.MountError as error:
        log(f"mount failed: {error}")
        return {"state": "mount_failed", "detail": str(error), "device": volume.path}

    try:
        device_id = natix.register_device(connection, candidate, mountpoint)
        free_gb = natix.free_bytes(mountpoint) / (1000 ** 3)
        log(f"mounted at {mountpoint}, {free_gb:.1f}GB free, device id {device_id}")

        started = time.time()

        def report(done: int, total: int, plan) -> None:
            # A first mirror is gigabytes over USB 2.0. Silence for six minutes
            # is indistinguishable from a hang, so say something periodically -
            # with a rate, which is the number that tells you whether it is
            # working or crawling.
            if done % 10 and done != total:
                return
            elapsed = max(time.time() - started, 0.001)
            rate = sum(p.size_bytes for p in [plan]) and (done / elapsed)
            log(
                f"  {done}/{total} clips  "
                f"{rate:.1f} clips/s  latest {plan.filename}"
            )

        result = natix.mirror(
            connection, device_id, mountpoint, limit=limit, dry_run=dry_run,
            on_progress=None if dry_run else report,
            should_stop=lambda: _should_stop,
        )
        verb = "would copy" if dry_run else "copied"
        log(
            f"{verb} {result.copied} clips "
            f"({result.copied_bytes / (1024 ** 2):.0f}MB), "
            f"skipped {result.skipped}, missing {result.missing}, "
            f"pruned {result.pruned}, failed {result.failed}"
        )
        # Every reclaimed item, by name and size, never a summary count. This
        # is the only place the system destroys footage it did not create, and
        # the journal is the only record of what went.
        for line in result.reclaimed_detail:
            log(f"  {line}")
        if result.reclaimed:
            log(f"  reclaimed {result.reclaimed} pre-existing events "
                f"({result.reclaimed_bytes / 1024 ** 3:.2f}GB) to make room")
        for error in result.errors[:5]:
            log(f"  error: {error}")
        if result.stopped_reason:
            log(f"  stopped early: {result.stopped_reason}")

            # We are still mounted here, so answer the obvious next question in
            # the same run rather than making somebody mount it again to look:
            # what is holding the space, and precisely what would go if
            # reclaiming were switched on. Both are read-only.
            log("")
            log("  what is on the stick (read-only):")
            for line in natix.describe_space(mountpoint):
                log(f"  {line}")

            suggested = natix.NATIX_RECLAIM_BUCKETS or ["SentryClips", "RecentClips"]
            needed = natix.free_bytes(mountpoint) + (
                natix.NATIX_RESERVE_MB * 1024 * 1024
            )
            log("")
            if natix.NATIX_RECLAIM_BUCKETS:
                log(f"  reclaiming is ENABLED for {suggested} but did not free enough:")
            else:
                log(f"  reclaiming is OFF. If you set "
                    f"NATIX_RECLAIM_BUCKETS={','.join(suggested)} it would:")
            for line in natix.describe_reclaim_plan(
                connection, device_id, mountpoint, needed, suggested
            ):
                log(f"  {line}")
            log("")
            log("  Nothing above has been deleted. SavedClips is excluded from the")
            log("  suggestion on purpose - those are clips someone chose to keep.")

        # Refresh the counters now that the copies are recorded.
        natix.register_device(connection, candidate, mountpoint)

        payload = result.as_dict()
        payload.update(
            {
                "state": "ok",
                "device_id": device_id,
                "device": volume.path,
                "label": volume.label,
                "mountpoint": str(mountpoint),
                "free_bytes": natix.free_bytes(mountpoint),
                "dry_run": dry_run,
            }
        )
        return payload
    finally:
        if mounted_here and natix.NATIX_UNMOUNT_AFTER:
            try:
                natix.unmount(mountpoint)
                log(f"unmounted {mountpoint}")
            except natix.MountError as error:
                log(f"WARNING: unmount failed: {error}")


def print_status(connection) -> None:
    state = natix.status(connection)
    print(f"mountpoint      {state['mountpoint']}")
    print(f"min confidence  {state['min_confidence']}")
    print(f"clips in archive {state['total_clips']}")
    print()
    if not state["attached"]:
        print("No candidate volume is attached.")
    for device in state["attached"]:
        flag = "USABLE" if device["usable"] else "not usable"
        print(
            f"{device['path']:<12} {device['confidence']:<7} {flag:<11} "
            f"{device['size_gb']}GB {device['fstype']} "
            f"label={device['label']} uuid_key={device['device_id']}"
        )
        for reason in device["reasons"]:
            print(f"    + {reason}")
        for problem in device["disqualifiers"]:
            print(f"    ! {problem}")
    print()
    for device in state["known"]:
        last = time.strftime("%Y-%m-%d %H:%M", time.localtime(device["last_seen_ts"]))
        print(
            f"known: {device['id']}  mirrored={device['mirrored_count']} "
            f"({(device['mirrored_bytes'] or 0) / (1024 ** 3):.1f}GB) "
            f"pending={device['pending']}  last seen {last}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="single pass, then exit")
    parser.add_argument("--status", action="store_true", help="report and exit")
    parser.add_argument("--dry-run", action="store_true", help="plan without copying")
    parser.add_argument(
        "--limit", type=int, default=NATIX_BATCH,
        help=f"max clips per pass (default {NATIX_BATCH}, 0 = no limit)",
    )
    arguments = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    connection = connect()

    if arguments.status:
        print_status(connection)
        return 0

    if arguments.once or arguments.dry_run:
        outcome = run_once(connection, dry_run=arguments.dry_run, limit=arguments.limit)
        publish_status(connection, outcome)
        if outcome["state"] == "no_device":
            log(f"no usable NATIX device: {outcome['detail']}")
            return 2
        if outcome["state"] == "mount_failed":
            return 3
        return 0

    log(f"watching for a NATIX VX360 every {NATIX_POLL_SECONDS}s")
    log(f"mount target {natix.NATIX_MOUNTPOINT}, reserve {natix.NATIX_RESERVE_MB}MB")

    quiet_since: float | None = None
    while not _should_stop:
        try:
            outcome = run_once(connection, limit=arguments.limit)
            publish_status(connection, outcome)

            if outcome["state"] == "no_device":
                # Say it once, then go quiet. In a car the stick is unplugged
                # for hours at a time and a log line a minute is just noise.
                if quiet_since is None:
                    log(f"no NATIX device attached ({outcome['detail']})")
                    quiet_since = time.time()
            else:
                if quiet_since is not None:
                    log(f"device back after {int(time.time() - quiet_since)}s")
                quiet_since = None
        except Exception as error:                       # noqa: BLE001
            # A worker that dies on a transient USB error is worse than one
            # that logs and tries again in a minute; the stick gets yanked
            # mid-pass routinely.
            log(f"pass failed: {type(error).__name__}: {error}")

        for _ in range(NATIX_POLL_SECONDS):
            if _should_stop:
                break
            time.sleep(1)

    log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
