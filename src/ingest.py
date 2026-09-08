"""
ingest.py

Copy finished TeslaCam clips from the car's volume into inbox/, safely.

This is the piece the architecture always described and never had. Commit
a5f4def moved the processor onto "an inbox basis" so it would stop scanning a
volume the car was still writing, and left a comment promising a host-side
ingest script "later". Seven months later nothing fed the inbox and nothing
had been ingested since February - and every service reported healthy,
because an empty inbox and a working pipeline with no new footage look
identical.

Why a host service and not the processor
----------------------------------------
The car does not write to a directory the Jetson can simply read. The Jetson
EXPORTS a block device to the car over USB (scripts/usb_mode_tesla.sh offers
/dev/disk/by-label/TESLACAM as a mass-storage LUN), and the car treats it as a
USB stick. While that export is live the car owns the filesystem. Mounting it
on the host at the same time - which the fstab entry will cheerfully do - is
two operating systems holding one exFAT/ext4 volume with no coordination, and
that is how footage gets corrupted.

So the ONE rule this service enforces, and the reason it exists as a separate
program: it never reads the volume while the gadget is offering it to the car.
That decision needs configfs, which a container should not see, and it needs
to be made by the thing that also owns the copy - so it lives here, on the
host, as root, and the processor simply consumes whatever appears in inbox/.

The operating cycle this expects (root, and not yet automated - see README):
    usb_gadget_stop.sh     stop offering the volume to the car
    mount /mnt/teslacam    now the host may read it
    <this service copies anything new and stable>
    umount /mnt/teslacam
    usb_mode_tesla.sh      offer it to the car again

Until the volume is mounted this service idles, logging why, exactly the way
gps.py idles without a receiver - and for the same reason: "nothing to ingest"
and "ingest is broken" must never look alike.

Configuration (environment, normally .env):
    TESLACAM_DIR              where the car's volume is mounted (common.py)
    INGEST_POLL_SECONDS       how often to look, default 30
    INGEST_GADGET_CONFIGFS    where the USB gadget tree lives; tests point
                              this at a temporary directory
"""
import json
import os
import shutil
import subprocess
import sys
import time

from pathlib import Path

from src.common import (
    INBOX_DIR,
    LOGS_DIR,
    TESLACAM_DIR,
    ensure_dirs,
    env_int,
    file_is_stable,
)

INGEST_POLL_SECONDS = env_int("INGEST_POLL_SECONDS", 30)
GADGET_CONFIGFS = Path(os.environ.get("INGEST_GADGET_CONFIGFS", "/sys/kernel/config/usb_gadget"))
HEARTBEAT_PATH = LOGS_DIR / "ingest.json"


# ---------------------------------------------------------------------------
# The copy - shared with processor.py, which used to own it
# ---------------------------------------------------------------------------
def iterate_new_clips(root: Path):
    """
    Yield MP4 paths under the TeslaCam directory.

    Tesla writes into subfolders (RecentClips/, SentryClips/, SavedClips/);
    rglob means those names are not hard-coded here.
    """
    if not root.exists():
        return
    for mp4file in root.rglob("*.mp4"):
        yield mp4file


def safe_copy_to_inbox(src: Path, inbox: Path = INBOX_DIR) -> Path | None:
    """
    Copy a Tesla-written file into the inbox once its size has stopped
    changing. Returns the inbox path, or None if it was skipped.

    Written to a dotfile first and renamed into place: a rename within one
    filesystem is atomic, so the processor can never pick up a half-copied
    clip. The dotfile prefix also keeps a crashed copy out of the processor's
    `*.mp4` glob - a `.tmp_` name is not matched by it.
    """
    if src.name.startswith("."):
        return None
    if not file_is_stable(src):
        return None

    dest = inbox / src.name
    if dest.exists():
        return None

    temporary = inbox / f".tmp_{src.name}"
    shutil.copy2(src, temporary)
    temporary.rename(dest)
    return dest


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def exported_backing_devices(configfs: Path = GADGET_CONFIGFS) -> set[str]:
    """
    Every block device the USB gadget is currently offering to the car, as
    resolved real paths. Empty when no gadget is bound or none carries a LUN.
    """
    devices: set[str] = set()
    for lun_file in configfs.glob("*/functions/mass_storage.*/lun.*/file"):
        try:
            backing = lun_file.read_text().strip()
        except OSError:
            continue
        if backing:
            devices.add(os.path.realpath(backing))
    return devices


def mount_source(path: Path) -> str | None:
    """The device `path` is a mountpoint FOR, or None if it is not one."""
    try:
        completed = subprocess.run(
            ["findmnt", "-no", "SOURCE", str(path)],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    source = completed.stdout.strip()
    return source or None


def source_state(root: Path, configfs: Path = GADGET_CONFIGFS,
                 resolve_mount=mount_source) -> tuple[str, str]:
    """
    Whether it is safe to read the TeslaCam volume right now.

    Returns one of:
        ("absent",   why)  - not mounted; nothing to do, not an error
        ("exported", why)  - mounted, but the car has it. DO NOT READ.
        ("ready",    device)
    """
    device = resolve_mount(root)
    if not device:
        return "absent", f"{root} is not a mountpoint"
    real = os.path.realpath(device)
    if real in exported_backing_devices(configfs):
        return "exported", f"{device} is currently offered to the car over USB"
    return "ready", device


# ---------------------------------------------------------------------------
# Heartbeat - same contract as gps.json, same reason
# ---------------------------------------------------------------------------
def write_heartbeat(payload: dict, path: Path = HEARTBEAT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def log(message: str) -> None:
    print(f"[📥 ingest] {message}", flush=True)


def main() -> int:
    ensure_dirs()
    log(f"source {TESLACAM_DIR}  inbox {INBOX_DIR}  poll {INGEST_POLL_SECONDS}s")

    copied_total = 0
    last_state: str | None = None
    beat = {
        "schema": 1, "pid": os.getpid(), "started_ts": int(time.time()),
        "source": str(TESLACAM_DIR), "state": None, "reason": None,
        "last_pass_ts": None, "copied_total": 0, "last_copied": None,
        "last_error": None, "last_error_ts": None,
    }

    while True:
        state, reason = source_state(TESLACAM_DIR)
        beat.update({"state": state, "reason": reason, "last_pass_ts": int(time.time())})

        if state != last_state:
            # Report transitions, not every pass - "still not mounted" every
            # 30s is noise, and the heartbeat carries the standing answer.
            log(f"{state}: {reason}")
            last_state = state

        if state == "ready":
            try:
                for clip in iterate_new_clips(TESLACAM_DIR):
                    landed = safe_copy_to_inbox(clip)
                    if landed:
                        copied_total += 1
                        beat["copied_total"] = copied_total
                        beat["last_copied"] = landed.name
                        log(f"copied -> inbox: {landed.name}")
            except OSError as error:
                beat["last_error"] = str(error)
                beat["last_error_ts"] = int(time.time())
                log(f"error: {error}")

        try:
            write_heartbeat(beat)
        except OSError as error:
            log(f"heartbeat write failed: {error}")

        time.sleep(INGEST_POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
