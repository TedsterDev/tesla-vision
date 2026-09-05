#!/usr/bin/env python3
"""
natix_verify.py - prove the VX360 actually received the footage.

The mirror reports what it *did*. This reads back what is actually *on the
stick* and checks it against the database and the source archive, because
"copied 132 clips" and "132 playable clips are on the stick" are different
claims and only the second one matters when you are asking NATIX's cloud to
upload them.

It is deliberately independent of the mirror's own bookkeeping where it can be:
sizes are compared against the source files on disk, not against the sizes the
mirror recorded, and a sample is compared byte-for-byte by hash.

    sudo python3 scripts/natix_verify.py            # mount, verify, unmount
    sudo python3 scripts/natix_verify.py --sample 8 # deep-check 8 random clips
    python3 scripts/natix_verify.py --mounted-at /mnt/natixv360   # already mounted
"""
import argparse
import hashlib
import os
import random
import re
import sys

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import natix                       # noqa: E402
from src.db import connect                  # noqa: E402

# TeslaCam/<bucket>/<YYYY-MM-DD_HH-MM-SS>/<same stamp>-<camera>.mp4
TESLA_PATH_RE = re.compile(
    r"^TeslaCam/(?P<bucket>SentryClips|SavedClips|RecentClips)/"
    r"(?P<event>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})/"
    r"(?P=event)-(?P<camera>[a-z_]+)\.mp4$"
)


@dataclass
class Check:
    description: str
    passed: bool
    detail: str = ""


class CheckList:
    """Same shape as selfcheck.py's: every check recorded, nothing skipped silently."""

    def __init__(self) -> None:
        self.checks: list[Check] = []

    def record(self, description: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(description, bool(passed), detail))
        return bool(passed)

    @property
    def everything_passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def print_table(self) -> None:
        width = max(len(check.description) for check in self.checks)
        line = "=" * (width + 60)
        print()
        print(line)
        print("NATIX MIRROR VERIFICATION  (read back from the stick, not from the log)")
        print(line)
        for check in self.checks:
            print(f"  [{'PASS' if check.passed else 'FAIL'}] "
                  f"{check.description:<{width}}  {check.detail}")
        failures = [check for check in self.checks if not check.passed]
        print("-" * (width + 60))
        if failures:
            print(f"  RESULT: FAIL - {len(failures)} of {len(self.checks)} checks failed.")
            print("  The stick does not hold what we believe it holds, so anything")
            print("  NATIX uploads from it is not what this archive contains.")
        else:
            print(f"  RESULT: PASS - all {len(self.checks)} checks passed.")
            print("  Every mirrored clip is on the stick, in Tesla's layout, byte-identical")
            print("  where sampled. The stick's own uploader is the only remaining unknown.")
        print(line)


def sha256_of(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify(connection, root: Path, device_id: str, sample_size: int) -> CheckList:
    checks = CheckList()

    rows = connection.execute(
        "SELECT clip_id, filename, dest_path, size_bytes, state "
        "FROM natix_mirror WHERE device_id=?",
        (device_id,),
    ).fetchall()
    done = [row for row in rows if row["state"] == "done"]

    checks.record(
        "mirror recorded at least one clip",
        bool(done),
        f"{len(done)} rows marked done for {device_id}",
    )
    if not done:
        return checks

    # --- 1. every recorded file is present, at the size the source has -------
    absent: list[str] = []
    wrong_size: list[str] = []
    for row in done:
        target = root / row["dest_path"]
        if not target.is_file():
            absent.append(row["dest_path"])
            continue
        source = natix.resolve_source(row["filename"])
        expected = source.stat().st_size if source else row["size_bytes"]
        if target.stat().st_size != expected:
            wrong_size.append(
                f"{row['dest_path']} is {target.stat().st_size}, source is {expected}"
            )

    checks.record("every recorded clip exists on the stick", not absent,
                  f"{len(done) - len(absent)}/{len(done)} present"
                  + (f"; missing e.g. {absent[0]}" if absent else ""))
    checks.record("sizes match the source archive", not wrong_size,
                  f"{len(done) - len(wrong_size)}/{len(done)} exact"
                  + (f"; e.g. {wrong_size[0]}" if wrong_size else ""))

    # --- 2. byte-for-byte on a random sample --------------------------------
    # Size equality catches truncation but not corruption, and USB flash does
    # corrupt. Hashing everything would mean re-reading 5GB over USB 2.0, so we
    # sample - randomly, so a systematic fault shows up across runs.
    candidates = [row for row in done if (root / row["dest_path"]).is_file()]
    sample = random.sample(candidates, min(sample_size, len(candidates)))
    mismatched: list[str] = []
    hashed = 0
    for row in sample:
        source = natix.resolve_source(row["filename"])
        if source is None:
            continue
        hashed += 1
        if sha256_of(source) != sha256_of(root / row["dest_path"]):
            mismatched.append(row["dest_path"])
    checks.record("sampled clips are byte-identical to the source", not mismatched,
                  f"{hashed - len(mismatched)}/{hashed} sha256 match"
                  + (f"; corrupt: {mismatched}" if mismatched else ""))

    # --- 3. the layout is one a Tesla consumer can parse --------------------
    unparseable = [
        row["dest_path"] for row in done if not TESLA_PATH_RE.match(row["dest_path"])
    ]
    checks.record("paths parse as TeslaCam/<bucket>/<event>/<stamp>-<camera>.mp4",
                  not unparseable,
                  f"{len(done) - len(unparseable)}/{len(done)} well-formed"
                  + (f"; e.g. {unparseable[0]}" if unparseable else ""))

    # --- 4. every event folder carries an event.json ------------------------
    event_dirs = {(root / row["dest_path"]).parent for row in done}
    without_json = [d for d in event_dirs if not (d / "event.json").is_file()]
    checks.record("every event folder has an event.json", not without_json,
                  f"{len(event_dirs) - len(without_json)}/{len(event_dirs)} folders"
                  + (f"; e.g. {without_json[0].name}" if without_json else ""))

    # --- 5. nothing half-written left behind --------------------------------
    partials = [p.name for p in root.rglob("*.part")]
    checks.record("no partial (.part) files remain", not partials, str(partials[:3]))

    # --- 6. the rolling window is correct -----------------------------------
    # NOT "is everything mirrored". The stick is smaller than the archive - 40
    # of 132 clips fit - so demanding the whole archive can never pass, and the
    # first version of this check failed a mirror that was working perfectly.
    #
    # The right invariant for a newest-first window is contiguity: everything
    # that did NOT make it must be older than everything that did. That is what
    # distinguishes "the stick holds the most recent 40 clips" from "the stick
    # holds 40 clips scattered at random", and only the first is useful when
    # you are asking who followed you home yesterday.
    total_clips = connection.execute("SELECT COUNT(*) AS n FROM clips").fetchone()["n"]
    mirrored = connection.execute(
        "SELECT MIN(c.captured_ts) AS oldest, COUNT(*) AS n FROM clips c "
        "JOIN natix_mirror m ON m.clip_id = c.id "
        "WHERE m.device_id=? AND m.state='done'",
        (device_id,),
    ).fetchone()
    newest_missed = connection.execute(
        "SELECT MAX(captured_ts) AS newest FROM clips WHERE id NOT IN "
        "(SELECT clip_id FROM natix_mirror WHERE device_id=? AND state='done')",
        (device_id,),
    ).fetchone()

    on_stick = mirrored["n"] or 0
    oldest_kept = mirrored["oldest"]
    newest_dropped = newest_missed["newest"]

    if on_stick == total_clips:
        checks.record("the whole archive fits and is mirrored", True,
                      f"{on_stick}/{total_clips} clips")
    elif oldest_kept is None:
        checks.record("the mirrored window is contiguous", False,
                      "nothing is mirrored at all")
    else:
        # `<=`, not `<`. One Tesla event is six clips sharing a timestamp, and
        # the window boundary lands inside an event whenever the remaining
        # space fits some of its cameras but not all. A dropped clip sharing
        # the oldest kept clip's timestamp is that boundary, not a hole.
        contiguous = newest_dropped is None or newest_dropped <= oldest_kept
        split_event = newest_dropped is not None and newest_dropped == oldest_kept
        checks.record(
            "the mirrored window is the newest end of the archive",
            contiguous,
            f"{on_stick}/{total_clips} clips - the stick is smaller than the "
            f"archive, so this is a rolling window"
            + ("; the oldest event on it is partial (some cameras did not fit)"
               if split_event else "")
            + ("" if contiguous else
               "; a dropped clip is NEWER than one that was kept, so the window "
               "has holes"))

    # --- 7. it settled rather than churning ---------------------------------
    # A window that keeps trading clips rewrites flash forever. If a clip has
    # been both copied and evicted from this stick, we are thrashing.
    churned = connection.execute(
        "SELECT COUNT(*) AS n FROM natix_mirror WHERE device_id=? AND state='pruned'",
        (device_id,),
    ).fetchone()["n"]
    checks.record(
        "no clip has been copied and then evicted (would mean churn)",
        churned == 0,
        f"{churned} pruned rows"
        + ("" if churned == 0 else
           "; each of those was written to flash and then deleted"))

    # --- 7. the stick still has room for its own work -----------------------
    free = natix.free_bytes(root)
    reserve = natix.NATIX_RESERVE_MB * 1024 * 1024
    checks.record("stick retains its reserve of free space", free >= reserve,
                  f"{free / 1024 ** 3:.2f} GB free, reserve is "
                  f"{natix.NATIX_RESERVE_MB} MB")

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=5,
                        help="how many clips to hash end-to-end (default 5)")
    parser.add_argument("--mounted-at", default=None,
                        help="skip mounting; verify the stick already mounted here")
    parser.add_argument("--device-id", default=None,
                        help="verify this device key instead of the attached stick's")
    arguments = parser.parse_args()

    connection = connect()

    if arguments.mounted_at:
        root = Path(arguments.mounted_at)
        if not natix.is_mounted(root):
            print(f"nothing is mounted at {root}")
            return 2
        device_id = arguments.device_id
        if device_id is None:
            row = connection.execute(
                "SELECT id FROM natix_devices ORDER BY last_seen_ts DESC LIMIT 1"
            ).fetchone()
            if row is None:
                print("no device on record; pass --device-id")
                return 2
            device_id = row["id"]
        checks = verify(connection, root, device_id, arguments.sample)
        checks.print_table()
        return 0 if checks.everything_passed else 1

    candidate = natix.find_device()
    if candidate is None:
        print("No usable NATIX device is attached. Run scripts/natix_probe.py.")
        return 2

    print(f"verifying {candidate.volume.path} ({candidate.volume.label}), "
          f"confidence {candidate.confidence}")
    mounted_here = not candidate.volume.mountpoint
    try:
        root = natix.mount(candidate)
    except natix.MountError as error:
        print(f"mount failed: {error}")
        return 3

    try:
        checks = verify(connection, root, candidate.device_id, arguments.sample)
        checks.print_table()
        return 0 if checks.everything_passed else 1
    finally:
        if mounted_here:
            try:
                natix.unmount(root)
            except natix.MountError as error:
                print(f"WARNING: unmount failed: {error}")


if __name__ == "__main__":
    sys.exit(main())
