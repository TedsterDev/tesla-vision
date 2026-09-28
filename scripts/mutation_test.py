#!/usr/bin/env python3
"""
mutation_test.py

Proves that scripts/selfcheck.py is not vacuous.

WHY THIS EXISTS
---------------
selfcheck.py is a positive control: it feeds known-answer input through the real
pipeline and asserts the known answer comes back. Its whole value rests on one
property that is easy to assume and easy to get wrong - that it FAILS when the
pipeline is broken.

A test that passes unconditionally is worse than no test, because it converts
"we have not checked" into "we have checked and it is fine". So this script
deliberately breaks the pipeline, one thing at a time, and confirms the control
notices. Any mutation that selfcheck does NOT catch is a hole in the control,
and this script names it.

HOW IT WORKS
------------
For each mutation we copy the whole repo to a scratch directory, append a small
monkeypatch to the end of one production module in the COPY, and run the copy's
selfcheck against it. selfcheck sets REPO_ROOT from its own location and inserts
that on sys.path, so a copied tree tests the copied source - the live repo is
never modified.

Appending a monkeypatch (rather than editing a function body) is deliberate: it
executes at import time, needs no AST surgery, and cannot accidentally change
behaviour other than the one binding it replaces.

A BASELINE run of an unmutated copy runs first. If the baseline fails, the copy
itself is broken and every subsequent "caught it!" would be meaningless.

USAGE
    docker compose exec processor python -u /app/scripts/mutation_test.py
    docker compose exec processor python -u /app/scripts/mutation_test.py --only plate-blind
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
import time

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Each mutation names the module to poison, the code appended to it, and the
# selfcheck stage that MUST report a failure as a result.
MUTATIONS = [
    (
        "plate-blind",
        "alpr.py",
        "PlateReader.read_plates = lambda self, *args, **kwargs: []",
        "alpr",
        "the plate reader finds nothing, ever",
    ),
    (
        "plate-illegible",
        "alpr.py",
        "is_legible = lambda *args, **kwargs: False",
        "alpr",
        "every plate crop is judged too dark to read",
    ),
    (
        "plate-unreadable",
        "alpr.py",
        'PlateReader.recognize = lambda self, *args, **kwargs: ("", 0.0)',
        "alpr",
        "OCR returns empty for every crop",
    ),
    (
        "plate-not-persisted",
        "processor.py",
        "record_plate_detections = lambda *args, **kwargs: None",
        "alpr",
        "plates are read but never written to the database",
    ),
    (
        "face-blind",
        "faces.py",
        "FaceEngine.detect = lambda self, *args, **kwargs: []",
        "faces",
        "the face detector finds nothing, ever",
    ),
    (
        "face-no-embedding",
        "faces.py",
        "FaceEngine.embed = lambda self, *args, **kwargs: None",
        "faces",
        "faces are detected but never embedded",
    ),
    (
        "face-no-identity",
        "faces.py",
        'FaceEngine.match_or_create_identity = lambda self, connection, embedding, seen_ts: '
        '("ghost", "Ghost", 1.0, False)',
        "faces",
        "embeddings never become identity rows",
    ),
    (
        "no-encounters",
        "correlate.py",
        "collapse_into_encounters = lambda *args, **kwargs: []",
        "correlate",
        "detections never collapse into encounters",
    ),
    (
        "zero-score",
        "correlate.py",
        "_original_score = score_encounters\n"
        "def score_encounters(*args, **kwargs):\n"
        "    result = _original_score(*args, **kwargs)\n"
        "    result.score = 0.0\n"
        "    result.severity = 'low'\n"
        "    return result\n",
        "correlate",
        "every entity scores zero",
    ),
    (
        "no-detections",
        "processor.py",
        "_original_analyze = analyze_clip\n"
        "def analyze_clip(*args, **kwargs):\n"
        "    return ClipAnalysis()\n",
        "yolo",
        "the detector returns an empty analysis for every clip",
    ),
]


def run_selfcheck(tree: Path, timeout_seconds: int = 900) -> tuple[int, str]:
    """Run the selfcheck inside a (possibly mutated) copy of the repo."""
    completed = subprocess.run(
        [sys.executable, "-u", str(tree / "scripts" / "selfcheck.py")],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return completed.returncode, completed.stdout + completed.stderr


def failing_stages(output: str) -> set[str]:
    """
    Extract which stages reported a FAIL.

    The results table only prints the stage name on the first row of each group,
    so we carry the last stage seen forward - otherwise every failure after the
    first row of a group looks stageless.
    """
    stages: set[str] = set()
    current_stage = ""

    for line in output.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("[PASS]") or stripped.startswith("[FAIL]")):
            continue

        remainder = stripped[6:].lstrip()
        # A stage name is a lowercase word followed by two or more spaces.
        first_token = remainder.split("  ")[0].strip()
        if first_token and " " not in first_token and first_token.islower():
            current_stage = first_token

        if stripped.startswith("[FAIL]"):
            stages.add(current_stage)

    return stages


def apply_mutation(tree: Path, module_filename: str, patch: str, name: str) -> None:
    """Append a monkeypatch to the end of a module in the copied tree."""
    target = tree / "src" / module_filename
    with target.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n# --- mutation '{name}' injected by mutation_test.py ---\n")
        handle.write(patch + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prove selfcheck.py detects a broken pipeline")
    parser.add_argument("--only", help="run a single mutation by name")
    parser.add_argument("--keep", action="store_true", help="keep the mutated trees")
    arguments = parser.parse_args()

    mutations = [m for m in MUTATIONS if not arguments.only or m[0] == arguments.only]
    if not mutations:
        print(f"No mutation named {arguments.only!r}. Known: {[m[0] for m in MUTATIONS]}")
        return 2

    workspace = Path(tempfile.mkdtemp(prefix="scout-mutation-"))
    print(f"[mutation] workspace {workspace}")
    print(f"[mutation] source    {REPO_ROOT}")

    results = []
    try:
        # --- baseline -----------------------------------------------------
        # If an unmutated copy does not pass, nothing below means anything.
        print("\n[mutation] BASELINE: unmutated copy must PASS")
        baseline_tree = workspace / "baseline"
        shutil.copytree(REPO_ROOT, baseline_tree, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", "models"))
        started = time.time()
        baseline_code, baseline_output = run_selfcheck(baseline_tree)
        baseline_ok = baseline_code == 0
        print(f"[mutation] baseline exit={baseline_code} "
              f"({'PASS' if baseline_ok else 'FAIL'}) in {time.time() - started:.0f}s")

        if not baseline_ok:
            print("\n[mutation] ABORT: the unmutated copy does not pass, so mutation results "
                  "would be meaningless. Last 25 lines:")
            print("\n".join(baseline_output.splitlines()[-25:]))
            return 1

        # --- mutations ----------------------------------------------------
        for name, module_filename, patch, expected_stage, description in mutations:
            tree = workspace / name
            shutil.copytree(REPO_ROOT, tree, ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", "models"))
            apply_mutation(tree, module_filename, patch, name)

            print(f"\n[mutation] {name}: {description}")
            started = time.time()
            try:
                exit_code, output = run_selfcheck(tree)
            except subprocess.TimeoutExpired:
                exit_code, output = -1, "(timed out)"
            elapsed = time.time() - started

            stages = failing_stages(output)
            caught = exit_code != 0
            right_stage = expected_stage in stages

            results.append((name, caught, right_stage, expected_stage, sorted(stages), elapsed))
            verdict = "CAUGHT" if caught else "MISSED"
            print(f"[mutation] {name}: {verdict} exit={exit_code} "
                  f"failing stages={sorted(stages) or 'none'} "
                  f"(expected '{expected_stage}') [{elapsed:.0f}s]")

    finally:
        if not arguments.keep:
            shutil.rmtree(workspace, ignore_errors=True)
        else:
            print(f"\n[mutation] kept {workspace}")

    # --- report -----------------------------------------------------------
    print("\n" + "=" * 100)
    print("MUTATION TEST RESULTS  (does selfcheck.py actually detect a broken pipeline?)")
    print("=" * 100)
    print(f"  {'mutation':22} {'caught':8} {'right stage':13} {'stages that failed'}")
    print("  " + "-" * 96)
    for name, caught, right_stage, expected, stages, elapsed in results:
        print(f"  {name:22} {'yes' if caught else 'NO':8} "
              f"{('yes' if right_stage else 'no (' + expected + ')'):13} "
              f"{', '.join(stages) or '-'}")

    missed = [r for r in results if not r[1]]
    wrong_stage = [r for r in results if r[1] and not r[2]]

    print("  " + "-" * 96)
    if missed:
        print(f"  RESULT: FAIL - {len(missed)} mutation(s) NOT caught: "
              f"{', '.join(r[0] for r in missed)}")
        print("  The self-check has a hole: the pipeline can break in these ways and it will")
        print("  still report PASS. Add an assertion for each before trusting a zero result.")
        return 1

    print(f"  RESULT: PASS - all {len(results)} mutations were caught.")
    if wrong_stage:
        print(f"  ({len(wrong_stage)} caught by a different stage than expected: "
              f"{', '.join(r[0] for r in wrong_stage)} - still detected, but the")
        print("   per-stage attribution could be sharper.)")
    print("  The self-check genuinely fails when the pipeline is broken, so a PASS from it")
    print("  is evidence, not decoration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
