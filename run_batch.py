"""
Batch runner for "Pss pipeline .py" — runs every PDF in an input folder
through the pipeline one by one, so you don't have to invoke run_script.md's
command by hand for each file.

Each PDF gets its own output folder named after its FULL filename stem
(e.g. output/20151425114_1016185945_20241212_v1/), not a truncated slice —
full filenames in a folder are already unique, so this can't collide the
way "last 4 digits + date" did (two PDFs in Data/ share the date 20241212).
Re-running the same PDF bumps the version suffix (_v1, _v2, ...) rather
than overwriting.

Usage:
    python3 run_batch.py \\
        --input-dir Data \\
        --output-root output \\
        --project proposal-auto-ai-internal \\
        --workers 5

    Add --skip-existing to resume a batch without re-running PDFs that
    already have a COMPLETED run (a run_report.json on disk) — a run that
    crashed or was killed mid-way is not skipped and is retried, since it
    has no report yet.

    Add --dry-run to preview the PDF -> output-folder plan (no API calls,
    no directories created) before committing to a real, billed run.
"""

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PIPELINE_SCRIPT = SCRIPT_DIR / "Pss pipeline .py"


def next_version_dir(root: Path, stem: str) -> Path:
    v = 1
    while (root / f"{stem}_v{v}").exists():
        v += 1
    return root / f"{stem}_v{v}"


def reserve_output_dir(root: Path, stem: str) -> Path:
    """Atomically claim the next free <stem>_vN dir via mkdir (no exist_ok).
    Prevents two concurrent run_batch.py invocations (or a retried slot left
    behind by a run that failed before the pipeline created its own output
    dir) from both picking the same version number and writing into it at
    once — mkdir is atomic, so a collision here raises and we just retry
    the next number instead of racing."""
    v = 1
    while True:
        candidate = root / f"{stem}_v{v}"
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            v += 1


def resolve_fixed_path(value: str) -> Path:
    """Resolve creds/taxonomy/prototypes to an absolute path anchored on
    this script's folder (not cwd, not the PDF's folder). The pipeline
    resolves relative --taxonomy/--prototypes against the INPUT PDF's own
    directory, with a same-dir-as-script fallback for taxonomy only, none
    for prototypes — since our PDFs live in Data/, a relative prototypes
    path would silently resolve to Data/pss_prototypes.json (and get
    created empty there) instead of the real one next to this script."""
    p = Path(value)
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()


def main():
    ap = argparse.ArgumentParser(description="Batch-run the PSS pipeline over a folder of PDFs")
    ap.add_argument("--input-dir", default="Data", help="folder of PDFs to process (default: Data)")
    ap.add_argument("--output-root", default="output", help="parent folder for per-PDF output dirs")
    ap.add_argument("--creds", default="service-account.json")
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--gemini2-location", default="us")
    ap.add_argument("--workers", type=int, default=5, help="parallel embedding workers per PDF")
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--no-gemini2", action="store_true")
    ap.add_argument("--taxonomy", default="pss_taxonomy.json")
    ap.add_argument("--prototypes", default="pss_prototypes.json")
    ap.add_argument("--accumulate-prototypes", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="page limit per PDF (forwarded as --limit)")
    ap.add_argument("--skip-existing", action="store_true",
                     help="skip a PDF if it already has a COMPLETED run (a <stem>_v*/run_report.json "
                          "on disk) — a prior run that crashed or was killed mid-way has no report, "
                          "so it is not skipped and gets a fresh _v(N+1) folder instead")
    ap.add_argument("--dry-run", action="store_true",
                     help="print the PDF -> output-folder plan and exit without calling the pipeline")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: --input-dir not found (or not a directory): {input_dir}")
        sys.exit(1)
    output_root = Path(args.output_root)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    creds_path = resolve_fixed_path(args.creds)
    taxonomy_path = resolve_fixed_path(args.taxonomy)
    prototypes_path = resolve_fixed_path(args.prototypes)
    for label, p in [("--creds", creds_path), ("--taxonomy", taxonomy_path)]:
        if not p.exists():
            print(f"ERROR: {label} file not found: {p}")
            sys.exit(1)
    print(f"Using creds={creds_path}\n      taxonomy={taxonomy_path}\n      prototypes={prototypes_path}\n")

    pdfs = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"No PDFs found in {input_dir}")
        return

    log_path = output_root / f"batch_run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    rows = []

    print(f"Found {len(pdfs)} PDFs in {input_dir}\n")

    def has_completed_run(stem: str) -> Path | None:
        """A version folder counts as done only if it has run_report.json —
        the last file the pipeline writes. Anything else (folder exists but
        no report) means a prior run died mid-way; treat it as not done."""
        for v in sorted(output_root.glob(f"{stem}_v*")):
            if (v / "run_report.json").exists():
                return v
        return None

    try:
        for i, pdf in enumerate(pdfs, 1):
            stem = pdf.stem
            done_dir = has_completed_run(stem) if args.skip_existing else None
            if done_dir:
                print(f"[{i}/{len(pdfs)}] SKIP (already completed): {pdf.name} -> {done_dir}")
                rows.append([pdf.name, str(done_dir), "skipped", 0])
                continue

            if args.dry_run:
                out_dir = next_version_dir(output_root, stem)
                print(f"[{i}/{len(pdfs)}] {pdf.name} -> {out_dir}  (dry-run, not executed)")
                rows.append([pdf.name, str(out_dir), "dry-run", 0])
                continue

            out_dir = reserve_output_dir(output_root, stem)
            print(f"[{i}/{len(pdfs)}] {pdf.name} -> {out_dir}")

            cmd = [
                sys.executable, str(PIPELINE_SCRIPT), str(pdf),
                "--creds", str(creds_path),
                "--project", args.project,
                "--location", args.location,
                "--gemini2-location", args.gemini2_location,
                "--output-dir", str(out_dir),
                "--workers", str(args.workers),
                "--taxonomy", str(taxonomy_path),
                "--prototypes", str(prototypes_path),
            ]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            if args.dpi:
                cmd += ["--dpi", str(args.dpi)]
            if args.no_gemini2:
                cmd += ["--no-gemini2"]
            if args.accumulate_prototypes:
                cmd += ["--accumulate-prototypes"]

            t0 = time.time()
            try:
                result = subprocess.run(cmd)
                elapsed = round(time.time() - t0, 1)
                status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            except Exception as e:
                elapsed = round(time.time() - t0, 1)
                status = f"FAILED (exception: {e})"
            print(f"    -> {status} in {elapsed}s\n")
            rows.append([pdf.name, str(out_dir), status, elapsed])
    except KeyboardInterrupt:
        print("\nInterrupted — writing log for PDFs processed so far.")
    finally:
        if not args.dry_run:
            with open(log_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["pdf_filename", "output_dir", "status", "seconds"])
                w.writerows(rows)

    n_fail = sum(1 for r in rows if str(r[2]).startswith("FAILED"))
    n_done = sum(1 for r in rows if r[2] not in ("dry-run",))
    if args.dry_run:
        print(f"\nBatch plan: {len(pdfs)} PDFs would be processed. Nothing was created or executed.")
    else:
        print(f"\nBatch complete: {n_done}/{len(pdfs)} PDFs accounted for, {n_fail} failed. Log: {log_path}")


if __name__ == "__main__":
    main()
