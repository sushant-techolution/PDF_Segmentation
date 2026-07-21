"""
Merge every per-PDF segments_summary.csv under an output folder into one
combined CSV — automates what was done manually for output/segments_summary_combined.csv.

Finds every segments_summary.csv nested under --output-root (one per PDF,
written by "Pss pipeline .py" / run_batch.py), writes a single header once,
then each PDF's rows in turn, with a blank line between PDFs so it's
easy to see where one PDF's data ends and the next begins.

Usage:
    python3 merge_summaries.py --output-root output
    python3 merge_summaries.py --output-root output --out output/combined.csv
"""

import argparse
import csv
from pathlib import Path


def merge(output_root: Path, out_path: Path) -> None:
    csv_paths = sorted(output_root.glob("*/segments_summary.csv"))
    if not csv_paths:
        print(f"No segments_summary.csv files found under {output_root}")
        return

    header = None
    blocks = []  # (pdf_dir_name, data_rows)

    for csv_path in csv_paths:
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            print(f"  skip (empty file): {csv_path}")
            continue
        file_header, data_rows = rows[0], rows[1:]
        if header is None:
            header = file_header
        elif file_header != header:
            print(f"  WARNING: header differs in {csv_path} — merging anyway, "
                  f"columns may not line up for this block")
        blocks.append((csv_path.parent.name, data_rows))

    if header is None:
        print("Nothing to merge (every file was empty).")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (name, rows) in enumerate(blocks):
            for row in rows:
                writer.writerow(row)
            if i != len(blocks) - 1:
                f.write("\n")

    total_rows = sum(len(rows) for _, rows in blocks)
    print(f"Merged {len(blocks)} PDFs, {total_rows} segment rows -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Merge per-PDF segments_summary.csv files into one")
    ap.add_argument("--output-root", default="output",
                     help="folder containing one subfolder per PDF, each with its own "
                          "segments_summary.csv (default: output)")
    ap.add_argument("--out", default=None,
                     help="path for the combined CSV (default: <output-root>/segments_summary_combined.csv)")
    args = ap.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_dir():
        print(f"ERROR: --output-root not found (or not a directory): {output_root}")
        raise SystemExit(1)

    out_path = Path(args.out) if args.out else output_root / "segments_summary_combined.csv"
    merge(output_root, out_path)


if __name__ == "__main__":
    main()
