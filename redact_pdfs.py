"""
redact_pdfs.py — Redact sensitive fields from PDF documents.

Scope: tax ID / EIN / SSN, phone numbers, email addresses — both as
standalone patterns and as values sitting next to a known field label
(TAXPAYER ID:, PHONE:, etc). Names, addresses, and dollar amounts are
out of scope; none of them have a fixed enough shape to redact reliably.

Uses PyMuPDF's redact-annotation API (add_redact_annot + apply_redactions),
which actually strips the underlying content in the region rather than
just drawing a box on top of it.

Two passes per page:
    1. Pattern pass — scan the page text for SSN/EIN/phone/email shapes.
    2. Label pass — find a field label's on-page position, then redact the
       value near it (same line, or the line below if the value wraps).

Never touches the source file — every output is a new "*_redacted.pdf"
alongside the original. Only redaction counts are logged, never the
matched text.

Usage:
    python redact_pdfs.py input.pdf [input2.pdf ...] [--out-dir DIR] [--dry-run]
    python redact_pdfs.py /path/to/folder --out-dir ./redacted [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

import fitz


# ── Pattern-based detection (standalone, no label required) ──────────────

PATTERNS = {
    "ssn":   re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ein":   re.compile(r"\b\d{2}-\d{7}\b"),
    # Requires a hyphen/period separator or parens around the area code —
    # a bare space between all three groups also matches things like a
    # budget code ("099 001 1000") and shouldn't count as a phone number.
    "phone": re.compile(
        r"\(\d{3}\)\s?\d{3}[-.]\d{4}\b|\b\d{3}[-.]\d{3}[-.]\d{4}\b"
    ),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z]{2,}\b"),
}

# ── Label-based detection (find the label, redact the value near it) ─────
# Only labels for the in-scope categories — no CONTACT:/ADDRESS: since
# names and addresses aren't redacted.
FIELD_LABELS = [
    "TAXPAYER ID:", "TAX ID:", "TAXPAYER ID", "TAX ID",
    "EIN:", "EIN", "SSN:", "SSN",
    "PHONE:", "PHONE", "PH:", "TEL:",
    "EMAIL:", "EMAIL", "E-MAIL:", "E-MAIL",
]

MAX_LABEL_VALUE_GAP = 14  # points; how far below a label its value may sit
                          # when printed on the next line (calibrated to
                          # this document set's form layouts)


def _is_value_shaped(word_text):
    """True if this could plausibly be part of a phone/tax-ID/SSN/EIN/email
    value — contains a digit or "@". Every in-scope category is inherently
    numeric or email-shaped, while field labels in these forms are always
    plain alphabetic text. This is what stops a colon-less label (a bare
    "PHONE" heading, or "SSN" sitting next to a checkbox) from being
    captured as if it were a value."""
    return any(ch.isdigit() for ch in word_text) or "@" in word_text


def _filter_value_words(words):
    """Keep only value-shaped words, stopping at the first colon-terminated
    token. The colon check matters on its own too: without it, a blank
    field's scan can wander into the next field's label and grab its real
    value instead (an empty "PHONE:" reaching all the way to a following
    "CONTRACT NUMBER: 099 001 1000")."""
    out = []
    for w in words:
        if w[4].rstrip().endswith(":"):
            break
        if _is_value_shaped(w[4]):
            out.append(w)
    return out


def _words_on_line(words, y0, y1, x_min):
    """Value-shaped words whose vertical center falls in [y0, y1] and start
    at/after x_min, in left-to-right order, stopping at the next label."""
    out = []
    for w in words:
        wx0, wy0, wx1, wy1, text = w[0], w[1], w[2], w[3], w[4]
        cy = (wy0 + wy1) / 2
        if y0 <= cy <= y1 and wx0 >= x_min - 1:
            out.append(w)
    out.sort(key=lambda w: w[0])
    return _filter_value_words(out)


def _words_below(words, label_rect, max_gap):
    """Value-shaped words on the next line down, roughly under the label's
    x-position, stopping at the next label."""
    out = []
    for w in words:
        wx0, wy0, wx1, wy1, text = w[0], w[1], w[2], w[3], w[4]
        if wy0 >= label_rect.y1 - 1 and wy0 <= label_rect.y1 + max_gap:
            if wx0 <= label_rect.x1 + 40:  # roughly aligned under the label
                out.append(w)
    out.sort(key=lambda w: (w[1], w[0]))
    return _filter_value_words(out)


def find_redaction_regions(page):
    """Returns list of (fitz.Rect, category) for everything to redact on
    this page. category is one of: ssn, ein, phone, email, labeled_value."""
    regions = []
    words = page.get_text("words")  # (x0,y0,x1,y1,text,block,line,word_no)

    # --- Pattern pass: search full page text, then locate each match's
    # on-page rect via search_for so we redact the exact visual location.
    full_text = page.get_text()
    for category, pat in PATTERNS.items():
        for m in pat.finditer(full_text):
            matched = m.group(0)
            for rect in page.search_for(matched):
                regions.append((rect, category))

    # --- Label pass: find each label's position, then its adjacent value.
    for label in FIELD_LABELS:
        for label_rect in page.search_for(label):
            # Same line, to the right of the label
            same_line = _words_on_line(
                words, label_rect.y0 - 1, label_rect.y1 + 1, label_rect.x1
            )
            value_words = same_line
            if not value_words:
                # Fall back to the line below (label-alone-then-value layout)
                value_words = _words_below(words, label_rect, MAX_LABEL_VALUE_GAP)
            if not value_words:
                continue
            x0 = min(w[0] for w in value_words)
            y0 = min(w[1] for w in value_words)
            x1 = max(w[2] for w in value_words)
            y1 = max(w[3] for w in value_words)
            regions.append((fitz.Rect(x0, y0, x1, y1), "labeled_value"))

    return regions


def redact_pdf(input_path: Path, output_path: Path, dry_run: bool = False) -> dict:
    """Redact one PDF. Returns per-category counts (never the matched text)."""
    doc = fitz.open(str(input_path))
    counts = {"ssn": 0, "ein": 0, "phone": 0, "email": 0, "labeled_value": 0}

    for page in doc:
        regions = find_redaction_regions(page)
        for rect, category in regions:
            counts[category] += 1
            if not dry_run:
                page.add_redact_annot(rect, fill=(0, 0, 0))
        if not dry_run and regions:
            # images=0 keeps embedded images that don't overlap a redaction
            # box; text/drawings under the box are permanently removed.
            page.apply_redactions()

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
    doc.close()
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="PDF file(s) or a directory of PDFs")
    ap.add_argument("--out-dir", default=None,
                     help="Output directory (default: alongside each input file)")
    ap.add_argument("--dry-run", action="store_true",
                     help="Report counts only; do not write redacted files")
    args = ap.parse_args()

    pdf_paths = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            pdf_paths.extend(sorted(p.rglob("*.pdf")) + sorted(p.rglob("*.PDF")))
        elif p.is_file():
            pdf_paths.append(p)
        else:
            print(f"WARNING: not found, skipping: {p}", file=sys.stderr)

    if not pdf_paths:
        print("No PDF files found.", file=sys.stderr)
        sys.exit(1)

    grand_total = {"ssn": 0, "ein": 0, "phone": 0, "email": 0, "labeled_value": 0}
    for src in pdf_paths:
        if args.out_dir:
            dst = Path(args.out_dir) / f"{src.stem}_redacted.pdf"
        else:
            dst = src.with_name(f"{src.stem}_redacted.pdf")

        counts = redact_pdf(src, dst, dry_run=args.dry_run)
        for k in grand_total:
            grand_total[k] += counts[k]

        total = sum(counts.values())
        mode = "[DRY RUN] would redact" if args.dry_run else "redacted"
        print(f"{src.name}: {mode} {total} region(s) "
              f"(ssn={counts['ssn']} ein={counts['ein']} phone={counts['phone']} "
              f"email={counts['email']} labeled_value={counts['labeled_value']})"
              + ("" if args.dry_run else f" -> {dst.name}"))

    print()
    print(f"TOTAL across {len(pdf_paths)} file(s): {sum(grand_total.values())} region(s) "
          f"({grand_total})")


if __name__ == "__main__":
    main()
