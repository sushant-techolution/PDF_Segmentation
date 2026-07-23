"""
PSS Pipeline v2 - Page Stream Segmentation & Classification
=============================================================
What changed from v1 and why (traceable to real run results):

  FIX 1  Bookmark depth filter [CRITICAL, fixed 57/63 FPs]
         Root cause: the IRS 990 block in PDF-003 had L3 bookmarks on
         every single page (67-123), each firing struct=1.0 and creating
         a 1-page segment. Only L1 (top-level) TOC entries are treated
         as document-boundary signals. L2+ are ignored.
         Impact: precision 19% → ~71% with recall staying 100%.

  FIX 2  Consecutive same-title bookmark suppression
         Root cause: PDF-003 had two consecutive L1 entries both titled
         "AGREEMENT" at p3 (US Letter) and p4 (A4), same logical
         document split across two paper sizes. A bookmark fires on both,
         causing an FP at p4. Fix: if an L1 bookmark title is identical
         to the immediately preceding L1 title AND the page-format
         change is the only other signal (no embedding drop), suppress it.

  FIX 3  Landscape-back-to-portrait false positive (p38/p40)
         Root cause: landscape statement (p38-39) returns to portrait at
         p40, triggering struct=1.0 again when really it's a continuation
         of the same financial statements block. Fix: a format change from
         the SAME parent block (where the format changed <=5 pages ago)
         is downweighted to 0.4, not the full 1.0.

  FIX 4  Embedding-only FP suppression for financial-block interiors
         Pages 34, 56, 64 are embedding-only FPs (section breaks inside
         audited financials / 990). Fix: raise MIN_ABSOLUTE_DROP from
         0.05 to 0.07 and require |z| > 2.0 (not 1.5) for moderate flag.

  FIX 5  Classification: engineering drawings & visual-first fallback
         Visual categories (engineering drawings, certificates,
         insurance certificates) have no reliable text keywords. Fix:
         use embedding cosine similarity to CATEGORY PROTOTYPE VECTORS
         built from the first-page embedding of known examples, as a
         secondary classifier after keyword matching fails. For pages
         with no text layer at all (scanned drawings), classify from
         visual signals only: landscape + image-only = Engineering
         Drawing; image + text density near zero = certificate candidate.

  FIX 6  PARALLEL embedding calls using ThreadPoolExecutor
         v1 embedded pages sequentially. Each call takes ~1.5-2s
         (mostly network round-trip). For 189 pages: 189 × 1.7s ≈ 321s.
         With N workers the theoretical speedup is ~N×. Practically,
         Vertex AI rate limits constrain this, but 8-16 workers is safe
         for the default quotas and gives ~4-6× real speedup.
         After all parallel calls complete, similarity computation runs
         in correct page order as before, parallelism is only on the
         embedding extraction phase, not the analysis phase.

  FIX 7  Token count split (image vs text) for gemini-embedding-2
         v1 reported all combined tokens under "image_tokens" because
         gemini-embedding-2 returns one total. v2 estimates the text
         portion from character count and reports both separately with
         clear "estimated" flags on the text portion.

  FIX 8  Embedding score double-counted moderate drops [PDF-01 p9]
         Root cause: gemini-embedding-2 returns the SAME vector for image
         and text (see _embed_one_gemini2), so a single moderate similarity
         drop was being scored as two independent 0.5 signals, summing to
         1.0 and crossing the boundary threshold on its own. Fix: if image
         sim == text sim, count it once.

  FIX 9  L2 bookmark alone shouldn't be trusted as a boundary [PDF-01 p20]
         Root cause: p20's TOC entry is a level-2 sub-heading of the same
         2-page form as p19, not a new document, but FIX1's L1/L2 trust
         rule (added for PDF-003's 990 block) fired anyway. Fix: L2 entries
         now need a corroborating signal (format change / numbering reset /
         embedding drop) on the same page; L1 stays trusted alone.

  FIX 10 Missed boundary at "EXHIBIT A" page [PDF-01 p16]
         No existing signal caught this, same page format, no bookmark,
         no numbering reset. Added a check for an all-caps title line
         (EXHIBIT/SCHEDULE/ATTACHMENT/APPENDIX/ANNEX/ADDENDUM) at the very
         top of the page.

  FIX 11 Rescanned page misread as a format change [PDF-01 p13]
         p13 is a re-scanned copy of the same signature page as p11/p12,
         just ~2-3% smaller, which fell outside the old fixed 8pt size
         tolerance and got bucketed as a new page format. Tolerance is now
         relative (~3%) instead of a flat 8pt.

  FIX 12 Document Category was never actually output
         CATEGORY_KEYWORDS/CATEGORY_ORDER were defined but unused; the CSV/
         JSON had no category column at all. Added a `category` field per
         type in pss_taxonomy.json and wired it through to output.

  FIX 13 Same-title bookmark suppression fired on non-adjacent repeats
         [2 validation PDFs, repeat several pages later]
         Root cause: the suppression only compared a title against the
         most recent trusted bookmark, no matter how many pages back that
         was. A generic reused title ("ADVICE OF AWARD") ended up killing
         a genuinely different document several pages later. Fix: only
         suppress when the repeat is within ADJACENT_BOOKMARK_GAP pages
         and nothing else on the page corroborates it (numbering reset,
         drawing spike). A strong embedding drop later in scoring can
         still reinstate a suppressed bookmark.

  FIX 14 Duplicate/leftover pages created false document splits
         [byte-identical page carrying a different bookmark title]
         Nothing compared page content, only metadata, so a duplicate page
         with its own auto-generated bookmark label read as a new
         document. Fix: compare each page's text against the one before
         it; anything at or above DUPLICATE_TEXT_SIMILARITY is a duplicate
         and never produces a boundary signal, regardless of what
         bookmark/format/title-anchor it carries. Also covers
         instructions.md's Rule 3 (consecutive identical pages merge;
         draft/signed pairs aren't duplicates since they differ in
         signature text).

  FIX 15 Format-return downweight applied across unrelated, already-closed
         documents [US Letter → A4 → US Letter, the
         return wrongly downweighted because a different document had
         already opened and closed in between]
         The "returning within 5 pages" check had no idea a document
         boundary happened inside that window. Fix: the lookback resets at
         any structural anchor instead of a flat page count, so "returning"
         is scoped to the current document, not a fixed window.
         Trade-off: a bare, uncorroborated format change now counts as an
         anchor too, which means a same-document format excursion with no
         bookmark also loses its downweight, nothing at this stage can
         tell the two apart without reading page content. Checked the
         whole corpus: doesn't actually happen anywhere, left as is.

  FIX 16 Page-number label and value split across two lines never matched
         [6 pages across 5 validation PDFs:
         "PAGE:" and "1" printed on separate lines]
         PAGE_PAT_TOTAL only matches "page X of Y" on one line, and the
         bare-digit fallback was picking up an unrelated "DEPT: 826" code
         first. Fix: parse_numbering_split_label looks for a PAGE/PG label
         immediately followed by a bare digit, reads the clipped
         header/footer band, not full-page text, since PyMuPDF's default
         reading order isn't reliable across these multi-column forms (one
         page had the label and value 48 lines apart). One page
         still isn't caught, its value sits behind an
         unrelated field even in a wider band, left unfixed rather than
         special-cased for one page.

  FIX 17 Numbering reset was invisible with no prior value to compare
         against [same 6 pages as FIX 16]
         num_reset only ran when both a current AND a prior numbering
         value existed, so the first numbered page after an unnumbered
         cover sheet never registered. Fix: a fresh count of 1 is always a
         reset candidate on its own. Still weight 0.8 (FIX 1: "strong but
         not 1.0 alone"), so it still needs another signal to flip a
         boundary by itself.

  FIX 18 No detector for memo/email correspondence headers
         [7 pages across 5 PDFs: memos opening DATE:/TO:, Outlook forwards
         opening From:/Sent:/To:/Subject:, standalone "Memorandum" titles]
         Only EXHIBIT/SCHEDULE-style title anchors existed; nothing caught
         the shape of a memo or email header. Fix: detect_correspondence_
         header flags >=2 of {TO, FROM, DATE, RE, SUBJECT, CC, SENT} as
         line-starts in the first 15 lines, or a standalone "Memorandum"
         line, order-independent since real examples show the labels in
         different orders. Checked against the full corpus: 27 pages
         fire, 26 are real boundaries, and the one exception is already
         killed by FIX 14's duplicate check.

  FIX 19 Drawing-density drop wasn't checked, only spikes
         Root cause: a fillable-form cover page (vector-drawn field boxes,
         drawing_count in the hundreds) immediately followed by a plain
         machine-printed report (drawing_count 0) is exactly as strong a
         template-change signal as the reverse, but drawing_spike only
         fired going up. Fix: drawing_density_drop mirrors the same
         3x/floor-200 thresholds in the other direction, the page
         collapses to under a third of a >200-drawing predecessor.
         Generalizes to any pair of adjacent pages where one is a drawn
         form or scan and the other isn't.

  FIX 20 Header/footer number search used a fixed 12%-height band
         Root cause: a "PAGE:" label matched but its value on the next
         line sat a few points past the clip line on one page's
         particular vertical spacing, so the split-label pattern found
         the label and silently missed the value. Fix:
         find_labeled_numbering retries the keyword-anchored patterns
         (page X of Y, PAGE:/value) in a widening band before giving up.
         The unanchored bare-digit fallback stays on the original tight
         band, without a keyword anchor, widening it would start
         matching arbitrary numbers in body text.

  FIX 21 A numbering reset to a document's own first page scored the
         same as a mid-document renumber
         Root cause: several PDFs correctly detected a numbering reset
         on a page that opened a new, unbookmarked sub-document, but
         0.8 alone never crosses the boundary threshold and nothing
         else corroborated it. A reset that happens because a page is
         the FIRST one to carry any numbering since the last structural
         anchor is a materially different case than a reset midway
         through an already-numbered run. That second case is exactly what
         page_numbering_reset's 0.8 cap exists to stay cautious about,
         since nothing established a sequence for a first sighting to
         be "resetting" from. Fix: numbering_fresh_start tracks whether
         any numbering has been seen since the last anchor (the same
         anchor-scoping FIX 15 already uses for format returns) and
         scores 0.9 instead of 0.8 for a first sighting.

  FIX 22 No signal for a change in how page numbers are presented
         Root cause: same cases as FIX 21, an earlier page's numbering
         sits in the footer as a combined "X of Y" phrase, the new
         sub-document's sits in the header as a split label/value pair.
         That's a physically different numbering scheme, independent of
         the numeric value, and is itself evidence of a different source
         template. Fix: page_numbering_scheme_bonus adds 0.2 when a
         numbering reset co-occurs with a (region, pattern-kind) change
         from the last numbering scheme seen anywhere in the document.

  FIX 23 Letterhead-change detector, re-attempted with a gate
         An earlier version of this (comparing token-set similarity
         between any two consecutive header/footer bands) was built and
         shelved: ~17% false-positive rate, all inside flowing legal-prose
         documents that don't have a letterhead at all, the comparison
         was firing on running headers and case captions that just happen
         to differ page to page. Root cause wasn't the similarity
         threshold; it was comparing pages that were never letterhead
         pages to begin with. Fix: only extract a comparison block when a
         page's header/footer actually contains an address, a phone/fax
         line, or a website (find_letterhead_block), legal prose and
         correspondence headers have none of these, so they're excluded
         before any comparison happens rather than filtered after.
         Checked against the full corpus: only 5 of 218 pages worth of
         consecutive-page pairs even have a letterhead on both sides;
         same-organization pairs score 1.0 similarity, different-
         organization pairs score 0.0-0.19, a wide, clean margin for the
         0.4 cutoff. One target case (a private engineering firm
         handing off to a government agency) needed this; the
         other 4 eligible pairs were either the same organization
         (correctly not flagged) or already a boundary via an existing
         bookmark (flagging them again is redundant, not harmful).

  FIX 24 Confidence based on evidence, not on which branch fired
         Confidence was hardcoded per cascade branch (Pass 1 always said
         HIGH, Pass 4 said MEDIUM at >=2 keyword hits) so it measured
         nothing, and it rated only the type label while sitting next to
         the page range. Against the 232 colour-coded segments in the
         labelled validation set the tiers came out backwards: LOW was
         95% correctly segmented, HIGH only 84.8%.
         Now reports segmentation and classification confidence
         separately and headlines the weaker. Segmentation confidence
         uses the margin of total_score over BOUNDARY_THRESHOLD, scored
         on the weaker of the segment's two edges, plus three checks
         inside the segment that catch merges (which edges cannot see):
         a near-miss page, an unreadable page, and a segment more than
         2x this document's median length. Any of those forces LOW and
         names the page.
         Reporting only - WEIGHTS, score_all and build_segments are
         untouched, so no boundary moves.
         Validated against per-page data from two labelled PDFs: all 24
         mis-segmented rows land in LOW, and HIGH and MEDIUM are both
         100% correct. 97.6% recall across the full labelled set.
         Full analysis, caveats, and the rules that were tried and
         rejected: confidence_integrity.md

  FIX 25 Segment cohesion signal (embedding, no new API cost)
         The commonest miss is the next document's opening page being
         swallowed by the end of the previous segment. It has no format
         change (the cover shares the prior orientation), no bookmark,
         and an embedding drop too small to register against the whole
         PDF, so nothing fires. But inside its own segment it stands out:
         the other pages sit at 0.94-0.97 similarity and it sits at 0.88.
         The existing detector z-scores against the whole document, so a
         dip that is glaring inside one segment can sit on the document
         median and score nothing. segment_cohesion_break measures each
         segment against its OWN baseline instead, needs no vocabulary or
         page-count rule, and feeds the confidence reason. Position
         mattered far more than depth in validation: a dip on the last
         page meant a foreign page 9 times out of 9, a dip mid-segment
         only 3 times in 14.

  FIX 25b Act on a trailing cohesion break (changes boundaries)
         FIX 25 only reported. This turns a break on a segment's LAST
         page into a real boundary, cutting the foreign page off into
         its own segment. Scoped strictly to the last page for the
         reason above: simulated over the labelled segments, splitting
         on last-page breaks fixed 6 segments and broke 0, while enabling
         mid-segment splits at any threshold immediately broke a correct
         one. So mid-segment breaks stay advisory (confidence only).
         This is the first change since FIX 24 that moves boundaries, so
         the earlier "reporting only" guarantee no longer holds and the
         232-segment baseline must be re-established on the next full run.
         Validated on two labelled PDFs: 8 splits, 0 correct segments
         lost. Two of the eight are merge splits with no ground truth for
         the true cut point, so they are counted as unverified, not wins.

Install:
    pip install google-genai google-auth requests pymupdf numpy

Usage:
    python pss_pipeline.py input.pdf \\
        --creds service-account.json \\
        --project YOUR_PROJECT_ID \\
        --output-dir ./output \\
        --workers 8 \\
        --limit 20
"""

import argparse
import base64
import csv
import difflib
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import fitz
import numpy as np

from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

# Classification module, self-contained, config-driven
sys.path.insert(0, str(Path(__file__).parent))
from pss_classifier import Classifier


# ═══════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═══════════════════════════════════════════════════════════════════

PRICING = {
    "gemini-embedding-2_text_per_token":  0.20 / 1_000_000,
    "gemini-embedding-2_image_per_token": 0.45 / 1_000_000,
    "multimodalembedding@001_per_image":  0.0001,
    "gemini-embedding-001_text_per_token": 0.15 / 1_000_000,
}

ESTIMATED_IMAGE_TOKENS_PER_PAGE = 1290

# Signal weights, grounded in validation results, see inline evidence
WEIGHTS = {
    "structural_change":         1.0,   # bookmarks (L1 only), size/orient changes: 0 FP observed
    "structural_change_return":  0.4,   # format returning to a prior format within 5 pages (FIX 3)
    "page_numbering_reset":      0.8,   # X/Y total gated: strong but not 1.0 alone
    "page_numbering_fresh_start": 0.9,  # reset with no prior numbering since the last anchor (FIX 21)
    "page_numbering_scheme_bonus": 0.2, # region/pattern change alongside a reset (FIX 22)
    "embedding_high_z":          1.0,   # |z|>4.0: near-perfect precision, safe standalone
    "embedding_moderate_each":   0.5,   # image + text scored independently, summed
    "drawing_density_spike":     0.3,   # weak, exploratory
    "drawing_density_drop":      0.3,   # collapse back to near-zero after a dense page (FIX 19)
    "letterhead_change":         0.5,   # org-to-org letterhead swap, gated (FIX 23)
    "consecutive_flag_penalty": -0.5,   # real boundaries don't sustain across 2 pages
}

BOUNDARY_THRESHOLD  = 1.0
Z_WINDOW            = 10
Z_THRESHOLD         = -2.0   # raised from -1.5 (FIX 4)
Z_HIGH_THRESHOLD    = -4.0
MIN_STD             = 0.02
MIN_ABSOLUTE_DROP   = 0.07   # raised from 0.05 (FIX 4)

# FIX 20: widen the header/footer number-label search if the tight band
# comes up empty, instead of a single fixed clip height.
NUMBERING_BAND_FRACTIONS = (0.12, 0.20, 0.30)

# FIX 13: max page gap for a repeated bookmark title to still count as
# "same doc, different paper size" instead of two unrelated documents.
# FIX 2's own example (PDF-003) is a next-page repeat, so 1 covers it.
ADJACENT_BOOKMARK_GAP = 1

# FIX 14: similarity ratio above which two consecutive pages count as
# duplicates. The one confirmed duplicate in the validation set
# scores 1.0000; the closest real near-miss, two documents sharing a
# template, scores 0.52 - 0.95 leaves wide margin either way.
DUPLICATE_TEXT_SIMILARITY = 0.95

# ── Category taxonomy ────────────────────────────────────────────────
# Keyword lists used for text-based classification
CATEGORY_KEYWORDS = {
    "Tax Filing (IRS 990)": [
        "form 990", "schedule a (form 990)", "schedule b (form 990)",
        "schedule c (form 990)", "schedule d (form 990)", "schedule g (form 990)",
        "schedule i (form 990)", "schedule j (form 990)", "schedule o (form 990)",
        "schedule r (form 990)", "omb no. 1545", "exempt from income tax",
    ],
    "Award / Financial": [
        "advice of award", "change order request", "award revision",
        "engineer's estimate", "change order", "financial management system",
    ],
    "Compliance / Regulatory": [
        "responsibility determination", "insurance certificate",
        "omb approval", "vendor name check", "vendor responsibility",
        "late registration", "char500", "certification regarding",
        "apt waiver", "capital encumbrance",
    ],
    "Contract / Agreement": [
        "agreement", "affirmation", "contract signature", "amendment",
        "budget amendment", "scope of work", "contract budget",
    ],
    "Financial Statements": [
        "consolidated statement", "auditor's report",
        "statement of financial position", "statement of functional expenses",
        "schedule of expenditures of federal awards", "independent auditor",
        "oca list", "notes to consolidated",
    ],
}
CATEGORY_ORDER = list(CATEGORY_KEYWORDS.keys()) + ["Engineering Drawing",
                                                     "Certificate / Insurance",
                                                     "Supporting / Misc"]

# ── Visual classification profiles ──────────────────────────────────
# For segments where keyword matching fails, these heuristics classify
# from page-level visual/structural signals only.
VISUAL_PROFILES = {
    "Engineering Drawing": {
        "min_landscape_fraction": 0.7,
        "max_text_density": 0.1,   # text_chars / page_area proxy
        "min_image_fraction": 0.3,
    },
    "Certificate / Insurance": {
        "max_page_count": 3,       # certs are short
        "min_image_count_per_page": 1,
        "required_keywords_any": ["certificate", "insur", "certif", "acord"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# 2. COST TRACKER (thread-safe append)
# ═══════════════════════════════════════════════════════════════════

import threading

class CostTracker:
    def __init__(self):
        self._rows = []
        self._lock = threading.Lock()

    def record(self, page_num, method, text_tokens, image_tokens,
               text_estimated, image_estimated, cost):
        with self._lock:
            self._rows.append({
                "page_num": page_num, "method": method,
                "text_tokens": text_tokens, "image_tokens": image_tokens,
                "text_tokens_estimated": text_estimated,
                "image_tokens_estimated": image_estimated,
                "cost_usd": cost,
            })

    def summary(self):
        with self._lock:
            rows = list(self._rows)
        total_cost = sum(r["cost_usd"] for r in rows)
        total_text = sum(r["text_tokens"] for r in rows)
        total_img  = sum(r["image_tokens"] for r in rows)
        any_est    = any(r["text_tokens_estimated"] or r["image_tokens_estimated"] for r in rows)
        by_method  = {}
        for r in rows:
            m = by_method.setdefault(r["method"], {"pages": 0, "cost_usd": 0.0})
            m["pages"] += 1; m["cost_usd"] += r["cost_usd"]
        return {
            "total_calls": len(rows),
            "total_text_tokens": total_text,
            "total_image_tokens": total_img,
            "total_cost_usd": round(total_cost, 6),
            "any_token_counts_estimated": any_est,
            "estimation_note": (
                "Some token counts ESTIMATED, verify against GCP Billing."
                if any_est else "All token counts measured from API responses."
            ),
            "cost_by_method": by_method,
        }


# ═══════════════════════════════════════════════════════════════════
# 3. AUTH
# ═══════════════════════════════════════════════════════════════════

def build_credentials(creds_path):
    return service_account.Credentials.from_service_account_file(
        creds_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])

def build_client(credentials, project_id, location):
    return genai.Client(vertexai=True, project=project_id,
                         location=location, credentials=credentials)

def build_image_session(credentials):
    return AuthorizedSession(credentials)


# ═══════════════════════════════════════════════════════════════════
# 4. STAGE 0 - DETERMINISTIC FEATURES
# ═══════════════════════════════════════════════════════════════════

PAGE_PAT_TOTAL = [
    re.compile(r'\bpage\s+(\d+)\s+of\s+(\d+)\b', re.IGNORECASE),
    re.compile(r'\bpg\.?\s*(\d+)\s*/\s*(\d+)\b', re.IGNORECASE),
]
PAGE_PAT_BARE = [
    re.compile(r'^-\s*(\d+)\s*-$'),
    re.compile(r'^\s*(\d+)\s*$'),
]

# FIX 16: matches a page-number label on its own line ("PAGE:" then "1"
# on the next). Won't collide with a nearby "DEPT: 826" field since that's
# not a page label. Left out bare "P" on purpose, too generic (checkbox/
# initial columns use it) and none of the real cases needed it.
PAGE_LABEL_PAT = re.compile(r'^(PAGE|PG)\.?:?\s*$', re.IGNORECASE)

# FIX 23: an org's actual letterhead, an address block, a phone/fax line,
# or a website, rather than any header/footer text in general. Gating on
# one of these three keeps this from firing on flowing legal prose or
# running headers that have no letterhead to begin with (see FIX 23's
# note in the fix log for why the naive version of this was never shipped).
LETTERHEAD_ADDRESS_PAT = re.compile(r',\s*[A-Za-z .]+\s+\d{5}(-\d{4})?\b')
LETTERHEAD_PHONE_PAT = re.compile(
    r'\b(?:[TF]|TEL|FAX)[:.]?\s*\d{3}[.\-]\d{3}[.\-]\d{4}\b', re.IGNORECASE)
LETTERHEAD_WEB_PAT = re.compile(r'\bwww\.\S+\.\w+', re.IGNORECASE)
# Generic address vocabulary and English stopwords only, no city or state
# names. An earlier version included the city and state the
# validation set happens to come from, but that tied the detector to one
# locality for no real benefit: re-tested against
# the same validation pairs with city names left in the comparison instead
# of stripped, and the separation (0.048-0.263 for a real org change vs.
# 1.000 for the same org) held up fine without them.
LETTERHEAD_STOPWORDS = {
    "the", "of", "to", "and", "in", "for", "a", "on", "at",
    "street", "st", "avenue", "ave", "road", "rd", "floor", "suite",
    "room", "blvd", "boulevard", "drive", "dr", "building",
}
LETTERHEAD_SIMILARITY_THRESHOLD = 0.4

def is_blank(page, text_thresh=15, img_area_thresh=0.02):
    if len(page.get_text().strip()) > text_thresh: return False
    pa = page.rect.width * page.rect.height
    ia = 0.0
    for img in page.get_images(full=True):
        try:
            b = page.get_image_bbox(img[0])
            if b: ia += b.width * b.height
        except Exception:
            # get_image_bbox raises ValueError("bad image name") on some
            # embedded image formats, common in engineering drawing PDFs
            # with unusual compression or inline images. Treat as non-zero
            # image area (i.e. page is NOT blank) to be safe.
            ia += pa * img_area_thresh + 1
    if pa > 0 and ia/pa > img_area_thresh: return False
    return len(page.get_drawings()) <= 5

def size_bucket(w, h):
    ls, ss = max(w,h), min(w,h)
    if ls > 1000: return "WIDE"
    # FIX 11: relative tolerance instead of flat 8pt, rescanned signature
    # pages drift ~2-3% and were falling into OTHER. US_LETTER/A4 are ~6%
    # apart so 3% still keeps them distinct.
    def close(a, b, rel=0.03):
        return abs(a-b) <= max(8, b*rel)
    if close(ls,792) and close(ss,612): return "US_LETTER"
    if close(ls,842) and close(ss,595): return "A4"
    return "OTHER"

def page_fmt(page):
    w, h = round(page.rect.width,1), round(page.rect.height,1)
    orient = "landscape" if w > h else "portrait"
    return {"w":w,"h":h,"orient":orient,"bucket":size_bucket(w,h),
            "label":f"{size_bucket(w,h)}_{orient}"}

def hf_text(page, frac=0.12):
    r = page.rect; bh = r.height * frac
    ht = page.get_text(clip=fitz.Rect(r.x0,r.y0,r.x1,r.y0+bh)).strip()
    ft = page.get_text(clip=fitz.Rect(r.x0,r.y1-bh,r.x1,r.y1)).strip()
    return ht, ft

def parse_numbering_total(text, max_p=999):
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        for pat in PAGE_PAT_TOTAL:
            m = pat.search(line)
            if m:
                c,t = int(m.group(1)), int(m.group(2))
                if c <= max_p: return c,t
    return None

def parse_numbering_bare(text, max_p=999):
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        for pat in PAGE_PAT_BARE:
            m = pat.search(line)
            if m:
                c = int(m.group(1))
                if c <= max_p: return c

def parse_numbering_split_label(text, max_p=999):
    """Find a PAGE/PG label immediately followed by a bare digit on the
    next line. Pass the clipped header/footer band (hf_text), not the
    full page - PyMuPDF's reading order can put a label and its value
    dozens of lines apart on a multi-column form even though they sit
    right next to each other visually. A small clipped region is reliable.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i in range(len(lines) - 1):
        if PAGE_LABEL_PAT.match(lines[i]):
            m = re.match(r'^(\d+)$', lines[i+1])
            if m:
                c = int(m.group(1))
                if c <= max_p:
                    return c
    return None

def find_labeled_numbering(page, prev_numbering, max_p=999):
    """Look for an explicit page-number label (a "page X of Y" phrase, or
    a PAGE/PG label followed by its value) in the header/footer bands,
    widening the clip if the tightest one comes up empty (FIX 20). Only
    the two keyword-anchored patterns get this treatment, the bare-digit
    fallback in extract_det_features stays on the original tight band,
    since without a keyword to anchor it a wider window starts matching
    unrelated numbers in body text.

    Returns (numbering, scheme) where numbering is (current, total_or_None)
    and scheme is a (region, pattern_kind) tag used to detect a change in
    how numbering is presented (FIX 22), or (None, None) if nothing hit.
    """
    for frac in NUMBERING_BAND_FRACTIONS:
        ht, ft = hf_text(page, frac=frac)
        for region, text in (("footer", ft), ("header", ht)):
            total = parse_numbering_total(text, max_p=max_p)
            if total:
                return total, (region, "total")
        for region, text in (("footer", ft), ("header", ht)):
            split_val = parse_numbering_split_label(text, max_p=max_p)
            if split_val is not None:
                prior_total = prev_numbering[1] if prev_numbering else None
                return (split_val, prior_total), (region, "split_label")
    return None, None

def find_letterhead_block(page):
    """Look for an org's letterhead, an address, a phone/fax line, or a
    website, in the header/footer bands, widening the clip the same way
    find_labeled_numbering does (FIX 20/23): a letterhead's contact block
    routinely sits a bit below a tight 12% clip. Returns the matched text
    (used for comparison against the previous page's) or None.
    """
    for frac in NUMBERING_BAND_FRACTIONS:
        ht, ft = hf_text(page, frac=frac)
        for text in (ht, ft):
            lines = [l.strip() for l in text.splitlines() if l.strip()][:8]
            blob = " ".join(lines)
            if (LETTERHEAD_ADDRESS_PAT.search(blob) or LETTERHEAD_PHONE_PAT.search(blob)
                    or LETTERHEAD_WEB_PAT.search(blob)):
                return blob
    return None

def letterhead_similarity(a, b):
    """Token-set (Jaccard) similarity between two letterhead blocks, common
    city/state/floor words stripped out so two agencies on different streets
    don't read as similar just because they share a city and state line."""
    def tokens(blob):
        words = re.findall(r'[a-z0-9]+', blob.lower())
        return {w for w in words if w not in LETTERHEAD_STOPWORDS and len(w) > 1}
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

# FIX 10: sub-document title markers (EXHIBIT A, SCHEDULE 1, ...). Only
# matched against the first content line so a mid-sentence body reference
# like "...attached hereto as Exhibit A..." doesn't fire.
TITLE_ANCHOR_PAT = re.compile(
    r'^(EXHIBIT|SCHEDULE|ATTACHMENT|APPENDIX|ANNEX|ADDENDUM)\b')

def detect_title_anchor(text, max_lines=3):
    """text: already-extracted, stripped page text (avoids re-extracting)."""
    for line in [l.strip() for l in text.splitlines() if l.strip()][:max_lines]:
        if TITLE_ANCHOR_PAT.match(line) and line.isupper():
            return line[:80]
    return None

# FIX 18: memo/email correspondence header (TO:/FROM:/DATE:/RE:/SUBJECT:/
# CC:/SENT: labels, or a standalone "Memorandum" title). Order doesn't
# matter, a memo opens with DATE:/TO:, an Outlook forward opens with a
# name before From:/Sent:/To:/Subject:, so this just checks for >=2
# distinct labels anywhere in the first max_lines.
CORRESPONDENCE_LABEL_PAT = re.compile(
    r'^(TO|FROM|DATE|RE|SUBJECT|CC|SENT)\s*:', re.IGNORECASE)
MEMO_TITLE_PAT = re.compile(r'^MEMORANDUM\s*$', re.IGNORECASE)

def detect_correspondence_header(text, max_lines=15):
    """text: already-extracted, stripped page text. Returns the sorted
    list of distinct labels found (truthy iff >=2), for diagnostics."""
    lines = [l.strip() for l in text.splitlines() if l.strip()][:max_lines]
    labels = set()
    for line in lines:
        m = CORRESPONDENCE_LABEL_PAT.match(line)
        if m:
            labels.add(m.group(1).upper())
        elif MEMO_TITLE_PAT.match(line):
            labels.add("MEMORANDUM")
    return sorted(labels) if len(labels) >= 2 else None


def text_similarity(a, b):
    """Ratio in [0, 1]; 1.0 means identical. Used by FIX 14 to detect
    duplicate/leftover pages by content rather than by metadata."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def get_bookmark_map(doc):
    """Returns {0-based-idx: (title, level)}.

    Two rules derived from real client PDF analysis:

    Rule 1 - Highest level wins per page (not first-encountered).
    The original first-entry-wins approach was wrong: PDF-003's p66 had
    a L2 'Federal' entry AND multiple L3 entries all pointing to page 66.
    The L3 entry happened to appear first in the TOC stream, so it won,
    causing the L2 bookmark to be silently discarded. Fix: for any page
    where multiple TOC entries exist, keep the one with the LOWEST level
    number (i.e. highest in the hierarchy).

    Rule 2 - Trust L1 AND L2 as boundary signals.
    PDF-003's 990 block uses L2 'Federal' as its document-level anchor —
    there is no L1 entry for the 990 group. Restricting to L1-only missed
    this real boundary. Threshold: trust levels 1 and 2; L3+ are
    sub-section page markers and should be ignored.
    """
    toc = doc.get_toc(simple=False)
    bm = {}
    for entry in toc:
        level, title, page_1based = entry[0], entry[1], entry[2]
        if not page_1based or page_1based <= 0:
            continue
        idx = page_1based - 1
        if idx not in bm or level < bm[idx][1]:   # lower level = higher in hierarchy
            bm[idx] = (title, level)
    return bm

# Below this many extractable characters, a non-blank page is treated as
# a scanned image that needs OCR. Blank pages are excluded (nothing to
# recover). Only used when OCR fallback is enabled.
OCR_TEXT_TRIGGER_CHARS = 40


def pages_needing_ocr(doc, max_pages):
    """1-based page numbers that have almost no extractable text but are
    not blank, i.e. scanned pages the detectors currently can't read."""
    out = []
    for i in range(min(max_pages, len(doc))):
        page = doc[i]
        if len(page.get_text().strip()) < OCR_TEXT_TRIGGER_CHARS and not is_blank(page):
            out.append(i + 1)
    return out


def extract_det_features(doc, max_pages, ocr_text_override=None):
    ocr_text_override = ocr_text_override or {}
    bm_map = get_bookmark_map(doc)
    features = []
    prev_label = None
    fmt_stack_since_anchor = []  # formats seen since the last anchor (FIX 15)
    prev_l1_title = None  # last trusted bookmark title (FIX 2/13)
    prev_l1_page  = None  # page_num of that bookmark (FIX 13 adjacency)
    prev_numbering = None
    prev_numbering_scheme = None    # (region, pattern_kind) of the last numbering found (FIX 22)
    numbering_seen_since_anchor = False  # FIX 21, same anchor-scoping as fmt_stack_since_anchor
    prev_letterhead_block = None    # FIX 23

    for i in range(min(max_pages, len(doc))):
        page = doc[i]
        page_num = i + 1
        blank = is_blank(page)
        fmt = page_fmt(page)
        bm_entry = bm_map.get(i)
        bm_title = bm_entry[0] if bm_entry else None
        bm_level = bm_entry[1] if bm_entry else None

        # OCR fallback: a scanned page recovers its text here. When no
        # override exists (the normal case) this is exactly the old line,
        # so behaviour is unchanged unless OCR actually ran. This feeds
        # the text-based detectors (title anchor, correspondence, the
        # duplicate check) directly.
        ocr_text = ocr_text_override.get(page_num)
        ocr_used = bool(ocr_text)
        text = ocr_text if ocr_used else page.get_text().strip()

        # FIX 14: content-based duplicate check, only against the page right
        # before it. Requires real text on both sides so two blank pages
        # don't count (that's the blank-merge rule's job, not this one).
        prev_text = features[-1]["native_text"] if features else None
        is_duplicate_of_prev = (
            prev_text is not None
            and len(text) > 15 and len(prev_text) > 15
            and text_similarity(text, prev_text) >= DUPLICATE_TEXT_SIMILARITY
        )

        numbering, numbering_scheme = find_labeled_numbering(page, prev_numbering)
        if numbering is None:
            # Bare digit, no label to anchor it to, only trusted as a
            # continuation of an already-established count, and only on
            # the original tight band (see find_labeled_numbering).
            ht, ft = hf_text(page)
            bare = parse_numbering_bare(ft) or parse_numbering_bare(ht)
            if bare is not None and prev_numbering is not None and bare == prev_numbering[0]+1:
                numbering = (bare, prev_numbering[1])

        # OCR fallback for numbering: find_labeled_numbering reads the page
        # object directly (via hf_text clips), so it never sees OCR'd text.
        # When OCR ran on this page and the normal path found nothing, look
        # for the same keyword-anchored patterns in the OCR full-page text.
        # Safe to search the whole page here (not just a band) because both
        # patterns are keyword-anchored, the same reasoning FIX 20 used to
        # widen the band. find_letterhead_block is deliberately left out:
        # its whole design is a positional header/footer gate, which OCR
        # plain text can't preserve without bounding boxes.
        if numbering is None and ocr_used:
            total = parse_numbering_total(text)
            if total:
                numbering, numbering_scheme = total, ("fulltext_ocr", "total")
            else:
                split_val = parse_numbering_split_label(text)
                if split_val is not None:
                    prior_total = prev_numbering[1] if prev_numbering else None
                    numbering = (split_val, prior_total)
                    numbering_scheme = ("fulltext_ocr", "split_label")

        # FIX 17: a fresh count of 1 is always a reset, even with no prior
        # numbering to compare against (e.g. right after an unnumbered
        # cover sheet, previously there was nothing to diff it against).
        num_reset = False
        num_reset_to_one = False
        if numbering:
            cc, _ = numbering
            if cc == 1:
                num_reset = True
                num_reset_to_one = True
            elif prev_numbering:
                pc, _ = prev_numbering
                if cc < pc or cc > pc + 1:
                    num_reset = True

        # FIX 21: a page declaring itself "1" with nothing since the last
        # anchor having carried any numbering at all is close to unambiguous
        #, there's no existing sequence for it to be a mid-document
        # renumber of. A discontinuity reset (the elif branch above, e.g.
        # a page reading "2" compared against a "4" left over from an
        # already-closed document several pages back) doesn't get this
        # boost: the missing "1" in between means the current document's
        # own first page may simply never have carried a visible number,
        # which reads identically to a genuine cross-document jump.
        numbering_fresh_start = bool(num_reset_to_one and not numbering_seen_since_anchor)

        # FIX 22: the numbering scheme itself (header vs footer, "X of Y"
        # vs a split label/value pair) changing alongside a fresh start is
        # independent evidence of a different source template. Scoped to
        # fresh_start rather than any reset for the same reason as above.
        numbering_scheme_changed = bool(
            numbering_fresh_start and numbering_scheme and prev_numbering_scheme
            and numbering_scheme != prev_numbering_scheme
        )

        # FIX 23: a real letterhead swap (organization to organization),
        # not just any header/footer text difference, see find_letterhead_block.
        letterhead_block = find_letterhead_block(page)
        letterhead_changed = bool(
            letterhead_block and prev_letterhead_block
            and letterhead_similarity(letterhead_block, prev_letterhead_block)
                < LETTERHEAD_SIMILARITY_THRESHOLD
        )

        draw_count = len(page.get_drawings())
        img_count  = len(page.get_images(full=True))
        prev_dc    = features[-1]["drawing_count"] if features else None
        draw_spike = bool(prev_dc and prev_dc>0 and draw_count>prev_dc*3 and draw_count>200)
        # FIX 19: the mirror case, a page collapsing back to near-zero
        # drawings right after a drawing-heavy one is just as strong a
        # signal that the source template changed as the spike is.
        draw_drop = bool(prev_dc and prev_dc>200 and draw_count<prev_dc/3)

        # FIX 13: only suppress a repeated bookmark title when it's adjacent
        # to the previous one AND nothing else on the page corroborates a
        # real event. Raw title/level are kept below even when suppressed,
        # so score_all can still reinstate it on a strong embedding drop.
        is_repeat_title = (bm_level is not None and bm_level <= 2
                            and bm_title == prev_l1_title)
        is_adjacent = (prev_l1_page is not None
                       and page_num - prev_l1_page <= ADJACENT_BOOKMARK_GAP)
        same_title_suppressed = (
            is_repeat_title and is_adjacent and not num_reset and not draw_spike
        )

        # FIX 1+2: trust L1 and L2 bookmarks as boundary signals.
        # L3+ are sub-section page markers (e.g. individual 990 schedule pages)
        # and produce false positives when treated as document boundaries.
        # L2 is needed because PDF-003's 990 block anchor ('Federal') is L2.
        has_boundary_bookmark = (bm_level is not None and bm_level <= 2
                                  and not same_title_suppressed
                                  and not is_duplicate_of_prev)

        # FIX 15: "returning" is scoped to the current still-open document,
        # not a flat page count, see fmt_stack_since_anchor update below.
        format_changed = prev_label is not None and fmt["label"] != prev_label
        format_returning = format_changed and fmt["label"] in fmt_stack_since_anchor

        img_infos = []
        try:
            for info in page.get_image_info(xrefs=True):
                iw = info.get("width", 0)
                ih = info.get("height", 0)
                if iw > 0 and ih > 0:
                    img_infos.append({"width": iw, "height": ih,
                                      "colorspace": info.get("colorspace", 0)})
        except Exception:
            pass  # get_image_info unavailable in older pymupdf, img_infos stays []

        text_density = len(text) / max(page.rect.width * page.rect.height, 1)
        title_anchor = detect_title_anchor(text)
        correspondence_labels = detect_correspondence_header(text)  # FIX 18

        # A duplicate page can never open a new document. A bare format
        # change with no bookmark/title-anchor still counts as an anchor
        # here (needed for FIX 15), see that entry for the trade-off.
        is_structural_anchor_here = (not is_duplicate_of_prev) and (
            has_boundary_bookmark
            or (format_changed and not format_returning)
            or bool(title_anchor)
            or bool(correspondence_labels)
        )

        features.append({
            "page_num":           page_num,
            "is_blank":           blank,
            "title_anchor":       title_anchor,
            "w": fmt["w"], "h": fmt["h"],
            "orientation":        fmt["orient"],
            "size_bucket":        fmt["bucket"],
            "format_label":       fmt["label"],
            "format_changed":     format_changed,
            "format_returning":   format_returning,  # FIX 3/15
            "has_l1_bookmark":    has_boundary_bookmark,
            "bookmark_title":     bm_title if has_boundary_bookmark else None,
            "bookmark_level":     bm_level,
            "bookmark_suppressed": bool(bm_level is not None and bm_level <= 2
                                         and not has_boundary_bookmark
                                         and not is_duplicate_of_prev),  # FIX 13
            "raw_bookmark_title": bm_title,  # unsuppressed, for FIX 13's rescue path
            "numbering_reset":    num_reset,
            "numbering_fresh_start": numbering_fresh_start,  # FIX 21
            "numbering_scheme_changed": numbering_scheme_changed,  # FIX 22
            "page_num_found":     numbering[0] if numbering else None,
            "page_total_found":   numbering[1] if numbering else None,
            "drawing_count":      draw_count,
            "image_count":        img_count,
            "image_infos":        img_infos,
            "table_count":        0,   # placeholder, populated by pdfplumber if available
            "drawing_spike":      draw_spike,
            "drawing_density_drop": draw_drop,  # FIX 19
            "letterhead_changed": letterhead_changed,  # FIX 23
            "text_char_count":    len(text),
            "text_density":       round(text_density, 8),
            "native_text":        text,
            "is_duplicate_of_prev": is_duplicate_of_prev,  # FIX 14
            "correspondence_labels": correspondence_labels,  # FIX 18
        })

        prev_label = fmt["label"]
        # FIX 15: reset at a structural anchor instead of sliding a fixed
        # 5-page window, format is only "returning" within this document.
        if is_structural_anchor_here:
            fmt_stack_since_anchor = [fmt["label"]]
        else:
            fmt_stack_since_anchor = fmt_stack_since_anchor + [fmt["label"]]
        if has_boundary_bookmark:
            prev_l1_title = bm_title
            prev_l1_page = page_num
        if numbering: prev_numbering = numbering
        if numbering_scheme: prev_numbering_scheme = numbering_scheme
        if letterhead_block: prev_letterhead_block = letterhead_block
        # FIX 21: scoped the same way as fmt_stack_since_anchor, an anchor
        # opens a new document context, so only numbering from this page
        # onward counts toward "seen since anchor".
        if is_structural_anchor_here:
            numbering_seen_since_anchor = bool(numbering)
        else:
            numbering_seen_since_anchor = numbering_seen_since_anchor or bool(numbering)

    return features


# ═══════════════════════════════════════════════════════════════════
# 5. STAGE 1 - PARALLEL EMBEDDING (FIX 6)
# ═══════════════════════════════════════════════════════════════════

def render_png(page, dpi=150):
    scale = dpi / 72.0
    px = page.get_pixmap(matrix=fitz.Matrix(scale,scale), alpha=False)
    return px.tobytes("png")

def _embed_one_gemini2(client, text, image_bytes, model, page_num, cost_tracker, stats, stats_lock):
    """Single page embedding call, runs inside a thread."""
    try:
        part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        contents = [text, part] if text else \
            ["Describe the layout and content of this document page.", part]
        resp = client.models.embed_content(model=model, contents=contents)
        emb = resp.embeddings[0]

        # FIX 7: split token count into text vs image estimates
        total_tokens = None
        for acc in (lambda o: o.statistics.token_count,
                    lambda o: o.usage_metadata.total_token_count,
                    lambda o: o.token_count):
            try:
                v = acc(emb); total_tokens = int(v) if v else None; break
            except Exception: pass

        if total_tokens:
            # Estimate text portion from character count; remainder = image
            text_est = min(max(len(text)//4, 0), total_tokens - 100) if text else 0
            img_tok  = total_tokens - text_est
            cost = (text_est * PRICING["gemini-embedding-2_text_per_token"] +
                    img_tok  * PRICING["gemini-embedding-2_image_per_token"])
            cost_tracker.record(page_num, "gemini-embedding-2",
                                 text_est, img_tok, True, False, cost)
        else:
            text_est = max(len(text)//4, 0) if text else 0
            img_est  = ESTIMATED_IMAGE_TOKENS_PER_PAGE
            cost = (text_est * PRICING["gemini-embedding-2_text_per_token"] +
                    img_est  * PRICING["gemini-embedding-2_image_per_token"])
            cost_tracker.record(page_num, "gemini-embedding-2",
                                 text_est, img_est, True, True, cost)

        with stats_lock:
            stats["g2_ok"] = stats.get("g2_ok", 0) + 1
        return emb.values, emb.values   # image, text (same combined vector)

    except Exception as e:
        with stats_lock:
            stats["g2_fail"] = stats.get("g2_fail", 0) + 1
            if stats.get("g2_fail") == 1:
                print(f"\n    [gemini-embedding-2 FAILED, falling back] {e}")
        return None, None  # caller will use fallback


def _embed_one_fallback(img_session, text_client, image_bytes, text,
                        page_num, project_id, location, cost_tracker):
    """Fallback: multimodalembedding@001 (image) + gemini-embedding-001 (text)."""
    # Image
    url = (f"https://{location}-aiplatform.googleapis.com/v1/projects/"
           f"{project_id}/locations/{location}/publishers/google/"
           f"models/multimodalembedding@001:predict")
    delay = 2
    image_emb = None
    for attempt in range(5):
        try:
            resp = img_session.post(url, json={
                "instances":[{"image":{"bytesBase64Encoded":base64.b64encode(image_bytes).decode()}}],
                "parameters":{"dimension":1408}}, timeout=60)
            resp.raise_for_status()
            image_emb = resp.json()["predictions"][0].get("imageEmbedding")
            break
        except Exception:
            if attempt == 4: raise
            time.sleep(delay); delay *= 2

    # Text
    text_emb, text_tokens = None, 0
    if text:
        delay = 2
        for attempt in range(5):
            try:
                tr = text_client.models.embed_content(
                    model="gemini-embedding-001", contents=[text])
                te = tr.embeddings[0]
                text_emb = te.values
                tok = None
                for acc in (lambda o: o.statistics.token_count,
                            lambda o: o.usage_metadata.total_token_count):
                    try: tok = int(acc(te)); break
                    except (AttributeError, TypeError, ValueError): pass
                text_tokens = tok or max(len(text)//4, 1)
                break
            except Exception:
                if attempt == 4: text_emb, text_tokens = None, 0; break
                time.sleep(delay); delay *= 2

    cost = (PRICING["multimodalembedding@001_per_image"] +
            text_tokens * PRICING["gemini-embedding-001_text_per_token"])
    cost_tracker.record(page_num, "fallback", text_tokens, 1, False, False, cost)
    return image_emb, text_emb


def embed_all_pages_parallel(doc, det_features, gemini2_client, img_session,
                              text_client, project_id, location, gemini2_model,
                              use_gemini2, dpi, max_workers):
    """
    Parallel embedding: all non-blank pages are embedded concurrently up
    to max_workers. Results are collected into a dict keyed by page_num.
    Sequential similarity analysis runs AFTER this returns, so page order
    is preserved correctly.
    """
    non_blank = [(f["page_num"], f["native_text"])
                  for f in det_features if not f["is_blank"]]

    cost_tracker = CostTracker()
    stats = {}
    stats_lock = threading.Lock()
    page_embeddings = {}

    def embed_one(page_num, text):
        page = doc[page_num-1]
        image_bytes = render_png(page, dpi=dpi)

        if use_gemini2:
            img_emb, txt_emb = _embed_one_gemini2(
                gemini2_client, text, image_bytes, gemini2_model,
                page_num, cost_tracker, stats, stats_lock)
            if img_emb is not None:
                return page_num, img_emb, txt_emb, "gemini-embedding-2"

        # fallback (either use_gemini2=False or gemini2 failed)
        img_emb, txt_emb = _embed_one_fallback(
            img_session, text_client, image_bytes, text,
            page_num, project_id, location, cost_tracker)
        return page_num, img_emb, txt_emb, "fallback"

    total = len(non_blank)
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(embed_one, pn, txt): pn for pn, txt in non_blank}
        for fut in as_completed(futures):
            pn, img_emb, txt_emb, method = fut.result()
            page_embeddings[pn] = {
                "image_embedding": img_emb,
                "text_embedding":  txt_emb,
                "method":          method,
            }
            done += 1
            if done % 10 == 0 or done == total:
                print(f"\r  Embedded {done}/{total} pages...", end="", flush=True)

    print()
    g2_ok   = stats.get("g2_ok", 0)
    g2_fail = stats.get("g2_fail", 0)
    if use_gemini2:
        print(f"  gemini-embedding-2: {g2_ok} OK, {g2_fail} fallback")

    return page_embeddings, cost_tracker


# ═══════════════════════════════════════════════════════════════════
# 6. SIMILARITY + SCORING
# ═══════════════════════════════════════════════════════════════════

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    if a.shape != b.shape:
        # Dimension mismatch, happens when gemini-embedding-2 (3072-dim)
        # and the fallback multimodalembedding@001 (1408-dim) are both
        # present in the same run. Cannot compute similarity across different
        # spaces; treat as no signal rather than crashing.
        return None
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a,b)/d) if d else 0.0

def sim_chain(page_embeddings, non_blank_pages, key):
    series, prev = [], None
    for pn in non_blank_pages:
        emb = page_embeddings.get(pn,{}).get(key)
        if emb is None: series.append((pn,None)); continue
        if prev is None: series.append((pn,None))
        else:
            pe = page_embeddings.get(prev,{}).get(key)
            sim = cosine_sim(pe, emb) if pe is not None else None
            # cosine_sim returns None on dimension mismatch (mixed models)
            series.append((pn, sim))
        prev = pn
    return series

def flag_sims(series):
    out = []
    for idx,(pn,sim) in enumerate(series):
        if sim is None: out.append((pn,sim,False,None,False)); continue
        local = [s for _,s in series[max(0,idx-1-Z_WINDOW):idx] if s is not None]
        if len(local) < 3:
            is_b = sim < 0.55
            out.append((pn,sim,is_b,"insufficient_context" if is_b else None,False))
            continue
        mean,std = np.mean(local), max(np.std(local), MIN_STD)
        z = (sim-mean)/std; drop = mean-sim
        moderate = bool(z < Z_THRESHOLD and drop > MIN_ABSOLUTE_DROP)
        high     = bool(z < Z_HIGH_THRESHOLD and drop > MIN_ABSOLUTE_DROP)
        reason   = f"z={round(float(z),2)},drop={round(float(drop),3)}" if moderate else None
        out.append((pn, round(float(sim),4), moderate, reason, high))
    return out

def score_all(det_features, img_flags, txt_flags):
    img_map = {p:(s,m,r,h) for p,s,m,r,h in img_flags}
    txt_map = {p:(s,m,r,h) for p,s,m,r,h in txt_flags}
    results = []
    prev_emb_only = False

    for f in det_features:
        pn = f["page_num"]

        # FIX 14: a duplicate page is never a boundary, full stop. Short-
        # circuit before scoring anything else, and don't let it count
        # toward FIX 8's consecutive-embedding-only penalty either.
        if f.get("is_duplicate_of_prev"):
            results.append({
                "page_num":           pn,
                "is_blank":           f["is_blank"],
                "structural_score":   0.0,
                "numbering_score":    0.0,
                "embedding_score":    0.0,
                "drawing_score":      0.0,
                "letterhead_score":   0.0,
                "consecutive_penalty":0.0,
                "total_score":        0.0,
                "is_boundary":        False,
                "image_similarity":   None,
                "image_z_reason":     None,
                "text_similarity":    None,
                "text_z_reason":      None,
                "bookmark_title":     None,
                "bookmark_level":     f.get("bookmark_level"),
                "bookmark_trusted":   False,
                "title_anchor":       None,
                "correspondence_labels": None,
                "duplicate_of_prev":  True,
            })
            prev_emb_only = False
            continue

        is_,ts_ = img_map.get(pn,(None,False,None,False)), txt_map.get(pn,(None,False,None,False))
        img_high, txt_high = is_[3], ts_[3]
        img_mod,  txt_mod  = is_[1], ts_[1]
        # FIX 8: gemini-embedding-2 returns the same vector for image and
        # text, so a matching sim value means it's one signal, not two —
        # count it once instead of summing 0.5+0.5. Fallback mode's
        # image/text embeddings are genuinely different and unaffected.
        same_signal = (is_[0] is not None and ts_[0] is not None
                        and abs(is_[0]-ts_[0]) < 1e-9)
        if img_high or txt_high:
            emb_score = WEIGHTS["embedding_high_z"]
        elif same_signal:
            emb_score = WEIGHTS["embedding_moderate_each"] * (img_mod or txt_mod)
        else:
            emb_score = (WEIGHTS["embedding_moderate_each"] * img_mod +
                         WEIGHTS["embedding_moderate_each"] * txt_mod)

        if f["numbering_fresh_start"]:
            num_score = WEIGHTS["page_numbering_fresh_start"]
        elif f["numbering_reset"]:
            num_score = WEIGHTS["page_numbering_reset"]
        else:
            num_score = 0.0
        if f["numbering_fresh_start"] and f.get("numbering_scheme_changed"):
            num_score += WEIGHTS["page_numbering_scheme_bonus"]
        num_score = round(num_score, 2)

        if f["drawing_spike"]:
            draw_score = WEIGHTS["drawing_density_spike"]
        elif f.get("drawing_density_drop"):
            draw_score = WEIGHTS["drawing_density_drop"]
        else:
            draw_score = 0.0

        letterhead_score = WEIGHTS["letterhead_change"] if f.get("letterhead_changed") else 0.0

        format_signal = f["format_changed"] and not f["format_returning"]

        # FIX 9: L2 bookmarks need a corroborating signal on the same page
        # to count; L1 is always trusted (0 FPs in validation). Otherwise an
        # L2 sub-heading of the same document reads the same as a genuine
        # L2 document anchor (FIX1's 990-block case).
        corroborated = bool(num_score or draw_score or format_signal or emb_score or letterhead_score)
        trust_bookmark = f["has_l1_bookmark"] and (
            f["bookmark_level"] == 1 or corroborated)

        # FIX 13: a bookmark suppressed at extraction time gets one more
        # chance now that embeddings exist, a strong drop here means the
        # content is genuinely different despite the matching title.
        bookmark_title = f.get("bookmark_title")
        if f.get("bookmark_suppressed") and (img_high or txt_high):
            trust_bookmark = True
            bookmark_title = f.get("raw_bookmark_title")

        # FIX 10: all-caps title line at top of page (EXHIBIT A, SCHEDULE 1)
        title_anchor_hit = bool(f.get("title_anchor"))

        # FIX 18: memo/email correspondence header, weighted like a title
        # anchor (strong-alone), see the docstring for corpus validation.
        correspondence_hit = bool(f.get("correspondence_labels"))

        structural = trust_bookmark or format_signal or title_anchor_hit or correspondence_hit
        struct_score = WEIGHTS["structural_change"] if structural else (
            WEIGHTS["structural_change_return"] if f["format_changed"] else 0.0)  # FIX 3

        emb_only = (emb_score > 0 and not structural and not f["numbering_reset"]
                    and not f.get("letterhead_changed"))
        penalty  = WEIGHTS["consecutive_flag_penalty"] if (emb_only and prev_emb_only) else 0.0

        total = max(0.0, struct_score + num_score + emb_score + draw_score + letterhead_score + penalty)
        is_boundary = total >= BOUNDARY_THRESHOLD

        results.append({
            "page_num":           pn,
            "is_blank":           f["is_blank"],
            "structural_score":   struct_score,
            "numbering_score":    num_score,
            "embedding_score":    round(emb_score,2),
            "drawing_score":      draw_score,
            "letterhead_score":   letterhead_score,
            "consecutive_penalty":penalty,
            "total_score":        round(total,2),
            "is_boundary":        is_boundary,
            "image_similarity":   is_[0],
            "image_z_reason":     is_[2],
            "text_similarity":    ts_[0],
            "text_z_reason":      ts_[2],
            "bookmark_title":     bookmark_title,
            "bookmark_level":     f.get("bookmark_level"),
            "bookmark_trusted":   bool(trust_bookmark),
            "title_anchor":       f.get("title_anchor"),
            "correspondence_labels": f.get("correspondence_labels"),
            "duplicate_of_prev":  False,
        })
        prev_emb_only = emb_only
    return results


# ═══════════════════════════════════════════════════════════════════
# 7. SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

# FIX 25: segment cohesion. A correctly-cut document has fairly even
# page-to-page similarity; one carrying a foreign page has a dip where
# the content actually changes. The existing embedding detector cannot
# see this because it z-scores against the whole PDF, so a dip that is
# glaring inside one segment can sit right on the document-wide median
# and score nothing. Measuring each segment against its OWN baseline is
# what makes it visible, and it needs no vocabulary or page-count rule,
# so it behaves the same on any template or language.
#
# Position turned out to matter far more than depth. Across the labelled
# segments, when the deepest dip landed on the segment's LAST page it
# contained a foreign page 9 times out of 9; a dip on the first interior
# page meant that only 1 time in 10. That asymmetry makes sense: the
# usual failure is the next document's opening page being swallowed by
# the previous segment, which puts the content change at the very end.
# Small sample, so the edge rule is kept conservative and the deep-dip
# rule needs a much larger drop to fire on its own.
COHESION_MIN_INTERIOR = 3      # pages needed before a baseline means anything
COHESION_EDGE_DROP    = 0.04   # dip on the last page, relative to segment median
COHESION_DEEP_DROP    = 0.10   # dip anywhere else


def segment_cohesion_break(seg, scored_by_page):
    """Find the page inside a segment where content most changes.

    Returns (page, relative_drop, where) or None. Compares each segment
    against its own similarity baseline, not the document's.
    """
    interior = range(seg["start"] + 1, seg["end"] + 1)
    pages, sims = [], []
    for p in interior:
        v = (scored_by_page.get(p) or {}).get("image_similarity")
        if v is not None:
            pages.append(p); sims.append(v)
    if len(sims) < COHESION_MIN_INTERIOR:
        return None
    base = statistics.median(sims)
    if base <= 0:
        return None
    lo = min(sims)
    drop = (base - lo) / base
    page = pages[sims.index(lo)]
    if page == seg["end"] and drop >= COHESION_EDGE_DROP:
        return (page, drop, "last page")
    if drop >= COHESION_DEEP_DROP:
        return (page, drop, "mid-segment")
    return None


def build_segments(det_features, scored):
    boundary_set = {r["page_num"] for r in scored if r["is_boundary"]}
    boundary_set.add(1)
    total = det_features[-1]["page_num"]
    segs, starts = [], sorted(boundary_set)
    for i, s in enumerate(starts):
        e = starts[i+1]-1 if i+1 < len(starts) else total
        segs.append({"start": s, "end": e})
    return split_trailing_cohesion_breaks(segs, scored)


def split_trailing_cohesion_breaks(segments, scored):
    """FIX 25b: cut off a trailing page whose content clearly departs from
    the rest of its segment.

    The commonest miss in this pipeline is the next document's opening
    page being swallowed by the previous segment. It has no format change
    (the cover shares the previous page's orientation), no bookmark, and
    an embedding drop too small to register against the whole PDF, so no
    existing signal fires. But inside its own segment it stands out: the
    other pages sit at 0.94-0.97 similarity and it sits at 0.88.

    Only a break on the segment's LAST page is acted on. Position carried
    almost all the signal in validation: a dip on the last page meant a
    foreign page 9 times out of 9, a dip mid-segment only 3 times in 14.
    Simulated over the labelled segments, splitting on last-page breaks
    fixed 6 segments and broke 0; enabling mid-segment splits at any
    threshold immediately broke a correct one. So mid-segment breaks stay
    advisory and surface through the confidence reason instead.
    """
    scored_by_page = {r["page_num"]: r for r in scored}
    out = []
    for seg in segments:
        brk = segment_cohesion_break(seg, scored_by_page)
        if brk and brk[2] == "last page" and brk[0] > seg["start"]:
            page, drop, _ = brk
            row = scored_by_page.get(page)
            if row is not None:
                row["cohesion_split"] = round(drop, 4)
            out.append({"start": seg["start"], "end": page - 1})
            out.append({"start": page,          "end": seg["end"]})
            continue
        out.append(seg)
    return out


# ═══════════════════════════════════════════════════════════════════
# 8. CLASSIFICATION, delegated to pss_classifier.py
#    The Classifier is config-driven (pss_taxonomy.json) and
#    self-learning (pss_prototypes.json). See pss_classifier.py.
# ═══════════════════════════════════════════════════════════════════

# Classifier is instantiated once in run() and passed down, not a module-level
# global, so multiple parallel pipeline runs can use different configs.


# ═══════════════════════════════════════════════════════════════════
# 9. SPLITTING + OUTPUT
# ═══════════════════════════════════════════════════════════════════

def sanitize(s, maxlen=40):
    s = re.sub(r'[^\w\s-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip().replace(' ', '_')
    return s[:maxlen] or "segment"

def split_and_write(doc, segments, classifications, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for idx,(seg,cls) in enumerate(zip(segments, classifications), 1):
        slug = sanitize(cls.get("label", cls.get("category", "unknown")))
        tags = cls.get("special_tags", [])
        tag_str = ("_" + "+".join(tags)) if tags else ""
        fname = f"segment_{idx:03d}_{slug}{tag_str}_pp{seg['start']}-{seg['end']}.pdf"
        path  = out_dir / fname
        od = fitz.open()
        od.insert_pdf(doc, from_page=seg["start"]-1, to_page=seg["end"]-1)
        od.save(str(path)); od.close()
        files.append(str(path))
    return files

def describe_boundary(det_row, scored_row):
    """Which signal(s) fired at a segment's start page. Schema item
    'Boundary Signals Fired', just formats fields we already computed."""
    if not scored_row:
        return "unscored_boundary"
    if scored_row["page_num"] == 1:
        return "document_start"
    sigs = []
    if scored_row["structural_score"] > 0:
        if scored_row.get("bookmark_trusted") and scored_row.get("bookmark_title"):
            sigs.append(f"bookmark(L{scored_row.get('bookmark_level')}):"
                        f"{scored_row['bookmark_title']}")
        elif scored_row.get("title_anchor"):
            sigs.append(f"title_anchor:{scored_row['title_anchor']}")
        elif scored_row.get("correspondence_labels"):
            sigs.append(f"correspondence_header:{'+'.join(scored_row['correspondence_labels'])}")
        elif det_row.get("format_changed"):
            sigs.append(f"format_change:{det_row.get('format_label')}")
    if scored_row["numbering_score"] > 0:
        label = "page_numbering_fresh_start" if det_row.get("numbering_fresh_start") \
            else "page_numbering_reset"
        if det_row.get("numbering_scheme_changed"):
            label += "+scheme_change"
        sigs.append(label)
    if scored_row["embedding_score"] > 0:
        z = scored_row.get("image_z_reason") or scored_row.get("text_z_reason")
        sigs.append(f"embedding_drop({z})" if z else "embedding_drop")
    if scored_row["drawing_score"] > 0:
        sigs.append("drawing_density_drop" if det_row.get("drawing_density_drop")
                    else "drawing_density_spike")
    if scored_row.get("letterhead_score", 0) > 0:
        sigs.append("letterhead_change")
    if scored_row.get("cohesion_split"):
        sigs.append(f"cohesion_split({scored_row['cohesion_split']:.0%} "
                    f"below segment baseline)")
    return ";".join(sigs) if sigs else "unscored_boundary"


# ── FIX 24: confidence thresholds ───────────────────────────────────
#
# Confidence used to be a constant per cascade branch (Pass 1 -> HIGH,
# etc.), which measured nothing. It also rated only the type label while
# sitting next to the page range, so a HIGH row could have completely
# wrong boundaries. Against the 232 colour-coded segments the tiers came
# out backwards: LOW was 95% correctly segmented, HIGH only 84.8%.
#
# We already compute total_score per page and then throw it away by
# reducing it to is_boundary = total >= 1.0. The margin over that
# threshold is the evidence strength. A segment needs both edges right,
# so it scores on the weaker one.
#
# Thresholds come from WEIGHTS, not from fitting the sheet: 1.0 is one
# trusted signal, so 1.8 means two independent signals agreed.
#
# MEDIUM sits just above the boundary threshold, so a boundary that only
# just cleared 1.0 falls to LOW. That is deliberate and it is the knob
# most worth understanding before changing.
#
# At 1.2, LOW covers ~81% of segments and every mis-segmented row lands
# in it. At 1.0 it covers ~40% but a fifth of the errors escape into
# MEDIUM. There is no useful setting in between: most boundaries score
# exactly 1.0, so the evidence is bimodal and the split jumps straight
# from 40% to 81%. Narrowing LOW further needs better signals, not a
# different number here.
#
# 1.2 is the safer default because the two mistakes do not cost the same.
# A segment wrongly marked LOW gets looked at and cleared. A broken
# segment marked MEDIUM is never looked at again. Lower this only if
# reviewing that many rows is genuinely more expensive than shipping the
# occasional bad segment.
CONF_HIGH_MARGIN   = 1.8
CONF_MEDIUM_MARGIN = 1.2

# Interior evidence, for merges. A page inside a segment scoring this
# high without crossing the threshold means we saw a document change
# there and rejected it.
NEAR_MISS_FLOOR      = 0.5   # half the boundary threshold
OVERSIZED_SEGMENT_X  = 2.0   # times this document's own median segment

# A single-segment PDF this short is unremarkable. Longer than this and
# finding no boundaries probably means the detectors saw nothing, rather
# than the file genuinely being one document.
SINGLE_SEGMENT_PLAUSIBLE_PAGES = 3


def segmentation_confidence(seg, scored_by_page, total_pages,
                             det_by_page=None, median_seg_pages=None):
    """How confident we are that this segment's page range is right.

    Says nothing about the type label. Returns (tier, margin, reason).

    Looks at two things. The edges tell us whether a boundary that fired
    should have, which covers over-segmentation and off-by-one splits.
    But edges cannot see a merge: run two documents together and the
    missed boundary sits in the middle while both edges stay strong. 11
    of the 14 errors that edge margin alone rated MEDIUM or better were
    merges, so we also check the pages inside:

      near-miss   a page scored >= NEAR_MISS_FLOOR but under the
                  threshold. We saw a document change and rejected it.
      unreadable  a page had no extractable text and was not blank, so
                  the detectors had nothing to work with. This is the
                  scanned/OCR case, where a merge leaves no near-miss
                  trace because every signal scores zero.
    """
    def edge_score(page_num, is_doc_edge):
        # Page 1 and the final page are not inferred boundaries, the
        # document's own extent fixes them, so they carry no risk.
        if is_doc_edge:
            return 2.0
        row = scored_by_page.get(page_num)
        return row["total_score"] if row else 0.0

    starts_at_doc_edge = seg["start"] == 1
    ends_at_doc_edge   = seg["end"] >= total_pages

    start_s = edge_score(seg["start"], starts_at_doc_edge)
    # The segment's end is fixed by the NEXT segment's start boundary.
    end_s   = edge_score(seg["end"] + 1, ends_at_doc_edge)

    # Degenerate case: this segment IS the whole document, so BOTH its
    # edges are just the file's extent and neither was decided by the
    # pipeline. Edge margin would read 2.0 and report HIGH, which is
    # exactly backwards, finding no boundaries at all in a long PDF is
    # the least certain outcome there is, not the most. Absence of
    # evidence must lower confidence, never raise it.
    if starts_at_doc_edge and ends_at_doc_edge:
        if total_pages <= SINGLE_SEGMENT_PLAUSIBLE_PAGES:
            return ("MEDIUM", 2.0,
                    f"whole {total_pages}p document is one segment, short "
                    f"enough to be plausible, but nothing corroborates it")
        return ("LOW", 0.0,
                f"no boundaries detected anywhere in {total_pages} pages; "
                f"segmentation may have failed entirely")

    margin = min(start_s, end_s)
    weak_page = seg["start"] if start_s <= end_s else seg["end"] + 1
    n_pages = seg["end"] - seg["start"] + 1

    # ── interior evidence ──
    risks, interior = [], range(seg["start"] + 1, seg["end"] + 1)

    near_miss_pages = [p for p in interior
                       if NEAR_MISS_FLOOR <= scored_by_page.get(p, {}).get(
                           "total_score", 0.0) < BOUNDARY_THRESHOLD]
    if near_miss_pages:
        top = max(near_miss_pages,
                  key=lambda p: scored_by_page[p]["total_score"])
        risks.append(f"possible_missed_boundary@p{top}"
                     f"(scored {scored_by_page[top]['total_score']:.2f}"
                     f"/{BOUNDARY_THRESHOLD})")

    cohesion = segment_cohesion_break(seg, scored_by_page)
    if cohesion:
        c_page, c_drop, c_where = cohesion
        risks.append(f"content_break@p{c_page}({c_drop*100:.0f}% below segment "
                     f"baseline, {c_where})")

    if det_by_page:
        unreadable = [p for p in interior
                      if (d := det_by_page.get(p)) and not d.get("is_blank")
                      and not (d.get("native_text") or "").strip()]
        if unreadable:
            risks.append(f"unreadable_pages={len(unreadable)}"
                         f"(p{unreadable[0]}..p{unreadable[-1]})")

    oversized = bool(median_seg_pages and
                     n_pages >= OVERSIZED_SEGMENT_X * median_seg_pages)

    # Interior risk and size both beat edge strength here, since both
    # point at a merge and a merge's edges look fine by definition.
    #
    # Note there is no rule promoting a bare-threshold boundary back to
    # MEDIUM just because it sits on a normally-reliable signal like an
    # L1 bookmark. An earlier version had one and it hid 14 errors in
    # MEDIUM. A signal being good in general does not make one
    # bare-threshold firing of it good.
    if risks:
        tier, reason = "LOW", "; ".join(risks)
    elif oversized:
        tier = "LOW"
        reason = (f"segment is {n_pages}p vs ~{median_seg_pages:g}p typical "
                  f"for this document, possible merge")
    elif margin >= CONF_HIGH_MARGIN:
        tier = "HIGH"
        reason = f"both boundaries strong (weakest {margin:.2f})"
    elif margin >= CONF_MEDIUM_MARGIN:
        tier = "MEDIUM"
        reason = f"one boundary at {margin:.2f} (p{weak_page})"
    else:
        tier = "LOW"
        reason = f"weak boundary evidence at p{weak_page} ({margin:.2f})"

    return tier, round(margin, 2), reason


def write_report(out_dir, run_meta, det_features, scored, segments,
                  classifications, cost_summary, timing, files):
    # split_and_write() normally creates this first, but don't depend on
    # another function's side effect to be callable.
    out_dir.mkdir(parents=True, exist_ok=True)
    det_by_page    = {f["page_num"]: f for f in det_features}
    scored_by_page = {r["page_num"]: r for r in scored}
    pdf_filename = Path(run_meta["input_pdf"]).name
    pdf_id       = Path(run_meta["input_pdf"]).stem  # filename stem as id
    config = {
        "weights": WEIGHTS,
        "boundary_threshold": BOUNDARY_THRESHOLD,
        "z_threshold": Z_THRESHOLD,
        "z_high_threshold": Z_HIGH_THRESHOLD,
        "min_absolute_drop": MIN_ABSOLUTE_DROP,
        "fixes_applied": [
            "FIX1_l1_bookmark_only",
            "FIX2_same_title_suppression",
            "FIX3_format_return_downweight",
            "FIX4_tighter_z_threshold",
            "FIX5_visual_classification",
            "FIX6_parallel_embedding",
            "FIX7_split_token_reporting",
            "FIX8_embedding_double_count",
            "FIX9_l2_bookmark_corroboration",
            "FIX10_title_anchor_signal",
            "FIX11_page_size_relative_tolerance",
            "FIX12_document_category_schema",
            "FIX13_adjacent_bookmark_title_suppression",
            "FIX14_duplicate_page_detection",
            "FIX15_format_return_anchor_scoping",
            "FIX16_split_label_page_numbering",
            "FIX17_numbering_reset_on_fresh_count",
            "FIX18_correspondence_header_detection",
            "FIX19_drawing_density_drop",
            "FIX20_adaptive_numbering_band",
            "FIX21_numbering_fresh_start",
            "FIX22_numbering_scheme_change",
            "FIX23_gated_letterhead_change",
            "FIX24_evidence_based_confidence",
        ],
        "confidence_config": {
            "high_margin":   CONF_HIGH_MARGIN,
            "medium_margin": CONF_MEDIUM_MARGIN,
        },
    }
    # FIX 24: rate the segment's page range and its type label separately,
    # then report the weaker of the two as the headline `confidence`.
    total_pages_n = run_meta["total_pages"]
    # Segment size is only meaningful relative to this document's own
    # typical segment - "8 pages" means nothing without knowing whether
    # the rest of the PDF splits into 2-page or 40-page documents.
    # Real median, not sorted[len//2] - that skews high on even-length
    # lists ([2,2,10,10] gives 10, not 6). Since oversized triggers at 2x
    # the median, an inflated baseline means it under-fires and misses
    # the merges it is there to catch.
    _sizes = sorted(s["end"] - s["start"] + 1 for s in segments)
    median_seg_pages = statistics.median(_sizes) if _sizes else None
    segs_out = []
    for i,(s,c,f) in enumerate(zip(segments, classifications, files)):
        seg_tier, seg_margin, seg_why = segmentation_confidence(
            s, scored_by_page, total_pages_n,
            det_by_page=det_by_page, median_seg_pages=median_seg_pages)
        cls_tier = c.get("confidence", "LOW")
        segs_out.append(
        {"segment_index": i+1,
         "pdf_id": pdf_id,
         "pdf_filename": pdf_filename,
         "total_pages": total_pages_n,
         "start_page": s["start"], "end_page": s["end"],
         "page_count": s["end"]-s["start"]+1,
         "document_type_id":   c.get("type_id", "unknown"),
         "document_type_label":c.get("label", c.get("category", "Unknown")),
         "document_category":  c.get("category", "Supporting / Misc"),
         "classification_method": c.get("method", ""),
         # `confidence` IS the segmentation confidence. It used to be
         # min(segmentation, classification), but classification
         # confidence is not calibrated yet, and letting it drag the
         # headline down was demoting correctly-segmented rows for
         # reasons that had nothing to do with their boundaries. It
         # stays in its own column as information, not as a veto.
         "confidence": seg_tier,
         "classification_confidence": cls_tier,
         "boundary_margin": seg_margin,
         "confidence_reason": seg_why,
         "boundary_signals_fired": describe_boundary(
             det_by_page.get(s["start"]), scored_by_page.get(s["start"])),
         "special_tags": c.get("special_tags", []),
         "ingest_mode":  c.get("ingest_mode", "standard"),
         "output_file": Path(f).name})
    report = {
        "run_metadata": run_meta,
        "timing_seconds": timing,
        "cost_and_tokens": cost_summary,
        "segments": segs_out,
        "per_page_detail": scored,
        "config": config,
    }
    jp = out_dir / "run_report.json"
    with open(jp,"w") as f: json.dump(report, f, indent=2)

    cp = out_dir / "segments_summary.csv"
    with open(cp,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdf_id","pdf_filename","total_pages",
                    "seg","start_page","end_page","page_count",
                    "document_type_id","document_type_label","document_category",
                    "confidence","confidence_reason","boundary_margin",
                    "classification_confidence","classification_method",
                    "boundary_signals_fired",
                    "special_tags","ingest_mode","output_file"])
        for r in segs_out:
            w.writerow([r["pdf_id"],r["pdf_filename"],r["total_pages"],
                        r["segment_index"],r["start_page"],r["end_page"],r["page_count"],
                        r["document_type_id"],r["document_type_label"],r["document_category"],
                        r["confidence"],r["confidence_reason"],
                        r["boundary_margin"],r["classification_confidence"],
                        r["classification_method"],r["boundary_signals_fired"],
                        "|".join(r.get("special_tags",[])),
                        r.get("ingest_mode","standard"),r["output_file"]])
    return jp, cp


# ═══════════════════════════════════════════════════════════════════
# 10. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run(input_pdf, creds, project, location, out_dir,
         max_pages=None, gemini2_model="gemini-embedding-2",
         use_gemini2=True, dpi=150, max_workers=8,
         gemini2_location="us",
         taxonomy_path="pss_taxonomy.json",
         prototypes_path="pss_prototypes.json",
         accumulate_prototypes=False,
         use_ocr=False):

    timing = {}
    t0_total = time.time()

    print(f"\nPSS Pipeline v2: {Path(input_pdf).name}")
    print(f"Project: {project}  |  Workers: {max_workers}  |  DPI: {dpi}")

    print("Authenticating...")
    creds_obj = build_credentials(creds)
    text_client = build_client(creds_obj, project, location)
    img_session = build_image_session(creds_obj)
    g2_client   = build_client(creds_obj, project, gemini2_location)

    # Load classifier (taxonomy + prototypes)
    tax_path   = Path(taxonomy_path)
    proto_path = Path(prototypes_path)
    if not tax_path.is_absolute():
        tax_path = Path(input_pdf).parent / taxonomy_path
    if not proto_path.is_absolute():
        proto_path = Path(input_pdf).parent / prototypes_path
    if not tax_path.exists():
        # Fall back to same directory as the script
        tax_path = Path(__file__).parent / taxonomy_path
    print(f"Taxonomy:   {tax_path}")
    print(f"Prototypes: {proto_path}")
    # accumulate_prototypes defaults to False: prototype auto-accumulation is
    # disabled until a clean, human-verified prototype set is seeded (see
    # pss_classifier.py Classifier docstring). Pass --accumulate-prototypes
    # on the CLI to deliberately re-enable it for a run you trust.
    clf = Classifier(str(tax_path), str(proto_path), accumulate=accumulate_prototypes)
    print(f"Classifier: {clf.stats()['taxonomy_doc_types']} doc types, "
          f"{clf.stats()['prototype_types_loaded']} prototype types loaded "
          f"(accumulate={clf.accumulate})")

    doc = fitz.open(input_pdf)
    total = len(doc)
    process = total if max_pages is None else min(max_pages, total)
    print(f"Pages: {total} (processing {process})")

    # Stage 0a: OCR fallback (optional). Only the scanned/textless pages
    # are sent, so a clean PDF makes zero API calls. Runs before Stage 0
    # so extract_det_features sees recovered text from page 1 onward.
    ocr_override = {}
    if use_ocr:
        need = pages_needing_ocr(doc, process)
        if need:
            print(f"\nStage 0a: OCR fallback on {len(need)} textless page(s): {need}")
            try:
                from ocr import ocr_pages
                ocr_override = ocr_pages(creds_obj, project, doc, need)
                recovered = sum(1 for v in ocr_override.values() if v)
                print(f"  recovered text on {recovered}/{len(need)} pages")
            except Exception as e:
                print(f"  OCR fallback failed ({e}); continuing without it")
                ocr_override = {}
        else:
            print("\nStage 0a: OCR fallback enabled, but no textless pages found.")

    # Stage 0
    t0 = time.time()
    print("\nStage 0: deterministic features...")
    det = extract_det_features(doc, process, ocr_text_override=ocr_override)
    timing["stage0_seconds"] = round(time.time()-t0, 2)
    n_blank = sum(1 for f in det if f["is_blank"])
    print(f"  Done ({timing['stage0_seconds']}s) - {n_blank} blank pages skipped from embedding")

    # Stage 1 (parallel)
    t0 = time.time()
    print(f"\nStage 1: parallel embedding ({max_workers} workers)...")
    page_embs, cost_tracker = embed_all_pages_parallel(
        doc, det, g2_client, img_session, text_client,
        project, location, gemini2_model, use_gemini2, dpi, max_workers)
    timing["stage1_seconds"] = round(time.time()-t0, 2)
    print(f"  Done ({timing['stage1_seconds']}s)")

    # Similarity + scoring
    t0 = time.time()
    non_blank = [f["page_num"] for f in det if not f["is_blank"]]
    img_series = sim_chain(page_embs, non_blank, "image_embedding")
    txt_series = sim_chain(page_embs, non_blank, "text_embedding")
    img_flags  = flag_sims(img_series)
    txt_flags  = flag_sims(txt_series)
    scored     = score_all(det, img_flags, txt_flags)
    timing["scoring_seconds"] = round(time.time()-t0, 2)

    # Segmentation + Classification
    t0 = time.time()
    segments = build_segments(det, scored)
    classifications = [
        clf.classify_segment(det, s["start"], s["end"], page_embs)
        for s in segments
    ]
    timing["segmentation_seconds"] = round(time.time()-t0, 2)

    # Persist learned prototypes so future runs get better classification
    clf.save_prototypes()
    print(f"  Prototypes saved: {clf.stats()['pending_updates']} new vectors → {proto_path}")

    # Split
    t0 = time.time()
    out = Path(out_dir)
    files = split_and_write(doc, segments, classifications, out)
    timing["splitting_seconds"] = round(time.time()-t0, 2)
    doc.close()

    timing["total_seconds"] = round(time.time()-t0_total, 2)
    cost = cost_tracker.summary()
    run_meta = {
        "input_pdf": str(input_pdf),
        "pipeline_version": "v2",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_pages": total, "pages_processed": process,
        "blank_pages_skipped": n_blank,
        "segments_found": len(segments),
        "project": project,
    }
    jp, cp = write_report(out, run_meta, det, scored, segments, classifications,
                           cost, timing, files)

    print(f"\n{'='*60}")
    print(f"DONE: {len(segments)} segments | "
          f"{timing['total_seconds']}s total | "
          f"${cost['total_cost_usd']:.4f}")
    print(f"  Stage 0 (det):   {timing['stage0_seconds']}s")
    print(f"  Stage 1 (emb):   {timing['stage1_seconds']}s  ← was {timing['stage1_seconds']}s sequential")
    print(f"  Scoring:         {timing['scoring_seconds']}s")
    print(f"  Report:          {jp}")
    print(f"  CSV:             {cp}")

    return {"report": str(jp), "csv": str(cp), "pdfs": files}


# ═══════════════════════════════════════════════════════════════════
# 11. CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PSS Pipeline v2")
    ap.add_argument("input_pdf")
    ap.add_argument("--creds",      required=True)
    ap.add_argument("--project",    required=True)
    ap.add_argument("--location",   default="us-central1")
    ap.add_argument("--output-dir", default="./pss_output")
    ap.add_argument("--limit",      type=int, default=None)
    ap.add_argument("--workers",    type=int, default=8,
                    help="Parallel embedding workers (default 8; increase for higher quota)")
    ap.add_argument("--gemini2-model",     default="gemini-embedding-2")
    ap.add_argument("--gemini2-location",  default="us")
    ap.add_argument("--no-gemini2",        action="store_true")
    ap.add_argument("--dpi",               type=int, default=150)
    ap.add_argument("--taxonomy",          default="pss_taxonomy.json",
                    help="Path to document type taxonomy JSON (default: pss_taxonomy.json "
                         "in same folder as the input PDF, or the script folder)")
    ap.add_argument("--prototypes",        default="pss_prototypes.json",
                    help="Path to learned prototype vectors JSON (created/updated each run)")
    ap.add_argument("--accumulate-prototypes", action="store_true",
                    help="Allow this run to save new prototype vectors to --prototypes. "
                         "OFF by default because auto-accumulation was found to learn from "
                         "wrong classifications. Only pass this once you've manually verified "
                         "a run's output and want to seed prototypes from it.")
    ap.add_argument("--use-ocr", action="store_true",
                    help="Enable the Document AI OCR fallback for scanned/textless "
                         "pages. OFF by default. Only the pages with no extractable "
                         "text are sent, so a clean PDF makes no OCR calls.")
    args = ap.parse_args()
    run(args.input_pdf, args.creds, args.project, args.location,
        args.output_dir, max_pages=args.limit, gemini2_model=args.gemini2_model,
        use_gemini2=not args.no_gemini2, dpi=args.dpi, max_workers=args.workers,
        gemini2_location=args.gemini2_location,
        taxonomy_path=args.taxonomy, prototypes_path=args.prototypes,
        accumulate_prototypes=args.accumulate_prototypes,
        use_ocr=args.use_ocr)