"""
PSS Pipeline v2 — Page Stream Segmentation & Classification
=============================================================
What changed from v1 and why (traceable to real run results):

  FIX 1  Bookmark depth filter [CRITICAL — fixed 57/63 FPs]
         Root cause: the IRS 990 block in PDF-003 had L3 bookmarks on
         every single page (67-123), each firing struct=1.0 and creating
         a 1-page segment. Only L1 (top-level) TOC entries are treated
         as document-boundary signals. L2+ are ignored.
         Impact: precision 19% → ~71% with recall staying 100%.

  FIX 2  Consecutive same-title bookmark suppression
         Root cause: PDF-003 had two consecutive L1 entries both titled
         "AGREEMENT" at p3 (US Letter) and p4 (A4) — same logical
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
         in correct page order as before — parallelism is only on the
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
         No existing signal caught this — same page format, no bookmark,
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
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import fitz
import numpy as np
import requests

from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession

# Classification module — self-contained, config-driven
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

# Signal weights — grounded in validation results, see inline evidence
WEIGHTS = {
    "structural_change":        1.0,   # bookmarks (L1 only), size/orient changes: 0 FP observed
    "structural_change_return": 0.4,   # format returning to a prior format within 5 pages (FIX 3)
    "page_numbering_reset":     0.8,   # X/Y total gated: strong but not 1.0 alone
    "embedding_high_z":         1.0,   # |z|>4.0: near-perfect precision, safe standalone
    "embedding_moderate_each":  0.5,   # image + text scored independently, summed
    "drawing_density_spike":    0.3,   # weak, exploratory
    "consecutive_flag_penalty": -0.5,  # real boundaries don't sustain across 2 pages
}

BOUNDARY_THRESHOLD  = 1.0
Z_WINDOW            = 10
Z_THRESHOLD         = -2.0   # raised from -1.5 (FIX 4)
Z_HIGH_THRESHOLD    = -4.0
MIN_STD             = 0.02
MIN_ABSOLUTE_DROP   = 0.07   # raised from 0.05 (FIX 4)

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
                "Some token counts ESTIMATED — verify against GCP Billing."
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
# 4. STAGE 0 — DETERMINISTIC FEATURES
# ═══════════════════════════════════════════════════════════════════

PAGE_PAT_TOTAL = [
    re.compile(r'\bpage\s+(\d+)\s+of\s+(\d+)\b', re.IGNORECASE),
    re.compile(r'\bpg\.?\s*(\d+)\s*/\s*(\d+)\b', re.IGNORECASE),
]
PAGE_PAT_BARE = [
    re.compile(r'^-\s*(\d+)\s*-$'),
    re.compile(r'^\s*(\d+)\s*$'),
]

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
            # embedded image formats — common in engineering drawing PDFs
            # with unusual compression or inline images. Treat as non-zero
            # image area (i.e. page is NOT blank) to be safe.
            ia += pa * img_area_thresh + 1
    if pa > 0 and ia/pa > img_area_thresh: return False
    return len(page.get_drawings()) <= 5

def size_bucket(w, h):
    ls, ss = max(w,h), min(w,h)
    if ls > 1000: return "WIDE"
    # FIX 11: relative tolerance instead of flat 8pt — rescanned signature
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
    return None

# FIX 10: sub-document title markers (EXHIBIT A, SCHEDULE 1, ...). Only
# matched against the first content line so a mid-sentence body reference
# like "...attached hereto as Exhibit A..." doesn't fire.
TITLE_ANCHOR_PAT = re.compile(
    r'^(EXHIBIT|SCHEDULE|ATTACHMENT|APPENDIX|ANNEX|ADDENDUM)\b')

def detect_title_anchor(page, max_lines=3):
    text = page.get_text().strip()
    for line in [l.strip() for l in text.splitlines() if l.strip()][:max_lines]:
        if TITLE_ANCHOR_PAT.match(line) and line.isupper():
            return line[:80]
    return None

def get_bookmark_map(doc):
    """Returns {0-based-idx: (title, level)}.

    Two rules derived from real client PDF analysis:

    Rule 1 — Highest level wins per page (not first-encountered).
    The original first-entry-wins approach was wrong: PDF-003's p66 had
    a L2 'Federal' entry AND multiple L3 entries all pointing to page 66.
    The L3 entry happened to appear first in the TOC stream, so it won,
    causing the L2 bookmark to be silently discarded. Fix: for any page
    where multiple TOC entries exist, keep the one with the LOWEST level
    number (i.e. highest in the hierarchy).

    Rule 2 — Trust L1 AND L2 as boundary signals.
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

def extract_det_features(doc, max_pages):
    bm_map = get_bookmark_map(doc)
    features = []
    prev_label = None
    prev_fmt_stack = []   # last 5 format labels (for FIX 3)
    prev_l1_title = None  # last L1 bookmark title (for FIX 2)
    prev_numbering = None

    for i in range(min(max_pages, len(doc))):
        page = doc[i]
        blank = is_blank(page)
        fmt = page_fmt(page)
        bm_entry = bm_map.get(i)
        bm_title = bm_entry[0] if bm_entry else None
        bm_level = bm_entry[1] if bm_entry else None

        # FIX 2: suppress if same title as previous boundary-level bookmark
        # AND no format change (avoids splitting continuation pages that share
        # the same document-level bookmark title as the page before)
        same_title_suppressed = False
        if bm_level is not None and bm_level <= 2 and bm_title == prev_l1_title:
            same_title_suppressed = True

        # FIX 1+2: trust L1 and L2 bookmarks as boundary signals.
        # L3+ are sub-section page markers (e.g. individual 990 schedule pages)
        # and produce false positives when treated as document boundaries.
        # L2 is needed because PDF-003's 990 block anchor ('Federal') is L2.
        has_boundary_bookmark = (bm_level is not None and bm_level <= 2
                                  and not same_title_suppressed)

        # FIX 3: format returning to a recent format (within 5 pages)
        format_changed = prev_label is not None and fmt["label"] != prev_label
        format_returning = format_changed and fmt["label"] in prev_fmt_stack

        ht, ft = hf_text(page)
        numbering = (parse_numbering_total(ft) or parse_numbering_total(ht))
        if numbering is None:
            bare = parse_numbering_bare(ft) or parse_numbering_bare(ht)
            if bare is not None and prev_numbering is not None and bare == prev_numbering[0]+1:
                numbering = (bare, prev_numbering[1])

        num_reset = False
        if numbering and prev_numbering:
            pc,pt = prev_numbering; cc,_ = numbering
            if pt is not None and pc==pt and cc==1: num_reset = True
            elif cc < pc or cc > pc+1: num_reset = True

        draw_count = len(page.get_drawings())
        img_count  = len(page.get_images(full=True))
        prev_dc    = features[-1]["drawing_count"] if features else None
        draw_spike = (prev_dc and prev_dc>0 and draw_count>prev_dc*3 and draw_count>200)

        # Collect per-image pixel dimensions for engineering drawing detection.
        # Uses get_image_info() which returns pixel dimensions without triggering
        # the bbox ValueError that affects some CAD PDF embedded image formats.
        img_infos = []
        try:
            for info in page.get_image_info(xrefs=True):
                iw = info.get("width", 0)
                ih = info.get("height", 0)
                if iw > 0 and ih > 0:
                    img_infos.append({"width": iw, "height": ih,
                                      "colorspace": info.get("colorspace", 0)})
        except Exception:
            pass  # get_image_info unavailable in older pymupdf — img_infos stays []

        text = page.get_text().strip()
        text_density = len(text) / max(page.rect.width * page.rect.height, 1)
        title_anchor = detect_title_anchor(page)

        features.append({
            "page_num":           i+1,
            "is_blank":           blank,
            "title_anchor":       title_anchor,
            "w": fmt["w"], "h": fmt["h"],
            "orientation":        fmt["orient"],
            "size_bucket":        fmt["bucket"],
            "format_label":       fmt["label"],
            "format_changed":     format_changed,
            "format_returning":   format_returning,  # FIX 3
            "has_l1_bookmark":    has_boundary_bookmark,
            "bookmark_title":     bm_title if has_boundary_bookmark else None,
            "bookmark_level":     bm_level,
            "numbering_reset":    num_reset,
            "page_num_found":     numbering[0] if numbering else None,
            "page_total_found":   numbering[1] if numbering else None,
            "drawing_count":      draw_count,
            "image_count":        img_count,
            "image_infos":        img_infos,
            "table_count":        0,   # placeholder — populated by pdfplumber if available
            "drawing_spike":      draw_spike,
            "text_char_count":    len(text),
            "text_density":       round(text_density, 8),
            "native_text":        text,
        })

        prev_label = fmt["label"]
        prev_fmt_stack = (prev_fmt_stack + [fmt["label"]])[-5:]
        if has_boundary_bookmark:
            prev_l1_title = bm_title
        if numbering: prev_numbering = numbering

    return features


# ═══════════════════════════════════════════════════════════════════
# 5. STAGE 1 — PARALLEL EMBEDDING (FIX 6)
# ═══════════════════════════════════════════════════════════════════

def render_png(page, dpi=150):
    scale = dpi / 72.0
    px = page.get_pixmap(matrix=fitz.Matrix(scale,scale), alpha=False)
    return px.tobytes("png")

def _embed_one_gemini2(client, text, image_bytes, model, page_num, cost_tracker, stats, stats_lock):
    """Single page embedding call — runs inside a thread."""
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
        except Exception as e:
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
                    except: pass
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
        # Dimension mismatch — happens when gemini-embedding-2 (3072-dim)
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

        num_score     = WEIGHTS["page_numbering_reset"] if f["numbering_reset"] else 0.0
        draw_score    = WEIGHTS["drawing_density_spike"] if f["drawing_spike"] else 0.0
        format_signal = f["format_changed"] and not f["format_returning"]

        # FIX 9: L2 bookmarks need a corroborating signal on the same page
        # to count; L1 is always trusted (0 FPs in validation). Otherwise an
        # L2 sub-heading of the same document reads the same as a genuine
        # L2 document anchor (FIX1's 990-block case).
        corroborated = bool(num_score or draw_score or format_signal or emb_score)
        trust_bookmark = f["has_l1_bookmark"] and (
            f["bookmark_level"] == 1 or corroborated)

        # FIX 10: all-caps title line at top of page (EXHIBIT A, SCHEDULE 1)
        title_anchor_hit = bool(f.get("title_anchor"))

        structural = trust_bookmark or format_signal or title_anchor_hit
        struct_score = WEIGHTS["structural_change"] if structural else (
            WEIGHTS["structural_change_return"] if f["format_changed"] else 0.0)  # FIX 3

        emb_only = emb_score > 0 and not structural and not f["numbering_reset"]
        penalty  = WEIGHTS["consecutive_flag_penalty"] if (emb_only and prev_emb_only) else 0.0

        total = max(0.0, struct_score + num_score + emb_score + draw_score + penalty)
        is_boundary = total >= BOUNDARY_THRESHOLD

        results.append({
            "page_num":           pn,
            "is_blank":           f["is_blank"],
            "structural_score":   struct_score,
            "numbering_score":    num_score,
            "embedding_score":    round(emb_score,2),
            "drawing_score":      draw_score,
            "consecutive_penalty":penalty,
            "total_score":        round(total,2),
            "is_boundary":        is_boundary,
            "image_similarity":   is_[0],
            "image_z_reason":     is_[2],
            "text_similarity":    ts_[0],
            "text_z_reason":      ts_[2],
            "bookmark_title":     f.get("bookmark_title"),
            "bookmark_level":     f.get("bookmark_level"),
            "bookmark_trusted":   bool(trust_bookmark),
            "title_anchor":       f.get("title_anchor"),
        })
        prev_emb_only = emb_only
    return results


# ═══════════════════════════════════════════════════════════════════
# 7. SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

def build_segments(det_features, scored):
    boundary_set = {r["page_num"] for r in scored if r["is_boundary"]}
    boundary_set.add(1)
    total = det_features[-1]["page_num"]
    segs, starts = [], sorted(boundary_set)
    for i, s in enumerate(starts):
        e = starts[i+1]-1 if i+1 < len(starts) else total
        segs.append({"start": s, "end": e})
    return segs


# ═══════════════════════════════════════════════════════════════════
# 8. CLASSIFICATION — delegated to pss_classifier.py
#    The Classifier is config-driven (pss_taxonomy.json) and
#    self-learning (pss_prototypes.json). See pss_classifier.py.
# ═══════════════════════════════════════════════════════════════════

# Classifier is instantiated once in run() and passed down — not a module-level
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
    'Boundary Signals Fired' — just formats fields we already computed."""
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
        elif det_row.get("format_changed"):
            sigs.append(f"format_change:{det_row.get('format_label')}")
    if scored_row["numbering_score"] > 0:
        sigs.append("page_numbering_reset")
    if scored_row["embedding_score"] > 0:
        z = scored_row.get("image_z_reason") or scored_row.get("text_z_reason")
        sigs.append(f"embedding_drop({z})" if z else "embedding_drop")
    if scored_row["drawing_score"] > 0:
        sigs.append("drawing_density_spike")
    return ";".join(sigs) if sigs else "unscored_boundary"


def write_report(out_dir, run_meta, det_features, scored, segments,
                  classifications, cost_summary, timing, files):
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
        ],
    }
    segs_out = [
        {"segment_index": i+1,
         "pdf_id": pdf_id,
         "pdf_filename": pdf_filename,
         "total_pages": run_meta["total_pages"],
         "start_page": s["start"], "end_page": s["end"],
         "page_count": s["end"]-s["start"]+1,
         "document_type_id":   c.get("type_id", "unknown"),
         "document_type_label":c.get("label", c.get("category", "Unknown")),
         "document_category":  c.get("category", "Supporting / Misc"),
         "classification_method": c.get("method", ""),
         "confidence": c.get("confidence", "LOW"),
         "boundary_signals_fired": describe_boundary(
             det_by_page.get(s["start"]), scored_by_page.get(s["start"])),
         "special_tags": c.get("special_tags", []),
         "ingest_mode":  c.get("ingest_mode", "standard"),
         "output_file": Path(f).name}
        for i,(s,c,f) in enumerate(zip(segments, classifications, files))
    ]
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
                    "confidence","classification_method","boundary_signals_fired",
                    "special_tags","ingest_mode","output_file"])
        for r in segs_out:
            w.writerow([r["pdf_id"],r["pdf_filename"],r["total_pages"],
                        r["segment_index"],r["start_page"],r["end_page"],r["page_count"],
                        r["document_type_id"],r["document_type_label"],r["document_category"],
                        r["confidence"],r["classification_method"],r["boundary_signals_fired"],
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
         accumulate_prototypes=False):

    timing = {}
    t0_total = time.time()

    print(f"\nPSS Pipeline v2 — {Path(input_pdf).name}")
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

    # Stage 0
    t0 = time.time()
    print("\nStage 0: deterministic features...")
    det = extract_det_features(doc, process)
    timing["stage0_seconds"] = round(time.time()-t0, 2)
    n_blank = sum(1 for f in det if f["is_blank"])
    print(f"  Done ({timing['stage0_seconds']}s) — {n_blank} blank pages skipped from embedding")

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
        "run_timestamp": datetime.utcnow().isoformat()+"Z",
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
    args = ap.parse_args()
    run(args.input_pdf, args.creds, args.project, args.location,
        args.output_dir, max_pages=args.limit, gemini2_model=args.gemini2_model,
        use_gemini2=not args.no_gemini2, dpi=args.dpi, max_workers=args.workers,
        gemini2_location=args.gemini2_location,
        taxonomy_path=args.taxonomy, prototypes_path=args.prototypes,
        accumulate_prototypes=args.accumulate_prototypes)