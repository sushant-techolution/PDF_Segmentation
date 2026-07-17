"""
PSS Classifier — Document Type Classification Module
======================================================
A standalone classification layer that works on top of the segmentation
output from pss_pipeline.py. Completely decoupled from segmentation so
both can evolve independently.

Design principles
-----------------
1. Config-driven, not code-driven. Everything about what document types
   exist, what keywords identify them, and what visual signals matter
   lives in pss_taxonomy.json. Adding a new document type = adding a
   JSON entry. No Python changes required.

2. Three-pass cascade (fast→reliable→fallback):
   Pass 1 — Anchor phrase match (fastest, near-100% precision)
             Unique phrases that identify a document type with certainty.
             One match is sufficient. Runs on already-extracted text.
   Pass 2 — Engineering drawing detection (visual fingerprint, free)
             Single large scanned image on a landscape page = CAD drawing.
             This is MORE reliable than embeddings for this type because
             the signal is structural, not semantic.
   Pass 3 — Prototype similarity (embedding-based, requires prior runs)
             Cosine similarity between segment centroid and stored
             prototypes. Only runs when passes 1 and 2 produce no result.
   Pass 4 — Keyword scoring (fallback, always available)
             Weighted keyword hit count across all candidates.
   Pass 5 — Default ("Supporting / Misc" with LOW confidence)

3. Self-learning prototypes. Every segment classified with HIGH confidence
   (via passes 1, 2, or 3) saves its embedding centroid as a prototype
   for that document type. Next run, Pass 3 uses these prototypes.
   The prototype file grows automatically — no manual labeling required.
   New document types in the config start with empty prototypes and
   bootstrap from keyword/anchor matches on first encounters.

4. Special tags are orthogonal to type. Engineering drawing and
   contains_tables are additive properties — a segment can be an
   "Advice of Award" AND "contains_tables". Tags are computed
   independently of the type classification.

Usage (standalone)
------------------
    from pss_classifier import Classifier
    clf = Classifier("pss_taxonomy.json", "pss_prototypes.json")
    result = clf.classify_segment(det_features, seg_start, seg_end, page_embeddings)
    # result: {"type_id", "label", "confidence", "method", "special_tags", "ingest_mode"}

Usage (batch, at end of pipeline run)
--------------------------------------
    clf.save_prototypes()   # persist learned prototypes after each run
"""

import json
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def cosine_sim(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    if a.shape != b.shape or a.shape[0] == 0:
        return 0.0
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def segment_centroid(page_embeddings: dict, start: int, end: int, key: str = "image_embedding"):
    """Average of all non-None embedding vectors in the segment range."""
    vecs = []
    for pnum in range(start, end + 1):
        emb = page_embeddings.get(pnum, {}).get(key)
        if emb is not None:
            vecs.append(np.array(emb, dtype=float))
    if not vecs:
        return None
    centroid = np.mean(vecs, axis=0)
    norm = np.linalg.norm(centroid)
    return (centroid / norm).tolist() if norm > 0 else centroid.tolist()


# ═══════════════════════════════════════════════════════════════════
# ENGINEERING DRAWING VISUAL FINGERPRINT
# ═══════════════════════════════════════════════════════════════════

def is_engineering_drawing_page(page_feature: dict) -> bool:
    """
    Rock-solid visual fingerprint for scanned CAD/architectural drawings.
    Validated empirically against PDF-004:

    Engineering drawing pages:  1 image, ~2200x1700px (3.7M pixels), landscape, 0 draws
    Budget/financial tables:    many small images OR thousands of vector draws
    Text documents:             0-2 small images OR logo-sized images

    Key discriminator: a SINGLE large scanned image covering most of a
    landscape page. This is structurally unique — no other document type
    in the NYC government contract taxonomy has this profile.
    """
    if page_feature.get("orientation") != "landscape":
        return False
    if page_feature.get("drawing_count", 0) > 20:
        return False  # high draw count = vector-rendered document, not a scan

    img_infos = page_feature.get("image_infos", [])
    if len(img_infos) != 1:
        return False  # engineering drawing = exactly 1 scanned image

    iw = img_infos[0].get("width", 0)
    ih = img_infos[0].get("height", 0)
    pixel_area = iw * ih
    return pixel_area > 1_500_000  # ~1200x1250 minimum — real CAD scans are 2200x1700+


def segment_has_engineering_drawings(det_features: list, start: int, end: int) -> bool:
    seg = [f for f in det_features if start <= f["page_num"] <= end]
    return any(is_engineering_drawing_page(f) for f in seg)


def engineering_drawing_fraction(det_features: list, start: int, end: int) -> float:
    seg = [f for f in det_features if start <= f["page_num"] <= end and not f["is_blank"]]
    if not seg:
        return 0.0
    drawing_pages = sum(1 for f in seg if is_engineering_drawing_page(f))
    return drawing_pages / len(seg)


def segment_has_tables(det_features: list, start: int, end: int,
                        min_draw_count: int = 150) -> bool:
    """
    Two signals that indicate tabular data:
    1. High vector-drawing count (FMS forms rendered as vectors)
    2. Detected tables via pdfplumber (stored as table_count in features, if present)

    min_draw_count is read from pss_taxonomy.json's
    special_tags.contains_tables.triggers.min_draw_count by the caller
    (Classifier._compute_special_tags) so it can be tuned without a code
    change. Previously this was hardcoded to 150 and silently ignored
    whatever value was in the taxonomy file — editing the config had no
    effect. Default here stays 150 so behavior is unchanged for any
    caller that doesn't pass a config value explicitly.
    """
    seg = [f for f in det_features if start <= f["page_num"] <= end]
    for f in seg:
        if f.get("drawing_count", 0) > min_draw_count:
            return True
        if f.get("table_count", 0) > 0:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════
# MAIN CLASSIFIER CLASS
# ═══════════════════════════════════════════════════════════════════

class Classifier:
    """
    Config-driven, self-learning document type classifier.

    taxonomy_path:  path to pss_taxonomy.json (the document type config)
    prototypes_path: path to pss_prototypes.json (learned per run, auto-saved)
                     Created fresh if it doesn't exist yet.
    """

    def __init__(self, taxonomy_path: str, prototypes_path: str, accumulate: bool = False):
        """
        accumulate: if False (default), classify_segment() will NEVER queue
        new prototype vectors, regardless of confidence. This was disabled
        by default because auto-accumulation was learning from wrong
        classifications and poisoning pss_prototypes.json (e.g. the p185
        ACORD misclassification traced to a bad prototype saved from an
        earlier run). Pass accumulate=True only after you've manually
        verified a run's classifications are correct and want to seed
        prototypes from it.
        """
        self.taxonomy_path = Path(taxonomy_path)
        self.prototypes_path = Path(prototypes_path)
        self.accumulate = accumulate
        self._load_taxonomy()
        self._load_prototypes()
        self._pending_prototype_updates = []  # accumulated during a run

    def _load_taxonomy(self):
        with open(self.taxonomy_path) as f:
            raw = json.load(f)
        self.config = raw.get("classification", {})
        self.proto_min_conf = self.config.get("prototype_min_confidence", 0.82)
        self.kw_min_hits = self.config.get("keyword_min_hits", 1)
        self.low_conf_thresh = self.config.get("low_confidence_threshold", 0.65)
        self.proto_save_min_pages = self.config.get("prototype_min_pages_to_save", 2)
        self.doc_types = raw.get("document_types", [])
        self.type_map = {dt["id"]: dt for dt in self.doc_types}
        self.tag_config = raw.get("special_tags", {})

    def _load_prototypes(self):
        """Load prototype vectors from the persistence file.
        Structure: {type_id: [[vec1], [vec2], ...]}
        """
        self.prototypes = {}
        if self.prototypes_path.exists():
            try:
                with open(self.prototypes_path) as f:
                    raw = json.load(f)
                for type_id, vecs in raw.items():
                    if type_id in self.type_map and vecs:
                        self.prototypes[type_id] = [np.array(v, dtype=float) for v in vecs]
            except Exception as e:
                print(f"  [classifier] Warning: could not load prototypes: {e}")

    def save_prototypes(self):
        """Persist all accumulated prototype updates to disk."""
        existing = {}
        if self.prototypes_path.exists():
            try:
                with open(self.prototypes_path) as f:
                    existing = json.load(f)
            except Exception:
                pass

        for type_id, vec in self._pending_prototype_updates:
            if type_id not in existing:
                existing[type_id] = []
            existing[type_id].append(vec)
            # Keep at most 20 prototypes per type (oldest entries removed)
            existing[type_id] = existing[type_id][-20:]

        with open(self.prototypes_path, "w") as f:
            json.dump(existing, f)
        self._pending_prototype_updates = []

    def _accumulate_prototype(self, type_id: str, centroid: list):
        """Queue a centroid vector to be saved as a prototype after the run."""
        if centroid is not None and type_id in self.type_map:
            self._pending_prototype_updates.append((type_id, centroid))

    # ── Pass 1: Anchor phrase matching ──────────────────────────────

    def _find_anchor_matches(self, text_lower: str) -> list:
        """Return all (phrase_len, doc_type, phrase) anchor matches in text_lower."""
        candidates = []
        for dt in self.doc_types:
            if dt["id"] == "supporting_misc":
                continue
            for phrase in dt.get("anchor_phrases", []):
                if phrase.lower() in text_lower:
                    candidates.append((len(phrase), dt, phrase))
        return candidates

    def _pass1_anchor(self, bm_titles: str, body_text: str) -> Optional[dict]:
        """
        Matches unique anchor phrases that identify a document type with
        near-certainty.

        Bookmark-title matches are checked FIRST and win outright over
        body-text matches, regardless of phrase length. Rationale (fixes
        the p130 bug: a 12-page Change Order Request was misclassified as
        Responsibility Determination): the old version searched bookmark
        titles and body text as one combined blob and picked whichever
        anchor phrase was longest. On p130 the bookmark correctly read
        "CHANGE ORDER REQUEST", but body text on page 1 happened to contain
        the longer phrase "responsibility determination" (29 chars vs 21),
        so length alone picked the wrong type even though the bookmark had
        the right answer. A bookmark/TOC entry is a stronger evidence
        source than an incidental phrase inside body text, so it should
        never lose to a body match purely because it's a shorter string.

        If no anchor phrase matches the bookmark title, this falls back to
        the original longest-match-in-body-text behavior unchanged — so
        segments with generic or mislabeled bookmarks (e.g. a bookmark
        literally titled "AGREEMENT" on what is actually an Advice of
        Award) still get classified from body text exactly as before.
        Verified against every bookmark title in the current ground-truth
        set to confirm this doesn't change any other segment's outcome.
        """
        bm_candidates = self._find_anchor_matches(bm_titles.lower()) if bm_titles.strip() else []
        if bm_candidates:
            bm_candidates.sort(key=lambda x: -x[0])
            _, best_dt, matched_phrase = bm_candidates[0]
            return {
                "type_id": best_dt["id"],
                "label": best_dt["label"],
                "confidence": "HIGH",
                "method": f"anchor_phrase_bookmark:{matched_phrase[:40]}",
                "ingest_mode": best_dt.get("ingest_mode", "standard"),
            }

        body_candidates = self._find_anchor_matches(body_text.lower())
        if not body_candidates:
            return None
        body_candidates.sort(key=lambda x: -x[0])
        _, best_dt, matched_phrase = body_candidates[0]
        return {
            "type_id": best_dt["id"],
            "label": best_dt["label"],
            "confidence": "HIGH",
            "method": f"anchor_phrase_body:{matched_phrase[:40]}",
            "ingest_mode": best_dt.get("ingest_mode", "standard"),
        }

    # ── Pass 2: Engineering drawing visual fingerprint ───────────────

    def _pass2_visual_fingerprint(self, det_features: list,
                                   start: int, end: int) -> Optional[dict]:
        """
        If the majority of non-blank pages in the segment are engineering
        drawings (per visual fingerprint), classify as Engineering Drawing.
        Threshold at 0.6 so that a segment that STARTS with a cover page
        and then has drawings still gets classified correctly.
        """
        frac = engineering_drawing_fraction(det_features, start, end)
        if frac >= 0.60:
            return {
                "type_id": "engineering_drawing",
                "label": "Engineering Drawing / Construction Plan",
                "confidence": "HIGH",
                "method": f"visual_fingerprint:{frac:.0%}_drawing_pages",
                "ingest_mode": "engineering_drawing",
            }
        return None

    # ── Pass 3: Prototype similarity (embedding-based) ──────────────

    def _pass3_prototype(self, centroid: Optional[list]) -> Optional[dict]:
        """
        Compare segment centroid against stored prototypes.
        Requires at least one prior run to have accumulated prototypes.
        """
        if centroid is None or not self.prototypes:
            return None

        centroid_vec = np.array(centroid, dtype=float)
        best_type_id, best_score = None, 0.0

        for type_id, proto_vecs in self.prototypes.items():
            if not proto_vecs or type_id not in self.type_map:
                continue
            # Average similarity against all stored prototypes for this type
            sims = [cosine_sim(centroid_vec, pv) for pv in proto_vecs]
            avg_sim = float(np.mean(sims))
            if avg_sim > best_score:
                best_score = avg_sim
                best_type_id = type_id

        if best_type_id is None or best_score < self.low_conf_thresh:
            return None

        dt = self.type_map[best_type_id]
        confidence = ("HIGH" if best_score >= self.proto_min_conf else
                       "MEDIUM" if best_score >= self.low_conf_thresh else "LOW")
        return {
            "type_id": best_type_id,
            "label": dt["label"],
            "confidence": confidence,
            "method": f"prototype_similarity:{best_score:.3f}",
            "ingest_mode": dt.get("ingest_mode", "standard"),
        }

    # ── Pass 4: Keyword scoring ──────────────────────────────────────

    def _pass4_keywords(self, combined_text: str) -> Optional[dict]:
        text_lower = combined_text.lower()
        scores = {}
        for dt in self.doc_types:
            if dt["id"] == "supporting_misc":
                continue
            hits = sum(1 for kw in dt.get("keywords", []) if kw.lower() in text_lower)
            if hits >= self.kw_min_hits:
                scores[dt["id"]] = hits

        if not scores:
            return None

        best_id = max(scores, key=scores.get)
        dt = self.type_map[best_id]
        confidence = "MEDIUM" if scores[best_id] >= 2 else "LOW"
        return {
            "type_id": best_id,
            "label": dt["label"],
            "confidence": confidence,
            "method": f"keyword_score:{scores[best_id]}_hits",
            "ingest_mode": dt.get("ingest_mode", "standard"),
        }

    # ── Special tag computation ──────────────────────────────────────

    def _compute_special_tags(self, det_features: list, start: int, end: int,
                                base_type_id: str) -> list:
        tags = []
        # Engineering drawing — structural visual fingerprint, very reliable
        if segment_has_engineering_drawings(det_features, start, end):
            tags.append("engineering_drawing")
        # Tables — only from actual page-level detection (draw count or pdfplumber).
        # Threshold now actually comes from taxonomy config (see segment_has_tables
        # docstring) instead of being silently hardcoded.
        table_min_draw = (self.tag_config.get("contains_tables", {})
                                          .get("triggers", {})
                                          .get("min_draw_count", 150))
        if segment_has_tables(det_features, start, end, min_draw_count=table_min_draw):
            tags.append("contains_tables")
        # Do NOT add inherited special_tags from taxonomy here.
        # Taxonomy special_tags are descriptive metadata about what the type
        # *typically* contains, not a detection result for this specific segment.
        return tags

    # ── Main classify entrypoint ─────────────────────────────────────

    def classify_segment(self, det_features: list, start: int, end: int,
                          page_embeddings: dict) -> dict:
        """
        Classify a single segment and return a full classification result.
        Also accumulates prototype updates for save_prototypes() to persist.

        Returns dict with keys:
          type_id, label, confidence, method, special_tags, ingest_mode
        """
        seg_feats = [f for f in det_features if start <= f["page_num"] <= end]

        # Build text for Pass 1 and 4: bookmark titles + first 3 pages of text
        bm_titles = " ".join(f["bookmark_title"] for f in seg_feats
                              if f.get("bookmark_title"))
        text_pages = [f["native_text"] for f in seg_feats
                       if f.get("native_text") and not f["is_blank"]]
        first_pages_text = " ".join(text_pages[:3])
        combined_text = bm_titles + " " + first_pages_text  # Pass 4 keyword scoring still uses the union

        # Build segment centroid for Pass 3
        centroid = segment_centroid(page_embeddings, start, end, "image_embedding")
        page_count = sum(1 for f in seg_feats if not f["is_blank"])

        # A segment opening on an EXHIBIT/SCHEDULE/ATTACHMENT title page is
        # an attachment by definition. Don't let 1-2 incidental keyword
        # hits (generic boilerplate like "agreement"/"shall") force it into
        # an unrelated type — fall back to Supporting/Misc unless the
        # keyword match is actually strong (3+ hits).
        opens_with_title_anchor = bool(seg_feats and seg_feats[0].get("title_anchor"))

        def _guarded_pass4():
            r = self._pass4_keywords(combined_text)
            if r and opens_with_title_anchor:
                hits = int(r["method"].split(":")[1].split("_")[0])
                if hits < 3:
                    return None
            return r

        # ── Cascade ──
        result = (
            self._pass1_anchor(bm_titles, first_pages_text)
            or self._pass2_visual_fingerprint(det_features, start, end)
            or self._pass3_prototype(centroid)
            or _guarded_pass4()
        )

        if result is None:
            misc = self.type_map.get("supporting_misc", {})
            result = {
                "type_id": "supporting_misc",
                "label": misc.get("label", "Supporting / Misc"),
                "confidence": "LOW",
                "method": "default",
                "ingest_mode": misc.get("ingest_mode", "standard"),
            }

        # Coarser grouping than type_id (several types roll up to e.g.
        # "Compliance / Regulatory"). Comes from taxonomy, not code.
        result["category"] = self.type_map.get(result["type_id"], {}).get(
            "category", "Supporting / Misc")

        # Compute special tags (always, orthogonal to type)
        result["special_tags"] = self._compute_special_tags(
            det_features, start, end, result["type_id"])

        # Accumulate prototype if high confidence and enough non-blank pages,
        # OR if classified by anchor phrase (anchor = high precision regardless of length)
        should_save = (self.accumulate and
                       result["confidence"] == "HIGH" and centroid is not None and
                       (page_count >= self.proto_save_min_pages or
                        result.get("method", "").startswith("anchor_phrase") or
                        result.get("method", "").startswith("visual_fingerprint")))
        if should_save:
            self._accumulate_prototype(result["type_id"], centroid)

        return result

    def reload_taxonomy(self):
        """Hot-reload taxonomy without restarting. Useful for batch runs
        where the config may be updated between documents."""
        self._load_taxonomy()

    def stats(self) -> dict:
        """Summary of loaded prototypes for logging."""
        return {
            "taxonomy_doc_types": len(self.doc_types),
            "prototype_types_loaded": len(self.prototypes),
            "prototype_vectors_total": sum(len(v) for v in self.prototypes.values()),
            "pending_updates": len(self._pending_prototype_updates),
        }