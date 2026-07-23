"""
OCR fallback via Google Document AI.

Used ONLY as a fallback: the pipeline calls this for the handful of pages
that have no usable extractable text and are not blank (scanned images).
Pages that already have good native text are never sent here, so on a
clean PDF this module makes zero API calls and costs nothing.

Design notes:
- Reuses the pipeline's existing service-account credentials (the
  cloud-platform scope already covers Document AI). No separate auth.
- Synchronous process_document has a per-request page cap (~15). Real
  PDFs are far larger, so we never send the whole file: we build a small
  temporary PDF of just the pages that need OCR and send that, chunked to
  stay under the cap.
"""
from google.api_core.client_options import ClientOptions
from google.cloud import documentai

import fitz  # PyMuPDF, already a pipeline dependency

# The Document OCR processor (not the Layout one). Document AI uses the
# "us" multi-region endpoint (not a compute region like us-central1).
OCR_LOCATION      = "us"
OCR_PROCESSOR_ID  = "4742e9cdd660aea3"
OCR_PAGE_LIMIT    = 15   # sync process_document cap; we chunk to respect it


def build_ocr_client(credentials, location=OCR_LOCATION):
    """A Document AI client using the pipeline's own credentials."""
    return documentai.DocumentProcessorServiceClient(
        credentials=credentials,
        client_options=ClientOptions(
            api_endpoint=f"{location}-documentai.googleapis.com"),
    )


def _per_page_text(document):
    """Split Document AI's single text blob into one string per page,
    using each page's layout text anchors."""
    full = document.text
    out = []
    for page in document.pages:
        segs = page.layout.text_anchor.text_segments
        txt = "".join(
            full[int(s.start_index):int(s.end_index)] for s in segs)
        out.append(txt.strip())
    return out


def ocr_pages(credentials, project_id, doc, page_numbers,
              location=OCR_LOCATION, processor_id=OCR_PROCESSOR_ID,
              page_limit=OCR_PAGE_LIMIT):
    """OCR specific pages of an already-open PyMuPDF document.

    doc:          an open fitz.Document
    page_numbers: 1-based page numbers to OCR
    Returns {page_num: recovered_text}. Empty dict if page_numbers is empty.
    Chunks the request so it never exceeds the online page limit.
    """
    if not page_numbers:
        return {}

    client = build_ocr_client(credentials, location)
    name = client.processor_path(project_id, location, processor_id)
    result = {}

    for i in range(0, len(page_numbers), page_limit):
        chunk = page_numbers[i:i + page_limit]
        sub = fitz.open()
        for pn in chunk:
            sub.insert_pdf(doc, from_page=pn - 1, to_page=pn - 1)
        pdf_bytes = sub.tobytes()
        sub.close()

        raw = documentai.RawDocument(content=pdf_bytes,
                                     mime_type="application/pdf")
        resp = client.process_document(
            documentai.ProcessRequest(name=name, raw_document=raw))
        for pn, txt in zip(chunk, _per_page_text(resp.document)):
            result[pn] = txt

    return result
