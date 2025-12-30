#!/usr/bin/env python3
"""
advanced_semantic_chunker.py — Production-ready advanced semantic chunker

Purpose:
- Produce Regulation -> Section -> Clause -> Interpretation -> Example chunks
- Extract headings, tables, formulas, bullets, compliance notes, penalties, exceptions,
  "Illustration"/"Example" paragraphs, and cross-references
- Build hierarchical document representation and per-chunk metadata suitable for RAG ingestion

Features:
- PDF/text input
- Multi-extractor fallback (PyMuPDF/pdfplumber)
- spaCy-based sentence segmentation & NER
- Camelot table extraction (optional, best-effort)
- Cross-reference resolution (regex)
- Outputs:
    - DOCID_chunks.json (flat list of chunks with metadata)
    - DOCID_hierarchy.json (nested hierarchy: sections -> clauses -> subclauses)
    - optionally saves tables as CSV/JSON (if Camelot enables)

IMPORTANT DEPENDENCIES (install before running):
pip install spacy pymupdf pdfplumber sentence-transformers camelot-py[cv] pandas
python -m spacy download en_core_web_sm

Optional (for OCR fallback):
pip install pytesseract pillow
# Install Tesseract on OS-level if using OCR

Notes:
- Camelot requires Ghostscript and tk; on Windows use conda or install manually.
- If some optional libs are not present, script falls back to text-only extraction.

Usage:
python advanced_semantic_chunker.py --input ./data/raw/mydoc.pdf --output_dir ./data/chunks
python advanced_semantic_chunker.py --input_dir ./data/raw --output_dir ./data/chunks
"""

from __future__ import annotations
import os
import re
import json
import uuid
import argparse
import logging
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

# --- optional libs (import safely) ---
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import camelot
except Exception:
    camelot = None

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None

import spacy

# Use small model by default; if you have a larger model, change this
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("advanced_chunker")

# ---------------- constants / regex ----------------
SECTION_RE = re.compile(r'^\s*(\d{1,3}(?:\.\d{1,3}){0,6})\s*[.\-:)]\s*(.*)$')  # 1 or 1.1 or 1.1.1 ...
BULLET_RE = re.compile(r'^\s*(?:[-•\*]|[0-9]+\)|\([a-z]\)|\([ivx]+\))\s+')
HEADING_RE = re.compile(r'^[A-Z][A-Z0-9 ,:\-/\(\)]{4,}$')  # all-caps-ish headings
EXAMPLE_RE = re.compile(r'^\s*(Illustration|Example|Case Study|Illustrative Example)\b', re.IGNORECASE)
NOTE_RE = re.compile(r'^\s*(Note|Important|Compliance Note|Observation)\b', re.IGNORECASE)
EXCEPTION_RE = re.compile(r'^\s*(However|Provided that|Except|Notwithstanding|Subject to)\b', re.IGNORECASE)
PENALTY_RE = re.compile(r'\b(penal[ties|ty]|fine|penalty|imprisonment|liable to pay|shall be punished|punishable)\b', re.IGNORECASE)
CROSSREF_RE = re.compile(r'\b(?:see|refer to|as per|under)\s+(?:clause|section|para|paragraph)?\s*([0-9]{1,3}(?:\.[0-9]{1,3}){0,6})', re.IGNORECASE)
FORMULA_RE = re.compile(r'[%\u00B0]|(?:\d[\d,]*\s*(?:%|per cent|per cent\.))|(?:[0-9\.\,]+\s*(?:₹|\$|INR|Rs\.?))')  # numbers with % or currency
TABLE_SIG_RE = re.compile(r'\|.*\|')  # pipe table detection
MIN_CHUNK_CHARS = 60
MAX_CHUNK_CHARS = 3000

# ---------------- load spaCy ----------------
try:
    nlp = spacy.load(SPACY_MODEL, disable=["parser"])  # keep parser off for speed; can enable later
except Exception as e:
    logger.info("Could not load spaCy model %s: %s. Attempting to download.", SPACY_MODEL, str(e))
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", SPACY_MODEL], check=True)
    nlp = spacy.load(SPACY_MODEL, disable=["parser"])

# ---------------- helper functions ----------------

def _make_id(prefix="ch") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def safe_read_text(path: str) -> str:
    """Try multi-extractor approach for PDFs; if text file, just read."""
    if path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    # PDF extraction fallback chain: PyMuPDF -> pdfplumber -> raise
    text_pages = []
    if fitz:
        try:
            doc = fitz.open(path)
            for p in doc:
                text = p.get_text("text") or ""
                text_pages.append(text)
            text_all = "\n\n".join(text_pages)
            if len(text_all.strip()) > 200:
                logger.info("Extracted text via PyMuPDF (fitz), chars=%d", len(text_all))
                return text_all
            else:
                logger.debug("PyMuPDF returned small text, falling back.")
        except Exception as e:
            logger.warning("PyMuPDF extract failed: %s", e)
    if pdfplumber:
        try:
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    txt = p.extract_text() or ""
                    text_pages.append(txt)
            text_all = "\n\n".join(text_pages)
            if len(text_all.strip()) > 200:
                logger.info("Extracted text via pdfplumber, chars=%d", len(text_all))
                return text_all
            else:
                logger.debug("pdfplumber returned small text, falling back.")
        except Exception as e:
            logger.warning("pdfplumber extract failed: %s", e)
    # OCR fallback if available
    if pytesseract:
        try:
            if fitz:
                doc = fitz.open(path)
                pages_text = []
                for i in range(len(doc)):
                    pix = doc[i].get_pixmap(dpi=200)
                    mode = "RGB" if pix.alpha == 0 else "RGBA"
                    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    txt = pytesseract.image_to_string(img)
                    pages_text.append(txt)
                text_all = "\n\n".join(pages_text)
                logger.info("Extracted text via PyMuPDF+OCR, chars=%d", len(text_all))
                return text_all
            else:
                # try pdfplumber to get images
                if pdfplumber:
                    pages_text = []
                    with pdfplumber.open(path) as pdf:
                        for p in pdf.pages:
                            try:
                                img = p.to_image(resolution=200).original
                                txt = pytesseract.image_to_string(img)
                                pages_text.append(txt)
                            except Exception:
                                pages_text.append("")
                    text_all = "\n\n".join(pages_text)
                    logger.info("Extracted text via pdfplumber+OCR, chars=%d", len(text_all))
                    return text_all
        except Exception as e:
            logger.warning("OCR fallback failed: %s", e)
    raise RuntimeError("No suitable extractor available or extraction failed for: " + path)


def normalize_text(raw: str) -> List[str]:
    """Normalize lines, fix hyphens and join broken lines into paragraphs."""
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    # remove multiple blank lines
    s = re.sub(r'\n{3,}', '\n\n', s)
    # hyphenation fix: words split across lines "exam-\nple"
    s = re.sub(r'(\w+)-\n(\w+)', r'\1\2', s)
    # join lines that belong to same paragraph heuristically
    lines = s.splitlines()
    normalized = []
    buf = ""
    for line in lines:
        line = line.strip()
        if not line:
            if buf:
                normalized.append(buf.strip())
                buf = ""
            normalized.append("")  # keep paragraph break
            continue
        # if line starts with section/bullet/heading -> flush current buffer
        if SECTION_RE.match(line) or BULLET_RE.match(line) or HEADING_RE.match(line) or EXAMPLE_RE.match(line):
            if buf:
                normalized.append(buf.strip())
                buf = ""
            normalized.append(line)
            continue
        # if buffer ends with punctuation, treat as new sentence
        if buf and re.search(r'[\.:\?\)]$', buf.strip()):
            normalized.append(buf.strip())
            buf = line
            continue
        # else join
        if buf:
            buf = buf + " " + line
        else:
            buf = line
    if buf:
        normalized.append(buf.strip())
    # remove extra empty duplicates
    out = []
    prev_empty = False
    for ln in normalized:
        if ln == "":
            if prev_empty:
                continue
            prev_empty = True
            out.append(ln)
        else:
            prev_empty = False
            out.append(ln)
    return out


# ---------------- semantic detectors ----------------

def detect_segment_type(text_line: str) -> str:
    """Return segment type label for a line/paragraph."""
    if not text_line or text_line.strip() == "":
        return "empty"
    if TABLE_SIG_RE.search(text_line) or "|" in text_line and len(text_line) < 400 and text_line.count("|") >= 2:
        return "table"
    if HEADING_RE.match(text_line) and len(text_line) < 150:
        return "heading"
    if SECTION_RE.match(text_line):
        return "section"
    if BULLET_RE.match(text_line):
        return "bullet"
    if EXAMPLE_RE.match(text_line):
        return "example"
    if NOTE_RE.match(text_line):
        return "note"
    if EXCEPTION_RE.match(text_line):
        return "exception"
    if FORMULA_RE.search(text_line):
        return "formula"
    return "paragraph"


def extract_tables_from_pdf(pdf_path: str, output_dir: Optional[str] = None) -> List[Dict]:
    """Use Camelot (if available) to extract tables from PDF. Returns list of tables with metadata."""
    tables = []
    if camelot is None:
        logger.debug("Camelot not installed; skipping table extraction.")
        return tables
    try:
        logger.info("Running Camelot table extraction on %s", pdf_path)
        # flavor stream/lattice; try lattice then stream
        try:
            tables_raw = camelot.read_pdf(pdf_path, pages='all', flavor='lattice', strip_text='\n')
            if not tables_raw or len(tables_raw) == 0:
                tables_raw = camelot.read_pdf(pdf_path, pages='all', flavor='stream', strip_text='\n')
        except Exception as e:
            logger.warning("Camelot lattice failed: %s. Trying stream.", e)
            tables_raw = camelot.read_pdf(pdf_path, pages='all', flavor='stream', strip_text='\n')

        for i, t in enumerate(tables_raw):
            try:
                df = t.df
                table_json = {
                    "table_id": _make_id("tbl"),
                    "page": t.page,
                    "shape": df.shape,
                    "data": df.to_dict(orient="records"),
                    "text_preview": "\n".join(df.astype(str).apply(lambda r: "|".join(r.values), axis=1).tolist()[:5])
                }
                tables.append(table_json)
                # optionally save CSV
                if output_dir:
                    csv_path = os.path.join(output_dir, f"{os.path.basename(pdf_path)}_table_{i}.csv")
                    t.to_csv(csv_path)
            except Exception as e:
                logger.warning("Camelot table postprocessing failed: %s", e)
    except Exception as e:
        logger.warning("Camelot extraction failed: %s", e)
    logger.info("Camelot extracted %d tables", len(tables))
    return tables


# ---------------- core chunking logic ----------------

def build_hierarchy_and_chunks(normalized_lines: List[str], doc_id: str, tables_meta: List[Dict] = None) -> Tuple[List[Dict], Dict]:
    """
    Traverse normalized lines and build:
      - flat list of chunks with metadata
      - nested hierarchy: {sections: [{section_no, title, clauses:[...]}]}
    """
    if tables_meta is None:
        tables_meta = []

    chunks = []
    hierarchy = {"doc_id": doc_id, "sections": []}
    current_section = None
    current_clause = None
    parent_stack = []  # stack of (level, obj) to allow nested subclauses

    def close_clause():
        nonlocal current_clause
        if current_clause:
            # finalize clause chunk if not already added
            # but in this design, clauses are created during processing
            current_clause = None

    def add_chunk(chunk_type, text, meta=None):
        ch = {
            "chunk_id": _make_id("ch"),
            "doc_id": doc_id,
            "type": chunk_type,
            "text": text.strip(),
            "meta": meta or {}
        }
        # compute quick flags
        ch["meta"]["char_len"] = len(ch["text"])
        ch["meta"]["has_penalty"] = bool(PENALTY_RE.search(ch["text"]))
        ch["meta"]["crossrefs"] = CROSSREF_RE.findall(ch["text"])
        ch["meta"]["formulas"] = bool(FORMULA_RE.search(ch["text"]))
        chunks.append(ch)
        return ch

    # helper to ensure section exists in hierarchy
    def ensure_section(section_no: str, title: Optional[str]):
        for s in hierarchy["sections"]:
            if s.get("section_no") == section_no:
                return s
        sec = {"section_no": section_no, "title": title or "", "clauses": []}
        hierarchy["sections"].append(sec)
        return sec

    # main loop: iterate over normalized lines
    i = 0
    while i < len(normalized_lines):
        line = normalized_lines[i]
        seg_type = detect_segment_type(line)
        # handle heading
        if seg_type == "heading":
            # create heading chunk and continue
            add_chunk("heading", line)
            i += 1
            continue
        # handle table (we may have tables_meta from Camelot)
        if seg_type == "table":
            # create chunk for table text, and try to match to any extracted table metadata by preview text
            table_meta_match = None
            for t in tables_meta:
                if t.get("text_preview") and t["text_preview"] in line[:500]:
                    table_meta_match = t
                    break
            meta = {"table_meta": table_meta_match} if table_meta_match else {}
            add_chunk("table", line, meta=meta)
            i += 1
            continue
        # handle section (numbered)
        sec_m = SECTION_RE.match(line)
        if sec_m:
            sec_no = sec_m.group(1).strip()
            sec_title = sec_m.group(2).strip() if sec_m.group(2) else ""
            section_obj = ensure_section(sec_no, sec_title)
            # create a section-chunk
            add_chunk("section", line, meta={"section_no": sec_no, "title": sec_title})
            # set current clause context cleared
            current_clause = None
            i += 1
            continue
        # handle example/note/exception
        if seg_type in ("example", "note", "exception"):
            add_chunk(seg_type, line)
            i += 1
            continue
        # handle bullet: accumulate consecutive bullets into an items chunk
        if seg_type == "bullet":
            bullet_lines = [line]
            j = i + 1
            while j < len(normalized_lines) and detect_segment_type(normalized_lines[j]) == "bullet":
                bullet_lines.append(normalized_lines[j])
                j += 1
            add_chunk("bullets", "\n".join(bullet_lines))
            i = j
            continue
        # default paragraph: try to interpret as clause/interpretation if near section/numbering
        # If a paragraph contains "1.2" style references at start, treat as clause
        clause_m = SECTION_RE.match(line)
        if clause_m:
            # handled earlier but keep defensive
            add_chunk("clause", line)
            i += 1
            continue
        # otherwise treat as paragraph; if includes 'Illustration' or 'Example' treat accordingly
        add_chunk("paragraph", line)
        i += 1

    # Post-processing: enrich chunks with spaCy NER & sentence segmentation, link crossrefs
    enrich_chunks_with_spacy(chunks)

    # Build basic mapping of crossref -> chunk ids for simple linking
    crossref_map = defaultdict(list)
    for ch in chunks:
        # look for section numbers inside chunk text and map
        for m in re.finditer(r'(\d{1,3}(?:\.\d{1,3}){0,6})', ch["text"]):
            key = m.group(1)
            crossref_map[key].append(ch["chunk_id"])

    # attach resolved crossref ids into chunk meta
    for ch in chunks:
        refs = ch["meta"].get("crossrefs", [])
        resolved = []
        for r in refs:
            if r in crossref_map:
                resolved.extend(crossref_map[r])
        ch["meta"]["crossref_ids"] = resolved

    return chunks, hierarchy


def enrich_chunks_with_spacy(chunks: List[Dict]):
    """Add NER, sentence count, keywords to chunk metadata using spaCy"""
    for ch in chunks:
        try:
            doc = nlp(ch["text"])
            ents = [{"text": e.text, "label": e.label_} for e in doc.ents]
            key_terms = list({t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha and len(t) > 2} )[:20]
            ch["meta"]["entities"] = ents
            ch["meta"]["sentences"] = [sent.text.strip() for sent in doc.sents] if hasattr(doc, "sents") else []
            ch["meta"]["keywords"] = key_terms
        except Exception as e:
            logger.debug("spaCy enrich failed for chunk %s : %s", ch.get("chunk_id"), e)
            ch["meta"]["entities"] = []
            ch["meta"]["sentences"] = []
            ch["meta"]["keywords"] = []


# ---------------- Save output functions ----------------

def save_outputs(doc_id: str, chunks: List[Dict], hierarchy: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    flat_path = os.path.join(output_dir, f"{doc_id}_chunks.json")
    hier_path = os.path.join(output_dir, f"{doc_id}_hierarchy.json")
    with open(flat_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    with open(hier_path, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d chunks to %s", len(chunks), flat_path)
    logger.info("Saved hierarchy to %s", hier_path)
    return flat_path, hier_path


# ---------------- main per-file pipeline ----------------

def process_file(input_path: str, output_dir: str, enable_tables: bool = True) -> Tuple[str, str]:
    logger.info("Processing %s", input_path)
    doc_id = os.path.splitext(os.path.basename(input_path))[0]
    # Extract text
    try:
        raw = safe_read_text(input_path)
    except Exception as e:
        logger.error("Text extraction failed for %s: %s", input_path, e)
        raise

    # Normalize
    normalized_lines = normalize_text(raw)

    # Table extraction (optional)
    tables_meta = []
    if enable_tables and input_path.lower().endswith(".pdf"):
        try:
            tables_meta = extract_tables_from_pdf(input_path, output_dir=output_dir)
        except Exception as e:
            logger.warning("Table extraction failed: %s", e)
            tables_meta = []

    # Build chunks & hierarchy
    chunks, hierarchy = build_hierarchy_and_chunks(normalized_lines, doc_id, tables_meta=tables_meta)

    # Optionally filter out too-short chunks
    chunks = [c for c in chunks if len(c["text"].strip()) >= MIN_CHUNK_CHARS or c["type"] in ("heading", "table", "example")]

    # Save outputs
    return save_outputs(doc_id, chunks, hierarchy, output_dir)


# ---------------- batch folder processing ----------------

def process_folder(input_dir: str, output_dir: str, enable_tables: bool = True):
    files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
    for fn in files:
        path = os.path.join(input_dir, fn)
        ext = os.path.splitext(fn)[1].lower()
        if ext not in (".pdf", ".txt"):
            logger.info("Skipping unsupported file: %s", fn)
            continue
        try:
            process_file(path, output_dir, enable_tables=enable_tables)
        except Exception as e:
            logger.exception("Failed to process %s: %s", fn, e)


# ---------------- CLI ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Advanced semantic chunker for regulatory documents")
    p.add_argument("--input", "-i", help="Input file (pdf or txt).")
    p.add_argument("--input_dir", help="Input directory containing pdf/txt files for batch processing.")
    p.add_argument("--output_dir", "-o", default="data/semantic_chunks", help="Output directory.")
    p.add_argument("--no_tables", action="store_true", help="Disable Camelot table extraction even if available.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.input:
        process_file(args.input, args.output_dir, enable_tables=not args.no_tables)
    elif args.input_dir:
        process_folder(args.input_dir, args.output_dir, enable_tables=not args.no_tables)
    else:
        print("Specify --input <file> or --input_dir <dir>. Use --help for more.")