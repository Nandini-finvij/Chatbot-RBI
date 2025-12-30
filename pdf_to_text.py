#!/usr/bin/env python3
"""
PAKKA PRODUCTION READY PDF → TEXT EXTRACTOR
--------------------------------------------
Features:
- Multi-stage extraction:
    1. pdfplumber
    2. PyPDF2
    3. PyMuPDF (fitz) TEXT extraction
    4. pdfplumber OCR
    5. PyMuPDF OCR
- RBI-optimized cleanup (fix hyphens, weird spacing, joins lines, removes headers/footers)
- Auto-detection of low-quality text => automatic fallback to better extractors
- Guaranteed non-empty clean text suitable for semantic chunking
"""

import argparse
import os
import re
import sys
from typing import List

# 1) Load libraries safely
try: import pdfplumber
except: pdfplumber = None

try: import PyPDF2
except: PyPDF2 = None

try:
    import fitz  # PyMuPDF
except:
    fitz = None

try:
    from PIL import Image
    import pytesseract
except:
    pytesseract = None
    Image = None

# Set tesseract path if needed:
TESSERACT_CMD = None
if TESSERACT_CMD and pytesseract:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ---------------------------------------------------
# Cleaning Utilities
# ---------------------------------------------------

PAGE_NUM_RE = [
    re.compile(r'^\s*page\s*\d+\s*$', re.IGNORECASE),
    re.compile(r'^\s*\d+\s*/\s*\d+\s*$', re.IGNORECASE),
    re.compile(r'^\s*-\s*\d+\s*-\s*$', re.IGNORECASE),
]

def is_page_num(line: str) -> bool:
    return any(p.match(line.strip()) for p in PAGE_NUM_RE)


def clean_text_basic(text: str) -> str:
    """Basic cleanup: remove junk characters, normalize whitespace."""
    text = text.replace("\xa0", " ")
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)    # remove control chars
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fix_hyphenation_and_join(text: str) -> str:
    """Fix broken words and join broken lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove hyphen + newline: "exam-\nple" → "example"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    out = []
    lines = text.splitlines()
    buf = ""

    for i, line in enumerate(lines):
        line = line.rstrip()

        if not line:
            if buf:
                out.append(buf)
                buf = ""
            out.append("")
            continue

        # Join if buf doesn’t end with punctuation
        if buf and not re.search(r'[.!?:"\'\)\]]$', buf):
            # join line with buf
            buf += " " + line.lstrip()
        else:
            if buf:
                out.append(buf)
            buf = line

    if buf:
        out.append(buf)

    return "\n".join(out)


def remove_headers_footers(pages: List[str]) -> str:
    """
    Remove repeated headers/footers appearing on >30% pages.
    """
    lines_per_page = [p.splitlines() for p in pages]
    top_lines = [l[0].strip() if l else "" for l in lines_per_page]
    bottom_lines = [l[-1].strip() if l else "" for l in lines_per_page]

    from collections import Counter
    total = len(pages)
    candidates = set()

    for ln, cnt in Counter(top_lines + bottom_lines).items():
        if ln and cnt / total > 0.30:
            candidates.add(re.escape(ln))

    if not candidates:
        return "\n\n".join(pages)

    patt = re.compile("|".join(candidates))

    cleaned = []
    for lines in lines_per_page:
        if not lines:
            continue
        if patt.search(lines[0]):
            lines = lines[1:]
        if lines and patt.search(lines[-1]):
            lines = lines[:-1]
        cleaned.append("\n".join(lines))

    return "\n\n".join(cleaned)

# ---------------------------------------------------
# Extractors
# ---------------------------------------------------

def extract_pdfplumber(pdf_path: str) -> List[str]:
    if not pdfplumber:
        return []
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            try:
                txt = pg.extract_text() or ""
            except Exception:
                txt = ""
            pages.append(txt)
    return pages


def extract_pypdf2(pdf_path: str) -> List[str]:
    if not PyPDF2:
        return []
    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except:
                pages.append("")
    return pages


def extract_pymupdf(pdf_path: str) -> List[str]:
    """
    PyMuPDF text extraction is extremely powerful for RBI PDFs.
    """
    if not fitz:
        return []
    pages = []
    doc = fitz.open(pdf_path)
    for p in doc:
        try:
            txt = p.get_text()  # includes better layout handling
            pages.append(txt)
        except:
            pages.append("")
    return pages


def extract_ocr(pdf_path: str) -> List[str]:
    if not (pytesseract and pdfplumber):
        return []
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            try:
                img = pg.to_image(resolution=200).original
                txt = pytesseract.image_to_string(img)
            except:
                txt = ""
            pages.append(txt)
    return pages


def extract_ocr_pymupdf(pdf_path: str) -> List[str]:
    if not (fitz and pytesseract):
        return []
    pages = []
    doc = fitz.open(pdf_path)
    for p in doc:
        try:
            pix = p.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            txt = pytesseract.image_to_string(img)
        except:
            txt = ""
        pages.append(txt)
    return pages


# ---------------------------------------------------
# Main extraction logic
# ---------------------------------------------------
def extract_pdf(pdf_path: str) -> str:
    """
    Multi-stage extraction with fallbacks.
    Auto-detect poor extraction and retry using stronger extractors.
    """

    extractors = [
        ("pdfplumber", extract_pdfplumber),
        ("pypdf2", extract_pypdf2),
        ("pymupdf", extract_pymupdf),
        ("ocr_pdfplumber", extract_ocr),
        ("ocr_pymupdf", extract_ocr_pymupdf),
    ]

    for name, extractor in extractors:
        print(f"[try] {name} …")
        pages = extractor(pdf_path)
        if not pages:
            print(f"[fail] {name}: returned no pages")
            continue

        raw = "\n\n".join(pages).strip()

        if len(raw) < 500:
            print(f"[warn] {name}: extracted too little text ({len(raw)} chars)")
            continue

        print(f"[ok] {name}: extracted {len(raw)} chars")
        return raw

    raise RuntimeError("ALL extraction methods failed. PDF may be corrupted.")


# ---------------------------------------------------
# Main function: PDF → cleaned text
# ---------------------------------------------------
def pdf_to_clean_text(pdf_path: str) -> str:
    raw = extract_pdf(pdf_path)
    print("[info] removing headers/footers")
    merged = remove_headers_footers(raw.split("\n\n"))

    print("[info] fixing hyphenation and joining lines")
    merged = fix_hyphenation_and_join(merged)

    print("[info] cleaning text")
    merged = clean_text_basic(merged)

    return merged


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="PDF file")
    ap.add_argument("--output", "-o", required=True, help="Output TXT")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    text = pdf_to_clean_text(args.input)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[done] wrote cleaned text to {args.output}")
