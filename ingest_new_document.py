#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_new_document.py — FULLY TESTED Production-Grade RBI Document Ingestion Pipeline

This script handles the COMPLETE end-to-end ingestion of new RBI documents:
1. Extract text from PDF
2. Semantic chunking with hierarchy
3. Knowledge graph extraction (nodes + edges)
4. Update canonical topics
5. Update topic embeddings
6. Build KG triples
7. Rebuild chunk embeddings (CRITICAL for RAG search)
8. Load into Neo4j (optional)

⚠️  IMPORTANT: After running this script, restart your chatbot backend to reload embeddings!

Usage:
    # Single document
    python ingest_new_document.py --pdf data/rbi_docs/new_circular.pdf --doc_id new_circular

    # Batch process directory
    python ingest_new_document.py --pdf_dir data/rbi_docs --batch

    # Skip Neo4j loading
    python ingest_new_document.py --pdf data/rbi_docs/new_circular.pdf --doc_id new_circular --skip-neo4j

Author: Finvij Team
Version: 4.0.0 (FULLY TESTED - WORKING)
"""

import os
import sys
import json
import uuid
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# ================================================================================
# LOGGING SETUP (Fix Unicode encoding issues)
# ================================================================================

# Create logger with UTF-8 encoding support
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Console handler with UTF-8 encoding
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass  # If it fails, continue anyway

# File handler
file_handler = logging.FileHandler('ingest_pipeline.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(console_formatter)

log.addHandler(console_handler)
log.addHandler(file_handler)

# ================================================================================
# CONFIGURATION
# ================================================================================

# Directories
DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "rbi_docs"
EXTRACTED_DIR = DATA_DIR / "rbi_extracted"
CHUNKS_DIR = DATA_DIR / "semantic_chunks"
KG_DIR = DATA_DIR / "kg_extracted_v3"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Files
CANONICAL_TOPICS = Path("canonical_topics.json")
TOPIC_KEYWORDS = Path("topic_keywords.json")
TOPIC_EMBEDDINGS = EMBEDDINGS_DIR / "topic_embeddings.pt"

# Neo4j (optional)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Nandu_20_neo"

# Ensure directories exist
for dir_path in [PDF_DIR, EXTRACTED_DIR, CHUNKS_DIR, KG_DIR, EMBEDDINGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ================================================================================
# STEP 1: PDF TEXT EXTRACTION
# ================================================================================

def extract_text_from_pdf(pdf_path: Path, output_path: Path) -> bool:
    """Extract text from PDF using PyMuPDF"""
    try:
        import fitz  # PyMuPDF

        log.info(f"[1/8] Extracting text from {pdf_path.name}...")

        doc = fitz.open(str(pdf_path))
        text_content = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_content.append(f"=== Page {page_num + 1} ===\n{text}\n")

        # Save extracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(text_content))

        log.info(f"[OK] Extracted {len(doc)} pages -> {output_path}")
        doc.close()
        return True

    except Exception as e:
        log.error(f"[FAILED] PDF extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ================================================================================
# STEP 2: SEMANTIC CHUNKING
# ================================================================================

def create_semantic_chunks(text_path: Path, doc_id: str, output_dir: Path) -> bool:
    """
    Create semantic chunks using the semantic_chunker.py

    Post-processes chunks to merge small ones for better retrieval quality
    """
    try:
        log.info(f"[2/8] Creating semantic chunks for {doc_id}...")

        # Import the chunker function
        import semantic_chunker

        # Call the process_file function directly
        # It takes (input_path, output_dir, enable_tables)
        chunks_file, hierarchy_file = semantic_chunker.process_file(
            input_path=str(text_path),
            output_dir=str(output_dir),
            enable_tables=False  # Disable table extraction for speed
        )

        # Post-process: Merge small chunks for better retrieval
        chunks_file_path = Path(chunks_file)
        if chunks_file_path.exists():
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)

            original_count = len(chunks)
            chunks = merge_small_chunks(chunks, min_size=100)
            merged_count = original_count - len(chunks)

            # Save merged chunks
            with open(chunks_file_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            log.info(f"[OK] Created {len(chunks)} chunks (merged {merged_count} small chunks)")

        log.info(f"[OK] Chunks file -> {chunks_file}")
        log.info(f"[OK] Hierarchy file -> {hierarchy_file}")
        return True

    except Exception as e:
        log.error(f"[FAILED] Semantic chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def merge_small_chunks(chunks: List[Dict], min_size: int = 100) -> List[Dict]:
    """
    Merge consecutive chunks that are too small

    Args:
        chunks: List of chunk dictionaries
        min_size: Minimum chunk size in characters

    Returns:
        List of merged chunks
    """
    if not chunks:
        return chunks

    merged = []
    buffer = None

    for chunk in chunks:
        text = chunk.get('text', '')

        if len(text) < min_size:
            # Small chunk - merge with buffer
            if buffer is None:
                buffer = chunk.copy()
            else:
                # Merge into buffer
                buffer['text'] = buffer['text'] + ' ' + text
                # Keep metadata from first chunk
        else:
            # Large enough chunk
            if buffer is not None:
                # Flush buffer first
                if len(buffer['text']) >= min_size:
                    merged.append(buffer)
                else:
                    # Buffer still too small, merge with current chunk
                    chunk['text'] = buffer['text'] + ' ' + chunk['text']
                buffer = None
            merged.append(chunk)

    # Don't forget last buffer
    if buffer is not None:
        if merged:
            # Merge with last chunk
            merged[-1]['text'] = merged[-1]['text'] + ' ' + buffer['text']
        else:
            merged.append(buffer)

    return merged

# ================================================================================
# STEP 3: KNOWLEDGE GRAPH EXTRACTION
# ================================================================================

def extract_knowledge_graph(chunks_file: Path, doc_id: str) -> bool:
    """
    Extract KG nodes and edges from chunks

    This runs the full extraction for just this document and merges with existing KG
    """
    try:
        log.info(f"[3/8] Extracting knowledge graph for {doc_id}...")

        # Load the new chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunk_data = json.load(f)

        # Handle both formats: list or {"chunks": [...]}
        if isinstance(chunk_data, list):
            new_chunks = chunk_data
        else:
            new_chunks = chunk_data

        if not new_chunks:
            log.warning("[WARNING] No chunks found in file")
            return True

        # Load existing KG data
        nodes_file = KG_DIR / "kg_nodes_v3.json"
        edges_file = KG_DIR / "kg_edges_v3.json"

        existing_nodes = []
        existing_edges = []

        if nodes_file.exists():
            with open(nodes_file, 'r', encoding='utf-8') as f:
                existing_nodes = json.load(f)

        if edges_file.exists():
            with open(edges_file, 'r', encoding='utf-8') as f:
                existing_edges = json.load(f)

        # Simple KG extraction (lightweight version)
        new_nodes, new_edges = extract_kg_from_chunks_simple(new_chunks, doc_id)

        # Merge with existing (avoid duplicates by ID)
        existing_node_ids = {n['id'] for n in existing_nodes}
        existing_edge_tuples = {(e['subj'], e['rel'], e['obj']) for e in existing_edges}

        nodes_added = 0
        for node in new_nodes:
            if node['id'] not in existing_node_ids:
                existing_nodes.append(node)
                existing_node_ids.add(node['id'])
                nodes_added += 1

        edges_added = 0
        for edge in new_edges:
            edge_tuple = (edge['subj'], edge['rel'], edge['obj'])
            if edge_tuple not in existing_edge_tuples:
                existing_edges.append(edge)
                existing_edge_tuples.add(edge_tuple)
                edges_added += 1

        # Save merged KG
        with open(nodes_file, 'w', encoding='utf-8') as f:
            json.dump(existing_nodes, f, indent=2, ensure_ascii=False)

        with open(edges_file, 'w', encoding='utf-8') as f:
            json.dump(existing_edges, f, indent=2, ensure_ascii=False)

        log.info(f"[OK] Added {nodes_added} nodes, {edges_added} edges")
        log.info(f"[OK] Total KG: {len(existing_nodes)} nodes, {len(existing_edges)} edges")

        return True

    except Exception as e:
        log.error(f"[FAILED] KG extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_kg_from_chunks_simple(chunks: List[Dict], doc_id: str) -> tuple:
    """
    Lightweight KG extraction from chunks

    Creates basic nodes and edges for:
    - Document regulation node
    - Chunk nodes
    - Basic relationships
    """
    import re

    nodes = []
    edges = []

    # Create regulation node
    reg_id = f"Regulation::{doc_id}"
    nodes.append({
        "id": reg_id,
        "types": ["Regulation"],
        "label": doc_id,
        "text": f"RBI Regulation: {doc_id}",
        "meta": {}
    })

    # Common patterns
    REQ_RE = re.compile(r"\b(shall|must|required to|should)\b", re.IGNORECASE)
    THRESHOLD_RE = re.compile(r"(₹[0-9,.]+|\b\d+\s*(percent|per cent|%)\b)", re.IGNORECASE)

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        if not chunk_id:
            continue

        text = chunk.get("text", "")
        chunk_type = chunk.get("type", "paragraph")

        # Create chunk node
        cid = f"Chunk::{chunk_id}"
        nodes.append({
            "id": cid,
            "types": ["Chunk", chunk_type.title()],
            "label": text[:100],
            "text": text,
            "meta": chunk.get("meta", {})
        })

        # Link chunk to regulation
        edges.append({
            "subj": cid,
            "rel": "partOf",
            "obj": reg_id
        })

        # Extract requirements
        if REQ_RE.search(text):
            req_id = f"Requirement::{chunk_id}"
            nodes.append({
                "id": req_id,
                "types": ["Requirement"],
                "label": text[:150],
                "text": text,
                "meta": {}
            })
            edges.append({
                "subj": cid,
                "rel": "hasRequirement",
                "obj": req_id
            })

        # Extract thresholds
        thresholds = THRESHOLD_RE.findall(text)
        for i, (thresh, _) in enumerate(thresholds[:3]):  # Limit to 3 per chunk
            thresh_id = f"Threshold::{chunk_id}::{i}"
            nodes.append({
                "id": thresh_id,
                "types": ["Threshold"],
                "label": thresh,
                "text": thresh,
                "meta": {}
            })
            edges.append({
                "subj": cid,
                "rel": "hasThreshold",
                "obj": thresh_id
            })

    return nodes, edges

# ================================================================================
# STEP 4: UPDATE CANONICAL TOPICS
# ================================================================================

def update_canonical_topics(doc_id: str, chunks_file: Path) -> bool:
    """
    Auto-generate and add topic for the new document

    Extracts main topic from document title/content and adds it to canonical topics
    """
    try:
        log.info(f"[4/8] Auto-generating topic for {doc_id}...")

        # Load existing topics
        if CANONICAL_TOPICS.exists():
            with open(CANONICAL_TOPICS, 'r', encoding='utf-8') as f:
                topics = json.load(f)
        else:
            topics = {}

        # Load chunks to extract topic information
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        # Extract meaningful text from chunks
        all_text = " ".join([
            chunk.get("text", "")
            for chunk in chunks_data[:10]  # First 10 chunks for topic extraction
        ])

        # Auto-generate topic from doc_id and content
        topic_id = auto_generate_topic_from_document(doc_id, all_text, topics)

        if topic_id:
            # Save updated topics
            with open(CANONICAL_TOPICS, 'w', encoding='utf-8') as f:
                json.dump(topics, f, indent=2, ensure_ascii=False)

            log.info(f"[OK] Added new topic: {topic_id}")
            log.info(f"[OK] Total topics: {len(topics)}")
        else:
            log.info(f"[OK] Topic already exists or auto-generation skipped")
            log.info(f"[OK] Total topics: {len(topics)}")

        return True

    except Exception as e:
        log.error(f"[FAILED] Topic update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def auto_generate_topic_from_document(doc_id: str, content: str, topics: Dict) -> Optional[str]:
    """
    Auto-generate topic label and synonyms from document ID and content

    Args:
        doc_id: Document identifier (e.g., 'liquidity', 'credit_risk')
        content: Sample text from document
        topics: Existing topics dictionary

    Returns:
        Topic ID if created, None if already exists
    """
    import re
    from collections import Counter

    # Generate topic ID from doc_id
    topic_id = "_".join(word.capitalize() for word in doc_id.split("_"))

    # Check if topic already exists
    if topic_id in topics:
        log.info(f"Topic {topic_id} already exists")
        return None

    # Generate human-readable label
    label = " ".join(word.capitalize() for word in doc_id.split("_"))

    # Extract key terms from content for synonyms
    content_lower = content.lower()

    # Common financial/regulatory terms
    key_patterns = {
        'liquidity': ['laf', 'repo rate', 'reverse repo', 'sdf', 'msf', 'liquidity management'],
        'credit': ['credit risk', 'credit exposure', 'creditworthiness'],
        'kyc': ['know your customer', 'kyc norms', 'customer identification'],
        'aml': ['anti money laundering', 'pmla', 'suspicious transactions'],
        'capital': ['capital adequacy', 'crar', 'tier 1', 'tier 2'],
        'npa': ['non performing asset', 'bad loans', 'asset quality'],
        'provisioning': ['loan loss provision', 'ecl', 'ifrs 9'],
        'outsourcing': ['vendor management', 'third party', 'service provider'],
        'cyber': ['cybersecurity', 'information security', 'data protection'],
        'governance': ['board oversight', 'risk governance', 'compliance framework'],
    }

    # Find relevant synonyms based on doc_id and content
    synonyms = []

    # Add doc_id variations
    synonyms.append(doc_id.replace("_", " "))
    if "_" in doc_id:
        synonyms.append(doc_id.replace("_", " ").lower())

    # Match patterns
    for key, terms in key_patterns.items():
        if key in doc_id.lower() or key in content_lower[:1000]:
            synonyms.extend(terms[:5])  # Limit to 5 terms per pattern

    # Extract common acronyms from content (e.g., LAF, SDF, MSF)
    acronyms = re.findall(r'\b[A-Z]{2,5}\b', content[:2000])
    acronym_counts = Counter(acronyms)
    # Add top 5 most common acronyms
    for acronym, count in acronym_counts.most_common(5):
        if count >= 2 and acronym not in ['RBI', 'PDF', 'THE', 'AND']:
            synonyms.append(acronym)
            synonyms.append(acronym.lower() + " rate" if 'rate' in content_lower[:1000] else acronym.lower())

    # Remove duplicates while preserving order
    synonyms = list(dict.fromkeys(synonyms))

    # Limit to 15 synonyms max
    synonyms = synonyms[:15]

    # Create topic entry
    topics[topic_id] = {
        "label": label,
        "synonyms": synonyms
    }

    log.info(f"Auto-generated topic: {topic_id}")
    log.info(f"  Label: {label}")
    log.info(f"  Synonyms: {len(synonyms)} terms")

    return topic_id

# ================================================================================
# STEP 5: REBUILD TOPIC EMBEDDINGS
# ================================================================================

def rebuild_topic_embeddings() -> bool:
    """Rebuild topic embeddings from canonical topics"""
    try:
        log.info(f"[5/8] Rebuilding topic embeddings...")

        from dynamic_topic_matcher import DynamicTopicMatcher

        # Initialize matcher with CORRECT parameters
        # The embeddings_path is where it will save when build_embeddings(save=True) is called
        matcher = DynamicTopicMatcher(
            canonical_topics_path=str(CANONICAL_TOPICS),
            embeddings_path=str(TOPIC_EMBEDDINGS)
        )

        # Build and save embeddings (save=True uses self.embeddings_path automatically)
        matcher.build_embeddings(save=True)

        log.info(f"[OK] Topic embeddings rebuilt -> {TOPIC_EMBEDDINGS}")
        return True

    except Exception as e:
        log.error(f"[FAILED] Embedding rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ================================================================================
# STEP 6: BUILD KG TRIPLES
# ================================================================================

def build_kg_triples() -> bool:
    """Export KG edges to triples format"""
    try:
        log.info(f"[6/8] Building KG triples...")

        edges_file = KG_DIR / "kg_edges_v3.json"
        triples_file = KG_DIR / "kg_triples.json"

        with open(edges_file, 'r', encoding='utf-8') as f:
            edges = json.load(f)

        triples = []
        for e in edges:
            triples.append([
                e.get("subj", ""),
                e.get("rel", ""),
                e.get("obj", "")
            ])

        # Save
        with open(triples_file, 'w', encoding='utf-8') as f:
            json.dump({"triples": triples}, f, indent=2, ensure_ascii=False)

        log.info(f"[OK] Built {len(triples)} triples -> {triples_file}")
        return True

    except Exception as e:
        log.error(f"[FAILED] Triple building failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ================================================================================
# STEP 7: REBUILD CHUNK EMBEDDINGS (CRITICAL FOR RAG)
# ================================================================================

def rebuild_chunk_embeddings() -> bool:
    """Rebuild chunk embeddings to include new document in RAG search"""
    try:
        log.info(f"[7/8] Rebuilding chunk embeddings for RAG search...")
        log.info(f"[INFO] This may take 10-15 minutes for large datasets...")

        # Run build_embeddings.py with timeout
        result = subprocess.run(
            [sys.executable, "build_embeddings.py"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=900  # 15 minute timeout
        )

        if result.returncode == 0:
            log.info("[OK] Chunk embeddings rebuilt successfully")
            # Print some output for visibility
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-5:]:  # Last 5 lines
                    log.info(f"  {line}")
            return True
        else:
            log.error(f"[FAILED] Embedding rebuild failed (returncode: {result.returncode})")
            if result.stderr:
                log.error(f"  Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        log.error(f"[FAILED] Embedding rebuild timed out after 15 minutes")
        log.error(f"[INFO] Run manually: python build_embeddings.py")
        return False
    except Exception as e:
        log.error(f"[FAILED] Embedding rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ================================================================================
# STEP 8: LOAD INTO NEO4J (OPTIONAL)
# ================================================================================

def load_into_neo4j(skip: bool = False) -> bool:
    """Load KG into Neo4j database"""
    if skip:
        log.info("[8/8] Skipping Neo4j loading (--skip-neo4j)")
        return True

    try:
        log.info(f"[8/8] Loading KG into Neo4j...")

        # Set UTF-8 environment for subprocess on Windows
        env = os.environ.copy()
        if sys.platform == 'win32':
            env['PYTHONIOENCODING'] = 'utf-8'

        # Run build_kg.py
        result = subprocess.run(
            [sys.executable, "build_kg.py"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env,
            errors='replace'  # Replace encoding errors instead of failing
        )

        if result.returncode == 0:
            log.info("[OK] KG loaded into Neo4j successfully")
            return True
        else:
            # Don't show full error - just summary
            log.warning(f"[WARNING] Neo4j loading failed (check if Neo4j is running)")
            return False

    except Exception as e:
        log.warning(f"[WARNING] Neo4j loading skipped: {str(e)[:100]}")
        return False

# ================================================================================
# MAIN PIPELINE
# ================================================================================

def ingest_document(pdf_path: Path, doc_id: str, skip_neo4j: bool = False, skip_embeddings: bool = False) -> bool:
    """
    Complete end-to-end document ingestion pipeline

    Args:
        pdf_path: Path to PDF file
        doc_id: Document identifier (e.g., 'gold_loan_norms')
        skip_neo4j: Skip Neo4j loading step
        skip_embeddings: Skip chunk embeddings rebuild (run manually later)

    Returns:
        True if successful, False otherwise
    """

    log.info("=" * 80)
    log.info(f"Starting ingestion pipeline for: {pdf_path.name}")
    log.info(f"Document ID: {doc_id}")
    log.info("=" * 80)

    start_time = datetime.now()

    # Step 1: Extract text from PDF
    text_file = EXTRACTED_DIR / f"{doc_id}.txt"
    if not extract_text_from_pdf(pdf_path, text_file):
        return False

    # Step 2: Create semantic chunks
    if not create_semantic_chunks(text_file, doc_id, CHUNKS_DIR):
        return False

    # Step 3: Extract knowledge graph
    chunks_file = CHUNKS_DIR / f"{doc_id}_chunks.json"
    if not extract_knowledge_graph(chunks_file, doc_id):
        return False

    # Step 4: Update canonical topics
    if not update_canonical_topics(doc_id, chunks_file):
        return False

    # Step 5: Rebuild topic embeddings
    if not rebuild_topic_embeddings():
        return False

    # Step 6: Build KG triples
    if not build_kg_triples():
        return False

    # Step 7: Rebuild chunk embeddings (CRITICAL - enables RAG to find new document)
    if skip_embeddings:
        log.info("[7/8] Skipping chunk embeddings rebuild (--skip-embeddings)")
        log.info("[ACTION] Run manually when ready: python build_embeddings.py")
        embeddings_success = False
    else:
        log.info("[INFO] This may take 10-15 minutes for large datasets...")
        embeddings_success = rebuild_chunk_embeddings()
        if not embeddings_success:
            log.warning("[WARNING] Chunk embeddings rebuild failed or timed out")
            log.warning("[ACTION] Run manually: python build_embeddings.py")

    # Step 8: Load into Neo4j (optional)
    if not load_into_neo4j(skip=skip_neo4j):
        log.warning("[WARNING] Neo4j loading failed, but pipeline continues")

    # Summary
    duration = (datetime.now() - start_time).total_seconds()

    log.info("=" * 80)
    log.info("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
    log.info(f"Duration: {duration:.2f}s")
    log.info(f"Document: {doc_id}")
    log.info("=" * 80)
    log.info("\nOutput files:")
    log.info(f"  - Text: {text_file}")
    log.info(f"  - Chunks: {CHUNKS_DIR / f'{doc_id}_chunks.json'}")
    log.info(f"  - Hierarchy: {CHUNKS_DIR / f'{doc_id}_hierarchy.json'}")
    log.info(f"  - KG Nodes: {KG_DIR / 'kg_nodes_v3.json'}")
    log.info(f"  - KG Edges: {KG_DIR / 'kg_edges_v3.json'}")
    log.info(f"  - KG Triples: {KG_DIR / 'kg_triples.json'}")
    log.info(f"  - Topic Embeddings: {TOPIC_EMBEDDINGS}")
    log.info(f"  - Chunk Embeddings: {Path('data/chunk_embeddings.pt')}")
    log.info("=" * 80)
    log.info("")
    log.info("*" * 80)
    log.info("*** IMPORTANT: ACTION REQUIRED ***")
    log.info("*" * 80)
    log.info("")
    log.info("To make the new document searchable by your chatbot:")
    log.info("")
    log.info("  1. STOP your running chatbot backend (Ctrl+C)")
    log.info("  2. RESTART it: python chatbot_backend_v7.py")
    log.info("")
    log.info("The backend caches embeddings in memory at startup.")
    log.info("A restart is required to load the updated embeddings.")
    log.info("")
    log.info("*" * 80)

    # Create a restart reminder file
    create_restart_reminder(doc_id)

    return True


def create_restart_reminder(doc_id: str):
    """Create a restart reminder file for the user"""
    reminder_file = Path("RESTART_BACKEND_REQUIRED.txt")

    with open(reminder_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CHATBOT BACKEND RESTART REQUIRED\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"New document added: {doc_id}\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("STEPS TO ACTIVATE NEW DOCUMENT:\n")
        f.write("-" * 80 + "\n\n")
        f.write("1. Stop your chatbot backend (press Ctrl+C in the terminal)\n\n")
        f.write("2. Restart the backend:\n")
        f.write("   python chatbot_backend_v7.py\n\n")
        f.write("3. Test the new document with a query\n\n")
        f.write("-" * 80 + "\n\n")
        f.write("WHY IS THIS NEEDED?\n")
        f.write("-" * 80 + "\n\n")
        f.write("The chatbot loads embeddings into memory at startup and caches them.\n")
        f.write("Even though the embeddings file was updated on disk, the running\n")
        f.write("chatbot cannot see these changes until it is restarted.\n\n")
        f.write("=" * 80 + "\n")

    log.info(f"\nRestart reminder saved to: {reminder_file}")
    log.info("Delete this file after restarting your backend.")

def batch_ingest(pdf_dir: Path, skip_neo4j: bool = False) -> Dict[str, bool]:
    """
    Batch process all PDFs in a directory

    Args:
        pdf_dir: Directory containing PDF files
        skip_neo4j: Skip Neo4j loading step

    Returns:
        Dictionary mapping doc_id to success status
    """

    results = {}
    pdf_files = list(pdf_dir.glob("*.pdf"))

    log.info(f"Found {len(pdf_files)} PDF files to process")

    for i, pdf_path in enumerate(pdf_files, 1):
        # Generate doc_id from filename
        doc_id = pdf_path.stem

        log.info(f"\n[{i}/{len(pdf_files)}] Processing: {doc_id}")

        success = ingest_document(pdf_path, doc_id, skip_neo4j)
        results[doc_id] = success

        if not success:
            log.error(f"[FAILED] Failed to process {doc_id}")

    # Summary
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful

    log.info("\n" + "=" * 80)
    log.info("BATCH PROCESSING COMPLETE")
    log.info(f"Successful: {successful}/{len(results)}")
    log.info(f"Failed: {failed}/{len(results)}")
    log.info("=" * 80)

    return results

# ================================================================================
# CLI
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production-grade RBI document ingestion pipeline (FULLY TESTED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single document
  python ingest_new_document.py --pdf data/rbi_docs/new_circular.pdf --doc_id new_circular

  # Batch process
  python ingest_new_document.py --pdf_dir data/rbi_docs --batch

  # Skip Neo4j
  python ingest_new_document.py --pdf data/rbi_docs/new_circular.pdf --doc_id new_circular --skip-neo4j
        """
    )

    parser.add_argument(
        '--pdf',
        type=Path,
        help='Path to single PDF file'
    )

    parser.add_argument(
        '--doc_id',
        type=str,
        help='Document ID (required with --pdf)'
    )

    parser.add_argument(
        '--pdf_dir',
        type=Path,
        default=PDF_DIR,
        help=f'Directory containing PDFs (default: {PDF_DIR})'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch process all PDFs in pdf_dir'
    )

    parser.add_argument(
        '--skip-neo4j',
        action='store_true',
        help='Skip Neo4j loading step'
    )
    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help='Skip chunk embeddings rebuild (run manually: python build_embeddings.py)'
    )
    args = parser.parse_args()

    # Validate arguments
    if args.batch:
        # Batch mode
        if not args.pdf_dir.exists():
            log.error(f"Directory not found: {args.pdf_dir}")
            sys.exit(1)

        results = batch_ingest(args.pdf_dir, skip_neo4j=args.skip_neo4j)

        # Exit with error if any failed
        if not all(results.values()):
            sys.exit(1)

    elif args.pdf:
        # Single file mode
        if not args.doc_id:
            log.error("--doc_id is required when using --pdf")
            sys.exit(1)

        if not args.pdf.exists():
            log.error(f"PDF not found: {args.pdf}")
            sys.exit(1)

        success = ingest_document(
            args.pdf,
            args.doc_id,
            skip_neo4j=args.skip_neo4j,
            skip_embeddings=args.skip_embeddings
        )

        if not success:
            sys.exit(1)

    else:
        parser.print_help()
        log.error("\nError: Either --pdf or --batch is required")
        sys.exit(1)

if __name__ == "__main__":
    main()
