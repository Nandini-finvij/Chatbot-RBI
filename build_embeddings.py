#!/usr/bin/env python3
"""
build_embeddings.py — Rebuild Chunk Embeddings for RAG Search

This script rebuilds the chunk_embeddings.pt file used by hybrid_retrieval.py
by encoding all chunks from the semantic_chunks directory.

Usage:
    python build_embeddings.py
    python build_embeddings.py --model all-mpnet-base-v2
    python build_embeddings.py --output data/chunk_embeddings.pt

Author: Finvij Team
Version: 1.0.0
"""

import os
import json
import torch
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Setup logging with UTF-8 support
import sys
if sys.platform == 'win32':
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('build_embeddings.log', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)

# ================================================================================
# CONFIGURATION
# ================================================================================

DEFAULT_MODEL = "all-mpnet-base-v2"
DEFAULT_CHUNK_DIR = "data/semantic_chunks"
DEFAULT_OUTPUT = "data/chunk_embeddings.pt"
BATCH_SIZE = 32  # Encode chunks in batches for efficiency

# ================================================================================
# CHUNK LOADING
# ================================================================================

def load_all_chunks(chunk_dir: Path) -> List[Dict]:
    """
    Load all chunks from semantic_chunks directory.

    Args:
        chunk_dir: Directory containing *_chunks.json files

    Returns:
        List of chunk dictionaries with chunk_id and text
    """
    all_chunks = []
    chunk_files = list(chunk_dir.glob("*_chunks.json"))

    log.info(f"Found {len(chunk_files)} chunk files in {chunk_dir}")

    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)

            # Handle both formats: {"chunks": [...]} or [...]
            if isinstance(chunk_data, dict):
                chunks = chunk_data.get("chunks", [])
            else:
                chunks = chunk_data

            # Extract relevant fields
            for chunk in chunks:
                if isinstance(chunk, dict) and "chunk_id" in chunk and "text" in chunk:
                    all_chunks.append({
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "doc_id": chunk.get("doc_id", "unknown"),
                        "source_file": chunk_file.name
                    })

            log.info(f"  [OK] Loaded {len(chunks)} chunks from {chunk_file.name}")

        except Exception as e:
            log.error(f"  ✗ Failed to load {chunk_file.name}: {e}")
            continue

    log.info(f"Total chunks loaded: {len(all_chunks)}")
    return all_chunks

# ================================================================================
# EMBEDDING GENERATION
# ================================================================================

def build_embeddings(
    chunks: List[Dict],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = BATCH_SIZE
) -> Tuple[List[str], torch.Tensor]:
    """
    Build embeddings for all chunks using SentenceTransformer.

    Args:
        chunks: List of chunk dictionaries
        model_name: Name of the SentenceTransformer model
        batch_size: Batch size for encoding

    Returns:
        Tuple of (chunk_ids, embeddings_tensor)
    """
    log.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Extract texts and IDs
    chunk_ids = [c["chunk_id"] for c in chunks]
    chunk_texts = [c["text"] for c in chunks]

    log.info(f"Encoding {len(chunk_texts)} chunks in batches of {batch_size}...")

    # Encode in batches with progress tracking
    all_embeddings = []
    total_batches = (len(chunk_texts) + batch_size - 1) // batch_size

    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        log.info(f"  Encoding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        try:
            # Encode batch
            embeddings = model.encode(
                batch,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
            all_embeddings.append(embeddings)

        except Exception as e:
            log.error(f"  ✗ Batch {batch_num} failed: {e}")
            raise

    # Concatenate all batches
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)

    log.info(f"✓ Generated embeddings: {all_embeddings_tensor.shape}")

    return chunk_ids, all_embeddings_tensor

# ================================================================================
# SAVE EMBEDDINGS
# ================================================================================

def save_embeddings(
    chunk_ids: List[str],
    embeddings: torch.Tensor,
    output_path: Path,
    metadata: Dict = None
) -> bool:
    """
    Save chunk IDs and embeddings to file.

    Args:
        chunk_ids: List of chunk IDs
        embeddings: Embeddings tensor
        output_path: Output file path
        metadata: Optional metadata dictionary

    Returns:
        True if successful
    """
    try:
        # Prepare data
        data = {
            "ids": chunk_ids,
            "embeddings": embeddings,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        log.info(f"Saving embeddings to {output_path}...")
        torch.save(data, output_path)

        # Verify file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        log.info(f"[OK] Saved successfully ({file_size_mb:.2f} MB)")

        return True

    except Exception as e:
        log.error(f"✗ Failed to save embeddings: {e}")
        return False

# ================================================================================
# STATISTICS
# ================================================================================

def print_statistics(chunks: List[Dict], chunk_ids: List[str], embeddings: torch.Tensor):
    """Print statistics about the embeddings"""

    # Count chunks per document
    doc_counts = {}
    for chunk in chunks:
        doc_id = chunk.get("doc_id", "unknown")
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    log.info("=" * 80)
    log.info("EMBEDDING STATISTICS")
    log.info("=" * 80)
    log.info(f"Total chunks:        {len(chunk_ids)}")
    log.info(f"Total documents:     {len(doc_counts)}")
    log.info(f"Embedding dimension: {embeddings.shape[1]}")
    log.info(f"Embedding shape:     {embeddings.shape}")
    log.info(f"")
    log.info("Chunks per document:")
    for doc_id, count in sorted(doc_counts.items(), key=lambda x: x[1], reverse=True):
        log.info(f"  {doc_id}: {count} chunks")
    log.info("=" * 80)

# ================================================================================
# MAIN PIPELINE
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rebuild chunk embeddings for RAG search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default settings
  python build_embeddings.py

  # Custom model
  python build_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2

  # Custom paths
  python build_embeddings.py --chunk_dir data/semantic_chunks --output data/embeddings/chunks.pt

  # Dry run (don't save)
  python build_embeddings.py --dry-run
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'SentenceTransformer model name (default: {DEFAULT_MODEL})'
    )

    parser.add_argument(
        '--chunk_dir',
        type=Path,
        default=DEFAULT_CHUNK_DIR,
        help=f'Directory containing chunk files (default: {DEFAULT_CHUNK_DIR})'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f'Output file path (default: {DEFAULT_OUTPUT})'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help=f'Encoding batch size (default: {BATCH_SIZE})'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Build embeddings but do not save'
    )

    args = parser.parse_args()

    log.info("=" * 80)
    log.info("CHUNK EMBEDDING BUILDER")
    log.info("=" * 80)
    log.info(f"Model:      {args.model}")
    log.info(f"Chunk dir:  {args.chunk_dir}")
    log.info(f"Output:     {args.output}")
    log.info(f"Batch size: {args.batch_size}")
    log.info(f"Dry run:    {args.dry_run}")
    log.info("=" * 80)

    start_time = datetime.now()

    # Step 1: Load chunks
    if not args.chunk_dir.exists():
        log.error(f"Chunk directory not found: {args.chunk_dir}")
        return 1

    chunks = load_all_chunks(args.chunk_dir)

    if not chunks:
        log.error("No chunks found! Cannot build embeddings.")
        return 1

    # Step 2: Build embeddings
    try:
        chunk_ids, embeddings = build_embeddings(
            chunks,
            model_name=args.model,
            batch_size=args.batch_size
        )
    except Exception as e:
        log.error(f"Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Print statistics
    print_statistics(chunks, chunk_ids, embeddings)

    # Step 4: Save (unless dry run)
    if args.dry_run:
        log.info("Dry run - skipping save")
    else:
        metadata = {
            "model": args.model,
            "chunk_dir": str(args.chunk_dir),
            "total_chunks": len(chunk_ids),
            "total_documents": len(set(c["doc_id"] for c in chunks)),
            "batch_size": args.batch_size
        }

        success = save_embeddings(chunk_ids, embeddings, args.output, metadata)

        if not success:
            return 1

    # Summary
    duration = (datetime.now() - start_time).total_seconds()

    log.info("=" * 80)
    log.info("[SUCCESS] EMBEDDING BUILD COMPLETED SUCCESSFULLY")
    log.info(f"Duration: {duration:.2f}s")
    log.info(f"Chunks processed: {len(chunk_ids)}")
    log.info(f"Output: {args.output}")
    log.info("=" * 80)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
