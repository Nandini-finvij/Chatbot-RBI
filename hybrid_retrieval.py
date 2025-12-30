# hybrid_retrieval.py (FAST, PRODUCTION)

import json
import torch
from sentence_transformers import SentenceTransformer, util
from kg_retrieval import get_topic_related_nodes, get_kg_facts
from pathlib import Path
# ---------- Load once ----------
MODEL_NAME = "all-mpnet-base-v2"
EMBED_FILE = "data/chunk_embeddings.pt"
CHUNK_DIR = "data/semantic_chunks"

model = SentenceTransformer(MODEL_NAME)

data = torch.load(EMBED_FILE)
ALL_IDS = data["ids"]
ALL_EMB = data["embeddings"]

ID2TEXT = {}

for f in Path(CHUNK_DIR).glob("*_chunks.json"):
    chunk_data = json.load(open(f, encoding="utf-8"))
    chunks = chunk_data["chunks"] if isinstance(chunk_data, dict) else chunk_data

    for c in chunks:
        ID2TEXT[c["chunk_id"]] = c["text"]

# ---------- Fast RAG ----------
def rag_search(query, top_k=5):
    q_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, ALL_EMB)[0]
    top = scores.topk(top_k)
    return [(ALL_IDS[i], ID2TEXT.get(ALL_IDS[i], "")) for i in top.indices]

# ---------- Hybrid Retrieval ----------
def hybrid_retrieve(query, topic_key):
    """
    Hybrid retrieval combining KG and RAG.

    Args:
        query: User query string
        topic_key: Topic key (can be None)

    Returns:
        Tuple of (chunks, kg_facts)
    """
    # Always do RAG search
    rag_chunks = rag_search(query, top_k=10)

    # Only do KG search if topic is provided
    kg_chunks = []
    if topic_key:
        try:
            kg_nodes = get_topic_related_nodes(topic_key)
            kg_chunks = [
                n.replace("Chunk::", "")
                for n in kg_nodes if n and n.startswith("Chunk::")
            ]
        except Exception as e:
            print(f"KG retrieval failed for topic {topic_key}: {e}")
            kg_chunks = []

    # Merge chunks (RAG + KG, deduplicated)
    rag_chunk_ids = {cid for cid, _ in rag_chunks}
    merged = list(rag_chunk_ids | set(kg_chunks))

    # Build final chunks with text
    final_chunks = []
    for cid in merged:
        text = ID2TEXT.get(cid, "")
        if text:  # Only include chunks with text
            final_chunks.append((cid, text))

    # Get KG facts only if we have chunks
    kg_facts = []
    if final_chunks and topic_key:
        try:
            kg_facts = get_kg_facts(["Chunk::" + cid for cid, _ in final_chunks])
        except Exception as e:
            print(f"KG facts retrieval failed: {e}")
            kg_facts = []

    return final_chunks, kg_facts
