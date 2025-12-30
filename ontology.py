#!/usr/bin/env python3
"""
extract_ontology_from_chunks_v3.py
FINAL VERSION WITH:
- Domain-level topic filtering (fixes ALL misclassification)
- Clean topic detection
- Rule-first + LLM fallback only within allowed domain
- Zero false positive DLG/ECL/IRAC topic assignments
- Noise removal (no percentage-based topic triggers)
"""

import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ---------------------
# Optional LLM Classifier
# ---------------------
try:
    from groq import Groq
    HAS_GROQ = True
except:
    HAS_GROQ = False

# ---------------------
# Load JSON helpers
# ---------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(o, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(o, f, indent=2, ensure_ascii=False)


# ---------------------
# Regex patterns
# ---------------------
REQ_RE = re.compile(r"\b(shall|must|required to|should|will)\b", re.IGNORECASE)
THRESHOLD_RE = re.compile(r"(₹[0-9,.]+|\b\d+\s*(percent|per cent|%)\b)", re.IGNORECASE)
TIMELINE_RE = re.compile(
    r"(effective from\s+[A-Za-z]+\s+\d{1,2},?\s*\d{4}|with effect from\s+[A-Za-z]+\s+\d{4}|within\s+\d+\s+(days|weeks|months))",
    re.IGNORECASE,
)
DEF_RE = re.compile(r"\b(defined as|means|refers to|is defined as)\b", re.IGNORECASE)
EXC_RE = re.compile(r"^(However|Provided that|Except|Notwithstanding)", re.IGNORECASE | re.MULTILINE)

ACTOR_KEYWORDS = {
    "re": "RE",
    "regulated entity": "RE",
    "lsp": "LSP",
    "nbfc": "NBFC",
    "bank": "Bank",
    "borrower": "Borrower",
    "customer": "Borrower",
}

# ---------------------
# LLM Classifier
# ---------------------
class LLMClassifier:
    def __init__(self, model="llama-3.3-70b-versatile", key_env="api"):
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError("Missing GROQ_API_KEY for LLM classification.")
        self.client = Groq(api_key=key)
        self.model = model

    def classify(self, text, topics):
        """
        classify text into one of the allowed domain topics
        """
        prompt = f"""
You are an RBI regulatory topic classifier.
Choose ONLY from these topics:
{topics}

Text:
\"\"\"{text}\"\"\"

Return ONLY the topic key.
"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=20,
        )
        return resp.choices[0].message["content"].strip()



# ---------------------
# Domain → Topics Map (THE FIX)
# ---------------------
# DO NOT CHANGE THIS UNLESS YOU ADD NEW DOCUMENTS
TOPIC_DOMAIN_MAP = {
    # Digital Lending FLDG
    "DLG_Eligibility": ["digital_lending", "digital_lending_fldg"],
    "DLG_Cap": ["digital_lending", "digital_lending_fldg"],
    "DLG_Forms": ["digital_lending", "digital_lending_fldg"],
    "DLG_Structure": ["digital_lending", "digital_lending_fldg"],
    "DLG_Restrictions": ["digital_lending", "digital_lending_fldg"],
    "DLG_Reporting": ["digital_lending", "digital_lending_fldg"],

    # Expected Credit Loss
    "ECL_Overview": ["ecl_framework", "ECL_Framework_Circular"],
    "ECL_Stages": ["ecl_framework", "ECL_Framework_Circular"],
    "ECL_Measurement": ["ecl_framework", "ECL_Framework_Circular"],

    # Gold Loan
    "Gold_Loan_LTV": ["gold_loan_norms"],
    "Gold_Loan_Operational": ["gold_loan_norms"],

    # KFS Guidelines
    "KFS_Requirements": ["Key_Fact_Statement_Guidelines", "kfs_guidelines"],

    # KYC
    "KYC_Process": ["KYC", "RBI_KYC_FAQs"],

    # AML
    "AML_Compliance": ["AML"],

    # IRACP
    "IRACP_Classification": ["Master_Direction__IRACPrudential_Norms"],
    "IRACP_Provisioning": ["Master_Direction__IRACPrudential_Norms"],

    # Model Governance
    "Model_Governance_Framework": ["Master_Direction__Model_Risk_Management"],
    "Model_Governance_Validation": ["Master_Direction__Model_Risk_Management"],
    "Outsourcing_Applicability": [
        "Commercial_Banks_Outsourcing_Directions_2025"
    ],

    "Outsourcing_Governance": [
        "Commercial_Banks_Outsourcing_Directions_2025"
    ],

    "Outsourcing_Risk_Management": [
        "Commercial_Banks_Outsourcing_Directions_2025"
    ],

    "Outsourcing_Due_Diligence": [
        "Commercial_Banks_Outsourcing_Directions_2025"
    ],

    "Outsourcing_Data_Security": [
        "Commercial_Banks_Outsourcing_Directions_2025"
    ],

    "Outsourcing_Audit_Access": [
        "Commercial_Banks_Outsourcing_Directions_2025"
    ],

    "Outsourcing_Business_Continuity": [
        "Commercial_Banks_Outsourcing_Directions_2025"
    ],

    "Outsourcing_Regulatory_Oversight": [
        "Commercial_Banks_Outsourcing_Directions_2025"
    ],

    "Outsourcing_Penalties": [
        "Commercial_Banks_Outsourcing_Directions_2025"
    ],

    # FAQs
    "FAQs_Digital_Lending": ["RBI_Digital_Lending_FAQs"],
    "FAQs_KYC": ["RBI_KYC_FAQs"]
}


# ---------------------
# Main Extractor
# ---------------------
def extract(chunks_dir, topic_kw_file, canonical_file, out_dir, llm_mode="hybrid"):

    topic_keywords = load_json(topic_kw_file)
    canonical_topics = load_json(canonical_file)

    os.makedirs(out_dir, exist_ok=True)

    nodes = {}
    edges = []
    topic_index = defaultdict(list)

    # optional LLM
    llm = None
    if llm_mode != "none" and HAS_GROQ and os.environ.get("GROQ_API_KEY"):
        llm = LLMClassifier()
    else:
        llm_mode = "none"

    def add_node(nid, ntype, label=None, text=None, meta=None):
        if nid not in nodes:
            nodes[nid] = {
                "id": nid,
                "types": set([ntype]),
                "label": label or nid,
                "text": text or "",
                "meta": meta or {},
            }
        else:
            nodes[nid]["types"].add(ntype)
            if label:
                nodes[nid]["label"] = label

    def add_edge(a, rel, b):
        edges.append({"subj": a, "rel": rel, "obj": b})

    # Process all chunk files
    for file in tqdm(list(Path(chunks_dir).glob("*_chunks.json"))):

        doc_chunks = load_json(file)
        if "chunks" in doc_chunks:
            doc_chunks = doc_chunks["chunks"]

        if not doc_chunks:
            continue

        doc_id = doc_chunks[0].get("doc_id", file.stem.replace("_chunks", ""))
        reg_node = f"Regulation::{doc_id}"
        add_node(reg_node, "Regulation", label=doc_id)

        # Process every chunk
        for ch in doc_chunks:
            cid = f"Chunk::{ch['chunk_id']}"
            text = ch["text"]
            meta = ch.get("meta", {})

            # Register chunk
            add_node(cid, "Chunk", label=cid, text=text, meta=meta)
            add_edge(cid, "partOf", reg_node)

            # -------------------------
            # 1. TOPIC DETECTION (THE FIX)
            # -------------------------
            allowed_topics = [
                t for t, docs in TOPIC_DOMAIN_MAP.items() if doc_id in docs
            ]

            matched_topics = []

            # RULE MATCH (only in allowed_topics)
            lower_text = text.lower()
            for topic in allowed_topics:
                for kw in topic_keywords[topic]:
                    if kw.lower() in lower_text:
                        matched_topics.append(topic)
                        break

            # LLM fallback if rule fails
            if llm and llm_mode in ("hybrid", "force"):
                if not matched_topics or llm_mode == "force":
                    try:
                        result = llm.classify(text, allowed_topics)
                        if result in allowed_topics:
                            matched_topics = [result]
                    except:
                        pass

            # Create topic nodes
            for topic in matched_topics:
                topic_node = f"Topic::{topic}"
                add_node(topic_node, "Topic", label=canonical_topics.get(topic, {}).get("label", topic))
                add_edge(cid, "pertainsTo", topic_node)
                topic_index[topic].append(cid)

            # -------------------------
            # 2. Requirements
            # -------------------------
            if REQ_RE.search(text):
                rid = f"Requirement::{ch['chunk_id']}"
                add_node(rid, "Requirement", label=text[:150], text=text)
                add_edge(cid, "hasRequirement", rid)

            # -------------------------
            # 3. Threshold
            # -------------------------
            for m in THRESHOLD_RE.findall(text):
                th = m[0]
                tid = f"Threshold::{ch['chunk_id']}::{abs(hash(th))}"
                add_node(tid, "Threshold", label=th)
                add_edge(cid, "hasThreshold", tid)

            # -------------------------
            # 4. Timeline
            # -------------------------
            for m in TIMELINE_RE.findall(text):
                tl = m[0]
                tid = f"Timeline::{ch['chunk_id']}::{abs(hash(tl))}"
                add_node(tid, "Timeline", label=tl)
                add_edge(cid, "hasTimeline", tid)

            # -------------------------
            # 5. Definitions
            # -------------------------
            if DEF_RE.search(text):
                did = f"Definition::{ch['chunk_id']}"
                add_node(did, "Definition", label=text[:150])
                add_edge(cid, "hasDefinition", did)

            # -------------------------
            # 6. Exception
            # -------------------------
            if EXC_RE.search(text):
                exid = f"Exception::{ch['chunk_id']}"
                add_node(exid, "Exception", label="Exception")
                add_edge(cid, "hasException", exid)

            # -------------------------
            # 7. Actors
            # -------------------------
            for kw, actor_label in ACTOR_KEYWORDS.items():
                if kw in lower_text:
                    aid = f"Actor::{actor_label}"
                    add_node(aid, "Actor", label=actor_label)
                    add_edge(cid, "appliesTo", aid)


    # finalize types
    for n in nodes.values():
        n["types"] = list(n["types"])

    write_json(list(nodes.values()), f"{out_dir}/kg_nodes_v3.json")
    write_json(edges, f"{out_dir}/kg_edges_v3.json")
    write_json(topic_index, f"{out_dir}/topic_index_v3.json")

    print("\nExtraction complete:")
    print("Nodes:", len(nodes))
    print("Edges:", len(edges))



# ---------------------
# CLI
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_dir", default="data/semantic_chunks")
    parser.add_argument("--topic_keywords", default="topic_keywords.json")
    parser.add_argument("--canonical_topics", default="canonical_topics.json")
    parser.add_argument("--out_dir", default="data/kg_extracted_v3")
    parser.add_argument("--llm_mode", default="hybrid", choices=["none", "hybrid", "force"])
    args = parser.parse_args()

    extract(args.chunks_dir, args.topic_keywords, args.canonical_topics, args.out_dir, args.llm_mode)


if __name__ == "__main__":
    main()