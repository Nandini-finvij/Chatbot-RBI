"""
prompt_builder.py — FINAL (DEFENSIVE + KG-SAFE)

- Never crashes on None values
- Filters incomplete KG facts
- Clean, readable prompts
"""

from collections import defaultdict
from typing import List, Tuple, Dict, Any


def _safe_str(x):
    """Convert to string safely, return None if empty."""
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def build_prompt(
    question: str,
    chunks: List[Tuple[str, str]],
    kg_facts: List[Dict[str, Any]]
) -> str:
    """
    Build a grounded prompt using:
    - Retrieved document chunks
    - KG facts (relations)

    This function is 100% defensive against bad / partial KG data.
    """

    # ---------------- KG SECTION ----------------
    kg_lines = []
    rel_map = defaultdict(list)

    for fact in kg_facts or []:
        rel = _safe_str(fact.get("relation"))
        label = _safe_str(fact.get("label")) or _safe_str(fact.get("target"))

        if not rel or not label:
            continue  # 🚫 skip bad KG facts safely

        rel_map[rel].append(label)

    for rel, items in rel_map.items():
        # Deduplicate + stringify
        clean_items = sorted(set(_safe_str(i) for i in items if _safe_str(i)))
        if not clean_items:
            continue
        kg_lines.append(f"{rel}: {', '.join(clean_items)}")

    kg_block = "\n".join(f"- {line}" for line in kg_lines) if kg_lines else "- (No structured KG facts found)"

    # ---------------- DOCUMENT SECTION ----------------
    doc_lines = []
    for cid, text in chunks:
        if not text:
            continue
        snippet = text.strip().replace("\n", " ")
        snippet = snippet[:1200]
        doc_lines.append(f"[{cid}] {snippet}")

    doc_block = "\n\n".join(doc_lines) if doc_lines else "(No document context found)"

    # ---------------- FINAL PROMPT ----------------
    prompt = f"""
You are an RBI regulatory assistant.

RULES:
- Answer ONLY using the context below.
- Do NOT guess or hallucinate.
- If the answer is not present, say:
  "I cannot find this information in the provided RBI documents."

READABILITY REQUIREMENTS:
- Write at a 10th grade reading level (Flesch-Kincaid Grade: 10 or below)
- Use simple, everyday words instead of complex jargon
- Keep sentences short (15-20 words maximum)
- Break complex concepts into simple explanations
- Use bullet points for lists and multiple points
- Explain technical terms in plain language when you must use them

====================
KNOWLEDGE GRAPH FACTS
====================
{kg_block}

====================
DOCUMENT CONTEXT
====================
{doc_block}

====================
USER QUESTION
====================
{question}

TASK:
Provide a clear, precise, RBI-compliant answer that is easy to understand for someone with basic knowledge.
Use simple language and short sentences.
"""

    return prompt.strip()
