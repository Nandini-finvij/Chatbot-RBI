"""
profile_aware_retrieval.py — Profile-Aware Retrieval & Re-ranking

Features:
- Profile relevance scoring
- Context-aware re-ranking
- Entity-specific filtering
- Product-specific boosting

Author: Finvij Team
Phase: 2 (Days 10-12)
"""

from typing import List, Tuple, Dict, Optional, Any
from user_profile import UserProfile
import re
import logging

log = logging.getLogger("profile_retrieval")

# ================================================================================
# PROFILE RELEVANCE SCORING
# ================================================================================

# Entity-specific keywords for relevance scoring
ENTITY_KEYWORDS = {
    "NBFC": ["nbfc", "non-banking financial company", "systemically important", "deposit taking", "non-deposit taking"],
    "Bank": ["bank", "commercial bank", "scheduled commercial bank", "banking company"],
    "Fintech": ["fintech", "technology", "digital", "app-based", "platform"],
    "LSP": ["lending service provider", "lsp", "digital lending platform", "marketplace"],
    "Cooperative_Bank": ["cooperative", "urban cooperative bank", "ucb"],
    "Small_Finance_Bank": ["small finance bank", "sfb"],
    "Payment_Bank": ["payment bank", "payments bank"],
    "HFC": ["housing finance", "hfc", "housing loan"],
}

# Product-specific keywords
PRODUCT_KEYWORDS = {
    "gold_loans": ["gold loan", "gold", "ornament", "jewelry", "bullion", "ltv", "loan to value"],
    "personal_loans": ["personal loan", "unsecured loan", "consumer loan"],
    "microfinance": ["microfinance", "mfi", "jlg", "joint liability", "micro credit"],
    "housing_loans": ["housing loan", "home loan", "mortgage", "property"],
    "vehicle_loans": ["vehicle loan", "auto loan", "car loan"],
    "business_loans": ["business loan", "msme", "sme", "working capital"],
}

# Compliance area keywords
COMPLIANCE_KEYWORDS = {
    "digital_lending": ["digital lending", "dlg", "fldg", "lsp", "online lending", "app-based lending"],
    "kyc_aml": ["kyc", "know your customer", "aml", "anti money laundering", "cdd", "customer due diligence"],
    "provisioning": ["provisioning", "ecl", "expected credit loss", "npa", "impairment"],
    "asset_classification": ["asset classification", "npa", "substandard", "doubtful", "loss", "iracp"],
    "gold_loans": ["gold loan", "gold", "ltv"],
    "model_risk": ["model", "model risk", "model validation", "model governance"],
    "outsourcing": ["outsourcing", "vendor", "third party", "service provider"],
}


def compute_profile_relevance(text: str, profile: UserProfile) -> float:
    """
    Compute relevance score of a text chunk to user profile.

    Scoring factors:
    - Entity type match: +0.3
    - Product match: +0.2 per product
    - Compliance area match: +0.15 per area
    - License category match: +0.15
    - Geography match: +0.1 per region

    Returns:
        Relevance score (0.0 to 1.0+, can exceed 1.0 for highly relevant)
    """
    text_lower = text.lower()
    score = 0.0

    # 1. Entity type relevance
    if profile.entity_type and profile.entity_type != "Unknown":
        entity_keywords = ENTITY_KEYWORDS.get(profile.entity_type, [])
        for keyword in entity_keywords:
            if keyword in text_lower:
                score += 0.3
                break  # Only count once

    # 2. License category relevance
    if profile.license_category:
        license_lower = profile.license_category.lower()
        if license_lower in text_lower:
            score += 0.15

    # 3. Product relevance (can add multiple times)
    for product in profile.products:
        product_keywords = PRODUCT_KEYWORDS.get(product, [])
        for keyword in product_keywords:
            if keyword in text_lower:
                score += 0.2
                break  # Count once per product

    # 4. Compliance area relevance
    for area in profile.compliance_areas:
        area_keywords = COMPLIANCE_KEYWORDS.get(area, [])
        for keyword in area_keywords:
            if keyword in text_lower:
                score += 0.15
                break  # Count once per area

    # 5. Geography relevance
    for geo in profile.geography:
        if geo.lower() in text_lower:
            score += 0.1

    # 6. Digital lending specific
    if profile.digital_lending:
        if any(kw in text_lower for kw in ["digital lending", "dlg", "fldg", "lsp"]):
            score += 0.25

    if profile.has_fldg_arrangements:
        if "fldg" in text_lower or "first loss" in text_lower or "dlg" in text_lower:
            score += 0.2

    # Normalize to 0-1 range (but allow going above 1.0 for very relevant chunks)
    return min(score, 2.0)


def profile_aware_rerank(
    chunks: List[Tuple[str, str]],
    profile: Optional[UserProfile],
    top_k: int = 10,
    alpha: float = 0.4
) -> List[Tuple[str, str, float]]:
    """
    Re-rank chunks based on user profile relevance.

    Args:
        chunks: List of (chunk_id, text) tuples from RAG
        profile: User profile (optional)
        top_k: Number of chunks to return
        alpha: Profile relevance weight (0-1)
               - 0.0 = no profile influence (pure RAG ranking)
               - 1.0 = only profile relevance
               - 0.4 = 40% profile, 60% RAG similarity

    Returns:
        List of (chunk_id, text, combined_score) tuples
    """
    if not profile or not chunks:
        # No profile or no chunks - return as-is
        return [(cid, text, 1.0) for cid, text in chunks[:top_k]]

    scored_chunks = []

    # RAG ranking gives implicit score based on position
    for idx, (chunk_id, text) in enumerate(chunks):
        # RAG score: higher for earlier results (1.0 for first, decaying)
        rag_score = 1.0 - (idx / len(chunks))

        # Profile relevance score
        profile_score = compute_profile_relevance(text, profile)

        # Combined score: weighted average
        combined_score = (1 - alpha) * rag_score + alpha * profile_score

        scored_chunks.append((chunk_id, text, combined_score))

    # Sort by combined score
    scored_chunks.sort(key=lambda x: x[2], reverse=True)

    # Return top-k
    return scored_chunks[:top_k]


def filter_by_entity_type(
    chunks: List[Tuple[str, str]],
    entity_type: str,
    strict: bool = False
) -> List[Tuple[str, str]]:
    """
    Filter chunks to only those relevant to entity type.

    Args:
        chunks: List of (chunk_id, text)
        entity_type: Entity type to filter for
        strict: If True, exclude chunks that mention OTHER entity types

    Returns:
        Filtered chunks
    """
    if entity_type == "Unknown":
        return chunks

    entity_keywords = ENTITY_KEYWORDS.get(entity_type, [])
    if not entity_keywords:
        return chunks

    filtered = []

    for chunk_id, text in chunks:
        text_lower = text.lower()

        # Check if relevant to this entity type
        is_relevant = any(kw in text_lower for kw in entity_keywords)

        if strict:
            # Also check if it mentions OTHER entity types
            mentions_others = False
            for other_entity, other_keywords in ENTITY_KEYWORDS.items():
                if other_entity != entity_type:
                    if any(kw in text_lower for kw in other_keywords):
                        mentions_others = True
                        break

            # Only include if relevant AND doesn't mention others
            if is_relevant and not mentions_others:
                filtered.append((chunk_id, text))
        else:
            # Include if relevant (even if mentions others)
            if is_relevant or not any(
                any(kw in text_lower for kw in kws)
                for other_type, kws in ENTITY_KEYWORDS.items()
            ):
                # Include if relevant OR if it doesn't mention ANY entity type (general guidance)
                filtered.append((chunk_id, text))

    return filtered if filtered else chunks  # Fallback to all if nothing matches


# ================================================================================
# INTEGRATED PROFILE-AWARE RETRIEVAL
# ================================================================================

def profile_aware_retrieve(
    query: str,
    topic_key: Optional[str],
    profile: Optional[UserProfile],
    rag_search_func,
    kg_retrieve_func,
    top_k: int = 10,
    rerank_alpha: float = 0.4,
    filter_entity: bool = False
) -> Tuple[List[Tuple[str, str]], List[Dict], Dict[str, Any]]:
    """
    Profile-aware hybrid retrieval with re-ranking.

    Args:
        query: User query
        topic_key: Detected topic (can be None)
        profile: User profile (can be None)
        rag_search_func: Function for RAG search
        kg_retrieve_func: Function for KG retrieval
        top_k: Final number of chunks to return
        rerank_alpha: Profile weight in re-ranking (0-1)
        filter_entity: Whether to filter by entity type

    Returns:
        Tuple of (chunks, kg_facts, debug_info)
    """
    debug_info = {
        "profile_used": profile is not None,
        "entity_type": profile.entity_type if profile else None,
        "rerank_applied": False,
        "filter_applied": False,
    }

    # Step 1: Base retrieval (RAG + KG)
    base_chunks, kg_facts = kg_retrieve_func(query, topic_key)

    if not base_chunks:
        return [], kg_facts, debug_info

    # Step 2: Filter by entity type (optional)
    if filter_entity and profile and profile.entity_type != "Unknown":
        filtered_chunks = filter_by_entity_type(base_chunks, profile.entity_type, strict=False)
        if filtered_chunks:
            base_chunks = filtered_chunks
            debug_info["filter_applied"] = True

    # Step 3: Re-rank based on profile
    if profile:
        scored_chunks = profile_aware_rerank(
            base_chunks,
            profile,
            top_k=top_k * 2,  # Get more candidates for re-ranking
            alpha=rerank_alpha
        )
        debug_info["rerank_applied"] = True

        # Extract top-k after re-ranking
        final_chunks = [(cid, text) for cid, text, score in scored_chunks[:top_k]]
    else:
        # No profile - just return top-k from base retrieval
        final_chunks = base_chunks[:top_k]

    return final_chunks, kg_facts, debug_info


# ================================================================================
# PROFILE CONTEXT FOR PROMPTS
# ================================================================================

def build_profile_context(profile: Optional[UserProfile]) -> str:
    """
    Build profile context string to inject into LLM prompts.

    This helps the LLM provide personalized answers.
    """
    if not profile or profile.entity_type == "Unknown":
        return ""

    context_parts = []

    # Entity context
    if profile.entity_type:
        entity_str = profile.entity_type
        if profile.license_category:
            entity_str += f" ({profile.license_category})"
        context_parts.append(f"User is a: {entity_str}")

    # Products context
    if profile.products:
        context_parts.append(f"Offers products: {', '.join(profile.products)}")

    # Digital lending context
    if profile.digital_lending:
        dl_note = "Does digital lending"
        if profile.has_fldg_arrangements:
            dl_note += " with FLDG arrangements"
        if profile.uses_lsp:
            dl_note += " via LSP"
        context_parts.append(dl_note)

    # Scale context
    if profile.asset_size:
        context_parts.append(f"Scale: {profile.asset_size}")

    if not context_parts:
        return ""

    return "\n".join(f"- {part}" for part in context_parts)
