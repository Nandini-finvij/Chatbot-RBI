# dynamic_topic_matcher.py
"""
Embedding-based topic detection
Replaces hardcoded TOPIC_RULES in chatbot_backend.py

Features:
- Dynamic topic matching via embeddings
- Multi-topic query support
- Configurable similarity thresholds
- Pre-computed embeddings for fast lookup
"""

import json
import torch
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from typing import List, Tuple, Optional
import logging

log = logging.getLogger("topic_matcher")

class DynamicTopicMatcher:
    def __init__(
        self,
        canonical_topics_path: str = "canonical_topics.json",
        embeddings_path: str = "data/embeddings/topic_embeddings.pt",
        model: Optional[SentenceTransformer] = None,
        similarity_threshold: float = 0.55
    ):
        """
        Initialize topic matcher.

        Args:
            canonical_topics_path: Path to canonical topics JSON
            embeddings_path: Path to save/load embeddings
            model: Pre-initialized SentenceTransformer (optional)
            similarity_threshold: Minimum similarity for topic match
        """
        self.canonical_path = canonical_topics_path
        self.embeddings_path = embeddings_path
        self.similarity_threshold = similarity_threshold

        # Use provided model or load new one
        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        # Initialize data structures
        self.topics = {}           # canonical_key -> {label, synonyms}
        self.topic_keys = []       # List of topic keys (with duplicates for synonyms)
        self.topic_texts = []      # List of topic text variations
        self.topic_embeddings = None  # Tensor of embeddings

        # Load or build embeddings
        if Path(embeddings_path).exists():
            self._load_embeddings()
        else:
            log.warning(f"Embeddings not found at {embeddings_path}. Building now...")
            self.build_embeddings(save=True)

    def _load_embeddings(self):
        """Load pre-computed topic embeddings from disk."""
        try:
            data = torch.load(self.embeddings_path)
            self.topics = data["topics"]
            self.topic_keys = data["topic_keys"]
            self.topic_embeddings = data["embeddings"]
            self.topic_texts = data.get("texts", [])
            log.info(f"✓ Loaded {len(set(self.topic_keys))} topics ({len(self.topic_keys)} variations)")
        except Exception as e:
            log.error(f"Failed to load embeddings: {e}")
            log.warning("Building embeddings from scratch...")
            self.build_embeddings(save=True)

    def build_embeddings(self, save: bool = False):
        """
        Build embeddings from canonical_topics.json.

        Args:
            save: Whether to save embeddings to disk
        """
        log.info(f"Building embeddings from {self.canonical_path}...")

        # Load canonical topics
        try:
            with open(self.canonical_path, encoding="utf-8") as f:
                canonical = json.load(f)
        except Exception as e:
            log.error(f"Failed to load canonical topics: {e}")
            raise

        # Flatten all topic variations
        all_texts = []
        all_keys = []

        for topic_key, topic_data in canonical.items():
            # Handle both dict format and array format (for backwards compatibility)
            if isinstance(topic_data, dict):
                label = topic_data.get("label", topic_key)
                synonyms = topic_data.get("synonyms", [])
            elif isinstance(topic_data, list):
                # Old format: array of synonyms
                label = topic_data[0] if topic_data else topic_key
                synonyms = topic_data[1:] if len(topic_data) > 1 else []
            else:
                log.warning(f"Invalid format for topic {topic_key}, skipping")
                continue

            # Add label as primary representation
            all_texts.append(label)
            all_keys.append(topic_key)

            # Add all synonyms
            for syn in synonyms:
                if syn and isinstance(syn, str):
                    all_texts.append(syn)
                    all_keys.append(topic_key)

        if not all_texts:
            raise ValueError("No valid topics found in canonical_topics.json")

        log.info(f"Encoding {len(all_texts)} topic variations for {len(set(all_keys))} unique topics...")

        # Encode all at once (batched for efficiency)
        embeddings = self.model.encode(
            all_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32
        )

        # Store in instance
        self.topics = canonical
        self.topic_keys = all_keys
        self.topic_texts = all_texts
        self.topic_embeddings = embeddings

        log.info(f"✓ Built {len(all_keys)} topic embeddings")

        # Save if requested
        if save:
            Path(self.embeddings_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "topics": canonical,
                "topic_keys": all_keys,
                "embeddings": embeddings,
                "texts": all_texts
            }, self.embeddings_path)
            log.info(f"✓ Saved embeddings to {self.embeddings_path}")

    def match(self, query: str, threshold: Optional[float] = None) -> Tuple[Optional[str], float]:
        """
        Find best matching topic for query.

        Args:
            query: User query string
            threshold: Override default similarity threshold

        Returns:
            Tuple of (topic_key, confidence_score)
            Returns (None, score) if no match above threshold
        """
        if threshold is None:
            threshold = self.similarity_threshold

        # Encode query
        query_emb = self.model.encode(query, convert_to_tensor=True)

        # Compute similarities with all topic variations
        scores = util.cos_sim(query_emb, self.topic_embeddings)[0]

        # Find best match
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()

        if best_score < threshold:
            return None, best_score

        return self.topic_keys[best_idx], best_score

    def match_multiple(self, query: str, top_k: int = 3, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Find top-k matching topics for query.

        Args:
            query: User query string
            top_k: Number of top matches to return
            threshold: Minimum similarity (optional)

        Returns:
            List of (topic_key, score) tuples, sorted by score descending
        """
        if threshold is None:
            threshold = self.similarity_threshold * 0.7  # Lower threshold for multi-match

        # Encode query
        query_emb = self.model.encode(query, convert_to_tensor=True)

        # Compute similarities
        scores = util.cos_sim(query_emb, self.topic_embeddings)[0]

        # Get top-k indices
        top_indices = scores.topk(min(top_k * 3, len(scores))).indices.tolist()

        # Deduplicate by topic_key and sort by score
        seen_topics = set()
        results = []

        for idx in top_indices:
            topic_key = self.topic_keys[idx]
            score = scores[idx].item()

            # Skip if already seen or below threshold
            if topic_key in seen_topics or score < threshold:
                continue

            seen_topics.add(topic_key)
            results.append((topic_key, score))

            if len(results) >= top_k:
                break

        return results

    def get_all_topics(self) -> List[str]:
        """Get list of all unique topic keys."""
        return list(set(self.topic_keys))

    def get_topic_label(self, topic_key: str) -> str:
        """Get human-readable label for topic key."""
        topic_data = self.topics.get(topic_key, {})
        if isinstance(topic_data, dict):
            return topic_data.get("label", topic_key)
        elif isinstance(topic_data, list) and topic_data:
            return topic_data[0]
        return topic_key

    def debug_match(self, query: str, top_k: int = 10) -> dict:
        """
        Debug helper to see all topic matching scores.

        Returns:
            Dictionary with query and top matches
        """
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.topic_embeddings)[0]

        # Get all scores with their topics and texts
        matches = []
        for idx, score in enumerate(scores.tolist()):
            matches.append({
                "topic": self.topic_keys[idx],
                "text": self.topic_texts[idx],
                "score": round(score, 4)
            })

        # Sort by score and deduplicate
        matches.sort(key=lambda x: x["score"], reverse=True)

        seen_topics = set()
        unique_matches = []
        for m in matches:
            if m["topic"] not in seen_topics:
                seen_topics.add(m["topic"])
                unique_matches.append(m)
                if len(unique_matches) >= top_k:
                    break

        return {
            "query": query,
            "threshold": self.similarity_threshold,
            "top_matches": unique_matches
        }


# ================================================================================
# SINGLETON PATTERN
# ================================================================================

_matcher_instance = None

def get_topic_matcher(
    canonical_topics_path: str = "canonical_topics.json",
    embeddings_path: str = "data/embeddings/topic_embeddings.pt",
    model: Optional[SentenceTransformer] = None,
    similarity_threshold: float = 0.55
) -> DynamicTopicMatcher:
    """
    Get or create singleton topic matcher instance.

    Args:
        canonical_topics_path: Path to canonical topics JSON
        embeddings_path: Path to embeddings file
        model: Pre-initialized model (optional)
        similarity_threshold: Minimum similarity for matches

    Returns:
        DynamicTopicMatcher instance
    """
    global _matcher_instance

    if _matcher_instance is None:
        _matcher_instance = DynamicTopicMatcher(
            canonical_topics_path=canonical_topics_path,
            embeddings_path=embeddings_path,
            model=model,
            similarity_threshold=similarity_threshold
        )

    return _matcher_instance
