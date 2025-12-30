"""
readability_analyzer.py — Flesch-Kincaid Readability Measurement

Provides readability scoring and analysis for:
- Chatbot responses
- Compliance plans
- Regulatory documents
- General text content

Metrics Calculated:
- Flesch Reading Ease Score (0-100, higher = easier)
- Flesch-Kincaid Grade Level (US grade level)
- Gunning Fog Index
- SMOG Index
- Coleman-Liau Index
- Automated Readability Index (ARI)
- Average sentence length
- Average syllables per word
- Complex word percentage

Author: Finvij Team
Feature: Readability Measurement
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics

log = logging.getLogger("readability")


# ================================================================================
# READABILITY SCORING DATA CLASS
# ================================================================================

@dataclass
class ReadabilityScore:
    """Container for all readability metrics"""

    # Primary metrics
    flesch_reading_ease: float  # 0-100, higher = easier
    flesch_kincaid_grade: float  # US grade level

    # Additional metrics
    gunning_fog: float
    smog_index: float
    coleman_liau_index: float
    automated_readability_index: float

    # Text statistics
    total_sentences: int
    total_words: int
    total_syllables: int
    total_characters: int
    complex_words: int

    # Derived metrics
    avg_sentence_length: float
    avg_syllables_per_word: float
    avg_word_length: float
    complex_word_percentage: float

    # Interpretation
    readability_level: str  # "Very Easy", "Easy", "Standard", "Difficult", "Very Difficult"
    grade_level_interpretation: str  # "Elementary", "Middle School", "High School", "College", "Graduate"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def to_summary(self) -> str:
        """Generate human-readable summary"""
        return f"""Readability Analysis:
- Flesch Reading Ease: {self.flesch_reading_ease:.1f} ({self.readability_level})
- Grade Level: {self.flesch_kincaid_grade:.1f} ({self.grade_level_interpretation})
- Avg Sentence Length: {self.avg_sentence_length:.1f} words
- Complex Words: {self.complex_word_percentage:.1f}%"""


# ================================================================================
# READABILITY ANALYZER CLASS
# ================================================================================

class ReadabilityAnalyzer:
    """
    Analyzes text readability using multiple metrics.
    """

    def __init__(self):
        """Initialize readability analyzer"""
        self.vowels = set('aeiouAEIOU')

    def analyze(self, text: str) -> ReadabilityScore:
        """
        Analyze text and return comprehensive readability scores.

        Args:
            text: Text to analyze

        Returns:
            ReadabilityScore object with all metrics
        """
        if not text or not text.strip():
            return self._empty_score()

        # Clean and prepare text
        clean_text = self._clean_text(text)

        # Extract text statistics
        sentences = self._count_sentences(clean_text)
        words = self._get_words(clean_text)
        total_words = len(words)
        total_characters = sum(len(word) for word in words)
        total_syllables = sum(self._count_syllables(word) for word in words)
        complex_words = sum(1 for word in words if self._is_complex_word(word))

        # Avoid division by zero
        if total_words == 0 or sentences == 0:
            return self._empty_score()

        # Calculate derived metrics
        avg_sentence_length = total_words / sentences
        avg_syllables_per_word = total_syllables / total_words
        avg_word_length = total_characters / total_words
        complex_word_percentage = (complex_words / total_words) * 100

        # Calculate readability scores
        flesch_reading_ease = self._flesch_reading_ease(
            total_words, sentences, total_syllables
        )

        flesch_kincaid_grade = self._flesch_kincaid_grade(
            total_words, sentences, total_syllables
        )

        gunning_fog = self._gunning_fog(
            total_words, sentences, complex_words
        )

        smog_index = self._smog_index(sentences, complex_words)

        coleman_liau = self._coleman_liau_index(
            total_words, sentences, total_characters
        )

        ari = self._automated_readability_index(
            total_words, sentences, total_characters
        )

        # Interpret scores
        readability_level = self._interpret_flesch_ease(flesch_reading_ease)
        grade_interpretation = self._interpret_grade_level(flesch_kincaid_grade)

        return ReadabilityScore(
            flesch_reading_ease=flesch_reading_ease,
            flesch_kincaid_grade=flesch_kincaid_grade,
            gunning_fog=gunning_fog,
            smog_index=smog_index,
            coleman_liau_index=coleman_liau,
            automated_readability_index=ari,
            total_sentences=sentences,
            total_words=total_words,
            total_syllables=total_syllables,
            total_characters=total_characters,
            complex_words=complex_words,
            avg_sentence_length=avg_sentence_length,
            avg_syllables_per_word=avg_syllables_per_word,
            avg_word_length=avg_word_length,
            complex_word_percentage=complex_word_percentage,
            readability_level=readability_level,
            grade_level_interpretation=grade_interpretation
        )

    # ============================================================================
    # FLESCH-KINCAID FORMULAS
    # ============================================================================

    def _flesch_reading_ease(self, words: int, sentences: int, syllables: int) -> float:
        """
        Flesch Reading Ease Score

        Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)

        Score interpretation:
        90-100: Very Easy (5th grade)
        80-89: Easy (6th grade)
        70-79: Fairly Easy (7th grade)
        60-69: Standard (8th-9th grade)
        50-59: Fairly Difficult (10th-12th grade)
        30-49: Difficult (College)
        0-29: Very Difficult (College graduate)
        """
        asl = words / sentences  # Average sentence length
        asw = syllables / words  # Average syllables per word

        score = 206.835 - (1.015 * asl) - (84.6 * asw)

        # Clamp between 0 and 100
        return max(0.0, min(100.0, score))

    def _flesch_kincaid_grade(self, words: int, sentences: int, syllables: int) -> float:
        """
        Flesch-Kincaid Grade Level

        Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

        Returns US grade level (e.g., 8.0 = 8th grade)
        """
        asl = words / sentences
        asw = syllables / words

        grade = (0.39 * asl) + (11.8 * asw) - 15.59

        return max(0.0, grade)

    # ============================================================================
    # ADDITIONAL READABILITY FORMULAS
    # ============================================================================

    def _gunning_fog(self, words: int, sentences: int, complex_words: int) -> float:
        """
        Gunning Fog Index

        Formula: 0.4 * [(words/sentences) + 100 * (complex_words/words)]

        Complex words = 3+ syllables
        """
        asl = words / sentences
        pcw = (complex_words / words) * 100

        fog = 0.4 * (asl + pcw)

        return max(0.0, fog)

    def _smog_index(self, sentences: int, complex_words: int) -> float:
        """
        SMOG (Simple Measure of Gobbledygook) Index

        Formula: 1.043 * sqrt(complex_words * (30/sentences)) + 3.1291

        Requires at least 30 sentences for accuracy
        """
        if sentences == 0:
            return 0.0

        import math

        # Adjust for fewer than 30 sentences
        polysyllable_count = complex_words * (30 / sentences) if sentences < 30 else complex_words

        smog = 1.043 * math.sqrt(polysyllable_count) + 3.1291

        return max(0.0, smog)

    def _coleman_liau_index(self, words: int, sentences: int, characters: int) -> float:
        """
        Coleman-Liau Index

        Formula: 0.0588 * L - 0.296 * S - 15.8

        Where:
        L = average number of letters per 100 words
        S = average number of sentences per 100 words
        """
        L = (characters / words) * 100
        S = (sentences / words) * 100

        cli = (0.0588 * L) - (0.296 * S) - 15.8

        return max(0.0, cli)

    def _automated_readability_index(self, words: int, sentences: int, characters: int) -> float:
        """
        Automated Readability Index (ARI)

        Formula: 4.71 * (characters/words) + 0.5 * (words/sentences) - 21.43
        """
        char_per_word = characters / words
        words_per_sentence = words / sentences

        ari = (4.71 * char_per_word) + (0.5 * words_per_sentence) - 21.43

        return max(0.0, ari)

    # ============================================================================
    # TEXT PROCESSING UTILITIES
    # ============================================================================

    def _clean_text(self, text: str) -> str:
        """Clean text for analysis"""
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Code
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Links

        # Remove special characters but keep sentence punctuation
        text = re.sub(r'[^\w\s.!?;:]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)

        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return max(1, len(sentences))  # At least 1 sentence

    def _get_words(self, text: str) -> List[str]:
        """Extract words from text"""
        # Split on whitespace and filter out empty strings
        words = text.split()

        # Remove punctuation from words
        words = [re.sub(r'[^\w]', '', word) for word in words]

        # Filter out empty strings and single characters
        words = [w for w in words if len(w) > 0]

        return words

    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word.

        Rules:
        1. Count vowel groups
        2. Silent 'e' doesn't count
        3. Minimum 1 syllable per word
        """
        word = word.lower()

        # Remove trailing 'e'
        if word.endswith('e'):
            word = word[:-1]

        # Count vowel groups
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in self.vowels

            if is_vowel and not previous_was_vowel:
                syllable_count += 1

            previous_was_vowel = is_vowel

        # Ensure at least 1 syllable
        return max(1, syllable_count)

    def _is_complex_word(self, word: str) -> bool:
        """
        Check if word is complex (3+ syllables).

        Exceptions:
        - Proper nouns
        - Familiar words
        - Words ending in -es, -ed (verb forms)
        """
        syllables = self._count_syllables(word)

        # Complex = 3+ syllables
        if syllables < 3:
            return False

        # Exception: common verb forms
        if word.lower().endswith(('es', 'ed', 'ing')):
            return False

        return True

    # ============================================================================
    # INTERPRETATION HELPERS
    # ============================================================================

    def _interpret_flesch_ease(self, score: float) -> str:
        """Interpret Flesch Reading Ease score"""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"

    def _interpret_grade_level(self, grade: float) -> str:
        """Interpret grade level"""
        if grade <= 5:
            return "Elementary School"
        elif grade <= 8:
            return "Middle School"
        elif grade <= 12:
            return "High School"
        elif grade <= 16:
            return "College"
        else:
            return "Graduate Level"

    def _empty_score(self) -> ReadabilityScore:
        """Return empty/zero score"""
        return ReadabilityScore(
            flesch_reading_ease=0.0,
            flesch_kincaid_grade=0.0,
            gunning_fog=0.0,
            smog_index=0.0,
            coleman_liau_index=0.0,
            automated_readability_index=0.0,
            total_sentences=0,
            total_words=0,
            total_syllables=0,
            total_characters=0,
            complex_words=0,
            avg_sentence_length=0.0,
            avg_syllables_per_word=0.0,
            avg_word_length=0.0,
            complex_word_percentage=0.0,
            readability_level="N/A",
            grade_level_interpretation="N/A"
        )


# ================================================================================
# BATCH ANALYSIS
# ================================================================================

class ReadabilityBatchAnalyzer:
    """Analyze multiple texts and generate aggregate statistics"""

    def __init__(self):
        self.analyzer = ReadabilityAnalyzer()

    def analyze_batch(self, texts: List[str]) -> Dict:
        """
        Analyze multiple texts and return aggregate statistics.

        Args:
            texts: List of text strings

        Returns:
            Dictionary with individual scores and aggregate stats
        """
        scores = [self.analyzer.analyze(text) for text in texts]

        if not scores:
            return {"individual_scores": [], "aggregate": {}}

        # Calculate aggregate statistics
        flesch_scores = [s.flesch_reading_ease for s in scores]
        grade_levels = [s.flesch_kincaid_grade for s in scores]

        aggregate = {
            "count": len(scores),
            "flesch_reading_ease": {
                "mean": statistics.mean(flesch_scores),
                "median": statistics.median(flesch_scores),
                "min": min(flesch_scores),
                "max": max(flesch_scores),
                "stdev": statistics.stdev(flesch_scores) if len(flesch_scores) > 1 else 0.0
            },
            "flesch_kincaid_grade": {
                "mean": statistics.mean(grade_levels),
                "median": statistics.median(grade_levels),
                "min": min(grade_levels),
                "max": max(grade_levels),
                "stdev": statistics.stdev(grade_levels) if len(grade_levels) > 1 else 0.0
            },
            "total_words": sum(s.total_words for s in scores),
            "total_sentences": sum(s.total_sentences for s in scores)
        }

        return {
            "individual_scores": [s.to_dict() for s in scores],
            "aggregate": aggregate
        }


# ================================================================================
# CONVENIENCE FUNCTIONS
# ================================================================================

_global_analyzer = None

def get_readability_analyzer() -> ReadabilityAnalyzer:
    """Get global ReadabilityAnalyzer instance"""
    global _global_analyzer

    if _global_analyzer is None:
        _global_analyzer = ReadabilityAnalyzer()

    return _global_analyzer


def analyze_readability(text: str) -> ReadabilityScore:
    """
    Quick function to analyze text readability.

    Args:
        text: Text to analyze

    Returns:
        ReadabilityScore object
    """
    analyzer = get_readability_analyzer()
    return analyzer.analyze(text)


def get_flesch_score(text: str) -> float:
    """
    Get just the Flesch Reading Ease score.

    Args:
        text: Text to analyze

    Returns:
        Flesch Reading Ease score (0-100)
    """
    score = analyze_readability(text)
    return score.flesch_reading_ease


def get_grade_level(text: str) -> float:
    """
    Get just the Flesch-Kincaid Grade Level.

    Args:
        text: Text to analyze

    Returns:
        Grade level (e.g., 8.5 = 8th-9th grade)
    """
    score = analyze_readability(text)
    return score.flesch_kincaid_grade
