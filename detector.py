"""
LANGUAGE DETECTOR FROM SCRATCH
================================
This implements the same core idea Google Translate, browsers, and other
language ID tools use under the hood: comparing the N-GRAM FINGERPRINT of
your text against pre-built fingerprints of known languages.

WHAT IS AN N-GRAM?
An n-gram is just a sliding window of N characters.
For the word "hello" with n=3 (trigrams), the n-grams are:
    "hel", "ell", "llo"
(We slide one character at a time across the text.)

WHY CHARACTER N-GRAMS INSTEAD OF WHOLE WORDS?
- Works even with typos, partial words, or words it's never seen
- Works for languages without spaces (Chinese, Japanese, Thai)
- Captures the "texture" of a language: English is full of "th", "ing";
  German is full of "ch", "sch"; Spanish is full of "es", "ón"

THE ALGORITHM (this is literally what we're doing):
1. TRAINING (done once, offline, like Google does with billions of words):
   - For each known language, slice all its training text into n-grams
   - Count how often each n-gram appears
   - Keep only the most common ones -> this is the language's "fingerprint"

2. DETECTION (done every time you type, must be FAST):
   - Slice your input text into n-grams the same way
   - Rank your n-grams by frequency (most common ones in YOUR text first)
   - For each candidate language, measure how far your ranked list is from
     that language's ranked list (this is called "out-of-place distance")
   - The language with the SMALLEST distance wins

This specific ranking-distance approach is a real, published technique
called the Cavnar-Trenkle algorithm (1994) - one of the foundational
methods in language identification, still conceptually similar to what
fast production systems use today.
"""

import re
from collections import Counter

# How many characters per n-gram chunk. We use multiple sizes (1 through 4)
# combined, because single chars catch alphabet differences (Cyrillic vs
# Latin vs Arabic script) while longer n-grams catch language-specific
# patterns (like "tion" in English or "sch" in German).
NGRAM_SIZES = (1, 2, 3, 4)

# How many of the TOP n-grams to keep per language fingerprint.
# Real systems often use ~300. We use a smaller number since our training
# text per language is small (a few hundred words, not millions).
PROFILE_SIZE = 200


def extract_ngrams(text: str) -> Counter:
    """
    Break text into character n-grams of multiple sizes and count them.

    Example: extract_ngrams("the cat") with NGRAM_SIZES=(1,2,3) would
    include n-grams like "t", "h", "e", " ", "th", "he", "e ", " c", "the",
    "he ", "e c", " ca", "cat", etc.
    """
    # Normalize: lowercase, collapse whitespace, strip punctuation noise.
    # This matters because "The" and "the" should count as the same
    # signal -- we care about LANGUAGE patterns, not casing or punctuation.
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    counts = Counter()
    for n in NGRAM_SIZES:
        # Slide a window of size n across the text, one character at a time.
        for i in range(len(text) - n + 1):
            gram = text[i : i + n]
            # Skip n-grams that are pure whitespace -- not informative.
            if gram.strip():
                counts[gram] += 1
    return counts


def build_profile(text: str) -> list[str]:
    """
    Build a language "fingerprint": the list of its most common n-grams,
    ordered from most frequent to least frequent.

    This is the RANKED LIST idea from the Cavnar-Trenkle algorithm.
    We don't care about exact frequency counts -- just the ORDER,
    because order is more stable across different text lengths than
    raw frequency would be.
    """
    counts = extract_ngrams(text)
    # most_common() returns [(ngram, count), ...] sorted by count descending
    ranked = [gram for gram, _count in counts.most_common(PROFILE_SIZE)]
    return ranked


def profile_distance(sample_ranked: list[str], language_ranked: list[str]) -> int:
    """
    Measure how "far apart" two ranked n-gram lists are.

    This is the "out-of-place" distance metric:
    For each n-gram in the SAMPLE text, find its rank/position in the
    LANGUAGE profile. If it's not found at all, give it a big penalty
    (we use PROFILE_SIZE as the max penalty -- "maximally out of place").

    Sum all these positional differences. LOWER total = more similar =
    more likely to be that language.

    Why this works: if your text is really English, most of its common
    n-grams ("th", "he", "in"...) will appear at similar high ranks in
    the English profile too. If your text is Spanish, those n-grams
    will be rare or missing in the English profile -> high penalty.
    """
    # Map each n-gram in the language profile to its rank position (0 = most common)
    language_rank_lookup = {gram: rank for rank, gram in enumerate(language_ranked)}

    total_distance = 0
    for sample_rank, gram in enumerate(sample_ranked):
        if gram in language_rank_lookup:
            language_rank = language_rank_lookup[gram]
            total_distance += abs(sample_rank - language_rank)
        else:
            # This n-gram never showed up in the language's training data at
            # all -- treat it as "maximally far away".
            total_distance += PROFILE_SIZE

    return total_distance


class LanguageDetector:
    """
    The full detector. Build it once with training text for each language
    (this is the slow, one-time "training" step). Then call .detect() as
    many times as you want -- each call is fast because it's just counting
    and comparing, no internet, no heavy ML model.
    """

    def __init__(self, training_data: dict[str, str]):
        # This dict comprehension is the entire "training" process:
        # for every language, slice its sample text into ranked n-grams.
        self.profiles: dict[str, list[str]] = {
            language: build_profile(text) for language, text in training_data.items()
        }

    def detect(self, text: str, top_n: int = 5) -> list[tuple[str, float]]:
        """
        Detect the language of `text`.

        Returns a list of (language, confidence_score) tuples, best first.
        confidence_score is normalized to roughly 0-100 for readability
        (it is NOT a true probability, just a relative "closeness" score).
        """
        if not text.strip():
            return []

        sample_ranked = build_profile(text)
        if not sample_ranked:
            return []

        # Compare the sample against EVERY language profile.
        # This is an O(languages * profile_size) operation -- with ~20
        # languages and a profile size of 200, that's only ~4000 comparisons.
        # That's why this is fast enough to run on every keystroke.
        raw_distances = {
            language: profile_distance(sample_ranked, profile)
            for language, profile in self.profiles.items()
        }

        # Convert "distance" (lower = better) into a friendlier "score"
        # (higher = better) for display purposes.
        max_distance = max(raw_distances.values()) or 1
        scores = {
            language: 100 * (1 - dist / max_distance)
            for language, dist in raw_distances.items()
        }

        ranked_languages = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked_languages[:top_n]


if __name__ == "__main__":
    # Quick manual test from the command line.
    from training_data import TRAINING_TEXT

    detector = LanguageDetector(TRAINING_TEXT)

    test_sentences = [
        "Hello, how are you doing today?",
        "Hola, como estas tu hoy?",
        "Bonjour, comment vas-tu aujourd'hui?",
        "Guten Tag, wie geht es dir heute?",
        "こんにちは、今日はどうですか？",
        "你好，今天怎么样？",
        "Привет, как дела сегодня?",
    ]

    for sentence in test_sentences:
        results = detector.detect(sentence, top_n=3)
        print(f"\nInput: {sentence}")
        for lang, score in results:
            print(f"  {lang:12s} {score:5.1f}")