import re
import Levenshtein
from typing import List
from .models import WordMatch, MultiWordResult
from .config import WORD_SIMILARITY_THRESHOLD, MAX_WORDS_PER_CHUNK

def levenshtein_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 1.0
    distance = Levenshtein.distance(a, b)
    max_len = max(len(a), len(b))
    return max(0.0, 1.0 - (distance / max_len))

def fuzzy_similarity(a: str, b: str) -> float:
    return levenshtein_similarity(a, b)

def tokenize_speech(text: str) -> List[str]:
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    words = [w.strip() for w in cleaned.split() if w.strip()]
    return words

def match_multiple_words(spoken_words: List[str], target_words: List[str], start_position: int,
                         similarity_threshold: float = WORD_SIMILARITY_THRESHOLD) -> MultiWordResult:
    result = MultiWordResult()
    if not spoken_words or start_position >= len(target_words):
        return result
    max_possible_words = min(len(spoken_words), MAX_WORDS_PER_CHUNK, len(target_words) - start_position)
    best_advance, best_matches = 0, []
    for j in range(len(spoken_words)):
        current_advance, current_matches = 0, []
        for i in range(max_possible_words):
            if j + i >= len(spoken_words) or start_position + i >= len(target_words):
                break
            spoken_word = spoken_words[j + i]
            target_word = target_words[start_position + i]
            similarity = levenshtein_similarity(spoken_word, target_word)
            is_correct = similarity >= similarity_threshold
            match = WordMatch(spoken_word, target_word, similarity, is_correct, start_position + i)
            current_matches.append(match)
            if is_correct:
                current_advance += 1
        if current_advance > best_advance:
            best_advance, best_matches = current_advance, current_matches
    for match in best_matches:
        result.add_match(match)
    result.advance_count = best_advance
    return result
