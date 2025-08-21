import pytest
from poetry.similarity import levenshtein_similarity, tokenize_speech, match_multiple_words

def test_levenshtein_similarity():
    assert levenshtein_similarity("chat", "chat") == 1.0
    assert 0 <= levenshtein_similarity("chat", "chien") <= 1

def test_tokenize_speech():
    text = "Bonjour, l'ami!"
    tokens = tokenize_speech(text)
    assert "bonjour" in tokens
    assert "l'ami" in tokens

def test_match_multiple_words():
    spoken = ["bonjour", "ami"]
    target = ["bonjour", "mon", "ami"]
    result = match_multiple_words(spoken, target, 0)
    assert result.correct_count >= 1
    assert result.total_words == len(result.matches)
