import json
import pytest
from poetry.transcriber import PoetryTranscriber

class DummyWhisper:
    def transcribe(self, *args, **kwargs):
        return [], None  # returns empty segments

def test_load_poems(monkeypatch, tmp_path):
    # --- Create dummy poems file ---
    poems_file = tmp_path / "poems.json"
    poems_file.write_text(json.dumps([
        {"poem_id": 1, "title": "Test Poem", "author": "Tester", "poem": "Hello world"}
    ]), encoding="utf-8")

    # --- Patch WhisperModel inside transcriber ---
    monkeypatch.setattr("poetry.transcriber.WhisperModel", lambda *a, **k: DummyWhisper())

    # --- Initialize Transcriber with mock model ---
    transcriber = PoetryTranscriber(poems_file=str(poems_file), device="cpu", model_name="tiny")

    # --- Assertions ---
    assert len(transcriber.poems) == 1
    poem = transcriber.poems[0]
    assert poem.title == "Test Poem"
    assert "hello" in poem.poem_words
    assert "world" in poem.poem_words

def test_find_poem_title(monkeypatch, tmp_path):
    poems_file = tmp_path / "poems.json"
    poems_file.write_text(json.dumps([
        {"poem_id": 2, "title": "Bonjour Ami", "author": "Tester", "poem": "Salut mon ami"}
    ]), encoding="utf-8")

    monkeypatch.setattr("poetry.transcriber.WhisperModel", lambda *a, **k: DummyWhisper())

    transcriber = PoetryTranscriber(poems_file=str(poems_file), device="cpu", model_name="tiny")
    poem = transcriber._find_poem_by_title("bonjour")
    
    assert poem is not None
    assert poem.title == "Bonjour Ami"
