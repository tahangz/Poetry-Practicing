from typing import List
import re

class WordMatch:
    def __init__(self, word: str, target: str, similarity: float, is_correct: bool, position: int):
        self.word = word
        self.target = target
        self.similarity = similarity
        self.is_correct = is_correct
        self.position = position


class MultiWordResult:
    def __init__(self):
        self.matches: List[WordMatch] = []
        self.correct_count = 0
        self.total_words = 0
        self.advance_count = 0
        
    def add_match(self, match: WordMatch):
        self.matches.append(match)
        self.total_words += 1
        if match.is_correct:
            self.correct_count += 1


class PoemData:
    def __init__(self, poem_dict: dict):
        self.poem_id = poem_dict["poem_id"]
        self.title = poem_dict["title"]
        self.author = poem_dict["author"]
        self.poem = poem_dict["poem"]
        self.title_words = self._clean_and_split(self.title)
        self.poem_words = self._clean_and_split(self.poem)
    
    def _clean_and_split(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s']", " ", text.lower())
        words = [w.strip() for w in cleaned.split() if w.strip()]
        return words
