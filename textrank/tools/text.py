# Class lưu trữ dữ liệu văn bản đã phân tích (câu, từ, dấu câu)

from typing import List, Dict


class Text:
    
    def __init__(self):
        self._word_matrix: Dict[int, Dict[int, str]] = {}
        self._sentences: Dict[int, str] = {}
        self._marks: List[str] = []
    
    def set_word_matrix(self, word_matrix: Dict[int, Dict[int, str]]) -> None:
        self._word_matrix = word_matrix
    
    def set_sentences(self, sentences: Dict[int, str]) -> None:
        self._sentences = sentences
    
    def set_marks(self, marks: List[str]) -> None:
        self._marks = marks
    
    def get_word_matrix(self) -> Dict[int, Dict[int, str]]:
        return self._word_matrix
    
    def get_sentences(self) -> Dict[int, str]:
        return self._sentences
    
    def get_marks(self) -> List[str]:
        return self._marks
