# Stop Words tiếng Việt - Đọc từ file Vietnamese-stopwords.txt

import os
from typing import List


class Vietnamese:
    
    def __init__(self):
        self.words: List[str] = []
        self._load_stopwords_from_file()
    
    def _load_stopwords_from_file(self) -> None:
        # Đọc stop words từ file Vietnamese-stopwords.txt
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stopwords_file = os.path.join(current_dir, 'Vietnamese-stopwords.txt')
        
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            self.words = [
                line.strip().lower() 
                for line in f 
                if line.strip() and not line.startswith('#')
            ]
    
    def exist(self, word: str) -> bool:
        # Kiểm tra từ có phải stop word không
        return word.lower() in self.words
