# Class phân tích văn bản thành câu và từ, loại bỏ stop words
# SỬ DỤNG VnCoreNLP để tách từ tiếng Việt chính xác

import re
import string
from typing import List, Dict, Optional, Any
from .text import Text
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stopwords.vietnamese import Vietnamese


class Parser:
    
    def __init__(self, vncorenlp_model: Any):
        """Khởi tạo Parser với VnCoreNLP model (BẮT BUỘC).
        
        Args:
            vncorenlp_model: VnCoreNLP instance với annotators=['wseg']
        
        Raises:
            ValueError: Nếu vncorenlp_model là None
        """
        if vncorenlp_model is None:
            raise ValueError("VnCoreNLP model là bắt buộc. Vui lòng khởi tạo model trước.")
        
        self._minimum_word_length: int = 0
        self._raw_text: str = ""
        self._marks: List[str] = []
        self._stop_words: Optional[Vietnamese] = None
        self._vncorenlp_model: Any = vncorenlp_model
    
    def set_minimum_word_length(self, word_length: int) -> None:
        self._minimum_word_length = word_length
    
    def set_raw_text(self, raw_text: str) -> None:
        self._raw_text = raw_text
    
    def set_stop_words(self, stop_words: Vietnamese) -> None:
        self._stop_words = stop_words
    
    def get_marks(self) -> List[str]:
        return self._marks
    
    def parse(self) -> Text:
        matrix: Dict[int, Dict[int, str]] = {}
        sentences = self._get_sentences()
        
        for sentence_idx, sentence in enumerate(sentences):
            matrix[sentence_idx] = self._get_words(sentence)
        
        text = Text()
        text.set_sentences({i: s for i, s in enumerate(sentences)})
        text.set_word_matrix(matrix)
        text.set_marks(self._marks)
        
        return text
    
    def _get_sentences(self) -> List[str]:
        # Tách văn bản thành các câu
        # Sử dụng lookbehind regex để tách sau dấu câu + khoảng trắng
        sentences = re.split(r'(?<=[.!?])\s+', self._raw_text.strip())
        
        cleaned = []
        for sentence in sentences:
            if sentence:
                cleaned_sentence = self._clean_sentence(sentence)
                if cleaned_sentence:
                    cleaned.append(cleaned_sentence)
        
        return cleaned
    
    def _get_words(self, sub_text: str) -> Dict[int, str]:
        """Trích xuất từ từ câu bằng VnCoreNLP word_segment.
        
        VnCoreNLP nhận diện từ ghép: "Hà_Nội", "trí_tuệ_nhân_tạo", ...
        """
        try:
            segments = self._vncorenlp_model.word_segment(sub_text)
            # word_segment trả về list[str], mỗi phần tử là 1 câu đã tách
            raw_words = []
            for seg in segments:
                raw_words.extend(seg.split())
        except Exception as e:
            # Nếu VnCoreNLP lỗi, raise exception thay vì fallback
            raise RuntimeError(f"VnCoreNLP word_segment failed: {e}")
        
        cleaned_words = [self._clean_word(w) for w in raw_words if w.strip()]
        cleaned_words = [w for w in cleaned_words if w]
        
        return self._filter_words(cleaned_words)
    
    def _filter_words(self, cleaned_words: List[str]) -> Dict[int, str]:
        """Lọc từ theo stop words và độ dài tối thiểu."""
        if self._stop_words:
            filtered_words = [
                word for word in cleaned_words
                if not all(c in string.punctuation for c in word)
                and len(word) > self._minimum_word_length
                and not self._stop_words.exist(word)
            ]
        else:
            filtered_words = [
                word for word in cleaned_words
                if not all(c in string.punctuation for c in word)
                and len(word) > self._minimum_word_length
            ]
        
        return {i: word for i, word in enumerate(filtered_words)}
    
    def _clean_sentence(self, sentence: str) -> str:
        trimmed = sentence.strip()
        if len(trimmed) == 1 and trimmed in string.punctuation:
            self._marks.append(trimmed)
            return ""
        return trimmed
    
    def _clean_word(self, word: str) -> str:
        return word.strip().lower()
