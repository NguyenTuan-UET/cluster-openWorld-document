# Class tạo tóm tắt văn bản bằng cách chọn các câu quan trọng nhất

from typing import Dict, List
from .graph import Graph
from .text import Text


class Summarize:
    
    # Hằng số cho kiểu tóm tắt
    GET_ALL_IMPORTANT = 0
    GET_FIRST_IMPORTANT_AND_FOLLOWINGS = 1
    
    def __init__(self):
        self._sentence_weight: Dict[int, int] = {}
    
    def get_summarize(
        self,
        scores: Dict[str, float],
        graph: Graph,
        text: Text,
        key_word_limit: int,
        sentence_limit: int,
        summarize_type: int
    ) -> List[str]:
        # Lấy các câu tóm tắt dựa trên từ khóa quan trọng
        graph_data = graph.get_graph()
        sentences = text.get_sentences()
        marks = text.get_marks()
        
        self._find_and_weight_sentences(scores, graph_data, key_word_limit)
        
        if summarize_type == Summarize.GET_ALL_IMPORTANT:
            return self._get_all_important(sentences, marks, sentence_limit)
        elif summarize_type == Summarize.GET_FIRST_IMPORTANT_AND_FOLLOWINGS:
            return self._get_first_important_and_followings(
                sentences, marks, sentence_limit
            )
        
        return []
    
    def _find_and_weight_sentences(
        self,
        scores: Dict[str, float],
        graph_data: Dict[str, Dict[int, Dict[int, List[int]]]],
        key_word_limit: int
    ) -> None:
        # Tìm và đánh trọng số các câu chứa từ khóa quan trọng
        i = 0
        
        for word, score in scores.items():
            if i >= key_word_limit:
                break
            
            i += 1
            
            if word in graph_data:
                word_map = graph_data[word]
                
                for sentence_idx in word_map.keys():
                    self._update_sentence_weight(sentence_idx)
        
        self._sentence_weight = dict(
            sorted(self._sentence_weight.items(), key=lambda x: x[1], reverse=True)
        )
    
    def _get_all_important(
        self,
        sentences: Dict[int, str],
        marks: List[str],
        sentence_limit: int
    ) -> List[str]:
        # Lấy các câu quan trọng nhất (giữ thứ tự gốc)
        summary: Dict[int, str] = {}
        i = 0
        
        for sentence_idx, weight in self._sentence_weight.items():
            if i >= sentence_limit:
                break
            
            i += 1
            summary[sentence_idx] = (
                sentences[sentence_idx] + self._get_mark(marks, sentence_idx)
            )
        
        sorted_summary = dict(sorted(summary.items()))
        
        return list(sorted_summary.values())
    
    def _get_first_important_and_followings(
        self,
        sentences: Dict[int, str],
        marks: List[str],
        sentence_limit: int
    ) -> List[str]:
        # Lấy câu quan trọng nhất và các câu kế tiếp
        summary: Dict[int, str] = {}
        start_idx = 0
        
        for sentence_idx, weight in self._sentence_weight.items():
            summary[sentence_idx] = (
                sentences[sentence_idx] + self._get_mark(marks, sentence_idx)
            )
            start_idx = sentence_idx
            break
        
        i = 0
        for sentence_idx, sentence in sentences.items():
            if sentence_idx <= start_idx:
                continue
            elif i >= sentence_limit - 1:
                break
            
            i += 1
            summary[sentence_idx] = (
                sentences[sentence_idx] + self._get_mark(marks, sentence_idx)
            )
        
        return list(summary.values())
    
    def _update_sentence_weight(self, sentence_idx: int) -> None:
        # Cập nhật trọng số của câu
        if sentence_idx in self._sentence_weight:
            self._sentence_weight[sentence_idx] += 1
        else:
            self._sentence_weight[sentence_idx] = 1
    
    def _get_mark(self, marks: List[str], idx: int) -> str:
        # Lấy dấu câu cho câu
        return marks[idx] if idx < len(marks) else ''
