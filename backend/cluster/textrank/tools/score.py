# Class tính điểm cho các từ dựa trên kết nối trong đồ thị (thuật toán TextRank)

from typing import Dict, List
from .graph import Graph
from .text import Text


class Score:
    
    def __init__(self):
        self._maximum_value: int = 0
        self._minimum_value: int = 0
    
    def calculate(self, graph: Graph, text: Text) -> Dict[str, float]:
        # Tính điểm cho tất cả các từ và chuẩn hóa về khoảng 0-1
        graph_data = graph.get_graph()
        word_matrix = text.get_word_matrix()
        
        word_connections = self._calculate_connection_numbers(graph_data)
        scores = self._calculate_scores(graph_data, word_matrix, word_connections)
        
        return self._normalize_and_sort_scores(scores)
    
    def _calculate_connection_numbers(
        self,
        graph_data: Dict[str, Dict[int, Dict[int, List[int]]]]
    ) -> Dict[str, int]:
        # Đếm số lượng kết nối cho mỗi từ
        word_connections: Dict[str, int] = {}
        
        for word_key, sentences in graph_data.items():
            connection_count = 0
            
            for sentence_idx, word_instances in sentences.items():
                for connections in word_instances.values():
                    connection_count += len(connections)
            
            word_connections[word_key] = connection_count
        
        return word_connections
    
    def _calculate_scores(
        self,
        graph_data: Dict[str, Dict[int, Dict[int, List[int]]]],
        word_matrix: Dict[int, Dict[int, str]],
        word_connections: Dict[str, int]
    ) -> Dict[str, int]:
        # Tính điểm thô cho các từ dựa trên các từ kết nối
        scores: Dict[str, int] = {}
        
        for word_key, sentences in graph_data.items():
            value = 0
            
            for sentence_idx, word_instances in sentences.items():
                for connections in word_instances.values():
                    for word_idx in connections:
                        if sentence_idx in word_matrix and word_idx in word_matrix[sentence_idx]:
                            word = word_matrix[sentence_idx][word_idx]
                            value += word_connections.get(word, 0)
            
            scores[word_key] = value
            
            if value > self._maximum_value:
                self._maximum_value = value
            
            if value < self._minimum_value or self._minimum_value == 0:
                self._minimum_value = value
        
        return scores
    
    def _normalize_and_sort_scores(self, scores: Dict[str, int]) -> Dict[str, float]:
        # Chuẩn hóa điểm về 0-1 và sắp xếp giảm dần
        normalized_scores: Dict[str, float] = {}
        
        for word, value in scores.items():
            normalized = self._normalize(
                value,
                self._minimum_value,
                self._maximum_value
            )
            normalized_scores[word] = normalized
        
        sorted_scores = dict(
            sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_scores
    
    def _normalize(self, value: int, min_val: int, max_val: int) -> float:
        # Chuẩn hóa giá trị về khoảng 0-1
        divisor = max_val - min_val
        
        if divisor == 0:
            return 0.0
        
        normalized = (value - min_val) / divisor
        return normalized
