# Class tạo đồ thị biểu diễn mối quan hệ giữa các từ trong văn bản

from typing import Dict, List
from .text import Text


class Graph:
    
    def __init__(self):
        self._graph: Dict[str, Dict[int, Dict[int, List[int]]]] = {}
    
    def create_graph(self, text: Text) -> None:
        # Tạo đồ thị từ văn bản đã phân tích (mỗi từ kết nối với từ trước và sau nó)
        word_matrix = text.get_word_matrix()
        
        for sentence_idx, words in word_matrix.items():
            idx_array = list(words.keys())
            
            for idx_key, idx_value in enumerate(idx_array):
                connections = []
                
                if idx_key > 0:
                    connections.append(idx_array[idx_key - 1])
                
                if idx_key < len(idx_array) - 1:
                    connections.append(idx_array[idx_key + 1])
                
                word = words[idx_value]
                
                if word not in self._graph:
                    self._graph[word] = {}
                if sentence_idx not in self._graph[word]:
                    self._graph[word][sentence_idx] = {}
                
                self._graph[word][sentence_idx][idx_value] = connections
    
    def get_graph(self) -> Dict[str, Dict[int, Dict[int, List[int]]]]:
        # Trả về cấu trúc đồ thị: {từ: {ID câu: {ID từ: [các ID từ kết nối]}}}
        return self._graph
