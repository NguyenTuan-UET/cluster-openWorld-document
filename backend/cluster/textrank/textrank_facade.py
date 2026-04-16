# Class Facade cho thuật toán tóm tắt văn bản TextRank
# SỬ DỤNG VnCoreNLP để tách từ tiếng Việt chính xác

import math
from typing import List, Dict, Any, Optional
from tools.parser import Parser
from tools.graph import Graph
from tools.score import Score
from tools.summarize import Summarize
from tools.text import Text
from stopwords.vietnamese import Vietnamese


class TextRankFacade:
    
    # Hằng số cho kiểu tóm tắt
    GET_ALL_IMPORTANT = Summarize.GET_ALL_IMPORTANT
    GET_FIRST_IMPORTANT_AND_FOLLOWINGS = Summarize.GET_FIRST_IMPORTANT_AND_FOLLOWINGS
    
    def __init__(self, vncorenlp_model: Any, stop_words: Vietnamese):
        """Khởi tạo TextRankFacade với VnCoreNLP và stopwords (BẮT BUỘC).
        
        Args:
            vncorenlp_model: VnCoreNLP instance với annotators=['wseg']
            stop_words: Vietnamese stopwords instance
            
        Example:
            from py_vncorenlp import VnCoreNLP
            from stopwords.vietnamese import Vietnamese
            
            vncorenlp = VnCoreNLP(annotators=["wseg"], save_dir="./vncorenlp")
            stopwords = Vietnamese()
            facade = TextRankFacade(vncorenlp, stopwords)
        """
        if vncorenlp_model is None:
            raise ValueError("VnCoreNLP model là bắt buộc. Vui lòng khởi tạo model trước.")
        if stop_words is None:
            raise ValueError("Vietnamese stopwords là bắt buộc.")
        
        self._vncorenlp_model: Any = vncorenlp_model
        self._stop_words: Vietnamese = stop_words
    
    def _create_parser(self, raw_text: str) -> Parser:
        """Tạo parser đã config với VnCoreNLP và stopwords."""
        parser = Parser(self._vncorenlp_model)
        parser.set_minimum_word_length(3)
        parser.set_raw_text(raw_text)
        parser.set_stop_words(self._stop_words)
        return parser
    
    def summarize(self, raw_text: str, max_sentences: Optional[int] = None) -> List[str]:
        """Tóm tắt văn bản.
        
        Args:
            raw_text      : Văn bản cần tóm tắt
            max_sentences : Số câu tối đa muốn lấy. Nếu None thì tự động:
                            - ≤ 5 câu  → tối đa 3 câu
                            - > 5 câu  → ~40% số câu, tối thiểu 5 câu
            
        Returns:
            Danh sách các câu tóm tắt (giữ thứ tự gốc)
        """
        text = self._create_parser(raw_text).parse()
        sentences = text.get_sentences()
        n = len(sentences)

        if max_sentences is not None:
            # Người dùng chỉ định trực tiếp, không vượt quá số câu gốc
            k = max(1, min(int(max_sentences), n))
        else:
            # Tự động theo tỉ lệ
            if n <= 5:
                k = min(n, 3)
            else:
                k = max(5, math.ceil(n * 0.4))
        
        # Số keyword phân tích: scale theo độ dài văn bản
        analyzed_keywords = min(max(5, n), 15)
        
        graph = Graph()
        graph.create_graph(text)
        
        score = Score()
        scores = score.calculate(graph, text)
        
        summarize = Summarize()
        return summarize.get_summarize(
            scores,
            graph,
            text,
            analyzed_keywords,
            k,
            Summarize.GET_ALL_IMPORTANT
        )
    
    def get_only_keywords(self, raw_text: str) -> Dict[str, float]:
        """Trích xuất từ khóa và điểm số từ văn bản."""
        text = self._create_parser(raw_text).parse()
        
        graph = Graph()
        graph.create_graph(text)
        
        score = Score()
        return score.calculate(graph, text)
    
    def get_highlights(self, raw_text: str) -> List[str]:
        """Lấy các câu nổi bật (15–25% số câu, min 2, max 6)."""
        text = self._create_parser(raw_text).parse()
        sentences = text.get_sentences()
        n_sent = len(sentences)

        # ====== HIGHLIGHT POLICY ======
        # 15–25% số câu, min 2, max 6
        maximum_sentences = min(
            max(2, math.ceil(n_sent * 0.2)),
            6
        )

        # Số keyword phân tích: scale theo độ dài
        analyzed_keywords = min(
            max(5, n_sent),
            12
        )

        graph = Graph()
        graph.create_graph(text)

        score = Score()
        scores = score.calculate(graph, text)

        summarize = Summarize()
        return summarize.get_summarize(
            scores,
            graph,
            text,
            analyzed_keywords,
            maximum_sentences,
            Summarize.GET_ALL_IMPORTANT
        )
    
    def summarize_text_compound(self, raw_text: str) -> List[str]:
        """Tìm 3 câu quan trọng nhất từ văn bản (LEGACY METHOD)."""
        text = self._create_parser(raw_text).parse()
        
        graph = Graph()
        graph.create_graph(text)
        
        score = Score()
        scores = score.calculate(graph, text)
        
        summarize = Summarize()
        return summarize.get_summarize(
            scores,
            graph,
            text,
            10,
            3,
            Summarize.GET_ALL_IMPORTANT
        )
    
    def summarize_text_basic(self, raw_text: str) -> List[str]:
        """Tìm câu quan trọng nhất và các câu kế tiếp (LEGACY METHOD)."""
        text = self._create_parser(raw_text).parse()
        
        graph = Graph()
        graph.create_graph(text)
        
        score = Score()
        scores = score.calculate(graph, text)
        
        summarize = Summarize()
        return summarize.get_summarize(
            scores,
            graph,
            text,
            10,
            3,
            Summarize.GET_FIRST_IMPORTANT_AND_FOLLOWINGS
        )
    
    def summarize_text_freely(
        self,
        raw_text: str,
        analyzed_keywords: int,
        expected_sentences: int,
        summarize_type: int
    ) -> List[str]:
        """Tùy chỉnh tóm tắt văn bản theo tham số."""
        text = self._create_parser(raw_text).parse()
        
        graph = Graph()
        graph.create_graph(text)
        
        score = Score()
        scores = score.calculate(graph, text)
        
        summarize = Summarize()
        return summarize.get_summarize(
            scores,
            graph,
            text,
            analyzed_keywords,
            expected_sentences,
            summarize_type
        )
