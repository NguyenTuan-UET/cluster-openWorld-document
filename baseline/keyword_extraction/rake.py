import string
from rake_nltk import Rake
from underthesea import word_tokenize  # Thêm thư viện tách từ tiếng Việt

# 1. Đọc danh sách stopwords định dạng dash từ file
with open('vietnamese-stopwords-dash.txt', 'r', encoding='utf-8') as f:
    stopwords = [line.strip().lower() for line in f if line.strip()]

# 2. Định nghĩa dấu câu theo đúng định dạng SET
punctuations = set(string.punctuation)

# 3. Khởi tạo RAKE
r = Rake(stopwords=stopwords, punctuations=punctuations)

# 4. Văn bản đầu vào
raw_text = """
Nhắc đến Cổ Loa, người ta nghĩ ngay đến truyền thuyết về An Dương Vương được thần Kim Quy bày cho cách xây thành, về chiếc lẫy nỏ thần làm từ móng chân rùa thần và mối tình bi thương Mỵ Châu – Trọng Thủy. Đằng sau những câu chuyện thiên về tâm linh ấy, thế hệ con cháu còn khám phá được những giá trị khảo cổ to lớn của Cổ Loa.
Khu di tích Cổ Loa cách trung tâm Hà Nội 17km thuộc huyện Đông Anh, Hà Nội, có diện tích bảo tồn gần 500ha được coi là địa chỉ văn hóa đặc biệt của thủ đô và cả nước. Cổ Loa có hàng loạt di chỉ khảo cổ học đã được phát hiện, phản ánh quá trình phát triển liên tục của dân tộc ta từ sơ khai qua các thời kỳ đồ đồng, đồ đá và đồ sắt mà đỉnh cao là văn hóa Đông Sơn, vẫn được coi là nền văn minh sông Hồng thời kỳ tiền sử của dân tộc Việt Nam.
Cổ Loa từng là kinh đô của nhà nước Âu Lạc thời kỳ An Dương Vương (thế kỷ III TCN) và của nước Đại Việt thời Ngô Quyền (thế kỷ X) mà thành Cổ Loa là một di tích minh chứng còn lại cho đến ngày nay.
""".lower()

# TIỀN XỬ LÝ: Tự động thêm dấu gạch dưới (_) cho các từ ghép tiếng Việt
# Ví dụ: "an dương vương" -> "an_dương_vương"
# text = word_tokenize(raw_text, format="text")

# 5. Tiến hành trích xuất cụm từ khóa
r.extract_keywords_from_text(raw_text)
# 6. Lấy toàn bộ kết quả
all_ranked_phrases = r.get_ranked_phrases_with_scores()

# 7. CHỈ LẤY TOP 10 TỪ KHÓA ĐẦU TIÊN
top_10_phrases = all_ranked_phrases[:10]

print(f"--- TOP {len(top_10_phrases)} TỪ KHÓA XUẤT SẮC NHẤT ---")
for score, phrase in top_10_phrases:
    # Không dùng .replace('_', ' ') nữa để GIỮ NGUYÊN dấu gạch dưới khi in ra
    print(f"Điểm: {score:<5.2f} | Từ khóa: {phrase}")