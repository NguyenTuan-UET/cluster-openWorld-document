from sentence_transformers import SentenceTransformer
import os

SAVE_DIR = "../../pretrained-models/symmetric_emb"

print("Đang tải dangvantuan/vietnamese-embedding...")

model = SentenceTransformer(
    "dangvantuan/vietnamese-embedding"
)
model.save(SAVE_DIR)

print("Đã tải xong dangvantuan vào:", os.path.abspath(SAVE_DIR))