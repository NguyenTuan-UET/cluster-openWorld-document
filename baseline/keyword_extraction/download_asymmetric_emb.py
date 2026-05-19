# download_aiteam_asymmetric.py
from sentence_transformers import SentenceTransformer
import os

SAVE_DIR_ASYMM = "../../pretrained-models/asymmetric_emb"

# Tải model AITeam (dùng tên chính thức)
print("Đang tải AITeamVN/Vietnamese_Embedding...")

model = SentenceTransformer(
    "AITeamVN/Vietnamese_Embedding",   # "AITeamVN/Vietnamese_Embedding_v2"
    trust_remote_code=True
)
model.max_seq_length = 2048
model.save(SAVE_DIR_ASYMM)

print(f"Đã tải xong AITeam vào: {os.path.abspath(SAVE_DIR_ASYMM)}")