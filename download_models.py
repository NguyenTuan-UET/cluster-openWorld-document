import os
import sys

# Đảm bảo thư mục hiện tại là thư mục chứa script
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(CUR_DIR, exist_ok=True)

def install_and_import(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        import subprocess
        print(f" Đang cài đặt thư mục phụ thuộc: {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Cài đặt các thư viện cần thiết để tải model
install_and_import("torch")
install_and_import("transformers")
install_and_import("sentence-transformers", "sentence_transformers")
install_and_import("py_vncorenlp", "py_vncorenlp")

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from sentence_transformers import SentenceTransformer
import py_vncorenlp

def download_vncorenlp():
    print("\n [1/5] Đang tải VnCoreNLP segmenter...")
    vncorenlp_dir = os.path.join(CUR_DIR, "vncorenlp")
    os.makedirs(vncorenlp_dir, exist_ok=True)
    try:
        py_vncorenlp.download_model(save_dir=vncorenlp_dir)
        print("Đã tải VnCoreNLP thành công và lưu vào:", vncorenlp_dir)
    except Exception as e:
        print(f" Lỗi tải VnCoreNLP: {e}")

def download_ner_model():
    print("\n [2/5] Đang tải NER model (NlpHUST/ner-vietnamese-electra-base)...")
    ner_pt_path = os.path.join(CUR_DIR, "ner-vietnamese-electra-base.pt")
    ner_tokenizer_dir = os.path.join(CUR_DIR, "ner-tokenizer")
    
    try:
        # Tải model từ Hugging Face
        print("   -> Đang tải model từ Hugging Face...")
        model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
        
        # Lưu dưới dạng checkpoint .pt giống dự án yêu cầu
        print("   -> Đang lưu model vào file .pt...")
        torch.save(model, ner_pt_path)
        print(f" Đã lưu NER model thành công vào: {ner_pt_path}")
        
        # Tải và lưu Tokenizer
        print("   -> Đang tải & lưu Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
        tokenizer.save_pretrained(ner_tokenizer_dir)
        print(f" Đã lưu NER Tokenizer thành công vào: {ner_tokenizer_dir}")
    except Exception as e:
        print(f" Lỗi tải NER model: {e}")

def download_symmetric_embedding():
    print("\n [3/5] Đang tải Symmetric Embedding model (dangvantuan/vietnamese-embedding)...")
    sym_dir = os.path.join(CUR_DIR, "symmetric_emb")
    try:
        model = SentenceTransformer("dangvantuan/vietnamese-embedding")
        model.save(sym_dir)
        print(" Đã tải và lưu Symmetric Embedding thành công vào:", sym_dir)
    except Exception as e:
        print(f" Lỗi tải Symmetric Embedding: {e}")

def download_asymmetric_embedding():
    print("\n [4/5] Đang tải Asymmetric Embedding model (AITeamVN/Vietnamese_Embedding)...")
    asym_dir = os.path.join(CUR_DIR, "asymmetric_emb")
    try:
        model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
        model.save(asym_dir)
        print("Đã tải và lưu Asymmetric Embedding thành công vào:", asym_dir)
    except Exception as e:
        print(f" Lỗi tải Asymmetric Embedding: {e}")

def download_phobert_pt():
    print("\n [5/5] Đang tải PhoBERT model (vinai/phobert-base-v2) làm file .pt dự phòng...")
    phobert_pt_path = os.path.join(CUR_DIR, "phobert.pt")
    try:
        # Tải PhoBERT base v2
        model = AutoModel.from_pretrained("vinai/phobert-base-v2")
        torch.save(model, phobert_pt_path)
        print(f" Đã tải và lưu PhoBERT .pt thành công vào: {phobert_pt_path}")
    except Exception as e:
        print(f" Lỗi tải PhoBERT model: {e}")

if __name__ == "__main__":
    print(" Bắt đầu quá trình tải các pretrained models...")
    download_vncorenlp()
    download_ner_model()
    download_symmetric_embedding()
    download_asymmetric_embedding()
    download_phobert_pt()
    print("\n Hoàn thành tải toàn bộ các pretrained models thành công!")
