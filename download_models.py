"""
Script tải tất cả models về pretrained-models/
Chạy sau khi clone repo:

    cd combined_pipeline
    source venv/bin/activate
    python download_models.py
"""

import os
import urllib.request
import tarfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "pretrained-models")
VNCORENLP_DIR = os.path.join(MODELS_DIR, "vncorenlp")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VNCORENLP_DIR, exist_ok=True)

# Models cần tải
MODELS = {
    "phobert.pt": "https://huggingface.co/dangvansansan/vietnamese-document-processing/resolve/main/phobert.pt",
    "ner-vietnamese-electra-base.pt": "https://huggingface.co/dangvansansan/vietnamese-document-processing/resolve/main/ner-vietnamese-electra-base.pt",
}

VNCORENLP_URL = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar"

def download_file(filename, url):
    filepath = os.path.join(MODELS_DIR, filename)
    if os.path.exists(filepath):
        print(f"  ✅ {filename} đã tồn tại, bỏ qua...")
        return True

    print(f"  ⏳ Đang tải {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"  ✅ {filename} đã tải xuống!")
        return True
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return False

def download_vncorenlp():
    jar_path = os.path.join(VNCORENLP_DIR, "VnCoreNLP-1.2.jar")
    if os.path.exists(jar_path):
        print(f"  ✅ VnCoreNLP-1.2.jar đã tồn tại, bỏ qua...")
        return True

    print(f"  ⏳ Đang tải VnCoreNLP...")
    try:
        urllib.request.urlretrieve(VNCORENLP_URL, jar_path)
        print(f"  ✅ VnCoreNLP-1.2.jar đã tải xuống!")
        return True
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return False

def download_vncorenlp_models():
    """Tải word models cho VnCoreNLP"""
    word_model_url = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab"
    word_model_path = os.path.join(VNCORENLP_DIR, "models", "wordsegmenter", "vi-vocab")

    os.makedirs(os.path.dirname(word_model_path), exist_ok=True)

    if os.path.exists(word_model_path):
        print(f"  ✅ vi-vocab đã tồn tại, bỏ qua...")
        return True

    print(f"  ⏳ Đang tải word models...")
    try:
        urllib.request.urlretrieve(word_model_url, word_model_path)
        print(f"  ✅ Word models đã tải xuống!")
        return True
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TẢI TẤT CẢ MODELS VỀ pretrained-models/")
    print("=" * 60)

    print("\n[1/3] Tải PhoBERT...")
    for filename, url in MODELS.items():
        download_file(filename, url)

    print("\n[2/3] Tải VnCoreNLP...")
    download_vncorenlp()

    print("\n[3/3] Tải VnCoreNLP word models...")
    download_vncorenlp_models()

    print("\n" + "=" * 60)
    print("Hoàn tất! Tất cả models đã được lưu tại:")
    print(f"  → {MODELS_DIR}")
    print("=" * 60)
