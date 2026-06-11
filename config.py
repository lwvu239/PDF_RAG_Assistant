"""
config.py — Cấu hình tập trung cho toàn bộ project PDF RAG Assistant.

Tất cả hằng số, tham số model, và cài đặt RAG pipeline được quản lý tại đây.
Hỗ trợ override thông qua biến môi trường trong file .env
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# 🔑 API Keys
# ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ──────────────────────────────────────────────
# 🤖 LLM Configuration (Gemini)
# ──────────────────────────────────────────────
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "2048"))

# ──────────────────────────────────────────────
# 📐 Embedding Model Configuration
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")  # "auto", "cuda", "cpu"
EMBEDDING_NORMALIZE = True

# ──────────────────────────────────────────────
# ✂️ Text Splitting (Chunking)
# ──────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]

# ──────────────────────────────────────────────
# 🔍 Retriever Configuration
# ──────────────────────────────────────────────
RETRIEVER_SEARCH_TYPE = "mmr"         # "similarity" | "mmr" (Maximum Marginal Relevance)
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
RETRIEVER_FETCH_K = 20                # Số docs fetch trước khi MMR filter
RETRIEVER_LAMBDA_MULT = 0.7           # 0=max diversity, 1=max relevance

# ──────────────────────────────────────────────
# 💬 Conversation Memory
# ──────────────────────────────────────────────
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "5"))

# ──────────────────────────────────────────────
# 💾 Cache & Storage
# ──────────────────────────────────────────────
FAISS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "faiss_cache")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# ──────────────────────────────────────────────
# 🎨 UI Configuration
# ──────────────────────────────────────────────
APP_TITLE = "PDF RAG Assistant"
APP_ICON = "📚"
APP_DESCRIPTION = "Trò chuyện thông minh với tài liệu PDF của bạn"
MAX_UPLOAD_SIZE_MB = 200


def validate_config():
    """Kiểm tra các cấu hình bắt buộc khi khởi động."""
    errors = []
    if not GOOGLE_API_KEY:
        errors.append("❌ GOOGLE_API_KEY chưa được thiết lập trong file .env")
    if CHUNK_SIZE < 200:
        errors.append(f"⚠️ CHUNK_SIZE={CHUNK_SIZE} quá nhỏ, khuyến nghị >= 500")
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(f"⚠️ CHUNK_OVERLAP ({CHUNK_OVERLAP}) phải nhỏ hơn CHUNK_SIZE ({CHUNK_SIZE})")
    return errors
