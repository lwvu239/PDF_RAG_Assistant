"""
embedding_model.py — Cấu hình và tải mô hình Embedding.

Sử dụng sentence-transformers multilingual cho hỗ trợ tiếng Việt tốt.
Tự động detect CUDA/CPU device.
"""

import torch
import streamlit as st

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE, EMBEDDING_NORMALIZE
from utils.logger import logger


def _detect_device() -> str:
    """Tự động phát hiện device tốt nhất (CUDA > CPU)."""
    if EMBEDDING_DEVICE != "auto":
        return EMBEDDING_DEVICE
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🚀 CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("💻 Using CPU for embeddings")
    return device


@st.cache_resource
def load_embeddings():
    """
    Tải mô hình Embedding với caching (chỉ load 1 lần).

    Returns:
        HuggingFaceEmbeddings instance
    """
    try:
        device = _detect_device()
        logger.info(f"📐 Loading embedding model: {EMBEDDING_MODEL_NAME} on {device}")

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE}
        )

        logger.info("✅ Embedding model loaded successfully")
        return embeddings

    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}", exc_info=True)
        raise RuntimeError(f"Không thể tải mô hình embedding: {e}")