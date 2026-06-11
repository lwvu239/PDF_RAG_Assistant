"""
vector_store.py — Tạo và quản lý FAISS Vector Database.

Hỗ trợ:
- Cache thông minh (hash-based) để tránh re-embed
- MMR search cho kết quả đa dạng
- Metadata enrichment (page, filename)
- Merge nhiều PDF vào 1 vector store
"""

import os
import hashlib
import streamlit as st
from langchain_community.vectorstores import FAISS

from config import (
    FAISS_CACHE_DIR,
    RETRIEVER_SEARCH_TYPE,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA_MULT,
)
from utils.logger import logger


def get_docs_hash(docs) -> str:
    """Tạo hash từ nội dung docs để kiểm tra cache."""
    content = "".join(doc.page_content for doc in docs)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_vector_store(docs, progress_callback=None):
    """
    Tạo hoặc load FAISS vector store từ cache.

    Args:
        docs: List[Document] đã được split thành chunks
        progress_callback: Callable(pct, msg) cho progress bar

    Returns:
        FAISS retriever với MMR search
    """
    os.makedirs(FAISS_CACHE_DIR, exist_ok=True)
    file_hash = get_docs_hash(docs)
    cache_path = os.path.join(FAISS_CACHE_DIR, file_hash)

    # ── Check Cache ──
    if os.path.exists(cache_path):
        try:
            if progress_callback:
                progress_callback(0.8, "📂 Đang load từ cache...")

            logger.info(f"📂 Loading FAISS from cache: {file_hash}")
            vector_db = FAISS.load_local(
                cache_path,
                st.session_state.embeddings,
                allow_dangerous_deserialization=True
            )

            if progress_callback:
                progress_callback(1.0, "✅ Đã load từ cache!")

            logger.info(f"✅ Loaded {vector_db.index.ntotal} vectors from cache")
            return _create_retriever(vector_db)

        except Exception as e:
            logger.warning(f"⚠️ Cache load failed, re-embedding: {e}")
            # Cache hỏng → tiếp tục embed mới

    # ── Embed + Build Index ──
    if progress_callback:
        progress_callback(0.3, f"🔢 Đang tạo embeddings cho {len(docs)} chunks...")

    logger.info(f"🔢 Embedding {len(docs)} chunks...")

    try:
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        # Batch embed tất cả cùng lúc
        all_embeddings = st.session_state.embeddings.embed_documents(texts)

        if progress_callback:
            progress_callback(0.7, "💾 Đang xây dựng FAISS index...")

        # Tạo FAISS từ embeddings đã tính sẵn
        vector_db = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, all_embeddings)),
            embedding=st.session_state.embeddings,
            metadatas=metadatas
        )

        logger.info(f"✅ Built FAISS index with {vector_db.index.ntotal} vectors")

        # Lưu cache
        if progress_callback:
            progress_callback(0.9, "💾 Đang lưu cache...")

        vector_db.save_local(cache_path)
        logger.info(f"💾 Saved cache: {file_hash}")

        if progress_callback:
            progress_callback(1.0, "✅ Hoàn tất!")

        return _create_retriever(vector_db)

    except Exception as e:
        logger.error(f"❌ Embedding failed: {e}", exc_info=True)
        raise RuntimeError(f"Không thể tạo vector store: {e}")


def merge_vector_stores(existing_retriever, new_docs, progress_callback=None):
    """
    Merge documents mới vào vector store đã có.

    Args:
        existing_retriever: Retriever hiện tại (có thể None)
        new_docs: List[Document] mới cần thêm
        progress_callback: Callable(pct, msg)

    Returns:
        FAISS retriever chứa cả data cũ + mới
    """
    if existing_retriever is None:
        return create_vector_store(new_docs, progress_callback)

    try:
        if progress_callback:
            progress_callback(0.3, f"🔢 Đang tạo embeddings cho {len(new_docs)} chunks mới...")

        logger.info(f"🔗 Merging {len(new_docs)} new chunks into existing store")

        texts = [doc.page_content for doc in new_docs]
        metadatas = [doc.metadata for doc in new_docs]
        new_embeddings = st.session_state.embeddings.embed_documents(texts)

        if progress_callback:
            progress_callback(0.6, "🔗 Đang merge vào vector store...")

        # Tạo FAISS mới từ docs mới
        new_vector_db = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, new_embeddings)),
            embedding=st.session_state.embeddings,
            metadatas=metadatas
        )

        # Merge vào store hiện tại
        existing_db = existing_retriever.vectorstore
        existing_db.merge_from(new_vector_db)

        if progress_callback:
            progress_callback(1.0, "✅ Merge hoàn tất!")

        logger.info(f"✅ Merged. Total vectors: {existing_db.index.ntotal}")
        return _create_retriever(existing_db)

    except Exception as e:
        logger.error(f"❌ Merge failed: {e}", exc_info=True)
        raise RuntimeError(f"Không thể merge vector store: {e}")


def _create_retriever(vector_db):
    """Tạo retriever với cấu hình từ config."""
    search_kwargs = {"k": RETRIEVER_K}

    if RETRIEVER_SEARCH_TYPE == "mmr":
        search_kwargs["fetch_k"] = RETRIEVER_FETCH_K
        search_kwargs["lambda_mult"] = RETRIEVER_LAMBDA_MULT

    return vector_db.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs=search_kwargs
    )