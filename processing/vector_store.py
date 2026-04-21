import os
import hashlib
import streamlit as st
from langchain_community.vectorstores import FAISS

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_cache")


def get_file_hash(docs):
    """Tạo hash từ nội dung docs để kiểm tra cache."""
    content = "".join(doc.page_content for doc in docs)
    return hashlib.md5(content.encode()).hexdigest()


def create_vector_store(docs, progress_callback=None):
    """
    Tạo vector store với cache support.
    - Nếu đã xử lý file này trước đó → load từ cache (< 1 giây)
    - Nếu file mới → embed + lưu cache cho lần sau
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    file_hash = get_file_hash(docs)
    cache_path = os.path.join(CACHE_DIR, file_hash)

    # Kiểm tra cache
    if os.path.exists(cache_path):
        if progress_callback:
            progress_callback(0.8, "📂 Đang load từ cache...")
        vector_db = FAISS.load_local(
            cache_path,
            st.session_state.embeddings,
            allow_dangerous_deserialization=True
        )
        if progress_callback:
            progress_callback(1.0, "✅ Đã load từ cache!")
        return vector_db.as_retriever()

    # Không có cache → embed và lưu
    if progress_callback:
        progress_callback(0.3, f"🔢 Đang tạo embeddings cho {len(docs)} chunks...")

    # Embed tất cả texts trước (batch) rồi tạo FAISS từ texts + embeddings
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    # Batch embed tất cả cùng lúc
    all_embeddings = st.session_state.embeddings.embed_documents(texts)

    if progress_callback:
        progress_callback(0.7, "💾 Đang xây dựng FAISS index...")

    # Tạo FAISS từ embeddings đã tính sẵn (không cần embed lại)
    vector_db = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, all_embeddings)),
        embedding=st.session_state.embeddings,
        metadatas=metadatas
    )

    # Lưu cache cho lần sau
    if progress_callback:
        progress_callback(0.9, "💾 Đang lưu cache...")
    vector_db.save_local(cache_path)

    if progress_callback:
        progress_callback(1.0, "✅ Hoàn tất!")

    return vector_db.as_retriever()