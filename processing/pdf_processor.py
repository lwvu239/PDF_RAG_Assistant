import tempfile
import os
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from processing.vector_store import create_vector_store
from rag.rag_chain import build_rag_chain


def process_pdf(uploaded_file):
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Bước 1: Lưu file tạm
    status_text.text("📥 Đang đọc file PDF...")
    progress_bar.progress(0.05)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Bước 2: Đọc PDF
    status_text.text("📖 Đang trích xuất nội dung PDF...")
    progress_bar.progress(0.1)

    loader = PyMuPDFLoader(tmp_file_path)
    documents = loader.load()

    # Bước 3: Chia chunks
    status_text.text(f"✂️ Đang chia {len(documents)} trang thành chunks...")
    progress_bar.progress(0.15)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    docs = splitter.split_documents(documents)

    status_text.text(f"✂️ Đã tạo {len(docs)} chunks")
    progress_bar.progress(0.2)

    # Bước 4: Tạo Vector Store (với progress callback)
    def progress_callback(pct, msg):
        # Map 0.0-1.0 → 0.2-0.9 trong tổng progress
        mapped = 0.2 + pct * 0.7
        progress_bar.progress(min(mapped, 0.95))
        status_text.text(msg)

    retriever = create_vector_store(docs, progress_callback=progress_callback)

    # Bước 5: Build RAG Chain
    status_text.text("🔗 Đang xây dựng RAG chain...")
    progress_bar.progress(0.95)

    rag_chain = build_rag_chain(retriever)

    # Cleanup
    os.unlink(tmp_file_path)
    progress_bar.progress(1.0)
    status_text.text(f"✅ Hoàn tất! Đã xử lý {len(docs)} chunks")

    return rag_chain, len(docs)