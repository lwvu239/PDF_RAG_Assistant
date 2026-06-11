"""
pdf_processor.py — Logic xử lý và đọc file PDF.

Hỗ trợ:
- Multi-PDF processing (nhiều file cùng lúc)
- Text cleaning (loại bỏ noise)
- Metadata enrichment (filename, page number)
- Progress tracking với callback
"""

import re
import tempfile
import os
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from processing.vector_store import create_vector_store, merge_vector_stores
from rag.rag_chain import build_rag_chain
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS
from utils.logger import logger


def clean_text(text: str) -> str:
    """
    Làm sạch text trích xuất từ PDF.

    - Loại bỏ khoảng trắng thừa
    - Normalize unicode
    - Loại bỏ ký tự đặc biệt vô nghĩa
    """
    # Loại bỏ multiple whitespace/newlines liên tiếp
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Loại bỏ ký tự control (trừ newline, tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text.strip()


def process_single_pdf(uploaded_file) -> tuple:
    """
    Xử lý 1 file PDF: đọc → clean → split → trả về docs + info.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        (docs: List[Document], info: dict) — chunks đã split và thông tin file
    """
    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Đọc PDF
        loader = PyMuPDFLoader(tmp_file_path)
        documents = loader.load()

        logger.info(f"📖 Read '{uploaded_file.name}': {len(documents)} pages")

        # Clean text + enrich metadata
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source_file"] = uploaded_file.name
            # PyMuPDF đã có 'page' trong metadata, đảm bảo nó tồn tại
            if "page" not in doc.metadata:
                doc.metadata["page"] = doc.metadata.get("page_number", 0)

        # Loại bỏ pages rỗng
        documents = [doc for doc in documents if len(doc.page_content.strip()) > 50]

        # Split thành chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=CHUNK_SEPARATORS,
            length_function=len,
        )
        docs = splitter.split_documents(documents)

        # Đảm bảo metadata được kế thừa đúng
        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = i

        info = {
            "name": uploaded_file.name,
            "pages": len(documents),
            "chunks": len(docs),
            "size_kb": round(len(uploaded_file.getvalue()) / 1024, 1),
        }

        logger.info(f"✂️ Split '{uploaded_file.name}': {info['chunks']} chunks from {info['pages']} pages")

        return docs, info

    finally:
        # Cleanup file tạm
        os.unlink(tmp_file_path)


def process_pdfs(uploaded_files):
    """
    Xử lý nhiều file PDF và tạo/merge vector store.

    Args:
        uploaded_files: List[UploadedFile] từ Streamlit

    Returns:
        (rag_chain, files_info: List[dict], total_chunks: int)
    """
    if not uploaded_files:
        return None, [], 0

    # Đảm bảo luôn là list
    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    progress_bar = st.progress(0)
    status_text = st.empty()
    all_docs = []
    files_info = []

    total_files = len(uploaded_files)

    # ── Bước 1: Đọc + split tất cả PDFs ──
    for i, uploaded_file in enumerate(uploaded_files):
        file_progress = (i / total_files) * 0.3
        progress_bar.progress(file_progress)
        status_text.text(f"📖 Đang xử lý file {i + 1}/{total_files}: {uploaded_file.name}...")

        try:
            docs, info = process_single_pdf(uploaded_file)
            all_docs.extend(docs)
            files_info.append(info)
        except Exception as e:
            logger.error(f"❌ Failed to process {uploaded_file.name}: {e}", exc_info=True)
            status_text.error(f"❌ Lỗi xử lý {uploaded_file.name}: {e}")
            continue

    if not all_docs:
        progress_bar.empty()
        status_text.error("❌ Không có nội dung nào được trích xuất từ PDF!")
        return None, [], 0

    total_chunks = len(all_docs)
    status_text.text(f"✂️ Tổng cộng {total_chunks} chunks từ {len(files_info)} file(s)")
    progress_bar.progress(0.3)

    # ── Bước 2: Tạo hoặc merge Vector Store ──
    def progress_callback(pct, msg):
        mapped = 0.3 + pct * 0.6
        progress_bar.progress(min(mapped, 0.95))
        status_text.text(msg)

    existing_retriever = st.session_state.get("retriever", None)

    if existing_retriever is not None:
        retriever = merge_vector_stores(existing_retriever, all_docs, progress_callback)
    else:
        retriever = create_vector_store(all_docs, progress_callback)

    st.session_state.retriever = retriever

    # ── Bước 3: Build RAG Chain ──
    status_text.text("🔗 Đang xây dựng RAG chain...")
    progress_bar.progress(0.95)

    rag_chain = build_rag_chain(retriever)

    # ── Done ──
    progress_bar.progress(1.0)
    status_text.text(f"✅ Hoàn tất! {total_chunks} chunks từ {len(files_info)} file(s)")

    return rag_chain, files_info, total_chunks