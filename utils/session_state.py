"""
session_state.py — Quản lý trạng thái phiên làm việc Streamlit.

Khởi tạo tất cả session state variables cần thiết cho ứng dụng.
"""

import streamlit as st


def init_session_state():
    """Khởi tạo tất cả session state keys với giá trị mặc định."""

    defaults = {
        # ── Models ──
        "models_loaded": False,
        "embeddings": None,
        "llm": None,

        # ── RAG Pipeline ──
        "rag_chain": None,
        "retriever": None,

        # ── Chat ──
        "messages": [],
        "chat_history": [],  # Lưu history cho conversation memory

        # ── File Management ──
        "uploaded_files_info": [],  # Danh sách file đã xử lý [{name, pages, chunks}]
        "total_chunks": 0,

        # ── Processing Status ──
        "processing": False,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value