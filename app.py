"""
app.py — Ứng dụng chính PDF RAG Assistant.

Giao diện Streamlit với:
- Premium UI design (custom CSS, gradients, animations)
- Multi-PDF upload & processing
- Streaming response (hiển thị từng token)
- Source attribution (hiển thị nguồn tham khảo)
- Conversation memory
"""

import streamlit as st

from config import APP_TITLE, APP_ICON, APP_DESCRIPTION, validate_config
from models.embedding_model import load_embeddings
from models.llm_model import load_llm
from processing.pdf_processor import process_pdfs
from rag.rag_chain import update_chat_history
from utils.session_state import init_session_state
from utils.logger import logger


# ══════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════
# Premium CSS
# ══════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Import Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 900px;
    }

    /* ── Header Hero ── */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.3);
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-title {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #8b5cf6, #6366f1);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        position: relative;
        z-index: 1;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #cbd5e1;
    }

    /* ── File Info Cards ── */
    .file-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }

    .file-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(139, 92, 246, 0.3);
    }

    .file-name {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
    }

    .file-meta {
        color: #94a3b8;
        font-size: 0.72rem;
    }

    /* ── Status Badges ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    .status-ready {
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .status-waiting {
        background: rgba(251, 191, 36, 0.1);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.2);
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
        animation: fadeInUp 0.3s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* ── Source Expander ── */
    .source-expander {
        background: rgba(99, 102, 241, 0.05);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 10px;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }

    .source-item {
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #6366f1;
        padding: 0.5rem 0.75rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.8rem;
    }

    .source-item-title {
        color: #a5b4fc;
        font-weight: 600;
        font-size: 0.75rem;
        margin-bottom: 0.2rem;
    }

    .source-item-preview {
        color: #94a3b8;
        font-size: 0.73rem;
        line-height: 1.4;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: none;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* ── Stats Row ── */
    .stats-row {
        display: flex;
        gap: 0.75rem;
        margin: 0.75rem 0;
    }

    .stat-item {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.6rem 0.75rem;
        text-align: center;
    }

    .stat-value {
        color: #8b5cf6;
        font-size: 1.25rem;
        font-weight: 700;
    }

    .stat-label {
        color: #94a3b8;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
        margin: 1rem 0;
        border: none;
    }

    /* ── Empty State ── */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        color: #64748b;
    }

    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }

    .empty-state-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }

    .empty-state-desc {
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Initialize
# ══════════════════════════════════════════════

init_session_state()

# Validate config
config_errors = validate_config()
if config_errors:
    for err in config_errors:
        st.error(err)
    st.stop()


# ══════════════════════════════════════════════
# Load Models (cached — chỉ chạy 1 lần)
# ══════════════════════════════════════════════

if not st.session_state.models_loaded:
    with st.spinner("🔄 Đang tải models (Embedding + LLM)..."):
        try:
            st.session_state.embeddings = load_embeddings()
            st.session_state.llm = load_llm()
            st.session_state.models_loaded = True
            logger.info("✅ All models loaded")
        except Exception as e:
            st.error(f"❌ Không thể tải models: {e}")
            logger.error(f"Failed to load models: {e}", exc_info=True)
            st.stop()


# ══════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📚 PDF RAG Assistant")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Model Status ──
    st.markdown('<div class="status-badge status-ready">✅ Models sẵn sàng</div>', unsafe_allow_html=True)
    st.markdown("")

    # ── File Upload ──
    st.markdown("#### 📄 Tải lên tài liệu")
    uploaded_files = st.file_uploader(
        "Chọn file PDF",
        type="pdf",
        accept_multiple_files=True,
        help="Hỗ trợ upload nhiều file cùng lúc. Tối đa 200MB/file.",
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"📎 Đã chọn **{len(uploaded_files)}** file(s)")

        if st.button("🚀 Xử lý PDF", use_container_width=True, type="primary"):
            with st.container():
                rag_chain, files_info, total_chunks = process_pdfs(uploaded_files)

                if rag_chain:
                    st.session_state.rag_chain = rag_chain
                    st.session_state.uploaded_files_info.extend(files_info)
                    st.session_state.total_chunks = sum(
                        f["chunks"] for f in st.session_state.uploaded_files_info
                    )
                    st.balloons()

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Processed Files Info ──
    if st.session_state.uploaded_files_info:
        st.markdown("#### 📂 Tài liệu đã xử lý")

        # Stats
        total_files = len(st.session_state.uploaded_files_info)
        total_pages = sum(f["pages"] for f in st.session_state.uploaded_files_info)
        total_chunks = st.session_state.total_chunks

        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-value">{total_files}</div>
                <div class="stat-label">Files</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{total_pages}</div>
                <div class="stat-label">Pages</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{total_chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # File cards
        for file_info in st.session_state.uploaded_files_info:
            st.markdown(f"""
            <div class="file-card">
                <div class="file-name">📄 {file_info['name']}</div>
                <div class="file-meta">{file_info['pages']} trang · {file_info['chunks']} chunks · {file_info['size_kb']} KB</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── Actions ──
    st.markdown("#### ⚡ Thao tác")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Xóa chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            if "_last_sources" in st.session_state:
                del st.session_state["_last_sources"]
            st.rerun()

    with col2:
        if st.button("🔄 Reset all", use_container_width=True):
            # Reset everything except models
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.rag_chain = None
            st.session_state.retriever = None
            st.session_state.uploaded_files_info = []
            st.session_state.total_chunks = 0
            if "_last_sources" in st.session_state:
                del st.session_state["_last_sources"]
            st.rerun()


# ══════════════════════════════════════════════
# Helper: Render Sources
# ══════════════════════════════════════════════

def _render_sources(sources):
    """Render sources trong lịch sử chat (HTML-based)."""
    if not sources:
        return

    sources_html = ""
    for s in sources:
        sources_html += f"""
        <div class="source-item">
            <div class="source-item-title">📄 {s['file']} — Trang {s['page']}</div>
            <div class="source-item-preview">{s['preview']}</div>
        </div>
        """

    st.markdown(f"""
    <details class="source-expander">
        <summary style="cursor: pointer; color: #a5b4fc; font-size: 0.82rem; font-weight: 500;">
            📋 Nguồn tham khảo ({len(sources)} đoạn)
        </summary>
        {sources_html}
    </details>
    """, unsafe_allow_html=True)


def _render_sources_streamlit(sources):
    """Render sources cho message mới (Streamlit native)."""
    if not sources:
        return

    with st.expander(f"📋 Nguồn tham khảo ({len(sources)} đoạn)", expanded=False):
        for i, s in enumerate(sources, 1):
            st.markdown(
                f"**{i}. 📄 {s['file']}** — Trang {s['page']}\n\n"
                f"_{s['preview']}_"
            )
            if i < len(sources):
                st.divider()


# ══════════════════════════════════════════════
# Main Content — Hero Header
# ══════════════════════════════════════════════

st.markdown(f"""
<div class="hero-container">
    <div class="hero-badge">🤖 AI-Powered RAG</div>
    <h1 class="hero-title">{APP_TITLE}</h1>
    <p class="hero-subtitle">{APP_DESCRIPTION}</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Chat Interface
# ══════════════════════════════════════════════

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

        # Hiển thị sources nếu có (cho assistant messages)
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            _render_sources(message["sources"])

# Empty state khi chưa có tài liệu
if not st.session_state.rag_chain and not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📄</div>
        <div class="empty-state-title">Chưa có tài liệu nào</div>
        <div class="empty-state-desc">
            Upload file PDF ở thanh bên trái và nhấn "Xử lý PDF"<br>
            để bắt đầu trò chuyện với AI về nội dung tài liệu.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Chat input — luôn hiện nhưng thông báo nếu chưa có PDF
prompt = st.chat_input(
    "Nhập câu hỏi về nội dung tài liệu..."
    if st.session_state.rag_chain
    else "⚠️ Vui lòng upload PDF trước khi hỏi..."
)

if prompt:
    if not st.session_state.rag_chain:
        st.warning("⚠️ Vui lòng upload và xử lý PDF trước khi đặt câu hỏi!")
    else:
        # ── User message ──
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # ── AI Response (Streaming) ──
        with st.chat_message("assistant", avatar="🤖"):
            try:
                # Stream response từng token
                full_response = st.write_stream(
                    st.session_state.rag_chain.stream(prompt)
                )

                # Lấy sources từ session state (được lưu bởi rag_chain)
                sources = st.session_state.get("_last_sources", [])

                # Hiển thị sources
                if sources:
                    _render_sources_streamlit(sources)

                # Lưu message + sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources,
                })

                # Cập nhật conversation memory
                update_chat_history(prompt, full_response)

                logger.info(f"✅ Answered question: '{prompt[:50]}...' with {len(sources)} sources")

            except Exception as e:
                error_msg = f"❌ Có lỗi xảy ra: {str(e)}"
                st.error(error_msg)
                logger.error(f"Chat error: {e}", exc_info=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                })
