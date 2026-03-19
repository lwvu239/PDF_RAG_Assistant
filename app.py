import streamlit as st

from models.embedding_model import load_embeddings
from models.llm_model import load_llm
from processing.pdf_processor import process_pdf
from utils.session_state import init_session_state

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
init_session_state()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- LOAD MODEL ----------------
if not st.session_state.models_loaded:
    with st.spinner("Đang tải models..."):
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.models_loaded = True

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Cài đặt")

    st.success("✅ Models đã sẵn sàng!")

    uploaded_file = st.file_uploader("📄 Upload tài liệu PDF", type="pdf")

    if uploaded_file and st.button("🔄 Xử lý PDF"):
        with st.spinner("Đang xử lý PDF..."):
            st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
            st.success(f"Đã xử lý {num_chunks} chunks")

    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []

# ---------------- MAIN UI ----------------
st.title("PDF RAG Assistant")
st.caption("Trò chuyện với Chatbot để trao đổi nội dung tài liệu PDF")

# ---------------- CHAT HISTORY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- CHAT INPUT ----------------
if st.session_state.rag_chain:
    prompt = st.chat_input("Nhập câu hỏi của bạn...")

    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Đang trả lời..."):
                output = st.session_state.rag_chain.invoke(prompt)

                answer = output.split("Answer:")[1].strip() if "Answer:" in output else output.strip()

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })