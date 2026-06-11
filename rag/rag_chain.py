"""
rag_chain.py — Thiết lập RAG pipeline.

Kết nối Retriever + LLM với:
- Prompt engineering nâng cao (bilingual, source citation)
- Conversation memory (nhớ context hội thoại)
- Source attribution (trả về nguồn tham khảo)
"""

import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from config import MEMORY_WINDOW_SIZE
from utils.logger import logger


# ──────────────────────────────────────────────
# Prompt Template
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable and helpful assistant specialized in answering questions based on PDF documents.

## Instructions:
1. Use ONLY the provided context to answer. Do NOT use external knowledge.
2. If the context doesn't contain enough information, clearly state: "Tôi không tìm thấy thông tin này trong tài liệu."
3. Answer in the SAME LANGUAGE as the user's question (Vietnamese → Vietnamese, English → English).
4. Be concise but thorough. Use bullet points or numbered lists when appropriate.
5. When possible, mention which page(s) or section(s) the information comes from.
6. If the question is ambiguous, ask for clarification.

## Context from documents:
{context}"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def format_docs(docs):
    """
    Format retrieved documents thành context string có kèm source info.

    Returns:
        (formatted_text, source_info_list)
    """
    formatted_parts = []
    sources = []

    for i, doc in enumerate(docs, 1):
        source_file = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "?")

        formatted_parts.append(
            f"[Tài liệu {i} | File: {source_file} | Trang: {page}]\n"
            f"{doc.page_content}"
        )

        sources.append({
            "file": source_file,
            "page": page,
            "preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
        })

    return "\n\n---\n\n".join(formatted_parts), sources


def get_chat_history_messages() -> list:
    """
    Lấy conversation history từ session state,
    giới hạn theo MEMORY_WINDOW_SIZE.

    Returns:
        List of LangChain Message objects
    """
    chat_history = st.session_state.get("chat_history", [])

    # Giới hạn số lượng messages (mỗi turn = 2 messages: human + ai)
    max_messages = MEMORY_WINDOW_SIZE * 2
    recent_history = chat_history[-max_messages:] if chat_history else []

    messages = []
    for entry in recent_history:
        if entry["role"] == "user":
            messages.append(HumanMessage(content=entry["content"]))
        elif entry["role"] == "assistant":
            messages.append(AIMessage(content=entry["content"]))

    return messages


# ──────────────────────────────────────────────
# RAG Chain Builder
# ──────────────────────────────────────────────

def build_rag_chain(retriever):
    """
    Xây dựng RAG chain với:
    - Retriever → format docs với source info
    - Conversation memory
    - LLM (streaming)
    - Output parser

    Args:
        retriever: FAISS retriever instance

    Returns:
        RunnableSequence — gọi .invoke(question) hoặc .stream(question)
    """
    logger.info("🔗 Building RAG chain with memory and source attribution")

    def retrieve_and_format(question: str):
        """Retrieve docs và format, lưu sources vào session state."""
        docs = retriever.invoke(question)
        formatted_context, sources = format_docs(docs)
        # Lưu sources để hiển thị trong UI
        st.session_state["_last_sources"] = sources
        return formatted_context

    rag_chain = (
        {
            "context": RunnableLambda(retrieve_and_format),
            "question": RunnablePassthrough(),
            "chat_history": RunnableLambda(lambda _: get_chat_history_messages()),
        }
        | RAG_PROMPT
        | st.session_state.llm
        | StrOutputParser()
    )

    logger.info("✅ RAG chain built successfully")
    return rag_chain


def update_chat_history(question: str, answer: str):
    """
    Cập nhật conversation history cho memory.

    Args:
        question: Câu hỏi của user
        answer: Câu trả lời của AI
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Giới hạn history size
    max_entries = MEMORY_WINDOW_SIZE * 2
    if len(st.session_state.chat_history) > max_entries:
        st.session_state.chat_history = st.session_state.chat_history[-max_entries:]