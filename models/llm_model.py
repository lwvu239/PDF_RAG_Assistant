"""
llm_model.py — Cấu hình và tải mô hình LLM (Gemini).

Sử dụng Google Generative AI (Gemini 2.5 Flash) qua API.
Hỗ trợ streaming và configurable parameters.
"""

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_OUTPUT_TOKENS
from utils.logger import logger


@st.cache_resource
def load_llm():
    """
    Tải LLM với caching (chỉ load 1 lần).
    Validate API key trước khi khởi tạo.

    Returns:
        ChatGoogleGenerativeAI instance

    Raises:
        ValueError: Nếu API key không được thiết lập
    """
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not configured")
        raise ValueError(
            "GOOGLE_API_KEY chưa được thiết lập. "
            "Vui lòng thêm vào file .env (xem .env.example)"
        )

    try:
        logger.info(f"🤖 Loading LLM: {LLM_MODEL_NAME} (temp={LLM_TEMPERATURE})")

        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
            streaming=True,  # Bật streaming cho real-time response
        )

        logger.info("✅ LLM loaded successfully")
        return llm

    except Exception as e:
        logger.error(f"❌ Failed to load LLM: {e}", exc_info=True)
        raise RuntimeError(f"Không thể tải LLM model: {e}")