import os
import streamlit as st

# load_dotenv: đọc biến môi trường từ file .env (chứa API key)
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

@st.cache_resource
def load_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("Vui lòng kiểm tra lại GOOGLE_API_KEY")
        st.stop()

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",      # Model Gemini 2.5 Flash (nhanh + rẻ)
        google_api_key=api_key,        # API key để xác thực
        temperature=0.3,               # Độ sáng tạo thấp (0.3) → trả lời chính xác, ít bịa đặt
    )