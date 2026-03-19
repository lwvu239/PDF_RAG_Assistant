import os
import streamlit as st

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
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
    )