
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import streamlit as st


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )