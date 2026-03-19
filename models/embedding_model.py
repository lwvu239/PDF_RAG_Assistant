from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import streamlit as st

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="bkai-foundation-models/vietnamese-bi-encoder"
    )