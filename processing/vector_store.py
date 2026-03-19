import streamlit as st
from langchain_chroma import Chroma


def create_vector_store(docs):
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=st.session_state.embeddings
    )

    return vector_db.as_retriever()