import streamlit as st


def init_session_state():

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "llm" not in st.session_state:
        st.session_state.llm = None