"""
Person 3: Embeddings Generator
Responsibility: Convert text chunks into vector embeddings.

Provides a factory function that returns HuggingFaceEmbeddings object
for use by the vector store and retriever.

Uses sentence-transformers for compatibility with existing database.
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import EMBEDDING_MODEL


@st.cache_resource(show_spinner=False)
def get_embedding_function():
    """
    Returns the embedding function to be used by the Vector DB.

    Returns:
        HuggingFaceEmbeddings: Configured embedding object using sentence-transformers.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
