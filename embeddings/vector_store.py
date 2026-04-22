"""
Person 4: Vector Store (ChromaDB)
Responsibility: Store chunks + metadata, and load the DB.

ALIGNED: Uses collection_name="student_docs" matching build_db.py and app.py.
"""

import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from core.config import COLLECTION_NAME


def store_chunks(chunks: list[dict], embedding_function, persist_directory: str = "./db"):
    """
    Creates the Chroma vector database with the given chunks.

    Args:
        chunks (list[dict]): The chunked documents with text, source, page.
        embedding_function: The embedding function from embedder.py.
        persist_directory (str): Where to save the DB.

    Returns:
        Chroma: The initialized Chroma database object.
    """
    doc_objects = [
        Document(
            page_content=c["text"],
            metadata={"source": c["source"], "page": c["page"]},
        )
        for c in chunks
    ]

    vector_db = Chroma.from_documents(
        documents=doc_objects,
        embedding=embedding_function,
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME,
    )

    return vector_db


def load_vector_db(embedding_function, persist_directory: str = "./db"):
    """
    Loads the persistent Chroma vector database.

    Args:
        embedding_function: The embedding function from embedder.py.
        persist_directory (str): Where the DB is saved.

    Returns:
        Chroma or None: The loaded DB, or None if it doesn't exist.
    """
    if not os.path.exists(persist_directory):
        return None

    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME,
    )

    return vector_db
