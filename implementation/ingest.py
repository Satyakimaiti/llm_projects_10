import os
import glob
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ==========================================
# CONFIGURATION
# ==========================================

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

load_dotenv(override=True)

# Free Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# LOAD DOCUMENTS
# ==========================================

def fetch_documents():

    folders = glob.glob(
        str(Path(KNOWLEDGE_BASE) / "*")
    )

    documents = []

    for folder in folders:

        doc_type = os.path.basename(folder)

        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        folder_docs = loader.load()

        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    return documents


# ==========================================
# CREATE CHUNKS
# ==========================================

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


# ==========================================
# CREATE VECTOR DATABASE
# ==========================================

def create_embeddings(chunks):

    if os.path.exists(DB_NAME):

        try:
            print("Deleting existing collection...")

            Chroma(
                persist_directory=DB_NAME,
                embedding_function=embeddings
            ).delete_collection()

        except Exception as e:
            print(f"Warning: {e}")

    print("Creating vector database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )

    collection = vectorstore._collection

    count = collection.count()

    sample_embedding = collection.get(
        limit=1,
        include=["embeddings"]
    )["embeddings"][0]

    dimensions = len(sample_embedding)

    print(
        f"\nThere are {count:,} vectors "
        f"with {dimensions:,} dimensions "
        f"in the vector store."
    )

    return vectorstore


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("Loading documents...")

    documents = fetch_documents()

    print(f"Loaded {len(documents)} documents")

    print("Creating chunks...")

    chunks = create_chunks(documents)

    print("Generating embeddings...")

    create_embeddings(chunks)

    print("\nIngestion complete!")