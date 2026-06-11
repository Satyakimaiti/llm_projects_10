from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    convert_to_messages
)

from langchain_core.documents import Document


# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv(override=True)

MODEL = "llama-3.3-70b-versatile"

DB_NAME = str(
    Path(__file__).parent.parent / "vector_db"
)

RETRIEVAL_K = 10


# ==========================================
# EMBEDDINGS
# MUST MATCH ingest.py
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.

You are chatting with a user about Insurellm.

Use the provided context when relevant.

If the answer is not contained in the context, say that you don't know.

Context:
{context}
"""


# ==========================================
# VECTOR STORE
# ==========================================

vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": RETRIEVAL_K}
)


# ==========================================
# GROQ MODEL
# ==========================================

llm = ChatGroq(
    model=MODEL,
    temperature=0
)


# ==========================================
# RETRIEVE CONTEXT
# ==========================================

def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant documents.
    """

    return retriever.invoke(question)


# ==========================================
# COMBINE USER HISTORY
# ==========================================

def combined_question(
    question: str,
    history: list[dict] = None
) -> str:

    if history is None:
        history = []

    prior = "\n".join(
        msg["content"]
        for msg in history
        if msg["role"] == "user"
    )

    return f"{prior}\n{question}"


# ==========================================
# MAIN RAG FUNCTION
# ==========================================

def answer_question(
    question: str,
    history: list[dict] = None
) -> tuple[str, list[Document]]:

    if history is None:
        history = []

    combined = combined_question(
        question,
        history
    )

    docs = fetch_context(combined)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    system_prompt = SYSTEM_PROMPT.format(
        context=context
    )

    messages = [
        SystemMessage(content=system_prompt)
    ]

    messages.extend(
        convert_to_messages(history)
    )

    messages.append(
        HumanMessage(content=question)
    )

    response = llm.invoke(messages)

    return response.content, docs


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    answer, docs = answer_question(
        "Who is Avery?"
    )

    print("\nANSWER:")
    print(answer)

    print("\nDOCUMENTS RETRIEVED:")
    print(len(docs))