import os
import dspy
from pathlib import Path

# LangChain document loaders, text splitter, vectorstore and embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)  # wrapper for sentence-transformers
from langchain.vectorstores import FAISS

LM_MODEL_STUDENT = "openrouter/openai/gpt-oss-20b:free"
LM_MODEL_TEACHER = "openrouter/openai/gpt-oss-120b:free"

# --- Config: change these paths to your downloaded PDFs ---
DIABETES_PDF_PATHS = [
    "docs/diabets1.pdf",
    "docs/diabets2.pdf",
]  # <-- put your two PDF filenames here
COPD_PDF_PATHS = ["docs/copd1.pdf", "docs/copd2.pdf"]
OUTPUT_DIABETES_FAISS_DIR = "faiss_index/diabetes"
OUTPUT_COPD_FAISS_DIR = "faiss_index/copd"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# chunk settings (tweak for your needs)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 200


def load_pdfs(paths):
    """Load PDFs into LangChain Document objects (keeps page-level granularity)."""
    all_docs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {p}")
        loader = PyPDFLoader(str(p))
        # load returns a list of Document objects (one per page typically)
        pages = loader.load()
        # add a source filename into metadata for traceability
        for i, doc in enumerate(pages):
            # ensure a copy of metadata dict (avoid mutating shared objects)
            meta = dict(doc.metadata or {})
            meta["source"] = str(p.name)
            meta["page"] = i
            doc.metadata = meta
        all_docs.extend(pages)
    return all_docs


def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into smaller chunks (keeps metadata)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    # split_documents returns list[Document] (with page_content and metadata)
    chunks = text_splitter.split_documents(documents)
    return chunks


def build_vectorstore(
    chunks, model_name=EMBEDDING_MODEL, save_dir=OUTPUT_DIABETES_FAISS_DIR
):
    """Create embeddings and store them in a FAISS vectorstore, then persist to disk."""
    # Instantiate HuggingFaceEmbeddings wrapper (requires sentence-transformers installed)
    hf_emb = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": "cpu"}
    )  # change to "cuda" if available

    # Build FAISS index from LangChain Document objects
    print(
        "Creating FAISS vector store from",
        len(chunks),
        "chunks. This may take a while...",
    )
    vectorstore = FAISS.from_documents(chunks, hf_emb)

    # Persist to disk
    vectorstore.save_local(save_dir)
    print(f"Saved FAISS vectorstore to: {save_dir}")
    return vectorstore, hf_emb


def make_diabets_vector():
    # Check if vector store already exists
    if os.path.exists(OUTPUT_DIABETES_FAISS_DIR) and os.path.exists(
        os.path.join(OUTPUT_DIABETES_FAISS_DIR, "index.faiss")
    ):
        print(
            f"Loading existing diabetes vector store from {OUTPUT_DIABETES_FAISS_DIR}..."
        )
        hf_emb = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
        )
        diabetes_vectorstore = FAISS.load_local(
            OUTPUT_DIABETES_FAISS_DIR, hf_emb, allow_dangerous_deserialization=True
        )
        print(f"Successfully loaded existing diabetes vector store.")
        return diabetes_vectorstore, hf_emb

    # Build new vector store if it doesn't exist
    print("Loading Diabetes PDFs...")
    docs = load_pdfs(DIABETES_PDF_PATHS)
    print(f"Loaded {len(docs)} page-documents from {len(DIABETES_PDF_PATHS)} PDFs.")

    print("Chunking Diabetes documents...")
    chunks = chunk_documents(docs)
    print(
        f"Produced {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )

    diabetes_vectorstore, diabetes_embeddings = build_vectorstore(
        chunks, save_dir=OUTPUT_DIABETES_FAISS_DIR
    )
    return diabetes_vectorstore, diabetes_embeddings


def make_copd_vector():
    # Check if vector store already exists
    if os.path.exists(OUTPUT_COPD_FAISS_DIR) and os.path.exists(
        os.path.join(OUTPUT_COPD_FAISS_DIR, "index.faiss")
    ):
        print(f"Loading existing COPD vector store from {OUTPUT_COPD_FAISS_DIR}...")
        hf_emb = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
        )
        copd_vectorstore = FAISS.load_local(
            OUTPUT_COPD_FAISS_DIR, hf_emb, allow_dangerous_deserialization=True
        )
        print(f"Successfully loaded existing COPD vector store.")
        return copd_vectorstore, hf_emb

    # Build new vector store if it doesn't exist
    print("Loading COPD PDFs...")
    docs = load_pdfs(COPD_PDF_PATHS)
    print(f"Loaded {len(docs)} page-documents from {len(COPD_PDF_PATHS)} PDFs.")

    print("Chunking COPD documents...")
    chunks = chunk_documents(docs)
    print(
        f"Produced {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )

    copd_vectorstore, copd_embeddings = build_vectorstore(
        chunks, save_dir=OUTPUT_COPD_FAISS_DIR
    )
    return copd_vectorstore, copd_embeddings


diabetes_vectorstore, diabetes_embeddings = make_diabets_vector()
copd_vectorstore, copd_embeddings = make_copd_vector()


def diabetes_vector_search_tool(query: str, k: int = 3) -> str:
    """
    A tool for the ReAct agent.
    Performs vector search and returns a formatted string of results.
    """
    assert diabetes_vectorstore
    results = diabetes_vectorstore.similarity_search_with_score(query, k=k)
    context = ""
    for i, (doc, score) in enumerate(results):
        doc_content = doc.page_content
        context += f"[PASSAGE {i+1}, score={score:.4f}]\n{doc_content}\\n\\n"
    return context


def copd_vector_search_tool(query: str, k: int = 3) -> str:
    """
    A tool for the ReAct agent.
    Performs vector search and returns a formatted string of results.
    """
    assert copd_vectorstore
    results = copd_vectorstore.similarity_search_with_score(query, k=k)
    context = ""
    for i, (doc, score) in enumerate(results):
        doc_content = doc.page_content
        context += f"[PASSAGE {i+1}, score={score:.4f}]\n{doc_content}\\n\\n"
    return context


def test_tools():

    # quick retrieval test
    c = diabetes_vector_search_tool(
        "What are the main treatments for Type 2 diabetes?", k=3
    )
    print(c)
    print("*" * 80)
    c = copd_vector_search_tool("What are the main treatments for COPD?", k=3)
    print(c)


# Define a signature (simple QA)
class RAGQA(dspy.Signature):
    """You are a helpful assistant. Answer a question using retrieved passages"""

    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()


class SimpleQA(dspy.Signature):
    """You are a helpful assistant. Answer a question"""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def rag_test(lm: dspy.LM):
    rag = dspy.ChainOfThought(RAGQA)
    question = "What is Gestational Diabetes Mellitus (GDM)?"
    retrieved_context = diabetes_vector_search_tool(question, k=3)

    rag(context=retrieved_context, question=question)
    lm.inspect_history(n=1)

    react = dspy.ReAct(signature=SimpleQA, tools=[diabetes_vector_search_tool])
    question = "What is Gestational Diabetes Mellitus (GDM)?"
    pred = react(question=question)
    lm.inspect_history(n=1)
    print(pred)
