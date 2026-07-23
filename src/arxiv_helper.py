import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

import csv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_helper import llm, embeddings

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_PATH = os.path.join(_SRC_DIR, "..", "dataset", "arxiv_cs_sample.csv")
ARXIV_VECTORDB_PATH = os.path.join(_SRC_DIR, "arxiv_faiss_index")

_PROMPT_TEMPLATE = """You are an expert research assistant specializing in computer science and AI.
Use the following research paper excerpts to answer the question.
Cite specific paper titles when relevant. If the answer is not found in the provided papers,
say "I don't have enough information in my knowledge base to answer this." — do not speculate.

Context (retrieved papers):
{context}

Question: {question}

Answer:"""

_QA_PROMPT = PromptTemplate(
    template=_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def _load_documents() -> list[Document]:
    docs = []
    with open(_DATASET_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = f"{row['title']}\n\n{row['abstract']}"
            docs.append(Document(
                page_content=text,
                metadata={
                    "arxiv_id": row.get("arxiv_id", ""),
                    "title": row.get("title", ""),
                    "authors": row.get("authors", ""),
                    "year": row.get("year", ""),
                    "categories": row.get("categories", ""),
                    "url": row.get("url", ""),
                },
            ))
    return docs


def create_arxiv_vector_db():
    docs = _load_documents()
    # Split long abstracts for better retrieval granularity
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    vectordb.save_local(ARXIV_VECTORDB_PATH)
    return len(split_docs)


def _load_vectordb():
    return FAISS.load_local(
        ARXIV_VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True
    )


def search_papers(query: str, k: int = 5) -> list[tuple[Document, float]]:
    """Semantic similarity search — returns (Document, relevance_score) pairs."""
    vectordb = _load_vectordb()
    return vectordb.similarity_search_with_relevance_scores(query, k=k)


def get_arxiv_qa_chain():
    """
    Returns a ConversationalRetrievalChain.
    Call with: chain({"question": q, "chat_history": [(h1, a1), (h2, a2), ...]})
    Response keys: "answer", "source_documents"
    """
    vectordb = _load_vectordb()
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 5},
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": _QA_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    return chain
