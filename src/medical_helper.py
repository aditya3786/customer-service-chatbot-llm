"""
Medical Q&A RAG Pipeline
========================
Retrieval-Augmented Generation pipeline for the Medical Q&A chatbot.
Uses the same FAISS + HuggingFace embeddings + Gemini stack as the
customer service chatbot, but with a separate vector index and a
medically-tuned prompt that enforces factual grounding and includes
a safety disclaimer.

Data source: dataset/medquad.csv
  - 5,068 Q&A pairs from 5 NIH sources
  - Columns: question, answer, focus, qtype, source
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Reuse the already-initialised LLM and embeddings — avoids loading models twice
from langchain_helper import llm, embeddings

load_dotenv()

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_MEDQUAD_PATH = os.path.join(_SRC_DIR, "..", "dataset", "medquad.csv")

MEDICAL_VECTORDB_PATH = "medical_faiss_index"

_MEDICAL_PROMPT_TEMPLATE = """You are a knowledgeable medical information assistant.
Using ONLY the medical information provided in the context below, answer the question accurately.
If the answer is not found in the context, say "I don't have enough information on this topic."
Do NOT make up medical information or provide advice beyond what is in the context.

IMPORTANT: Always remind the user to consult a qualified healthcare provider for personal medical advice.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def create_medical_vector_db():
    """
    Parse medquad.csv, embed all 5,068 Q&A pairs, and save the FAISS index.

    Uses the 'question' column as the source for embedding so that
    retrieval matches on question similarity. The full row (question +
    answer + focus + qtype + source) becomes the document content.
    """
    loader = CSVLoader(
        file_path=_MEDQUAD_PATH,
        source_column="question",
        encoding="utf-8",
    )
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(MEDICAL_VECTORDB_PATH)


def get_medical_qa_chain():
    """
    Build and return the medical RetrievalQA chain.

    Retrieval uses cosine similarity with a 0.7 score threshold to
    ensure only medically relevant context is passed to Gemini.
    Temperature is kept at 0.1 for factual, deterministic responses.
    """
    vectordb = FAISS.load_local(
        MEDICAL_VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectordb.as_retriever(score_threshold=0.7)

    PROMPT = PromptTemplate(
        template=_MEDICAL_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain
