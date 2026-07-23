"""
LangChain RAG Helper
====================
Implements a Retrieval-Augmented Generation (RAG) pipeline for the
customer service chatbot using:

  - LLM        : Google Gemini 2.5 Flash (via langchain-google-genai)
  - Embeddings : Google Generative AI embedding-001 (via langchain-google-genai)
  - Vector DB  : FAISS (local index stored in faiss_index/)
  - Data source: dataset/dataset.csv — 76 FAQ rows (prompt + response)

Sentiment-aware responses
-------------------------
get_qa_chain(sentiment) accepts a sentiment label from sentiment_analyzer.py
and injects a tone instruction into the prompt via PromptTemplate.partial().
This avoids any extra API calls — only the prompt text changes.
"""

import os
os.environ["USE_TF"] = "0"   # prevent TF crash on macOS (protobuf conflict)
os.environ["USE_JAX"] = "0"

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_PATH = os.path.join(_SRC_DIR, "..", "dataset", "dataset.csv")

# --- Model initialisation (runs once at import time) ---

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.1,   # low temperature for factual, consistent answers
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

vectordb_file_path = "faiss_index"

# Tone instructions injected into the prompt based on detected sentiment
_SENTIMENT_INSTRUCTIONS = {
    "positive": (
        "The customer seems satisfied or enthusiastic. "
        "Match their positive energy with a warm, encouraging tone."
    ),
    "negative": (
        "The customer seems frustrated or upset. "
        "Start by acknowledging their concern with empathy, "
        "then provide a clear and helpful answer."
    ),
    "neutral": "",   # no special instruction for neutral queries
}


def create_vector_db():
    """
    Load the FAQ CSV, embed all rows, and persist the FAISS index locally.

    Preprocessing steps:
    1. Load CSV with cp1252 encoding (source file uses Windows-1252 charset)
    2. Each row becomes a LangChain Document with page_content = full row text
    3. Embed documents using all-MiniLM-L6-v2 (384-dim vectors)
    4. Save FAISS index to faiss_index/ for reuse across sessions
    """
    loader = CSVLoader(file_path=_DATASET_PATH, source_column="prompt", encoding="cp1252")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain(sentiment: str = "neutral"):
    """
    Build and return a sentiment-aware RetrievalQA chain.

    Parameters
    ----------
    sentiment : str
        One of 'positive', 'negative', 'neutral' (from sentiment_analyzer).

    Pipeline
    --------
    User question
        → FAISS retriever (cosine similarity, threshold=0.7)
        → top-k FAQ chunks as context
        → Gemini prompt (context + sentiment_instruction + question)
        → generated answer

    The sentiment_instruction is pre-filled via PromptTemplate.partial() so
    RetrievalQA still sees only {context} and {question} as runtime variables.
    """
    vectordb = FAISS.load_local(
        vectordb_file_path, embeddings, allow_dangerous_deserialization=True
    )

    # score_threshold=0.7 filters out weakly related FAQ chunks
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
    {sentiment_instruction}
    CONTEXT: {context}

    QUESTION: {question}"""

    base_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "sentiment_instruction"],
    )

    # Bake in the sentiment instruction — result has only [context, question]
    instruction = _SENTIMENT_INSTRUCTIONS.get(sentiment, "")
    PROMPT = base_prompt.partial(sentiment_instruction=instruction)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you offer internships?"))
