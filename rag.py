from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

loader = PyPDFLoader("data/genai-principles.pdf")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    documents = chunks,
    embedding = embeddings
)

retriever = vector_store.as_retriever()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key=api_key
)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI Assistant.
Answer ONLY using the context provided below.

If the answer is not present in the context, say:
"I don't know the answer based on the provided document."
Context:{context}
Question:{question}
Answer:
"""
)

chain = (
    {
        "context":retriever,
        "question":RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

