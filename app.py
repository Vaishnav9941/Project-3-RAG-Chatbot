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
# This line simply tells that here is my PDF

documents = loader.load()
# At this step the langchain reads the PDF
# Here each page present in PDF becomes one document object by default

print("Number of Pages : ",len(documents))
# This prints the number of pages present in the documents/PDF

print("\nFirst Page Content : ")
print(documents[0].page_content)
# This prints only the content of first page of the PDF

print("\nFirst Page Metadata")
print(documents[0].metadata)
# This prints only the metdata of the content

# SPLIT INTO CHUNKS
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

## NOTE : chunk_size = 500 means each chunk will contain approx 500 characters
## NOTE : chunk_overlap = 100 means it prevents splitting the text awkwardly and maintain proper order of sentences

chunks = text_splitter.split_documents(documents)

print("Total Pages : ", len(documents))
print("Total Chunks : ", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0].page_content)

print("\nMetadata:\n")
print(chunks[0].metadata)

## Create Embedding Model using HuggingFaceEmbedding
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

## Convert The First Chunk into vector
vector = embeddings.embed_query(chunks[0].page_content)

print("Vector Length: ", len(vector))
print(vector[:10]) # Show First 10 numbers

## Note : BUILDING YOUR FIRST FAISS DATABASE

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding = embeddings
)

retriever = vector_store.as_retriever()
## Thi above line returns only top 4 chunks

# retriever = vector_store.as_retriever(
#     search_kwargs = {"k":3}
# )
## This retrieve only top 3 chunks

query = "What is the main topic of the document"

docs = retriever.invoke(query)
# print(docs[0].page_content)

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key = api_key
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

## Create the chain using LCEL

chain = (
    {
        "context":retriever,
        "question":RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Ask A question
response = chain.invoke("What is this document about?")
print(response)