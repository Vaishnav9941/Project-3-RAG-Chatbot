from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import streamlit as st
import tempfile
import os

# -------------------------------
# Load Environment Variables
# -------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄"
)


st.title("📄 Chat with Your PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []

## Here we are creating a button , if the user click this button the list gets empty and UI reruns from start
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []  ## This makes the list empty
    st.rerun() ## This reruns the entire stramlit UI

## This creates an empty list called message

## Now lets display the previous chats in the UI using following

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

# -------------------------------
# Create RAG Pipeline
# -------------------------------

@st.cache_resource ## This says if I have already created this RAG chain for the same PDF, dont rebuid It. Just reuse the existing one
def create_rag_chain(pdf_path):

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    # Create Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store vectors in FAISS
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k":3}
    )

    # Load Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful AI Assistant.

Answer ONLY using the context provided below.

If the answer is not present in the context, say:
"I don't know the answer based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # LCEL Chain
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# -------------------------------
# After Upload
# -------------------------------

if uploaded_file is not None:

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.success("✅ PDF Uploaded Successfully!")

    # Build RAG Chain
    # chain = create_rag_chain(temp_path)
    with st.spinner(" ⏳Creating Embeddings and preparing your pDF...."):
        chain = create_rag_chain(temp_path)
    
    ## This above line does: While create_rag_chain(temp_path) is running, Streamlit will display the above written sentence

    # User Question
    question = st.chat_input("Ask a question about your PDF")

    if question:

        ## Storing the user's Message
        st.session_state.messages.append(
            {
                "role":"user",
                "content":question
            }
        )

        response = chain.invoke(question)

        ## Storing the Assistant's Reply
        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":response
            }
        )

        st.rerun()

        ##The below lines are removed because we are already appending the messages above the rerun
        # with st.chat_message("user"):
        #     st.write(question)

        # with st.chat_message("assistant"):
        #     st.write(response)