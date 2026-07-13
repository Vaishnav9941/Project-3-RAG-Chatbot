import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄"
)

st.title("📄 Chat With Your PDF")

st.write("Upload a PDF and ask Questions")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:
    st.success("PDF Uploaded Successfully")

uploaded_file = st.file_uploader()
st.write(type(uploaded_file))

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.success("PDF Uploaded Successfully")

## namedTemporaryFile() creates a temporary file on our computer

loader = PyPDFLoader(temp_path)
documents = loader.load()

st.write("Pages:", len(documents))

