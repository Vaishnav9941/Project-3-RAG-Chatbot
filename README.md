# 📄 RAG Chatbot using LangChain, FAISS, Groq & Streamlit

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload a PDF and ask questions about its contents.

The chatbot retrieves the most relevant chunks from the uploaded PDF using semantic search and answers questions using Groq's Llama 3.3 model.

---

# 🚀 Features

- 📄 Upload any PDF
- 💬 Ask questions about the uploaded PDF
- 🧠 Semantic Search using HuggingFace Embeddings
- 📚 FAISS Vector Database
- ⚡ Groq Llama 3.3 LLM
- 💻 Interactive Streamlit Chat UI
- 📝 Chat History
- 🔄 Cached Embeddings for faster responses

---

# 🛠 Tech Stack

- Python
- LangChain
- HuggingFace Embeddings
- FAISS
- Groq
- Streamlit

---

# 📂 Project Structure

```text
Project-3-RAG_Chatbot/
│
├── data/
│   └── genai-principles.pdf
│
├── streamlitUI.py
├── rag.py
├── requirements.txt
├── .gitignore
├── README.md
└── screenshots/
```

---

# ⚙️ How It Works

1. User uploads a PDF.
2. PyPDFLoader reads the PDF.
3. RecursiveCharacterTextSplitter splits the PDF into chunks.
4. HuggingFace Embeddings convert each chunk into vectors.
5. FAISS stores the vectors.
6. User asks a question.
7. The question is converted into an embedding.
8. Retriever searches FAISS for the most relevant chunks.
9. Retrieved context and the question are passed to the Groq LLM.
10. The answer is displayed in the Streamlit interface.

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/Vaishnav9941/Project-3-RAG-Chatbot.git
```

Go to the project folder

```bash
cd Project-3-RAG-Chatbot
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```text
GROQ_API_KEY=your_api_key_here
```

Run the application

```bash
streamlit run streamlitUI.py
```

---

# 📸 Screenshots

Coming Soon...

---

# 🎯 Future Improvements

- Support multiple PDFs
- Show source chunks
- Display page numbers
- Conversation-aware retrieval
- Deploy using Streamlit Cloud

---

# 👨‍💻 Author

**Vaishnav Bhat**

Learning Generative AI through hands-on projects.