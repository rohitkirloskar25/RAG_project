🤖 PDF RAG Pipeline (Gemma2 + ChromaDB)
A complete Retrieval-Augmented Generation pipeline built with LangChain, ChromaDB, and Groq. This system allows you to ingest multiple PDF documents, store them in a vector database, and perform semantic search to answer questions using the Gemma2-9b-it model.

🛠️ Tech Stack
LLM: Groq (Gemma2-9b-it)

Vector Store: ChromaDB

Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)

Framework: LangChain

PDF Processing: PyPDF & RecursiveCharacterTextSplitter

📁 Project Structure
Plaintext

RAG_project/
├── data/
│   ├── pdf_files/      # Put your PDFs here
│   └── vector_store/   # Persistent ChromaDB data
├── .env                # API Keys (GROQ_API_KEY)
├── main.ipynb          # Core Logic
└── README.md
🚀 Quick Start
1. Setup Environment
Bash

pip install numpy sentence-transformers chromadb langchain-community \
            langchain-text-splitters langchain-groq python-dotenv
2. Configure Credentials
Create a .env file in the root directory:

Plaintext

GROQ_API_KEY=your_groq_api_key_here
3. Usage Workflow
Ingest: Place PDF files in data/pdf_files/.

Process: The process_all_pdfs function loads and attaches metadata (source, page, file type).

Chunk: split_documents breaks text into 1000-character chunks with overlap.

Embed: EmbeddingManager converts text to 384-dimension vectors.

Retrieve & Ask: ```python

Example Query
query = "What is the invitation for and when is it?" response = rag_simple(query, rag_retriever, llm) print(response)


## 🧠 Key Components

### **`EmbeddingManager`**
Handles the initialization of the HuggingFace model and converts text strings into NumPy arrays.

### **`VectorStore`**
Manages the persistent ChromaDB collection. It handles the generation of unique UUIDs for every chunk and stores metadata alongside the embeddings for filtered retrieval.

### **`RAGRetriever`**
Performs similarity search. It converts a user's natural language query into a vector and finds the top-K most similar document chunks in the database.

---
