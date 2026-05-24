# AI URL Research Tool (LangChain RAG System)

An AI-powered research assistant that allows users to input up to 3 web URLs and ask questions directly about their content. The system extracts, processes, and indexes web data, enabling intelligent question-answering using Retrieval-Augmented Generation (RAG) with Large Language Models.

---

# Project Overview

This project enables users to turn a set of web pages into a temporary knowledge base. Once URLs are processed, the system:

- Loads and extracts content from the webpages
- Splits content into manageable chunks
- Converts text into embeddings
- Stores embeddings in a FAISS vector database
- Retrieves relevant context based on user queries
- Uses an LLM to generate grounded answers from the retrieved content

The system ensures responses are **strictly based on provided sources**, reducing hallucinations and improving factual accuracy.

---

# Core Features

- Accepts up to **3 web URLs** as input
- Web content extraction and preprocessing
- Document chunking for efficient retrieval
- Embedding-based semantic search (FAISS)
- Persistent vector storage for fast querying
- LLM-powered question answering
- Two RAG strategies:
  - **Stuffing approach**
  - **Map-Reduce approach**
- Source-aware responses (answers linked to URLs)
- Streamlit-based interactive UI

---

# Project Structure

```bash
├── main.py
├── text.ipynb
├── requirement.txt
├── faiss_index/              # Saved vector database (generated at runtime)
└── .env                      # API keys and environment variables
```

---

# System Workflow

```text
1. User enters up to 3 URLs in the Streamlit interface
        |
        v
2. Web pages are loaded and scraped
        |
        v
3. Text content is split into smaller chunks
        |
        v
4. Each chunk is converted into embeddings
        |
        v
5. Embeddings are stored in a FAISS vector database
        |
        v
6. User enters a natural language question
        |
        v
7. System performs semantic search to retrieve relevant chunks
        |
        v
8. Retrieved context is passed into the LLM
        |
        v
9. LLM generates a grounded answer using:
        - Stuffing RAG OR
        - Map-Reduce RAG
        |
        v
10. Final answer + source URLs are returned to the user
```
