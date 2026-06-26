# RAG-Based Web Research Assistant

An LLM-powered research chatbot that ingests content from multiple unstructured URLs and lets users query the aggregated information in natural language through an interactive Streamlit interface.

## Overview

Researching a topic often means having a dozen browser tabs open and manually cross-referencing them. This project automates that process: feed it a list of URLs, and it scrapes, embeds, and indexes the content so you can ask natural-language questions across all of them at once and get a synthesized, retrieval-grounded answer.

## Features

- **Multi-URL Ingestion** — Pulls and parses unstructured content from multiple web pages in a single run.
- **RAG Pipeline** — Combines semantic retrieval with LLM generation so answers are grounded in the actual scraped content.
- **Map-Reduce & Stuffing Strategies** — Switches between document map-reduce and stuffing approaches depending on corpus size, balancing accuracy and efficiency for both small and large text volumes.
- **Semantic Search** — FAISS vector store enables fast, high-performance similarity search over embedded document chunks.
- **Interactive UI** — Streamlit front end for entering URLs and querying results conversationally.

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| LLM | OpenAI |
| Vector Store | FAISS |
| Embeddings | OpenAI Embeddings |
| UI | Streamlit |
| Language | Python |

## How It Works

1. **Ingest** — User submits a list of URLs; content is scraped and cleaned into raw text.
2. **Chunk & Embed** — Text is split into chunks and embedded using OpenAI embeddings.
3. **Index** — Embeddings are stored in a FAISS vector index for fast retrieval.
4. **Query** — A user's question is embedded and matched against the index to retrieve the most relevant chunks.
5. **Generate** — Depending on the amount of relevant content, the system either "stuffs" it directly into the prompt or uses a map-reduce strategy to summarize across multiple chunks before generating a final answer.

## Getting Started

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

### Running the App
```bash
streamlit run app.py
```

Then paste in one or more URLs and start asking questions about their content.

## Example Usage

```
URLs:
- https://example.com/article-on-llms
- https://example.com/related-research-post

User: What are the main differences in approach between these two articles?
Assistant: The first article focuses on fine-tuning strategies, while the
second emphasizes retrieval-based methods. Specifically...
```

## Project Structure
```
.
├── ingestion/        # URL scraping and text extraction
├── embeddings/        # Embedding generation and FAISS indexing
├── retrieval/          # Map-reduce / stuffing retrieval logic
├── app.py              # Streamlit entry point
└── requirements.txt
```

## What I Learned

This project gave me practical experience designing RAG pipelines that scale beyond a single document — handling variable corpus sizes with different retrieval strategies, integrating FAISS for performant vector search, and shipping a usable end-to-end prototype with Streamlit. It reflects core applied AI engineering skills: information retrieval, LLM orchestration, and building tools people can actually interact with.

## Author

**Ubrillo** — [GitHub](https://github.com/<your-username>) · [LinkedIn](#)
