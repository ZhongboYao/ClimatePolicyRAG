# Datathon KUL 2025 — Climate Policy RAG System

A Retrieval-Augmented Generation (RAG) system for analyzing climate policies from extensive PDF documents, built for the KU Leuven 2025 Datathon.

## Overview

This project ingests climate policy documents from COP meetings, G7 country reports, and the IPCC 6th Assessment Report, then enables semantic querying, multi-agent country negotiation simulation, and topic modeling over the extracted policies.

## Features

- **PDF Ingestion & Chunking** — Extracts and chunks text from large policy PDFs using PyMuPDF and LlamaIndex
- **Policy Extraction** — Uses GPT-4o-mini to classify chunks and extract structured policy records (policy, effect, country, year)
- **Hybrid Vector Search** — Stores dense (OpenAI `text-embedding-3-small`) and sparse (SPLADE) embeddings in Qdrant for hybrid retrieval
- **Cross-Encoder Reranking** — Reranks retrieved results with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Query Answering** — Answers natural language questions grounded in retrieved policy context
- **Multi-Agent Simulation** — Simulates climate negotiations between G7 countries using LangChain agents
- **Topic Modeling** — Discovers policy themes across countries and time periods using BERTopic

## Project Structure

```
.
├── main.ipynb          # End-to-end pipeline notebook
└── src/
    ├── config.py       # Constants (chunk size, folder paths, collection name)
    ├── utils.py        # File I/O and JSON helpers
    ├── ingestion.py    # PDF text extraction and chunking (Chunk, PDF classes)
    ├── policy.py       # Policy data model and grouping utilities
    ├── vector_store.py # Qdrant embedding, storage, and retrieval operations
    ├── retrieval.py    # Hybrid search, reranking, and cosine similarity filtering
    ├── chatbot.py      # Query answering functions
    └── agent.py        # Multi-agent country negotiation simulation
```

## Data Sources

Place documents under a `Dataset/` directory with the following structure:

```
Dataset/
├── COP Meetings/
├── G7/
│   ├── Canada/
│   ├── France/
│   ├── Germany/
│   ├── Italy/
│   ├── Japan/
│   ├── United Kingdom/
│   └── United States/
└── IPCC report/
    └── 6th assessment/
```

## Setup

### 1. Install dependencies

```bash
pip install pymupdf llama-index openai langchain langchain-openai langchain-qdrant \
    qdrant-client fastembed sentence-transformers bertopic umap-learn hdbscan \
    scikit-learn pandas tqdm
```

### 2. Configure API keys

Create `api.py` in the project root with your credentials:

```python
OPENAI_API = "your-openai-api-key"
QDRANT_URL = "your-qdrant-url"
QDRANT_API = "your-qdrant-api-key"
```

### 3. Run the pipeline

Open and run `main.ipynb` end-to-end. The notebook covers:

1. Extract text from PDFs
2. Chunk documents
3. Classify and extract structured policies via GPT-4o-mini
4. Embed and store in Qdrant
5. Query the system
6. Run multi-agent climate discussions
7. Perform topic modeling

## Tech Stack

| Layer | Tools |
|---|---|
| PDF parsing | PyMuPDF |
| Text splitting | LlamaIndex SentenceSplitter |
| LLM | GPT-4o-mini (OpenAI) |
| Embeddings | `text-embedding-3-small` (dense), SPLADE (sparse) |
| Vector DB | Qdrant |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Topic modeling | BERTopic, UMAP, HDBSCAN |
| Orchestration | LangChain |
