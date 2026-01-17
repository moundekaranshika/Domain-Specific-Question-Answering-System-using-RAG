# Domain-Specific Question Answering using RAG

This project implements a **Retrieval-Augmented Generation (RAG)** based Question Answering system that answers queries grounded in user-provided documents.

The goal is to reduce hallucinations commonly seen in Large Language Models by retrieving relevant context before generating responses.

---

##  Features
- Document ingestion from PDFs
- Text chunking and semantic embedding
- Vector search using FAISS
- Context-aware answer generation using LLMs
- Designed for extensibility with Knowledge Graph integration

---

## Architecture

Documents → Text Splitter → Embeddings → FAISS Vector Store  
User Query → Retriever → LLM → Answer

---

##  Tech Stack
- Python
- Hugging Face Transformers
- LangChain
- FAISS
- Sentence Transformers

---

## 📂 Project Structure
