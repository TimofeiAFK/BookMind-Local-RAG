# BookMind Local RAG

A local RAG-based AI assistant that processes EPUB books and answers questions using semantic search and a locally hosted LLM.

## Overview

BookMind Local RAG is a Python-based project for asking questions about EPUB books using a local language model. The system extracts text from an EPUB file, cleans it, splits it into chunks, retrieves the most relevant passages using semantic search, and sends the context to a local LLM to generate an answer.

This project was created as a portfolio project to demonstrate skills in NLP, local LLM inference, RAG pipelines, text preprocessing, and AI application development.

## Features

- Extracts text from EPUB files
- Cleans and prepares book text
- Splits long text into searchable chunks
- Uses embeddings for semantic search
- Retrieves relevant context for a user question
- Generates answers using a local LLM
- Runs locally without sending book content to external APIs

## Tech Stack

- Python
- EPUB text processing
- Sentence Transformers
- Semantic search
- Local LLM inference
- GGUF / llama.cpp compatible models

## Project Pipeline

```text
EPUB book
→ text extraction
→ text cleaning
→ chunking
→ embeddings
→ semantic retrieval
→ prompt construction
→ local LLM answer

BookMind-Local-RAG/
├── README.md
├── .gitignore
├── requirements.txt
├── script_1.py
└── script_2.py
