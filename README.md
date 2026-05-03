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

## Project Structure

BookMind-Local-RAG/
├── README.md
├── .gitignore
├── requirements.txt
├── script_1.py
└── script_2.py

## Installation

##Clone the repository:

git clone https://github.com/YOUR_USERNAME/BookMind-Local-RAG.git
cd BookMind-Local-RAG

## Create a virtual environment:
python -m venv .venv

## Activate it:

## For macOS/Linux:

source .venv/bin/activate

## For Windows:

.venv\Scripts\activate

## Install dependencies:

pip install -r requirements.txt

## Usage
1)
python Book_RAG_1.py

2)
python Book_RAG_2.py

## Important Notes

This repository does not include:
	•	copyrighted books
	•	local LLM model files
	•	generated embeddings
	•	prepared datasets

Users should provide their own EPUB files and local GGUF model.

## Current Limitations

	•	EPUB quality affects text extraction
	•	Some EPUB files may contain images instead of selectable text
	•	Answer quality depends on the local LLM model
	•	Current version uses basic file paths inside scripts
	•	No persistent vector database yet













