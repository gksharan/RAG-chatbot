# RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that searches local text
documents and generates answers grounded in the most relevant passages.

## How it works

1. Loads `.txt` files from the `data` directory.
2. Splits documents into overlapping chunks.
3. Creates semantic embeddings with Sentence Transformers.
4. Stores the embeddings in a FAISS similarity index.
5. Retrieves the most relevant chunks for each question.
6. Generates an answer with a Hugging Face text-generation model.

## Requirements

- Python 3.9 or newer
- `numpy`
- `faiss-cpu`
- `sentence-transformers`
- `transformers`
- `torch`

Install the dependencies:

```powershell
python -m pip install numpy faiss-cpu sentence-transformers transformers torch
```

## Usage

Ensure the sample documents are present in `data/`, then run:

```powershell
python rag.py
```

Ask questions at the `You:` prompt. Enter `quit`, `exit`, or `q` to stop.

The first run may download the embedding and text-generation models from
Hugging Face.

## Project structure

```text
RAG-chatbot/
├── data/
│   ├── overview.txt
│   ├── setup.txt
│   └── troubleshooting.txt
├── rag.py
└── README.md
```

The local Python virtual environment is intentionally excluded from GitHub.
