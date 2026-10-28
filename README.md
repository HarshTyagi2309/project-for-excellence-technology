# DocuMind — Mini RAG Q&A Assistant

A document question-answering app built with Streamlit, Sentence Transformers, FAISS and Groq.

## Features

- Upload PDF or TXT documents.
- Automatically extract text and split it into overlapping chunks.
- Generate local embeddings using Sentence Transformers.
- Retrieve the top 3 relevant chunks using FAISS.
- Generate answers using Groq, grounded in retrieved document context.
- View retrieved source chunks and similarity scores.
- Clean chat interface with suggested questions.

## Requirements

- Python 3.12 recommended
- A Groq API key
- Internet connection for the Groq API and initial model download

## Setup on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Add your Groq API key to `.env`:

```text
GROQ_API_KEY=your_actual_key_here
```

Get a key from https://console.groq.com/keys.
Never commit `.env` or publish your API key.

Run the app:

```powershell
streamlit run app.py
```

Upload a PDF or TXT file in the sidebar and ask questions in the chat.

## Sample questions

- Summarize this document.
- What are the main points?
- What does the document say about annual leave?
- Is a particular policy mentioned?

## Limitations

Scanned PDFs without selectable text may require OCR. Answers depend on text extraction and retrieval quality; always verify important information against the original document.
