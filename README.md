# Local RAG MCQ Assistant

A local Retrieval-Augmented Generation (RAG) application for answering multiple-choice questions based on PDF documents. Built with ChromaDB, LangGraph, and support for multiple LLM providers.

## Features

- **Document Processing**: PDF ingestion with page-level chunking using PyMuPDF
- **Vector Search**: ChromaDB-based similarity search for relevant context retrieval
- **Multiple LLM Providers**: Support for OpenAI, Anthropic Claude, and Google Gemini
- **Interactive UI**: Streamlit-based chatbot interface for easy interaction

## Project Structure

```
local-rag/
├── src/
│   ├── config.py
│   ├── pipelines/
│   │   ├── download_models.py
│   │   └── ingest_documents.py
│   ├── services/
│   │   ├── document_processor.py
│   │   ├── vector_store.py
│   │   ├── llm_service.py
│   │   └── rag_workflow.py
│   └── frontend/
│       └── app.py
├── models/
├── documents/
├── chroma-db/
├── pyproject.toml
└── .env
```

## Prerequisites

- Python 3.11 or higher
- uv package manager: https://docs.astral.sh/uv/getting-started/installation/ (Remember to add to path)
- HuggingFace account (for downloading embedding model)
- API keys for at least one LLM provider: OpenAI, Anthropic, or Gemini

## Installation

### 1. Clone and Setup

```bash
cd local-rag
cp .env.template .env
```

### 2. Configure Environment Variables

Edit `.env` by adding your credentials for the LLM providers you want to use.

#### Supported LLM Providers

The application supports three LLM providers that you can select from the UI:

- **OpenAI** - Use OpenAI models (GPT-4, GPT-3.5, etc.)
  - Requires: `OPENAI_API_KEY`
  - Configure `OPENAI_MODEL_NAME` (e.g., `gpt-4-turbo-preview`, `gpt-4o-mini`)
  - Pay-per-use API calls

- **Anthropic** - Use Anthropic Claude models
  - Requires: `ANTHROPIC_API_KEY`
  - Configure `ANTHROPIC_MODEL_NAME` (e.g., `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`)
  - Pay-per-use API calls

- **Gemini** - Use Google Gemini models
  - Requires: `GEMINI_API_KEY`
  - Configure `GEMINI_MODEL_NAME` (e.g., `gemini-1.5-pro`, `gemini-1.5-flash`)
  - Pay-per-use API calls

**Note**: You can switch between providers anytime from the UI settings sidebar. Add API keys in `.env` for the providers you want to use.

### 3. Install Dependencies

```bash
uv sync --all-extras
```

If the `.venv` is not activated automatically, run:

```bash
cd local-rag
source .venv/source/activate
```

### 4. Download Embedding Model (Required)

The embedding model is **required** for vector store operations and document ingestion.

```bash
poe download
```

This downloads the embedding model: `sentence-transformers/all-MiniLM-L6-v2` (~90MB)

**Note**: Model download may take some time depending on your internet speed.

### 5. Ingest Documents

Place your PDF documents in the `documents/` directory. These PDFs will be processed and ingested into the ChromaDB vector store for RAG-based MCQ answering.

The documents will be:
- Processed page by page using PyMuPDF
- Chunked at the page level for optimal retrieval
- Stored with metadata including source filename and page numbers

```bash
poe ingest
```

This pipeline will:
- Scan all PDFs in the `documents/` directory
- Extract text from each page
- Generate embeddings using the configured embedding model
- Store chunks in ChromaDB with page-level metadata for citation tracking

## Task Management

This project uses [Poe the Poet](https://poethepoet.natn.io/) as a task runner to manage common development tasks. All tasks are configured in `pyproject.toml` and run within the uv package manager.

To see all available tasks:
```bash
poe
```

## Usage

### Launch the Application

```bash
poe chat
```

The Streamlit app will open in your browser at `http://localhost:8501`

## Testing

Run the test suite using:

```bash
poe tests
```

This will execute all tests case samples for the RAG flow.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Vector store powered by [ChromaDB](https://www.trychroma.com/)
- Document processing with [PyMuPDF](https://pymupdf.readthedocs.io/)
- Package management with [uv](https://github.com/astral-sh/uv)
