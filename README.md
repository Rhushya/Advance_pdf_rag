# PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) application for answering questions from PDF documents using Streamlit, LlamaIndex, and Groq.

## Features

This repository contains three versions of the application with increasing features:

1. **Basic Version** (`app.py`): Simple PDF RAG with chat functionality
2. **Enhanced Version** (`enhanced_app.py`): Advanced features including evaluation and improved UI
3. **LlamaCloud Version** (`llamacloud_app.py`): Integration with LlamaCloud/LlamaParse for enhanced PDF extraction

## Setup

### Prerequisites

- Python 3.9+
- API Keys:
  - Groq API key for LLM capabilities ([Get here](https://console.groq.com/))
  - LlamaCloud API key for LlamaParse (optional, [Get here](https://cloud.llamaindex.ai/))

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
   ```

## Usage

### Basic Version

Run the basic version with:
```
streamlit run app.py
```

Features:
- Upload PDF documents
- Ask questions about the document content
- View source context for answers

### Enhanced Version

Run the enhanced version with:
- ```
  streamlit run enhanced_app.py
  ```

Additional features:
- Adjustable chunk size and overlap
- Response evaluation (Correctness or Relevancy)
- Improved UI with expanded source context
- Document processing options

### LlamaCloud Version

Run the LlamaCloud version with:
- ```
  streamlit run llamacloud_app.py
  ```

Additional features:
- LlamaParse for enhanced PDF extraction
- Document explorer with search functionality
- Document metadata display
- Tabbed interface for better organization

## How It Works

This application uses the following components:

1. **PDF Processing**: Extracts text from PDF documents using LlamaIndex or LlamaParse
2. **Chunking**: Splits text into manageable chunks using sentence splitters
3. **Embedding**: Creates vector embeddings of text chunks using HuggingFace models
4. **Indexing**: Stores embeddings in a vector index for efficient retrieval
5. **Query Processing**: Converts user questions into vector queries
6. **Retrieval**: Finds relevant text chunks based on similarity
7. **Generation**: Uses Groq's LLM models to generate answers based on retrieved context

## Customization

You can customize the application by:

- Changing embedding models in the code
- Adjusting chunk sizes and overlaps
- Modifying the Groq model (default: llama3-70b-8192)
- Adding additional evaluation methods

## License

MIT# Advance_pdf_rag
