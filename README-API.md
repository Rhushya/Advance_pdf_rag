# PDF Q&A Assistant (API Version)

This is a streamlined version of the PDF Q&A application that processes PDFs locally and uses API calls only for question answering. It allows you to upload PDF documents and ask questions about their content without needing to run tensor operations locally.

## Features

- Upload PDF documents for processing
- Local PDF text extraction
- Ask questions about document content
- Get accurate answers through API-based Q&A
- Multiple API provider options (Groq and Hugging Face)

## Setup

1. Clone this repository
2. Install the requirements: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```
4. Run the application: `streamlit run api_app.py`

## API Options

The application supports two API providers for question answering:

1. **Groq**: High-performance, low-latency option using Llama3-70B
2. **Hugging Face**: Alternative option using Mistral-7B for text generation

You can select your preferred API provider in the sidebar when using the application.

## How It Works

1. **Document Processing**: The application processes PDFs locally using the PyPDF library
2. **Context Retrieval**: When you ask a question, the app finds relevant sections of the document using keyword matching
3. **Question Answering**: The relevant context and question are sent to the selected API for generating an answer
4. **Response Generation**: The system returns the generated answer based on the document context

## Required API Keys

- **Groq API Key**: Used for question answering with Llama3 model
- **Hugging Face API Key**: Used for question answering with Mistral model

You need at least one of these API keys to use the application.

## Advantages

- No need to install tensor libraries (PyTorch, Hugging Face Transformers, etc.)
- Lower memory requirements
- Faster setup and initialization
- Consistent performance regardless of local hardware
- No dependency on external document processing APIs
- Local document processing for privacy

## Limitations

- Requires internet connection for question answering
- Simple keyword-based context retrieval (no vector embeddings)
- Subject to API rate limits for question answering
- Processing time depends on API service response times

## Files in this Project

- `api_app.py`: The main application file
- `requirements.txt`: Contains the required Python packages
- `README-API.md`: This documentation file 