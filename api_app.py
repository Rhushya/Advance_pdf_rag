import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import requests
import json
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="centered", 
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📄 PDF Q&A Assistant")
st.markdown("""
Upload a PDF file and ask questions about its content.
This app processes PDFs locally and uses API calls for question answering.
""")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "api_method" not in st.session_state:
    st.session_state.api_method = "groq"

def process_pdf_locally(uploaded_file):
    """Process PDF using local PyPDF library."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        
        # Extract text from PDF using pypdf
        import pypdf
        
        with st.status("Extracting text from PDF...", expanded=True) as status:
            pdf_reader = pypdf.PdfReader(temp_file_path)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                status.update(label=f"Processing page {page_num+1} of {len(pdf_reader.pages)}...", state="running")
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            if not text.strip():
                st.error("Could not extract text from the PDF.")
                return None
            
            status.update(label="Document processed successfully!", state="complete")
            
            return text
    
    except Exception as e:
        st.error(f"Error in processing PDF: {str(e)}")
        return None

def query_with_huggingface(user_query, context):
    """Use Hugging Face Inference API for generating responses."""
    try:
        hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_api_key:
            return "API key not found. Please set your HUGGINGFACE_API_KEY in the .env file."
        
        headers = {
            "Authorization": f"Bearer {hf_api_key}",
            "Content-Type": "application/json"
        }
        
        # Select a good model for the task - using a large language model
        model_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
        # Prepare prompt
        prompt = f"""<s>[INST] You are a helpful assistant that answers questions based on the provided document context.
Context:
{context[:12000]}  # Limiting context size to avoid token limits

Question: {user_query}

Please answer the question based only on the provided context. [/INST]</s>"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.1,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        response = requests.post(model_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            return f"Error from Hugging Face API: {response.text}"
        
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                return result[0]["generated_text"].replace(prompt, "").strip()
            else:
                return str(result[0])
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"].replace(prompt, "").strip()
        else:
            return str(result)
    
    except Exception as e:
        return f"Error querying Hugging Face: {str(e)}"

def query_with_groq(user_query, context):
    """Use Groq API for generating responses."""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "API key not found. Please set your GROQ_API_KEY in the .env file."
        
        headers = {
            'Authorization': f'Bearer {groq_api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'llama3-70b-8192',
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that answers questions based on the provided document context. Be concise and accurate.'
                },
                {
                    'role': 'user',
                    'content': f"Context from document:\n\n{context[:20000]}\n\nQuestion: {user_query}\n\nPlease answer based on the context provided."
                }
            ],
            'temperature': 0.1,
            'max_tokens': 800
        }
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            return f"Error from Groq API: {response.text}"
        
        result = response.json()
        return result.get('choices', [{}])[0].get('message', {}).get('content', "No response from API")
    
    except Exception as e:
        return f"Error querying Groq: {str(e)}"

def find_relevant_context(document_text, query, max_chars=20000):
    """
    Simple keyword-based context retrieval to find the most relevant parts of the text.
    This is a basic implementation that works locally without embeddings.
    """
    if not document_text:
        return ""
    
    # Split document into paragraphs
    paragraphs = [p for p in document_text.split("\n\n") if p.strip()]
    
    # Score paragraphs based on keyword matches
    query_terms = set(query.lower().split())
    scored_paragraphs = []
    
    for para in paragraphs:
        score = 0
        para_lower = para.lower()
        
        # Count exact query matches
        if query.lower() in para_lower:
            score += 10
        
        # Count term matches
        for term in query_terms:
            if term in para_lower:
                score += 1
        
        scored_paragraphs.append((score, para))
    
    # Sort by score in descending order
    scored_paragraphs.sort(reverse=True)
    
    # Build context from highest scoring paragraphs
    context = ""
    for _, para in scored_paragraphs:
        if len(context) + len(para) <= max_chars:
            context += para + "\n\n"
        else:
            break
    
    # If no good matches, return the beginning of the document
    if not context.strip() and document_text:
        context = document_text[:max_chars]
    
    return context

def main():
    # File uploader in sidebar
    with st.sidebar:
        st.header("Document Upload")
        
        # API Selection
        st.subheader("API Selection")
        api_option = st.radio(
            "Select API Provider for Question Answering:",
            ["Groq", "Hugging Face"],
            index=0,
            help="Choose which API to use for question answering"
        )
        
        # Update API method based on selection
        if api_option == "Groq":
            st.session_state.api_method = "groq"
        else:
            st.session_state.api_method = "huggingface"
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        
        if uploaded_file is not None and not st.session_state.file_processed:
            if st.button("Process Document"):
                with st.spinner("Processing PDF... This may take a moment."):
                    # Process the file locally
                    document_text = process_pdf_locally(uploaded_file)
                    if document_text:
                        st.session_state.document_text = document_text
                        st.session_state.file_processed = True
                        st.success(f"File '{uploaded_file.name}' processed successfully!")
        
        # Reset button
        if st.session_state.file_processed:
            if st.button("Process Another Document"):
                st.session_state.document_text = None
                st.session_state.file_processed = False
                st.session_state.messages = []
                st.experimental_rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if st.session_state.file_processed:
        user_query = st.chat_input("Ask a question about the document")
        
        if user_query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Find relevant context for the query
                    relevant_context = find_relevant_context(
                        st.session_state.document_text, 
                        user_query
                    )
                    
                    # Get response from selected API
                    if st.session_state.api_method == "huggingface":
                        response = query_with_huggingface(user_query, relevant_context)
                    else:
                        response = query_with_groq(user_query, relevant_context)
                    
                    # Display response
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Please upload a PDF document to get started.")

if __name__ == "__main__":
    main() 