import os
import json
import PyPDF2
import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import requests
import numpy as np
from bs4 import BeautifulSoup
from supabase import create_client
import datetime
import uuid

# ✅ Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ✅ Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ Initialize Gemini & Embeddings model
genai.configure(api_key=GEMINI_API_KEY)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize session state for chats if not already done
if 'chats' not in st.session_state:
    st.session_state['chats'] = {}  # Dictionary to store all chats

if 'current_chat_id' not in st.session_state:
    st.session_state['current_chat_id'] = None

if 'chat_messages' not in st.session_state:
    st.session_state['chat_messages'] = []  # List to store current chat messages

# Keep track of source document content for each chat
if 'source_contents' not in st.session_state:
    st.session_state['source_contents'] = {}

# **Extract Text from PDF**
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# **Scrape Website Data**
def scrape_website(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        
        # Debugging: Print the response status code
        if response.status_code != 200:
            return f"Error: Received status code {response.status_code}"

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator=" ")
        return " ".join(text.split())  # Clean and format text

    except requests.exceptions.RequestException as e:
        return f"Request Exception: {str(e)}"
    except Exception as e:
        return f"General Exception: {str(e)}"

# **Store Extracted Text and Embeddings in Supabase**
def store_data(text, source):
    """Store extracted text and embeddings in Supabase."""
    # Store the full text in session state for context
    if st.session_state['current_chat_id']:
        st.session_state['source_contents'][st.session_state['current_chat_id']] = text
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(text)

    for doc in docs:
        embedding = embeddings_model.embed_query(doc)
        embedding_array = np.array(embedding).tolist()  # Ensure correct format

        try:
            response = supabase.table("documents").insert({
                "source": source,
                "text": doc,
                "embedding": embedding_array  # Ensure it's a valid JSON format
            }).execute()

            # ✅ Fix: Ensure the response is checked properly
            if "error" in response and response["error"]:
                return f"Error storing data: {response['error']}"

        except Exception as e:
            return f"Exception: {str(e)}"

    return "✅ Data stored successfully!"

# **Query Data from Supabase with conversation history context**
def query_data(question):
    try:
        # Get current chat ID
        chat_id = st.session_state['current_chat_id']
        if not chat_id:
            return "No active chat session. Please upload a document or select a chat first."
            
        # Get previous messages for context
        previous_messages = []
        current_messages = st.session_state['chat_messages']
        
        # Limit to last 5 exchanges to avoid context length issues
        for msg in current_messages[-10:]:
            if msg['role'] == 'user':
                previous_messages.append(f"User: {msg['content']}")
            else:
                previous_messages.append(f"AI: {msg['content']}")
        
        conversation_history = "\n".join(previous_messages)
        
        # Get semantic search results
        question_embedding = embeddings_model.embed_query(question)
        question_embedding_array = np.array(question_embedding).tolist()

        # ✅ Call the stored procedure in Supabase
        result = supabase.rpc("match_documents", {
            "query_embedding": question_embedding_array,
            "match_threshold": 0.5,
            "match_count": 5  # Increased to get more context
        }).execute()

        # ✅ Check if response has errors
        if isinstance(result, dict) and "error" in result and result["error"]:
            return f"Error retrieving data: {result['error']}"

        # ✅ Extract retrieved text for LLM context
        document_context = "\n".join([row["text"] for row in result.data])
        
        # Build the prompt with conversation history and document context
        prompt = f"""
        You are an assistant that answers questions based on specific documents.
        
        DOCUMENT CONTEXT:
        {document_context}
        
        CONVERSATION HISTORY:
        {conversation_history}
        
        CURRENT QUESTION: {question}
        
        Answer the current question based on the document context and conversation history.
        If the question refers to elements from previous conversation, use that context.
        If you don't have enough information to answer accurately, say so.
        """
        
        # Generate response
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        return response.text if hasattr(response, "text") else "No valid response from Gemini."

    except Exception as e:
        return f"Error: {e}"

# Create a new chat
def create_new_chat(source_name, source_content=None):
    chat_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state['chats'][chat_id] = {
        'source': source_name,
        'timestamp': timestamp,
        'messages': []
    }
    st.session_state['current_chat_id'] = chat_id
    st.session_state['chat_messages'] = []  # Reset current chat messages
    
    # Store source content if provided
    if source_content:
        st.session_state['source_contents'][chat_id] = source_content
        
    return chat_id

# Switch to a selected chat
def switch_chat(chat_id):
    if chat_id in st.session_state['chats']:
        st.session_state['current_chat_id'] = chat_id
        st.session_state['chat_messages'] = st.session_state['chats'][chat_id]['messages']
        return True
    return False

# Add message to current chat
def add_message(role, content):
    message = {'role': role, 'content': content, 'timestamp': datetime.datetime.now().strftime("%H:%M:%S")}
    st.session_state['chat_messages'].append(message)
    
    # Also update the message in the chats dictionary
    if st.session_state['current_chat_id']:
        st.session_state['chats'][st.session_state['current_chat_id']]['messages'] = st.session_state['chat_messages']

# Sidebar for chat history
def render_sidebar():
    with st.sidebar:
        st.header("💬 Chat History")
        
        # Button to start a new chat (without a source - user can upload later)
        if st.button("New Empty Chat"):
            create_new_chat("New Chat")
            st.rerun()
            
        # Add a button to delete the current chat
        if st.session_state['current_chat_id'] and st.button("🗑️ Delete Current Chat"):
            chat_id = st.session_state['current_chat_id']
            if chat_id in st.session_state['chats']:
                del st.session_state['chats'][chat_id]
                if chat_id in st.session_state['source_contents']:
                    del st.session_state['source_contents'][chat_id]
                st.session_state['current_chat_id'] = None
                st.session_state['chat_messages'] = []
                st.rerun()
        
        st.divider()
        
        # Display all available chats
        if st.session_state['chats']:
            for chat_id, chat_data in sorted(
                st.session_state['chats'].items(),
                key=lambda x: x[1]['timestamp'],
                reverse=True
            ):
                source = chat_data['source']
                timestamp = chat_data['timestamp']
                messages_count = len(chat_data['messages'])
                
                # Create a button for each chat with message count
                if st.button(f"{source} ({messages_count} msgs) - {timestamp}", key=chat_id):
                    switch_chat(chat_id)
                    st.rerun()
        else:
            st.info("No chat history yet. Upload a PDF or enter a website URL to start.")

# Main UI
def main():
    render_sidebar()
    
    st.title("📄 PDF & Website Chatbot")
    
    # Show current document being queried
    if st.session_state['current_chat_id']:
        current_source = st.session_state['chats'][st.session_state['current_chat_id']]['source']
        st.info(f"Current document: {current_source}")
    
    # Input options
    option = st.radio("Choose an option:", ("Upload PDF", "Enter Website URL"))
    
    if option == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file:
            if st.button("Process PDF"):
                pdf_text = extract_text_from_pdf(uploaded_file)
                if pdf_text.strip():
                    # Create a new chat for this PDF
                    create_new_chat(uploaded_file.name, pdf_text)
                    result = store_data(pdf_text, source=uploaded_file.name)
                    st.success(result)
                    st.rerun()  # Refresh the page
                else:
                    st.error("No text extracted from PDF.")
    
    elif option == "Enter Website URL":
        website_url = st.text_input("Enter Website URL:")
        if st.button("Scrape Website") and website_url.strip():
            website_text = scrape_website(website_url)
            if "Error" not in website_text:
                # Create a new chat for this website
                create_new_chat(website_url, website_text)
                result = store_data(website_text, source=website_url)
                st.success(result)
                st.rerun()  # Refresh the page
            else:
                st.error(website_text)
    
    # Display chat messages
    st.divider()
    st.markdown("### 💬 Chat with your Data")
    
    # Display current chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state['chat_messages']:
            if message['role'] == 'user':
                st.markdown(f"**User:** {message['content']} *({message['timestamp']})*")
            else:
                st.markdown(f"**AI:** {message['content']} *({message['timestamp']})*")
    
    # Chat input
    user_input = st.text_input("Ask a question:")
    if st.button("Send") and user_input.strip():
        # Check if we have an active chat
        if not st.session_state['current_chat_id']:
            st.warning("Please upload a document or select a chat first.")
        else:
            # Add user message
            add_message('user', user_input)
            
            # Get response from Gemini
            response = query_data(user_input)
            
            # Add AI response
            add_message('ai', response)
            
            # Rerun to update the UI
            st.rerun()

if __name__ == "__main__":
    main()