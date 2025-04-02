import os
import PyPDF2
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# **Set Page Configuration**
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("OPENAI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# Custom Chat UI CSS
st.markdown(
    """
    <style>
        .chat-container {
            background-color: #EEF0F4;
            padding: 20px;
            border-radius: 10px;
            max-width: 800px;
            margin: auto;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .chat-message {
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        .user-message {
            background-color: #D1E8FF;
            text-align: right;
        }
        .bot-message {
            background-color: #F5F7FA;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# **Extract Text from PDF**
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# **Store Data in Vector DB**
def store_data_in_vector_db(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local("faiss_index")
    return vector_db

# **Load Stored Vector DB**
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# **Query PDF Data**
def query_pdf(question):
    try:
        vector_db = load_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.get_relevant_documents(question)
        
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Context: {context}\nQuestion: {question}")
        
        return response.text if hasattr(response, "text") else "No valid response from Gemini."
    except Exception as e:
        return f"Error: {e}"

# **Main UI**
st.title("📄 PDF Chatbot")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully!")
    pdf_text = extract_text_from_pdf(uploaded_file)
    if pdf_text.strip():
        store_data_in_vector_db(pdf_text)
        st.write(f"✅ Stored {len(pdf_text.split())} words in Vector DB")
    else:
        st.error("No text extracted from PDF. Try another file.")

# **Initialize Chat History**
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# **Chat UI**
st.markdown("### 💬 Chat with your PDF")
with st.container():
    for chat in st.session_state.chat_history:
        role_class = "user-message" if chat["role"] == "user" else "bot-message"
        st.markdown(f'<div class="chat-container {role_class}"><b>{chat["role"].capitalize()}:</b> {chat["message"]}</div>', unsafe_allow_html=True)

# **User Input**
user_input = st.text_input("Ask a question:", value=st.session_state.get("user_query", ""))

# **Send Button Logic**
if st.button("Send") and user_input.strip():
    # Append user message to chat history
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    
    # Get the response from PDF or external sources
    response = query_pdf(user_input)
    
    # Append bot response to chat history
    st.session_state.chat_history.append({"role": "bot", "message": response})
    
    # Update session state with the new user query value and reset input
    st.session_state.user_query = ""  # Clear the user input session state value
    
    # Re-render the app to reset the input field without clearing chat history
    st.rerun()
