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

# **Query Data from Supabase**
def query_data(question):
    try:
        question_embedding = embeddings_model.embed_query(question)
        question_embedding_array = np.array(question_embedding).tolist()  # Ensure correct format

        # ✅ Call the stored procedure in Supabase
        result = supabase.rpc("match_documents", {
            "query_embedding": question_embedding_array,
            "match_threshold": 0.5,  # Adjust threshold if needed
            "match_count": 3
        }).execute()

        # ✅ Fix: Check if response has errors
        if isinstance(result, dict) and "error" in result and result["error"]:
            return f"Error retrieving data: {result['error']}"

        # ✅ Extract retrieved text for LLM context
        context = "\n".join([row["text"] for row in result.data])

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Context: {context}\nQuestion: {question}")

        return response.text if hasattr(response, "text") else "No valid response from Gemini."

    except Exception as e:
        return f"Error: {e}"


# **Streamlit UI**
st.title("📄 PDF & Website Chatbot")

option = st.radio("Choose an option:", ("Upload PDF", "Enter Website URL"))

if option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text.strip():
            result = store_data(pdf_text, source=uploaded_file.name)
            st.success(result)
        else:
            st.error("No text extracted from PDF.")

elif option == "Enter Website URL":
    website_url = st.text_input("Enter Website URL:")
    if st.button("Scrape Website") and website_url.strip():
        website_text = scrape_website(website_url)
        if "Error" not in website_text:
            result = store_data(website_text, source=website_url)
            st.success(result)
        else:
            st.error(website_text)

# **Chat UI**
st.markdown("### 💬 Chat with your Data")
user_input = st.text_input("Ask a question:")
if st.button("Send") and user_input.strip():
    response = query_data(user_input)
    st.markdown(f"**Response:** {response}")
