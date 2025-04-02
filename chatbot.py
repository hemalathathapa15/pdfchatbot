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
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes

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

# **Extract Text from PDF (including images using OCR)**
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    # OCR for scanned PDFs
    images = convert_from_bytes(pdf_file.read())
    ocr_text = "\n".join([pytesseract.image_to_string(image) for image in images])
    
    return text + "\n" + ocr_text

# **Scrape Website Data (including OCR for images)**
def scrape_website(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if response.status_code != 200:
            return f"Error: Received status code {response.status_code}"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text
        for script in soup(["script", "style", "noscript"]):
            script.extract()
        text = soup.get_text(separator=" ")
        cleaned_text = " ".join(text.split())
        
        # OCR for images
        img_text = ""
        for img_tag in soup.find_all("img"):
            img_url = img_tag.get("src")
            if img_url:
                try:
                    img_response = requests.get(img_url, stream=True)
                    image = Image.open(img_response.raw)
                    img_text += pytesseract.image_to_string(image) + "\n"
                except Exception as e:
                    img_text += f"[Failed to process image: {img_url}]\n"
        
        return cleaned_text + "\n" + img_text
    except Exception as e:
        return f"Exception: {str(e)}"

# Store extracted text & embeddings
def store_data(text, source):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(text)
    
    for doc in docs:
        embedding = embeddings_model.embed_query(doc)
        embedding_array = np.array(embedding).tolist()
        
        try:
            supabase.table("documents").insert({
                "source": source,
                "text": doc,
                "embedding": embedding_array
            }).execute()
        except Exception as e:
            return f"Error storing data: {str(e)}"
    return "✅ Data stored successfully!"

# Query Data
def query_data(question):
    try:
        chat_id = st.session_state.get('current_chat_id')
        if not chat_id:
            return "No active chat session."
        
        question_embedding = embeddings_model.embed_query(question)
        question_embedding_array = np.array(question_embedding).tolist()
        
        result = supabase.rpc("match_documents", {
            "query_embedding": question_embedding_array,
            "match_threshold": 0.5,
            "match_count": 5
        }).execute()
        
        document_context = "\n".join([row["text"] for row in result.data])
        
        prompt = f"""
        DOCUMENT CONTEXT:
        {document_context}
        
        CURRENT QUESTION: {question}
        
        Answer based on the document context.
        """
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else "No valid response."
    except Exception as e:
        return f"Error: {e}"

# Main UI
def main():
    st.sidebar.title("💬 Chat History")
    
    option = st.radio("Choose an option:", ("Upload PDF", "Enter Website URL"))
    
    if option == "Upload PDF":
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file and st.button("Process PDF"):
            pdf_text = extract_text_from_pdf(uploaded_file)
            if pdf_text.strip():
                store_data(pdf_text, source=uploaded_file.name)
                st.success("PDF processed successfully!")
            else:
                st.error("No text extracted.")
    
    elif option == "Enter Website URL":
        website_url = st.text_input("Enter Website URL:")
        if st.button("Scrape Website") and website_url.strip():
            website_text = scrape_website(website_url)
            if "Error" not in website_text:
                store_data(website_text, source=website_url)
                st.success("Website processed successfully!")
            else:
                st.error(website_text)
    
    user_input = st.text_input("Ask a question:")
    if st.button("Send") and user_input.strip():
        response = query_data(user_input)
        st.write("**AI:**", response)

if __name__ == "__main__":
    main()
