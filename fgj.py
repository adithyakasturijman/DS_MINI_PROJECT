import os
import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()
GENAI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GENAI_API_KEY)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


embedding_dim = embedding_model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(embedding_dim)
document_chunks = [] 


def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def embed_and_store(text_chunks):
    global faiss_index, document_chunks
    document_chunks = text_chunks 
    embeddings = embedding_model.encode(text_chunks)
    faiss_index.add(np.array(embeddings, dtype=np.float32))


def retrieve_relevant_chunks(query, top_k=3):
    if faiss_index.ntotal == 0:
        return ""
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(np.array(query_embedding, dtype=np.float32), top_k)
    return "\n\n".join([document_chunks[i] for i in indices[0] if i < len(document_chunks)])


def generate_answer_with_rag(query):
    retrieved_text = retrieve_relevant_chunks(query)
    
    if not retrieved_text:
        return "No relevant information found in the document."

    prompt = (
        f"You are an AI assistant using Retrieval-Augmented Generation (RAG) to answer questions.\n\n"
        f"Document Context:\n{retrieved_text}\n\n"
        f"Question: {query}\n\n"
        f"Provide a detailed yet concise answer based only on the document context."
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip() if response and hasattr(response, "text") else "No response from AI."
    except Exception as e:
        return f"Error: {e}"


st.title("📄 RAG-based PDF Q&A with Gemini AI")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting and embedding text..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        text_chunks = chunk_text(pdf_text)
        embed_and_store(text_chunks)
    st.success("✅ PDF processed and embedded into FAISS!")

query = st.text_input("💬 Ask a question about the document:")

if query:
    answer = generate_answer_with_rag(query)
    st.write("### 📝 Answer:")
    st.write(answer)
