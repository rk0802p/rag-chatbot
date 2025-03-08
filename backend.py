import os
import fitz  # PyMuPDF for reading PDFs
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load Hugging Face embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

def create_vector_store(pdf_path, faiss_index_path="faiss_index.pkl"):
    """Process PDF, create vector embeddings, and store in FAISS."""
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    # Convert text chunks into embeddings
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    # Initialize FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index and chunks
    with open(faiss_index_path, "wb") as f:
        pickle.dump((index, chunks), f)

def load_vector_store(faiss_index_path="faiss_index.pkl"):
    """Load FAISS index and text chunks."""
    with open(faiss_index_path, "rb") as f:
        index, chunks = pickle.load(f)
    return index, chunks

def query_pdf(question, faiss_index_path="faiss_index.pkl"):
    """Find the most relevant chunk from FAISS for a user query."""
    index, chunks = load_vector_store(faiss_index_path)
    
    # Convert question into embedding
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    
    # Search FAISS index
    D, I = index.search(question_embedding, k=1)  # Get top 1 result

    return chunks[I[0][0]] if I[0][0] < len(chunks) else "No relevant info found."
