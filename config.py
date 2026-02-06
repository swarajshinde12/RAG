import os

# === Base project paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder where we'll keep PDFs
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")

# Folder where we'll store FAISS index + metadata
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")

# Create folders if they don't exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# === Gemini models & API key ===
# ⚠️ Put your Gemini API key here (the one you used in Jupyter)
GEMINI_API_KEY = "AIzaSyBQTtFks3LzJe4FFGm9twWLfNemnobxalQ"

# LLM for answering questions
GEMINI_CHAT_MODEL = "models/gemini-2.5-flash"

# Embedding model for vector search
GEMINI_EMBED_MODEL = "models/text-embedding-004"

# === Chunking + Retrieval settings ===
CHUNK_SIZE = 600        # characters per chunk
CHUNK_OVERLAP = 150     # overlapping characters between chunks
TOP_K = 5               # how many chunks to retrieve for each question
