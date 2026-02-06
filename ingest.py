import os
import json
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, PDF_DIR, INDEX_DIR

# Load embedding model (same as before)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
        full_text += text + "\n"

    return pages, full_text


def chunk_text_with_metadata(pages, source_file, chunk_size=400):
    chunks = []
    metadata = []

    for page_num, text in pages:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                metadata.append({
                    "file": source_file,
                    "page": page_num,
                    "chunk_id": len(chunks)
                })
    return chunks, metadata


def ingest_pdfs():
    all_chunks = []
    all_metadata = []

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    print(f"\n📄 Found {len(pdf_files)} PDFs to ingest.\n")

    for pdf_name in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_name)

        print(f"➡ Processing: {pdf_name}")
        pages, full_text = extract_pdf_text(pdf_path)

        chunks, meta = chunk_text_with_metadata(pages, pdf_name)
        all_chunks.extend(chunks)
        all_metadata.extend(meta)

        print(f"   ✔ Extracted {len(chunks)} chunks from {pdf_name}\n")

    print("🔹 Embedding all chunks...")
    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)

    print("🔹 Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

    # Save chunks
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    # Save metadata
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n✅ Ingestion complete!")
    print(f"📌 Total Chunks: {len(all_chunks)}")
    print(f"📌 Metadata saved.")
    print(f"📌 FAISS index saved at: {INDEX_DIR}\n")


if __name__ == "__main__":
    ingest_pdfs()
