import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel
import google.generativeai as genai

from config import INDEX_DIR, GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
model = GenerativeModel("models/gemini-2.5-flash")

# Load FAISS index
index = faiss.read_index(f"{INDEX_DIR}/index.faiss")

# Load chunks + metadata
chunks = json.load(open(f"{INDEX_DIR}/chunks.json", "r", encoding="utf-8"))
metadata = json.load(open(f"{INDEX_DIR}/metadata.json", "r", encoding="utf-8"))

def retrieve(query, top_k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    distances, idxs = index.search(query_vec, top_k)

    results = []
    for i, index_id in enumerate(idxs[0]):
        results.append({
            "chunk": chunks[index_id],
            "meta": metadata[index_id],
            "score": float(distances[0][i])
        })
    return results


def build_prompt(query, retrieved):
    context = ""
    for r in retrieved:
        file = r["meta"]["file"]
        page = r["meta"]["page"]
        context += f"\n[Source: {file} | Page: {page}]\n{r['chunk']}\n"

    return f"""
You are a RAG chatbot. Use ONLY the context below to answer.

QUESTION:
{query}

CONTEXT:
{context}

FINAL ANSWER:
"""


def chat():
    print("\n🤖 KnowItBot Pro — Multi-PDF RAG Chatbot")
    print("Ask anything about your uploaded documents.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("🧑 You: ")
        if query.lower() == "exit":
            break

        print("🤖 Bot is thinking...\n")

        retrieved = retrieve(query)
        prompt = build_prompt(query, retrieved)

        response = model.generate_content(prompt)

        print("💬 ANSWER:\n")
        print(response.text)
        print("\n📚 SOURCES USED:\n")

        for r in retrieved:
            print(f"• {r['meta']['file']} — page {r['meta']['page']} (chunk {r['meta']['chunk_id']})")

        print("\n----------------------------------------\n")


if __name__ == "__main__":
    chat()
