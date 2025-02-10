import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the FAISS index and metadata
index = faiss.read_index("faiss_index.bin")
with open("faiss_metadata.json", "r") as f:
    doc_names = json.load(f)

# Load the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_articles(query, top_k=3):
    """Finds the most relevant articles for a given query."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(doc_names[idx])
    
    return results

# Example query
query = "How to handle stress in parenting?"
top_articles = search_articles(query, top_k=3)

print("\n🔍 Top Matching Articles:")
for article in top_articles:
    print(f"📄 {article}")
