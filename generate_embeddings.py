import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load extracted text
with open("pdf_texts.json", "r") as f:
    pdf_texts = json.load(f)

# Load the Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text into embeddings
documents = list(pdf_texts.values())
embeddings = model.encode(documents, convert_to_numpy=True)

# Convert embeddings into FAISS index
d = embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(embeddings)  # Add vectors to FAISS

# Save the FAISS index
faiss.write_index(index, "faiss_index.bin")

# Save document mappings
with open("faiss_metadata.json", "w") as f:
    json.dump(list(pdf_texts.keys()), f)

print("✅ Embeddings saved in FAISS index.")
