import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# === Load legal data from JSON ===
act_name = "Right To Information Act,2005"
act_path = f"./Acts/{act_name}/"
with open(f"{act_path}{act_name}.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Handle both sections and schedules
texts = [
    f"{item.get('section', item.get('schedule'))} - {item['title']}: {item['text']}"
    for item in data
]

# === Load sentence-transformer embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Generate embeddings ===
print("🔄 Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# === Create and populate FAISS index ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === Save the FAISS index and metadata ===
faiss.write_index(index, f"{act_path}{act_name}.index")
with open(f"{act_path}{act_name}.pkl", "wb") as f:
    pickle.dump(data, f)

print(f"✅ Embeddings saved to '{act_name}.index'")
print(f"✅ Metadata saved to '{act_name}.pkl'")
