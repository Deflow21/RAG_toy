# index_embeddings.py
"""
Строит FAISS-индекс по JSON-описаниям
"""
import json, glob, numpy as np, faiss, pathlib
from sentence_transformers import SentenceTransformer
from config import JSON_DIR, INDEX_DIR

INDEX_DIR.mkdir(parents=True, exist_ok=True)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
records, vectors = [], []

for jfile in glob.glob(str(JSON_DIR/"*.json")):
    doc = json.load(open(jfile, encoding="utf-8"))
    text = (
        " ".join(f"{f['kind']} {f['params']}" for f in doc["features"]) + ". " +
        " ".join(f"{o['type']} {o['params']}" for o in doc["ops"])
    )
    vec = model.encode(text)
    records.append(doc["id"])
    vectors.append(vec)

X = np.vstack(vectors).astype("float32")
index = faiss.IndexFlatL2(X.shape[1])
index.add(X)
faiss.write_index(index, INDEX_DIR/"abc.index")
(pathlib.Path(INDEX_DIR)/"ids.json").write_text(json.dumps(records))
print(f"Indexed {len(records)} parts -> {INDEX_DIR}")
