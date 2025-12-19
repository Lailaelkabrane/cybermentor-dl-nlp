import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# mêmes chemins que dans app.py
RAG_BASE = "../RAG_project"
KB_PATH = os.path.join(RAG_BASE, "chroma_db")
DOCS_PATH = os.path.join(RAG_BASE, "docs.json")

MODEL = "all-mpnet-base-v2"
print("🔁 Loading embedding model:", MODEL)
model = SentenceTransformer(MODEL)

os.makedirs(KB_PATH, exist_ok=True)

print("📂 Chroma DB path:", KB_PATH)
client = chromadb.PersistentClient(path=KB_PATH)
collection_name = "security_kb"

# Charger docs.json
print("📥 Loading docs.json...")
with open(DOCS_PATH, "r", encoding="utf-8") as f:
    attacks = json.load(f)

texts = []
ids = []
metas = []

for attack in attacks:
    mitigation_texts = []
    for m in attack.get("mitigation_recommendations", []):
        mitigation_texts.append(
            f"Recommendation: {m['recommendation']}. "
            f"Implementation: {m['implementation']}. "
            f"Priority: {m['priority']}. "
            f"Source: {m['source']} ({m['source_detail']})"
        )
    text = (
        f"Attack ID: {attack['id']}\n"
        f"Type: {attack['attack_type']}\n"
        f"Description: {attack['description']}\n"
        f"Indicators: {', '.join(attack.get('indicators', []))}\n"
        f"Mitigation Recommendations:\n" + "\n".join(mitigation_texts) + "\n"
        f"Defense Layers: {', '.join(attack.get('defense_layers', []))}\n"
        f"Source: {attack.get('source', '')}"
    )
    texts.append(text)
    ids.append(attack["id"])
    metas.append({
        "source": attack.get("source", ""),
        "attack_type": attack.get("attack_type", "")
    })

print(f"🧮 Encoding {len(texts)} documents...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

# Créer ou récupérer la collection
try:
    coll = client.get_collection(collection_name)
    print("ℹ️ Collection already exists, adding data to it...")
except:
    coll = client.create_collection(name=collection_name)
    print("✅ Collection created:", collection_name)

coll.add(ids=ids, documents=texts, metadatas=metas, embeddings=embeddings)

print("🎉 Indexed", len(ids), "documents in collection", collection_name)
