"""
Step 2: Project TF-IDF vectors to 2D with UMAP, cluster with KMeans,
score with silhouette. Output: nodes.json (the map data).
"""
import json
import numpy as np
import scipy.sparse as sp
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT_DIR = "/home/claude/pipeline/output"

print("Loading TF-IDF matrix + metadata...")
tfidf_matrix = sp.load_npz(f"{OUT_DIR}/tfidf_matrix.npz")
with open(f"{OUT_DIR}/subreddit_meta.json") as f:
    meta = json.load(f)

print("Running UMAP (this can take a minute)...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=42,
)
embedding = reducer.fit_transform(tfidf_matrix)
print("Embedding shape:", embedding.shape)

# Try a small range of K and pick best silhouette score
print("Selecting K via silhouette score...")
best_k, best_score, best_labels = None, -1, None
for k in [15, 20, 25, 30, 39, 45]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embedding)
    score = silhouette_score(embedding, labels)
    print(f"  k={k}: silhouette={score:.4f}")
    if score > best_score:
        best_k, best_score, best_labels = k, score, labels

print(f"Best K = {best_k} (silhouette={best_score:.4f})")

nodes = []
for i, m in enumerate(meta):
    nodes.append({
        "subreddit": m["subreddit"],
        "category_1": m["category_1"],
        "category_2": m["category_2"],
        "x": float(embedding[i, 0]),
        "y": float(embedding[i, 1]),
        "cluster": int(best_labels[i]),
    })

with open(f"{OUT_DIR}/nodes.json", "w") as f:
    json.dump({
        "nodes": nodes,
        "k": best_k,
        "silhouette_score": best_score,
    }, f)

print(f"Saved nodes.json with {len(nodes)} nodes")
