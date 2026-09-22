"""
Step 4: Build a row-normalized transition matrix from cosine similarity
between subreddit TF-IDF vectors, compute the stationary distribution
(steady-state "gravity well" score), and export a sparse top-k edge list
for path-finding (full 1013x1013 dense graph is unusable for pathfinding/viz).
"""
import json
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

OUT_DIR = "/home/claude/pipeline/output"
TOP_K = 8          # neighbors kept per node for the sparse graph
TEMPERATURE = 0.05  # softmax temperature: lower = more peaked/predictable hops

with open(f"{OUT_DIR}/subreddit_meta.json") as f:
    meta = json.load(f)
subreddits = [m["subreddit"] for m in meta]
n = len(subreddits)

tfidf_matrix = sp.load_npz(f"{OUT_DIR}/tfidf_matrix.npz")

print("Computing pairwise cosine similarity...")
sim = cosine_similarity(tfidf_matrix)  # 1013 x 1013, dense but manageable
np.fill_diagonal(sim, -np.inf)  # no self-loops

print(f"Keeping top-{TOP_K} neighbors per node + softmax...")
edges = []  # (from_idx, to_idx, prob)
trans_rows = []  # for building sparse row-normalized transition matrix

for i in range(n):
    row = sim[i]
    top_idx = np.argpartition(row, -TOP_K)[-TOP_K:]
    top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
    top_scores = row[top_idx] / TEMPERATURE
    top_scores -= top_scores.max()  # numerical stability
    weights = np.exp(top_scores)
    probs = weights / weights.sum()
    for j, p in zip(top_idx, probs):
        edges.append({"from": subreddits[i], "to": subreddits[int(j)], "prob": float(p)})
    trans_rows.append((i, top_idx, probs))

print(f"Built sparse edge list with {len(edges)} edges")

# Build sparse transition matrix for stationary distribution via power iteration
print("Computing stationary distribution via power iteration...")
P = sp.lil_matrix((n, n))
for i, idx, probs in trans_rows:
    for j, p in zip(idx, probs):
        P[i, int(j)] = p
P = P.tocsr()

pi = np.ones(n) / n
for _ in range(200):
    pi_next = pi @ P
    pi_next = pi_next / pi_next.sum()
    if np.abs(pi_next - pi).sum() < 1e-10:
        pi = pi_next
        break
    pi = pi_next

stationary = {subreddits[i]: float(pi[i]) for i in range(n)}
top_gravity_wells = sorted(stationary.items(), key=lambda x: -x[1])[:10]
print("Top 10 'gravity well' subreddits (highest steady-state probability):")
for name, p in top_gravity_wells:
    print(f"  {name}: {p:.5f}")

with open(f"{OUT_DIR}/transitions.json", "w") as f:
    json.dump({"edges": edges, "top_k": TOP_K, "temperature": TEMPERATURE}, f)

with open(f"{OUT_DIR}/stationary.json", "w") as f:
    json.dump(stationary, f)

print("Saved transitions.json, stationary.json")
