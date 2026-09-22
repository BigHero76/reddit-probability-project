"""
Step 3: Shannon entropy of each subreddit's word-frequency distribution,
plus top-10 distinctive words (by TF-IDF weight) for tooltips.
"""
import json
import numpy as np
import scipy.sparse as sp

OUT_DIR = "/home/claude/pipeline/output"

tfidf_matrix = sp.load_npz(f"{OUT_DIR}/tfidf_matrix.npz")
with open(f"{OUT_DIR}/subreddit_meta.json") as f:
    meta = json.load(f)
with open(f"{OUT_DIR}/vocab.json") as f:
    vocab = np.array(json.load(f))

tfidf_dense_row = tfidf_matrix.toarray()  # 1013 x 6000, fine at this size

entropy_data = {}
for i, m in enumerate(meta):
    row = tfidf_dense_row[i]
    # normalize row to a probability distribution over vocab
    total = row.sum()
    if total == 0:
        entropy_data[m["subreddit"]] = {"entropy": 0.0, "top_words": []}
        continue
    p = row / total
    p_nonzero = p[p > 0]
    entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))

    top_idx = np.argsort(row)[-10:][::-1]
    top_words = [vocab[j] for j in top_idx if row[j] > 0]

    entropy_data[m["subreddit"]] = {
        "entropy": round(entropy, 4),
        "top_words": top_words,
    }

entropies = [v["entropy"] for v in entropy_data.values()]
print(f"Entropy range: {min(entropies):.2f} - {max(entropies):.2f}, mean {np.mean(entropies):.2f}")

with open(f"{OUT_DIR}/entropy.json", "w") as f:
    json.dump(entropy_data, f)

print("Saved entropy.json")
