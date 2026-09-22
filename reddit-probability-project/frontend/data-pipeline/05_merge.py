"""
Step 5: Merge nodes + entropy + stationary distribution into one
frontend-ready file. Transitions stay separate (edges, loaded on demand
by the Hopper page rather than the initial map load).
"""
import json

OUT_DIR = "/home/claude/pipeline/output"

with open(f"{OUT_DIR}/nodes.json") as f:
    node_data = json.load(f)
with open(f"{OUT_DIR}/entropy.json") as f:
    entropy_data = json.load(f)
with open(f"{OUT_DIR}/stationary.json") as f:
    stationary_data = json.load(f)

for node in node_data["nodes"]:
    sub = node["subreddit"]
    ent = entropy_data.get(sub, {"entropy": None, "top_words": []})
    node["entropy"] = ent["entropy"]
    node["top_words"] = ent["top_words"]
    node["stationary_prob"] = stationary_data.get(sub, 0.0)

# normalize stationary_prob into a 0-1 "size" field for viz (min-max scale)
probs = [n["stationary_prob"] for n in node_data["nodes"]]
lo, hi = min(probs), max(probs)
for node in node_data["nodes"]:
    node["size"] = (node["stationary_prob"] - lo) / (hi - lo) if hi > lo else 0.5

with open(f"{OUT_DIR}/map.json", "w") as f:
    json.dump(node_data, f)

print(f"Saved map.json — {len(node_data['nodes'])} nodes, k={node_data['k']}, "
      f"silhouette={node_data['silhouette_score']:.4f}")

# quick sanity check on file sizes for upload/serving feasibility
import os
for fname in ["map.json", "transitions.json"]:
    path = f"{OUT_DIR}/{fname}"
    size_kb = os.path.getsize(path) / 1024
    print(f"{fname}: {size_kb:.1f} KB")
