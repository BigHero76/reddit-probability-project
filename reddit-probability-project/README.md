# Reddit Probability Space

A statistical/probability project mapping 1,013 Reddit communities using
TF-IDF, UMAP, clustering, Markov chain transitions, and Shannon entropy.

## Structure

```
reddit-probability-project/
├── data-pipeline/
│   ├── 01_tfidf.py          # aggregate posts per subreddit, build TF-IDF matrix
│   ├── 02_umap_cluster.py   # UMAP 2D projection + KMeans clustering + silhouette
│   ├── 03_entropy.py        # Shannon entropy of vocabulary per subreddit
│   ├── 04_transitions.py    # cosine-similarity transition matrix + stationary distribution
│   ├── 05_merge.py          # merges everything into frontend-ready map.json
│   └── output/
│       ├── map.json           # 1,013 nodes: x, y, cluster, entropy, top words, size
│       ├── transitions.json   # sparse edge list (top-8 neighbors/node) for the Hopper
│       └── cluster_labels.json# dominant category + purity per cluster
└── frontend/
    └── index.html          # landing page — hero, mesh gradient, parallax, feature cards
```

## Re-running the pipeline

Source data: `reddit_sample_1.csv` (stratified sample — 50 posts × 1,013
subreddits from the Kaggle rspct dataset). Run scripts 01 → 05 in order;
each reads the previous step's output from `output/`.

## Frontend

`frontend/index.html` is self-contained (no build step) — open it directly
in a browser. It's the landing page only; the Map and Hopper pages (linked
in the nav) are the next step, built against `map.json` / `transitions.json`.

## Key results

- 1,013 subreddits, 45 emergent clusters, silhouette score 0.50
- Cluster purity vs. human-labeled categories: up to 100% (video games), 95% (TV shows)
- Entropy range: 6.45–10.19 bits across subreddit vocabularies
- 8,104 weighted transitions (top-8 neighbors/node, softmax temperature 0.05)
