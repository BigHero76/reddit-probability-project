"""
Step 1: Load raw posts, aggregate text per subreddit, build TF-IDF matrix.
Output: tfidf_matrix.npz, vocab.json, subreddit_meta.json
"""
import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

DATA_PATH = "/mnt/user-data/uploads/reddit_sample_1.csv"
OUT_DIR = "/home/claude/pipeline/output"

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Basic cleanup: reddit-specific markup like <lb> (line break tokens) and URLs
def clean_text(t):
    t = str(t)
    t = re.sub(r"<lb>|<tab>", " ", t)
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

df["clean_text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).apply(clean_text)

# Aggregate: one "document" per subreddit = concatenation of its 50 posts
print("Aggregating per subreddit...")
agg = df.groupby("subreddit").agg(
    doc=("clean_text", " ".join),
    category_1=("category_1", "first"),
    category_2=("category_2", "first"),
    post_count=("clean_text", "count"),
).reset_index()

# also keep true post_count from full dataset assumption (each subreddit=50 sampled here,
# real weight should reflect original size; since sample is uniform 50/subreddit we
# don't have true relative popularity, so 'size' in the viz will just be uniform
# unless we later join back original subreddit post totals)
print(f"{len(agg)} subreddits aggregated")

print("Building TF-IDF matrix...")
vectorizer = TfidfVectorizer(
    max_features=6000,
    min_df=3,          # word must appear in at least 3 subreddit-docs
    max_df=0.6,         # drop words in >60% of docs (too generic)
    stop_words="english",
    ngram_range=(1, 1),
)
tfidf_matrix = vectorizer.fit_transform(agg["doc"])
print("TF-IDF shape:", tfidf_matrix.shape)

sp.save_npz(f"{OUT_DIR}/tfidf_matrix.npz", tfidf_matrix)
agg[["subreddit", "category_1", "category_2", "post_count"]].to_json(
    f"{OUT_DIR}/subreddit_meta.json", orient="records"
)
with open(f"{OUT_DIR}/vocab.json", "w") as f:
    json.dump(vectorizer.get_feature_names_out().tolist(), f)

print("Saved tfidf_matrix.npz, subreddit_meta.json, vocab.json")
