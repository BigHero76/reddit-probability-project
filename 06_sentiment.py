import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import beta, norm, entropy
from nltk.sentiment import SentimentIntensityAnalyzer


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"<lb>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Reddit sentiment distributions and KL divergence analysis.")
    parser.add_argument("--input", required=True, help="Path to reddit_sample.csv")
    parser.add_argument("--output", default="output/sentiment_analysis.json")
    args = parser.parse_args()

    print("Loading VADER...")
    analyzer = SentimentIntensityAnalyzer()

    print("Reading dataset...")
    df = pd.read_csv(args.input)

    required_columns = ["subreddit", "title", "selftext"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found.")

    print(f"Posts loaded: {len(df):,}")
    print("Combining title and selftext...")

    df["title"] = df["title"].fillna("").apply(clean_text)
    df["selftext"] = df["selftext"].fillna("").apply(clean_text)
    df["text"] = (df["title"] + " " + df["selftext"]).str.strip()

    print("Calculating sentiment scores...")
    df["sentiment"] = df["text"].apply(lambda text: analyzer.polarity_scores(text)["compound"])

    print("Calculating subreddit distributions...")
    results = {}

    for subreddit, group in df.groupby("subreddit"):
        scores = group["sentiment"].dropna().to_numpy(dtype=float)
        if len(scores) == 0:
            continue

        mean = float(np.mean(scores))
        median = float(np.median(scores))
        std = float(np.std(scores))
        skewness = float(pd.Series(scores).skew()) if len(scores) > 2 else 0.0

        histogram, bin_edges = np.histogram(scores, bins=30, range=(-1, 1))
        histogram = histogram.astype(float) + 1e-10
        histogram /= histogram.sum()

        beta_alpha = None
        beta_beta = None
        transformed = np.clip((scores + 1) / 2, 1e-6, 1 - 1e-6)

        if len(transformed) >= 10:
            try:
                beta_alpha, beta_beta, _, _ = beta.fit(transformed, floc=0, fscale=1)
                beta_alpha = float(beta_alpha)
                beta_beta = float(beta_beta)
            except Exception:
                pass

        normal_mu, normal_sigma = norm.fit(scores)

        results[str(subreddit)] = {
            "post_count": int(len(scores)),
            "mean": mean,
            "median": median,
            "std": std,
            "skewness": skewness,
            "beta": {"alpha": beta_alpha, "beta": beta_beta},
            "normal": {"mu": float(normal_mu), "sigma": float(normal_sigma)},
            "histogram": histogram.tolist(),
            "bin_edges": bin_edges.tolist()
        }

    print("Calculating KL divergence...")
    subreddits = list(results.keys())
    kl_divergence = {}

    for subreddit_a in subreddits:
        distribution_a = np.asarray(results[subreddit_a]["histogram"], dtype=float)
        kl_divergence[subreddit_a] = {}
        for subreddit_b in subreddits:
            if subreddit_a == subreddit_b:
                continue
            distribution_b = np.asarray(results[subreddit_b]["histogram"], dtype=float)
            kl_divergence[subreddit_a][subreddit_b] = float(entropy(distribution_a, distribution_b))

    output = {
        "metadata": {
            "total_posts": int(len(df)),
            "total_subreddits": int(len(results)),
            "sentiment_method": "VADER",
            "sentiment_range": [-1, 1],
            "histogram_bins": 30,
            "text_source": "title + selftext",
            "cleaning": "<lb> markers replaced with spaces"
        },
        "subreddits": results,
        "kl_divergence": kl_divergence
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print("================================")
    print("SENTIMENT ANALYSIS COMPLETE")
    print("================================")
    print(f"Posts analyzed: {len(df):,}")
    print(f"Subreddits analyzed: {len(results):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
