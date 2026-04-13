"""Exploratory data analysis helpers."""

from __future__ import annotations

from collections import Counter

import pandas as pd


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return key statistics about the dataset."""
    tag_counts = df["tags"].apply(len)
    all_tags = [t for tags in df["tags"] for t in tags]
    tag_freq = Counter(all_tags)

    return {
        "n_questions": len(df),
        "n_unique_tags": len(tag_freq),
        "avg_tags_per_question": round(tag_counts.mean(), 2),
        "median_tags_per_question": int(tag_counts.median()),
        "min_tags": int(tag_counts.min()),
        "max_tags": int(tag_counts.max()),
        "top_20_tags": tag_freq.most_common(20),
        "bottom_20_tags": tag_freq.most_common()[-20:],
    }


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable exploration summary."""
    stats = dataset_summary(df)
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total questions:          {stats['n_questions']}")
    print(f"  Unique tags:              {stats['n_unique_tags']}")
    print(f"  Avg tags per question:    {stats['avg_tags_per_question']}")
    print(f"  Median tags per question: {stats['median_tags_per_question']}")
    print(f"  Min / Max tags:           {stats['min_tags']} / {stats['max_tags']}")
    print()
    print("Top 20 tags:")
    for tag, count in stats["top_20_tags"]:
        print(f"    {tag:<40s} {count:>5d}")
    print()
    print("Bottom 20 (rarest) tags:")
    for tag, count in stats["bottom_20_tags"]:
        print(f"    {tag:<40s} {count:>5d}")
    print()

    ql = df["question"].str.split().apply(len)
    print(f"  Question length (words):  avg={ql.mean():.1f}  median={ql.median():.0f}  "
          f"min={ql.min()}  max={ql.max()}")
    print("=" * 60)
