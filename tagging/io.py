"""Data loading and tag parsing utilities."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "question-tags-sample-data.csv"


class TaggedQuestion(NamedTuple):
    question: str
    tags: list[str]


def parse_tags(raw: str) -> list[str]:
    """Convert colon-delimited tag string like ':a:b:c:' into ['a', 'b', 'c']."""
    return [t for t in raw.strip().split(":") if t]


def load_dataframe(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV and return a DataFrame with columns [question, tags_raw, tags].

    ``tags`` is a list[str] column with parsed tag lists.
    """
    df = pd.read_csv(path, quotechar='"', quoting=csv.QUOTE_MINIMAL)
    df = df.rename(columns={"tags": "tags_raw"})
    df["tags"] = df["tags_raw"].apply(parse_tags)
    return df


def load_tagged_questions(path: Path = DATA_PATH) -> list[TaggedQuestion]:
    """Load the CSV as a list of TaggedQuestion tuples."""
    df = load_dataframe(path)
    return [TaggedQuestion(row.question, row.tags) for row in df.itertuples(index=False)]
