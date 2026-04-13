"""Multi-label tag prediction pipeline.

Architecture
------------
1. **TF-IDF** on question text (unigrams + bigrams, sub-linear TF).
2. **OneVsRestClassifier(LogisticRegression)** — one binary classifier per tag.
3. Tags that appear fewer than ``min_tag_freq`` times are dropped to reduce
   noise and dimensionality.

Trade-offs
----------
* TF-IDF + linear model is fast, interpretable, and strong for short-text
  classification with thousands of labels.
* Dropping rare tags avoids overfitting on tags the model can't learn from
  just 1-2 examples.
* A transformer-based approach (e.g. fine-tuned BERT) would likely improve
  recall on rarer tags but takes significantly more compute.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer


class TagPredictor:
    """End-to-end multi-label tag predictor for questions."""

    def __init__(
        self,
        min_tag_freq: int = 5,
        max_features: int = 20_000,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 5.0,
        threshold: float = 0.3,
    ):
        self.min_tag_freq = min_tag_freq
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.threshold = threshold

        self.mlb = MultiLabelBinarizer()
        self.pipeline: Pipeline | None = None
        self._all_tags: list[str] = []

    # ------------------------------------------------------------------
    # Filtering rare tags
    # ------------------------------------------------------------------
    def _filter_rare_tags(self, tag_lists: list[list[str]]) -> list[list[str]]:
        """Keep only tags that appear >= min_tag_freq across all samples."""
        from collections import Counter

        freq = Counter(t for tags in tag_lists for t in tags)
        keep = {t for t, c in freq.items() if c >= self.min_tag_freq}
        self._all_tags = sorted(keep)
        return [[t for t in tags if t in keep] for tags in tag_lists]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        questions: list[str],
        tag_lists: list[list[str]],
        test_size: float = 0.2,
        random_state: int = 42,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Train the pipeline and return evaluation metrics on a held-out set."""
        filtered_tags = self._filter_rare_tags(tag_lists)

        Y = self.mlb.fit_transform(filtered_tags)
        if verbose:
            print(f"Label matrix shape: {Y.shape}  "
                  f"({Y.shape[1]} tags kept with freq >= {self.min_tag_freq})")

        X_train, X_test, Y_train, Y_test = train_test_split(
            questions, Y, test_size=test_size, random_state=random_state,
        )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=True,
                strip_accents="unicode",
                token_pattern=r"(?u)\b\w[\w\-]+\b",  # keep hyphenated words
            )),
            ("clf", OneVsRestClassifier(
                LogisticRegression(
                    C=self.C,
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                ),
                n_jobs=-1,
            )),
        ])

        self.pipeline.fit(X_train, Y_train)

        metrics = self.evaluate(X_test, Y_test, verbose=verbose)
        return metrics

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        questions: list[str],
        Y_true: np.ndarray,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Compute multi-label metrics on a test set."""
        Y_prob = self.pipeline.decision_function(questions)
        Y_pred = (Y_prob >= self.threshold).astype(int)

        # Ensure at least one tag per question (pick the top-scoring tag)
        for i in range(Y_pred.shape[0]):
            if Y_pred[i].sum() == 0:
                Y_pred[i, np.argmax(Y_prob[i])] = 1

        metrics = {
            "hamming_loss": round(hamming_loss(Y_true, Y_pred), 4),
            "f1_micro": round(f1_score(Y_true, Y_pred, average="micro", zero_division=0), 4),
            "f1_macro": round(f1_score(Y_true, Y_pred, average="macro", zero_division=0), 4),
            "f1_samples": round(f1_score(Y_true, Y_pred, average="samples", zero_division=0), 4),
            "precision_micro": round(precision_score(Y_true, Y_pred, average="micro", zero_division=0), 4),
            "recall_micro": round(recall_score(Y_true, Y_pred, average="micro", zero_division=0), 4),
        }

        if verbose:
            print("\n" + "=" * 60)
            print("EVALUATION METRICS")
            print("=" * 60)
            for k, v in metrics.items():
                print(f"  {k:<25s} {v:.4f}")
            print("=" * 60)

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, questions: list[str] | str) -> list[list[str]]:
        """Predict tags for one or more questions."""
        if isinstance(questions, str):
            questions = [questions]

        Y_prob = self.pipeline.decision_function(questions)
        if Y_prob.ndim == 1:
            Y_prob = Y_prob.reshape(1, -1)

        results = []
        for scores in Y_prob:
            mask = scores >= self.threshold
            if not mask.any():
                mask[np.argmax(scores)] = True
            predicted = self.mlb.classes_[mask].tolist()
            # Sort by score descending for nicer output
            idx = np.where(mask)[0]
            order = np.argsort(-scores[idx])
            predicted = [self.mlb.classes_[idx[j]] for j in order]
            results.append(predicted)
        return results

    def predict_with_scores(self, question: str) -> list[tuple[str, float]]:
        """Return (tag, score) pairs sorted by score descending."""
        scores = self.pipeline.decision_function([question])
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        scores = scores[0]
        order = np.argsort(-scores)
        top = [(self.mlb.classes_[i], round(float(scores[i]), 3)) for i in order if scores[i] >= self.threshold]
        if not top:
            best = order[0]
            top = [(self.mlb.classes_[best], round(float(scores[best]), 3))]
        return top

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TagPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)
