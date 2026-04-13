#!/usr/bin/env python
"""End-to-end multi-label tag prediction for questions.

Run:
    python main.py              # full pipeline: explore → train → evaluate → demo
    python main.py --predict    # interactive prediction mode (loads saved model)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tagging.io import load_dataframe
from tagging.explore import print_summary
from tagging.pipeline import TagPredictor

MODEL_PATH = Path("models/tag_predictor.pkl")

DEMO_QUESTIONS = [
    "What is the cost per semester?",
    "How do I apply for financial aid?",
    "Can I transfer credits from a community college?",
    "What are the admission requirements for international students?",
    "How do I reset my campus email password?",
    "When is the deadline to register for spring classes?",
    "Is there parking available on campus?",
    "What GPA do I need to get into the nursing program?",
]


def run_explore(df):
    """Phase 1: Data exploration."""
    print("\n--- PHASE 1: DATA EXPLORATION ---\n")
    print_summary(df)
    print()

    print("Sample questions and their tags:")
    for _, row in df.head(5).iterrows():
        print(f"  Q: {row['question']}")
        print(f"     Tags: {row['tags']}")
        print()


def run_train(df) -> TagPredictor:
    """Phase 2 & 3: Train and evaluate."""
    print("\n--- PHASE 2: MODEL TRAINING ---\n")

    model = TagPredictor(
        min_tag_freq=5,
        max_features=20_000,
        ngram_range=(1, 2),
        C=5.0,
        threshold=0.3,
    )

    questions = df["question"].tolist()
    tag_lists = df["tags"].tolist()

    metrics = model.fit(questions, tag_lists, test_size=0.2, verbose=True)

    model.save(MODEL_PATH)
    return model


def run_demo(model: TagPredictor) -> None:
    """Phase 4: Demonstrate predictions on new questions."""
    print("\n--- PHASE 3: PREDICTIONS ON NEW QUESTIONS ---\n")
    print("=" * 70)

    for q in DEMO_QUESTIONS:
        tags = model.predict(q)[0]
        scored = model.predict_with_scores(q)
        print(f"  Q: {q}")
        print(f"     Predicted tags: {tags}")
        print(f"     Scores:         {scored}")
        print()

    print("=" * 70)


def run_interactive(model: TagPredictor) -> None:
    """Interactive loop: type a question, get tag predictions."""
    print("\n--- INTERACTIVE PREDICTION MODE ---")
    print("    Type a question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            q = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q or q.lower() in ("quit", "exit", "q"):
            break

        tags = model.predict(q)[0]
        scored = model.predict_with_scores(q)
        print(f"  Predicted tags: {tags}")
        print(f"  Scores:         {scored}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Question tag predictor")
    parser.add_argument(
        "--predict", action="store_true",
        help="Launch interactive prediction mode (loads saved model)",
    )
    args = parser.parse_args()

    if args.predict:
        if not MODEL_PATH.exists():
            print(f"No saved model at {MODEL_PATH}. Run without --predict first to train.")
            sys.exit(1)
        model = TagPredictor.load(MODEL_PATH)
        run_interactive(model)
        return

    df = load_dataframe()

    run_explore(df)
    model = run_train(df)
    run_demo(model)

    print("\nDone! Model saved to", MODEL_PATH)
    print("Run `python main.py --predict` for interactive predictions.")


if __name__ == "__main__":
    main()
