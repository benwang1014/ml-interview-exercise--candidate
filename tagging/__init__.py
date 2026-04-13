"""Question → multi-label tag prediction utilities."""

from tagging.io import load_dataframe, load_tagged_questions, parse_tags
from tagging.pipeline import TagPredictor

__all__ = ["load_dataframe", "load_tagged_questions", "parse_tags", "TagPredictor"]
