"""Shim for the missing `loaddatas_LP_arxiv` module. Re-exports from loaddatas.py."""
from loaddatas import get_edges_split, get_adj_split, loaddatas  # noqa: F401
