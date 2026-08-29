"""Tokenisation helpers shared by the index, the parser and the ranker.

Everything here is pure Python / standard library: no network, no model calls.
"""

from __future__ import annotations

import math
import re

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stopwords for retrieval. Kept deliberately small: the simulator quotes product
# copy almost verbatim, so most content words carry signal.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can do for from get has have i if in is it
    its just like looking me my no not of on or please prefer preference really
    should so some that the their then there these they this to too use used
    very want was we what when which will with would you your
    """.split()
)

# Filler that shows up in the simulator's canned sentences and should never end
# up in a retrieval query.
CHATTER = frozenset(
    """
    actually additional ask attribute earlier exploring hello hey hi ignore
    judgment key matters need okay one options prefer preference quite
    requirement right specific still thanks those yet
    """.split()
)


def normalize(value: object) -> str:
    """Flatten an arbitrary catalog value into a single lowercase string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items()).lower()
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value).lower()
    return str(value).lower()


def tokens(text: str) -> list[str]:
    """Content tokens, in order, with stopwords and 1-character noise removed."""
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def query_tokens(text: str) -> list[str]:
    """Tokens suitable for a retrieval query (also drops simulator filler)."""
    return [token for token in tokens(text) if token not in CHATTER]


def idf(document_frequency: int, total_documents: int) -> float:
    """Smoothed inverse document frequency; unseen terms get the maximum value."""
    if document_frequency <= 0:
        document_frequency = 1
    return math.log(1.0 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
