"""Tokenisation helpers shared by the BM25 index and the offline embedder."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers him his how i if in into is it its
    just me more most my no nor not of off on once only or other our out over own same
    she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom
    why will with would you your yours
    """.split()
) | frozenset(
    # question "fluff" that carries no topic signal
    "long take takes taking happen happens happened go goes going get gets getting need needs "
    "want tell know many much please".split()
)

_SUFFIXES = ("ingly", "edly", "ing", "ed", "es", "s")


def stem(token: str) -> str:
    """A deliberately tiny suffix stripper (refunds -> refund, processed -> process)."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str, *, remove_stopwords: bool = True, use_stemming: bool = True) -> list[str]:
    """Lower-case, split on non-alphanumerics, drop stopwords and stem."""
    tokens = _WORD_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if use_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens
