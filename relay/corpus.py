"""Corpus: PDF text extraction, chunking, and a BM25 index with optional rerank.

The retrieval tool is deliberately lexical (BM25), not embeddings: it is
deterministic, needs no extra service, and makes the cost of each retrieval
setting directly measurable (more chunks = more prompt tokens).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pymupdf

from relay.config import PAPERS

PAPERS_DIR = Path(__file__).resolve().parents[1] / "papers"

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "the a an of to in and or is are was were be been it its this that these those "
    "we our you your they their as at by for from with on not no can do does how "
    "what which who whom why when where than then there here into over under".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def est_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — used for TPM accounting."""
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper: str
    page: int
    text: str

    @property
    def cite(self) -> str:
        return f"[{self.paper} p.{self.page}]"


def extract_pages(pdf: Path) -> list[tuple[int, str]]:
    doc = pymupdf.open(str(pdf))
    pages = []
    for i, page in enumerate(doc, start=1):
        text = re.sub(r"[ \t]+", " ", page.get_text())
        text = re.sub(r"\n{2,}", "\n", text).strip()
        if text:
            pages.append((i, text))
    doc.close()
    return pages


def chunk_pages(chunk_chars: int, overlap: int) -> list[Chunk]:
    """Split each page into fixed-character windows with optional overlap."""
    chunks: list[Chunk] = []
    for paper in PAPERS:
        for page_no, text in _cached_pages(paper["file"]):
            step = max(1, chunk_chars - overlap)
            for start in range(0, max(1, len(text)), step):
                piece = text[start:start + chunk_chars].strip()
                if len(piece) < 120:
                    continue
                cid = f"{paper['key']}:p{page_no}:{start}"
                chunks.append(Chunk(cid, paper["key"], page_no, piece))
    return chunks


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokens(c.text) for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / len(self.docs)) if self.docs else 1.0
        self.df: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.docs)
        self.tf: list[dict[str, int]] = []
        for d in self.docs:
            counts: dict[str, int] = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)

    def _idf(self, t: str) -> float:
        n = self.df.get(t, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query: str) -> list[float]:
        q = tokens(query)
        out = []
        for i in range(self.N):
            s = 0.0
            dl = self.doc_len[i] or 1
            for t in q:
                f = self.tf[i].get(t, 0)
                if not f:
                    continue
                s += self._idf(t) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            out.append(s)
        return out

    def search(self, query: str, top_k: int, rerank: bool = False) -> list[tuple[Chunk, float]]:
        base = self.scores(query)
        order = sorted(range(self.N), key=lambda i: base[i], reverse=True)[: max(top_k * 3, top_k)]
        if not rerank:
            return [(self.chunks[i], base[i]) for i in order[:top_k] if base[i] > 0]
        q = set(tokens(query))
        rescored = []
        for i in order:
            if base[i] <= 0:
                continue
            toks = set(self.docs[i])
            coverage = len(q & toks) / (len(q) or 1)
            # prefer chunks that actually contain the query's rarer terms
            rare = sum(self._idf(t) for t in (q & toks))
            rescored.append((self.chunks[i], base[i] + 1.2 * coverage + 0.05 * rare))
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:top_k]


def build_index(chunk_chars: int, overlap: int) -> BM25:
    return BM25(chunk_pages(chunk_chars, overlap))


@lru_cache(maxsize=8)
def _cached_pages(filename: str) -> tuple[tuple[int, str], ...]:
    """Parse each PDF once; chunking is re-run per retrieval setting."""
    return tuple(extract_pages(PAPERS_DIR / filename))
