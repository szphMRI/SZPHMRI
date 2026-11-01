from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.services.rag.embedder import BaseEmbedder


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    try:
        import jieba  # type: ignore

        return [t for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        return [c for c in text if not c.isspace()]


class KnowledgeBaseIndex:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.docs: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self.matrix: np.ndarray | None = None

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"知识库文件不存在: {p}")
        with p.open("r", encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f if line.strip()]
        self._build()

    def _build(self) -> None:
        if not self.docs:
            self.bm25, self.matrix = None, None
            return
        corpus = [tokenize(d.get("title", "") + " " + d.get("content", "")) for d in self.docs]
        self.bm25 = BM25Okapi(corpus)
        texts = [d.get("title", "") + " " + d.get("content", "") for d in self.docs]
        self.matrix = self.embedder.encode(texts)

    def add_document(self, doc: dict) -> None:
        self.docs.append(doc)
        self._build()

    def keyword_search(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        if not self.bm25 or not self.docs:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], float(scores[i])) for i in idx if scores[i] > 0]

    def vector_search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[dict, float]]:
        if self.matrix is None or len(self.docs) == 0:
            return []
        sims = self.matrix @ query_vec
        idx = np.argsort(sims)[::-1][:top_k]
        return [(self.docs[i], float(sims[i])) for i in idx]