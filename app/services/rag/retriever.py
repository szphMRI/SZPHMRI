from __future__ import annotations

from app.schemas import Evidence, RetrievalQuery
from app.services.rag.embedder import BaseEmbedder
from app.services.rag.index import KnowledgeBaseIndex
from app.services.rag.reranker import EvidenceReranker


class HybridRetriever:
    def __init__(
        self,
        index: KnowledgeBaseIndex,
        embedder: BaseEmbedder,
        reranker: EvidenceReranker,
        top_k_keyword: int = 10,
        top_k_vector: int = 10,
        final_k: int = 5,
    ):
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.top_k_keyword = top_k_keyword
        self.top_k_vector = top_k_vector
        self.final_k = final_k

    def retrieve(self, query: RetrievalQuery) -> list[Evidence]:
        """知识库本身在 KnowledgeBaseIndex._build() 中同时建了两套索引：
        BM25Okapi 关键词索引 + self.matrix 向量矩阵，对应“对医学知识库执行关键词检索和语义向量检索”。知识库内容来自 data/knowledge_base.jsonl"""

        kw_hits = self.index.keyword_search(query.keyword_query, self.top_k_keyword)
        qv = self.embedder.encode_one(query.semantic_query)
        vec_hits = self.index.vector_search(qv, self.top_k_vector)

        merged: dict[str, Evidence] = {}
        for doc, score in kw_hits:
            e = merged.setdefault(doc["doc_id"], self._to_evidence(doc))
            e.keyword_score = score
        for doc, score in vec_hits:
            e = merged.setdefault(doc["doc_id"], self._to_evidence(doc))
            e.vector_score = score

        ranked = self.reranker.rerank(query, list(merged.values()))
        return ranked[: self.final_k]

    @staticmethod
    def _to_evidence(doc: dict) -> Evidence:
        return Evidence(
            doc_id=doc["doc_id"],
            title=doc.get("title", ""),
            content=doc.get("content", ""),
            source=doc.get("source", ""),
            evidence_level=doc.get("evidence_level", "OCEBM-?"),
            year=doc.get("year"),
        )