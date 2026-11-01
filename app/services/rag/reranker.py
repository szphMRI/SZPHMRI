from __future__ import annotations

from app.schemas import Evidence, RetrievalQuery

LEVEL_WEIGHTS = {
    "1a": 1.00, "1b": 0.95, "1c": 0.90,
    "2a": 0.85, "2b": 0.75, "2c": 0.65,
    "3a": 0.60, "3b": 0.55,
    "4": 0.40, "5": 0.25,
}


class EvidenceReranker:
    """可信度筛选”：不仅看检索相关度，还把循证等级和时效性纳入可信度评分，低于阈值或证据等级不达标的直接剔除。
    最终结果在 HybridRetriever.retrieve() 末尾 return ranked[: self.final_k] 返回。"""

    def __init__(self, min_score: float = 0.45, max_level_rank: int = 3):
        self.min_score = min_score
        self.max_level_rank = max_level_rank

    def rerank(
        self,
        query: RetrievalQuery,
        evidences: list[Evidence],
        current_year: int = 2026,
    ) -> list[Evidence]:
        if not evidences:
            return []
        max_kw = max((e.keyword_score for e in evidences), default=1.0) or 1.0
        for e in evidences:
            norm_kw = min(e.keyword_score / max_kw, 1.0)
            norm_vec = (e.vector_score + 1.0) / 2.0
            level = e.evidence_level.replace("OCEBM-", "").strip()
            level_w = LEVEL_WEIGHTS.get(level, 0.3)
            recency = 1.0 if not e.year else max(0.0, 1.0 - (current_year - e.year) / 10.0)
            e.rerank_score = round(
                0.40 * norm_vec + 0.25 * norm_kw + 0.25 * level_w + 0.10 * recency, 4
            )
        ranked = sorted(evidences, key=lambda x: x.rerank_score, reverse=True)
        return [
            e
            for e in ranked
            if e.rerank_score >= self.min_score
            and LEVEL_WEIGHTS.get(e.evidence_level.replace("OCEBM-", "").strip(), 0.0) > 0
        ]