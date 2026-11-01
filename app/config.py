from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
    name: str = "mri-report-gen"
    version: str = "0.1.0"
    hospital_name: str = "示例医院"


@dataclass
class KnowledgeBaseConfig:
    path: str = "data/knowledge_base.jsonl"


@dataclass
class RetrievalConfig:
    top_k_keyword: int = 10
    top_k_vector: int = 10
    final_k: int = 5
    min_rerank_score: float = 0.45
    embed_dim: int = 768


@dataclass
class ReportConfig:
    min_evidence_level_rank: int = 3
    disclaimer: str = "本报告由 AI 辅助生成，仅供临床参考。"


@dataclass
class VisualizationConfig:
    output_dir: str = "outputs/figures"
    alpha: float = 0.45


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        raw: dict[str, Any] = {}
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

        kb = KnowledgeBaseConfig(**raw.get("knowledge_base", {}))
        if not Path(kb.path).is_absolute():
            kb.path = str(PROJECT_ROOT / kb.path)

        vis = VisualizationConfig(**raw.get("visualization", {}))
        if not Path(vis.output_dir).is_absolute():
            vis.output_dir = str(PROJECT_ROOT / vis.output_dir)

        return cls(
            app=AppConfig(**raw.get("app", {})),
            knowledge_base=kb,
            retrieval=RetrievalConfig(**raw.get("retrieval", {})),
            report=ReportConfig(**raw.get("report", {})),
            visualization=vis,
        )