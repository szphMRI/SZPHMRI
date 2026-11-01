from __future__ import annotations

from app.config import Config
from app.core.pipeline import ReportPipeline
from app.schemas import Report
from app.services.analyzer import RuleBasedAnalyzer
from app.services.imaging import ImagingService
from app.services.lesion import MockLesionDetector
from app.services.lesion_visualizer import LesionVisualizer
from app.services.notifier import ConsoleNotifier
from app.services.rag.embedder import HashingEmbedder
from app.services.rag.index import KnowledgeBaseIndex
from app.services.rag.query_builder import LesionQueryBuilder
from app.services.rag.reranker import EvidenceReranker
from app.services.rag.retriever import HybridRetriever
from app.services.report import ReportBuilder


class AppContainer:
    """依赖装配：替换任意实现只需改这里。"""

    def __init__(self, config: Config):
        self.config = config

        self.embedder = HashingEmbedder(config.retrieval.embed_dim)
        self.index = KnowledgeBaseIndex(self.embedder)
        self.index.load(config.knowledge_base.path)

        self.reranker = EvidenceReranker(
            min_score=config.retrieval.min_rerank_score,
            max_level_rank=config.report.min_evidence_level_rank,
        )
        self.retriever = HybridRetriever(
            index=self.index,
            embedder=self.embedder,
            reranker=self.reranker,
            top_k_keyword=config.retrieval.top_k_keyword,
            top_k_vector=config.retrieval.top_k_vector,
            final_k=config.retrieval.final_k,
        )
        self.visualizer = LesionVisualizer(
            output_dir=config.visualization.output_dir,
            alpha=config.visualization.alpha,
        )
        self.pipeline = ReportPipeline(
            imaging=ImagingService(),
            detector=MockLesionDetector(),
            query_builder=LesionQueryBuilder(),
            retriever=self.retriever,
            analyzer=RuleBasedAnalyzer(),
            reporter=ReportBuilder(
                hospital_name=config.app.hospital_name,
                disclaimer=config.report.disclaimer,
            ),
            notifier=ConsoleNotifier(),
            visualizer=self.visualizer,
        )
        self.reports: dict[str, Report] = {}