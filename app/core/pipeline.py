from __future__ import annotations

from app.schemas import LesionInfo, MRIStudy, Report
from app.services.analyzer import RuleBasedAnalyzer
from app.services.imaging import ImagingService
from app.services.lesion import LesionDetector
from app.services.lesion_visualizer import LesionVisualizer
from app.services.notifier import Notifier
from app.services.rag.query_builder import LesionQueryBuilder
from app.services.rag.retriever import HybridRetriever
from app.services.report import ReportBuilder


class ReportPipeline:
    """分析报告生成主流程（对应专利步骤 1 → 7）。"""

    def __init__(
        self,
        imaging: ImagingService,
        detector: LesionDetector,
        query_builder: LesionQueryBuilder,
        retriever: HybridRetriever,
        analyzer: RuleBasedAnalyzer,
        reporter: ReportBuilder,
        notifier: Notifier,
        visualizer: LesionVisualizer,
    ):
        self.imaging = imaging
        self.detector = detector
        self.query_builder = query_builder
        self.retriever = retriever
        self.analyzer = analyzer
        self.reporter = reporter
        self.notifier = notifier
        self.visualizer = visualizer

    def run(
        self,
        study_id: str,
        patient_id: str,
        dicom_dir: str | None = None,
        lesions: list[LesionInfo] | None = None,
    ) -> Report:
        # 1. 获取影像数据
        study: MRIStudy = self.imaging.load_study(study_id, patient_id, dicom_dir)

        # 2. 确定病灶信息
        study.lesions = lesions if lesions else self.detector.detect(study)

        # 3. 病灶信息 → 病灶检索数据
        query = self.query_builder.build(study)

        # 4+5. 混合检索 + 可信度筛选
        # 混合检索，即 根据病灶检索数据对医学知识库执行关键词检索和语义向量检索

        # 调用类 /rag/query_builder.py → LesionQueryBuilder.build()

        evidences = self.retriever.retrieve(query)

        # 6. 分析结果（类型 / 状态 / 依据）
        analysis = self.analyzer.analyze(study, evidences)

        # 病灶掩码三视图（可选）
        figures_md = ""
        if dicom_dir:
            figures_md = self.visualizer.build_markdown(dicom_dir, study.lesions)

        # 7. 生成报告并推送
        report = self.reporter.build(study, analysis, figures_markdown=figures_md)
        self.notifier.send(report)
        return report