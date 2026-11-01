from app.config import Config
from app.container import AppContainer
from app.schemas import LesionInfo


def test_end_to_end_report():
    cfg = Config.load()
    c = AppContainer(cfg)
    lesion = LesionInfo(
        lesion_id="L1",
        location="右侧额叶灰白质交界区",
        count=2,
        max_diameter_mm=3.2,
        morphology="类圆形，边界欠清",
        signal_features={"T1WI": "低信号", "T2WI/FLAIR": "高信号", "DWI": "稍高信号"},
        enhancement="环形强化",
        radiomics={"GLCM熵": 7.23},
    )
    report = c.pipeline.run("S1", "P1", lesions=[lesion])
    assert report.report_id.startswith("RPT-")
    assert "脑转移瘤" in report.content_markdown
    assert report.analysis.lesion_analyses[0].evidence_ids


def test_retriever_returns_evidence():
    cfg = Config.load()
    c = AppContainer(cfg)
    from app.schemas import MRIStudy
    study = MRIStudy(study_id="S1", patient_id="P1", lesions=[])
    query = c.pipeline.query_builder.build(study)
    evidences = c.retriever.retrieve(query)
    assert evidences
    assert all(e.rerank_score >= cfg.retrieval.min_rerank_score for e in evidences)