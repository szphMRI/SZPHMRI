from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LesionInfo(BaseModel):
    lesion_id: str
    location: str
    count: int = 1
    max_diameter_mm: float = 0.0
    morphology: str = ""
    signal_features: dict[str, str] = Field(default_factory=dict)
    enhancement: str = ""
    radiomics: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0
    mask_path: Optional[str] = None
    reference_sequence: str = "T1WI"


class MRIStudy(BaseModel):
    study_id: str
    patient_id: str
    modality: str = "MRI"
    sequences: list[str] = Field(default_factory=list)
    field_strength_t: float = 3.0
    study_time: datetime = Field(default_factory=datetime.now)
    dicom_dir: Optional[str] = None
    lesions: list[LesionInfo] = Field(default_factory=list)


class RetrievalQuery(BaseModel):
    keyword_query: str
    semantic_query: str
    differential_query: str = ""


class Evidence(BaseModel):
    doc_id: str
    title: str
    content: str
    source: str = ""
    evidence_level: str = "OCEBM-?"
    year: Optional[int] = None
    keyword_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0


class LesionAnalysis(BaseModel):
    lesion_id: str
    lesion_type: str
    type_probability: float
    status_assessment: str
    severity: Severity
    diagnosis_basis: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    differential: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    study_id: str
    patient_id: str
    lesion_analyses: list[LesionAnalysis] = Field(default_factory=list)
    summary: str = ""
    risk_flags: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class Report(BaseModel):
    report_id: str
    study_id: str
    patient_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    content_markdown: str
    analysis: AnalysisResult
    status: str = "pending_review"