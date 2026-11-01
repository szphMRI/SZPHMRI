from __future__ import annotations

import math
from typing import Any

from app.schemas import (
    AnalysisResult,
    Evidence,
    LesionAnalysis,
    LesionInfo,
    MRIStudy,
    Severity,
)

DIAGNOSIS_RULES: list[dict[str, Any]] = [
    {
        "type": "脑转移瘤",
        "must": ["环形强化"],
        "support": ["灰白质交界", "多发", "水肿", "类圆形", "稍高信号"],
        "status": "活动性病灶，恶性风险高",
        "severity": Severity.HIGH,
    },
    {
        "type": "高级别胶质瘤",
        "must": ["不规则强化"],
        "support": ["坏死", "占位效应", "水肿", "边界欠清"],
        "status": "活动性病灶，需警惕恶性",
        "severity": Severity.HIGH,
    },
    {
        "type": "低级别胶质瘤",
        "must": ["无强化"],
        "support": ["边界清晰", "T2高信号", "占位效应轻"],
        "status": "惰性病灶，建议随访",
        "severity": Severity.MEDIUM,
    },
    {
        "type": "脑膜瘤",
        "must": ["均匀强化"],
        "support": ["脑外", "宽基底", "硬膜尾征"],
        "status": "良性肿瘤可能性大",
        "severity": Severity.MEDIUM,
    },
    {
        "type": "炎性肉芽肿",
        "must": ["环形强化"],
        "support": ["边界清晰", "水肿", "感染病史"],
        "status": "炎性病变可能，建议结合实验室检查",
        "severity": Severity.MEDIUM,
    },
    {
        "type": "腔隙性脑梗死",
        "must": ["无强化"],
        "support": ["基底节", "点状", "DWI高信号"],
        "status": "缺血性病灶，建议脑血管评估",
        "severity": Severity.LOW,
    },
]


class RuleBasedAnalyzer:
    def analyze(self, study: MRIStudy, evidences: list[Evidence]) -> AnalysisResult:
        if not study.lesions:
            return AnalysisResult(
                study_id=study.study_id,
                patient_id=study.patient_id,
                summary="本次检查未见明确异常病灶。",
                evidence=evidences,
            )

        analyses = [self._analyze_lesion(les, evidences) for les in study.lesions]
        risk_flags = self._risk_flags(study.lesions, analyses)
        summary = "；".join(
            f"病灶{a.lesion_id}倾向{a.lesion_type}"
            f"（置信度{a.type_probability:.1%}），{a.status_assessment}"
            for a in analyses
        )
        return AnalysisResult(
            study_id=study.study_id,
            patient_id=study.patient_id,
            lesion_analyses=analyses,
            summary=summary,
            risk_flags=risk_flags,
            evidence=evidences,
        )

    def _analyze_lesion(self, les: LesionInfo, evidences: list[Evidence]) -> LesionAnalysis:
        text = self._lesion_text(les)
        scored: list[tuple[dict, float]] = []
        for rule in DIAGNOSIS_RULES:
            score = 2.0 if any(m in text for m in rule["must"]) else 0.0
            score += sum(1.0 for s in rule["support"] if s in text)
            scored.append((rule, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        probs = self._softmax([s for _, s in scored])

        best_rule, _ = scored[0]
        best_prob = probs[0]

        matched = [e for e in evidences if best_rule["type"] in (e.title + e.content)]
        basis = [f"病灶位于{les.location}，最大径{les.max_diameter_mm}mm，{les.enhancement}"]
        hits = [s for s in best_rule["support"] if s in text]
        if hits:
            basis.append("影像学支持点：" + "、".join(hits))
        for e in matched[:2]:
            basis.append(f"循证依据（{e.evidence_level}）：{e.title}——{e.content[:60]}…")

        differential = [
            {
                "lesion_type": rule["type"],
                "probability": round(p, 4),
                "support": [s for s in rule["support"] if s in text],
                "against": [m for m in rule["must"] if m not in text],
            }
            for (rule, _), p in zip(scored[1:4], probs[1:4])
        ]

        return LesionAnalysis(
            lesion_id=les.lesion_id,
            lesion_type=best_rule["type"],    ## 病灶类型
            type_probability=round(best_prob, 4),
            status_assessment=best_rule["status"],  ## 病灶状态评估
            severity=best_rule["severity"],
            diagnosis_basis=basis,  ## 病灶诊断依据
            evidence_ids=[e.doc_id for e in matched],   ## 病灶诊断依据中的循证证据编号
            differential=differential,
        )

    @staticmethod
    def _lesion_text(les: LesionInfo) -> str:
        sig = " ".join(f"{k}{v}" for k, v in les.signal_features.items())
        rad = " ".join(f"{k}{v}" for k, v in les.radiomics.items())
        return " ".join([les.location, les.morphology, les.enhancement, sig, rad])

    @staticmethod
    def _softmax(scores: list[float]) -> list[float]:
        if not scores:
            return []
        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        total = sum(exps) or 1.0
        return [e / total for e in exps]

    @staticmethod
    def _risk_flags(lesions: list[LesionInfo], analyses: list[LesionAnalysis]) -> list[str]:
        flags: list[str] = []
        for les in lesions:
            if 0 < les.max_diameter_mm < 5:
                flags.append(
                    f"病灶{les.lesion_id}为微小病灶（{les.max_diameter_mm}mm），"
                    f"存在漏诊风险，建议薄层增强复查。"
                )
        for a in analyses:
            if a.severity in (Severity.HIGH, Severity.CRITICAL):
                flags.append(f"病灶{a.lesion_id}恶性风险较高，建议多学科会诊。")
        return flags