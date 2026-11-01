from __future__ import annotations

import uuid
from datetime import datetime

from app.schemas import AnalysisResult, MRIStudy, Report

REPORT_TEMPLATE = """# {hospital} 磁共振影像分析报告

- 报告编号：{report_id}
- 检查编号：{study_id}
- 患者编号：{patient_id}
- 检查时间：{study_time}
- 设备场强：{field_strength}T
- 扫描序列：{sequences}

## 一、影像所见

{findings}

## 二、病灶影像标注

{figures}

## 三、病灶分析

{lesion_sections}

## 四、鉴别诊断

{differential}

## 五、风险提示

{risks}

## 六、循证证据

{evidences}

## 七、说明

{disclaimer}


"""


class ReportBuilder:
    def __init__(
        self,
        hospital_name: str = "示例医院",
        disclaimer: str = "本报告由 AI 辅助生成，仅供临床参考。",
    ):
        self.hospital_name = hospital_name
        self.disclaimer = disclaimer

    def build(
        self,
        study: MRIStudy,
        analysis: AnalysisResult,
        figures_markdown: str = "",
    ) -> Report:
        report_id = f"RPT-{uuid.uuid4().hex[:12].upper()}"
        content = REPORT_TEMPLATE.format(
            hospital=self.hospital_name,
            report_id=report_id,
            study_id=study.study_id,
            patient_id=study.patient_id,
            study_time=study.study_time.strftime("%Y-%m-%d %H:%M"),
            field_strength=study.field_strength_t,
            sequences="、".join(study.sequences) or "未提供",
            findings=self._findings(study),
            figures=figures_markdown or "未提供可视化图像。",
            lesion_sections=self._lesion_sections(analysis),
            differential=self._differential(analysis),
            risks="\n".join(f"- {r}" for r in analysis.risk_flags) or "- 无",
            evidences=self._evidences(analysis),
            disclaimer=self.disclaimer,
            # created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        return Report(
            report_id=report_id,
            study_id=study.study_id,
            patient_id=study.patient_id,
            content_markdown=content,
            analysis=analysis,
        )

    @staticmethod
    def _findings(study: MRIStudy) -> str:
        if not study.lesions:
            return "颅内未见明确异常信号灶，脑室系统形态正常。"
        lines = []
        for les in study.lesions:
            sig = "、".join(f"{k}呈{v}" for k, v in les.signal_features.items())
            lines.append(
                f"- {les.location}见{les.count}个病灶，最大径约{les.max_diameter_mm}mm，"
                f"{les.morphology}；{sig}；增强扫描{les.enhancement}。"
            )
        return "\n".join(lines)

    @staticmethod
    def _lesion_sections(analysis: AnalysisResult) -> str:
        if not analysis.lesion_analyses:
            return "无。"
        blocks = []
        for a in analysis.lesion_analyses:
            basis = "\n".join(f"  - {b}" for b in a.diagnosis_basis)
            blocks.append(
                f"### 病灶 {a.lesion_id}\n"
                f"- **病灶类型**：{a.lesion_type}（后验概率 {a.type_probability:.1%}）\n"
                f"- **状态评估**：{a.status_assessment}（严重程度：{a.severity.value}）\n"
                f"- **诊断依据**：\n{basis}\n"
                f"- **关联证据编号**：{', '.join(a.evidence_ids) or '无'}\n"
            )
        return "\n".join(blocks)

    @staticmethod
    def _differential(analysis: AnalysisResult) -> str:
        lines = []
        for a in analysis.lesion_analyses:
            for d in a.differential:
                support = "、".join(d.get("support") or []) or "无"
                against = "、".join(d.get("against") or []) or "无"
                lines.append(
                    f"- 病灶{a.lesion_id} 鉴别 {d['lesion_type']}（{d['probability']:.1%}）："
                    f"支持 {support}；不支持 {against}"
                )
        return "\n".join(lines) or "- 无"

    @staticmethod
    def _evidences(analysis: AnalysisResult) -> str:
        if not analysis.evidence:
            return "- 无"
        return "\n".join(
            f"- [{e.doc_id}] {e.title}"
            f"（{e.evidence_level}{'，' + str(e.year) if e.year else ''}）："
            f"{e.content[:100]}…"
            for e in analysis.evidence
        )