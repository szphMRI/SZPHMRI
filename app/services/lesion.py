from __future__ import annotations

from typing import Protocol

from app.schemas import LesionInfo, MRIStudy


class LesionDetector(Protocol):
    def detect(self, study: MRIStudy) -> list[LesionInfo]: ...


class MockLesionDetector:
    """开发期占位检测器；生产替换为 3D-UNet / nnU-Net 等，输出 LesionInfo 即可。"""

    def __init__(self, mask_path: str | None = None):
        self.mask_path = mask_path

    def detect(self, study: MRIStudy) -> list[LesionInfo]:
        return [
            LesionInfo(
                lesion_id=f"{study.study_id}-L1",
                location="右侧额叶皮层下、左侧顶叶灰白质交界区",
                count=2,
                max_diameter_mm=3.2,
                morphology="类圆形，边界欠清，无明显占位效应",
                signal_features={"T1WI": "低信号", "T2WI/FLAIR": "高信号", "DWI": "稍高信号"},
                enhancement="环形强化",
                radiomics={"GLCM熵": 7.23, "对比度": 4.89, "球形度": 0.78},
                confidence=0.93,
                mask_path=self.mask_path,
                reference_sequence="T1WI",
            )
        ]