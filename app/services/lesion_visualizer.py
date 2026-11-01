from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from app.schemas import LesionInfo
from app.services.dicom_reader import DicomSeries, DicomSeriesReader
from app.services.registration import resample_mask_to_reference
from app.services.visualization import render_views_to_markdown


class LesionVisualizer:
    """根据病灶掩码 + 参考序列生成报告内嵌三视图 Markdown。"""

    def __init__(self, output_dir: str | Path = "outputs/figures", alpha: float = 0.45):
        self.output_dir = Path(output_dir)
        self.alpha = alpha

    def build_markdown(self, dicom_dir: str | Path, lesions: list[LesionInfo]) -> str:
        if not lesions or not dicom_dir:
            return ""

        try:
            series_map = DicomSeriesReader(dicom_dir).read_all()
        except Exception as exc:
            print(f"[LesionVisualizer] 读取 DICOM 失败: {exc}")
            return ""

        blocks: list[str] = []
        for les in lesions:
            if not les.mask_path:
                continue
            base = self._pick_series(series_map, les.reference_sequence)
            if base is None or base.image is None:
                continue

            mask_image = sitk.ReadImage(les.mask_path)
            aligned = resample_mask_to_reference(mask_image, base.image)

            volume = base.array
            mask = sitk.GetArrayFromImage(aligned).astype(np.uint8)

            md = render_views_to_markdown(
                volume=volume,
                mask=mask,
                lesion_id=les.lesion_id,
                output_dir=self.output_dir,
                alpha=self.alpha,
            )
            blocks.append(md)

        return "\n\n".join(blocks)

    @staticmethod
    def _pick_series(series_map: dict[str, DicomSeries], preferred: str) -> DicomSeries | None:
        for key, s in series_map.items():
            if preferred.lower() in key.lower():
                return s
        return next(iter(series_map.values()), None)