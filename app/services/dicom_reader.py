from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


@dataclass
class DicomSeries:
    series_id: str
    description: str
    files: list[str]
    image: sitk.Image | None = None

    @property
    def array(self) -> np.ndarray:
        if self.image is None:
            raise ValueError("Series 尚未读取")
        return sitk.GetArrayFromImage(self.image)

    @property
    def spacing(self) -> tuple[float, float, float]:
        if self.image is None:
            return (1.0, 1.0, 1.0)
        sx, sy, sz = self.image.GetSpacing()
        return (sz, sy, sx)


class DicomSeriesReader:
    def __init__(self, dicom_dir: str | Path):
        self.dicom_dir = str(dicom_dir)

    def list_series(self) -> list[DicomSeries]:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(self.dicom_dir)
        if not series_ids:
            raise FileNotFoundError(f"目录中未找到 DICOM 序列: {self.dicom_dir}")

        result: list[DicomSeries] = []
        for sid in series_ids:
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(self.dicom_dir, sid)
            desc = self._series_description(files[0])
            result.append(DicomSeries(series_id=sid, description=desc, files=files))
        return result

    def read_series(self, series: DicomSeries) -> DicomSeries:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series.files)
        series.image = reader.Execute()
        return series

    def read_all(self) -> dict[str, DicomSeries]:
        out: dict[str, DicomSeries] = {}
        for i, s in enumerate(self.list_series()):
            self.read_series(s)
            key = s.description or s.series_id
            out[f"{key}_{i}"] = s
        return out

    @staticmethod
    def _series_description(dcm_path: str) -> str:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        return (
            getattr(ds, "SeriesDescription", "")
            or getattr(ds, "SequenceName", "")
            or getattr(ds, "ProtocolName", "")
        ).strip()