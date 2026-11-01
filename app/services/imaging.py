from __future__ import annotations

import os
from datetime import datetime

import pydicom

from app.schemas import MRIStudy
from app.services.dicom_reader import DicomSeriesReader


class ImagingService:
    """读取患者 MRI 检查元数据（序列名、场强等）。"""

    def load_study(
        self,
        study_id: str,
        patient_id: str,
        dicom_dir: str | None = None,
    ) -> MRIStudy:
        sequences: list[str] = []
        field_strength = 3.0

        if dicom_dir and os.path.isdir(dicom_dir):
            try:
                reader = DicomSeriesReader(dicom_dir)
                series_list = reader.list_series()
                sequences = [s.description for s in series_list if s.description]
                if series_list:
                    ds = pydicom.dcmread(series_list[0].files[0], stop_before_pixels=True)
                    field_strength = float(
                        getattr(ds, "MagneticFieldStrength", 3.0) or 3.0
                    )
            except Exception as exc:
                print(f"[ImagingService] 读取 DICOM 元数据失败: {exc}")

        return MRIStudy(
            study_id=study_id,
            patient_id=patient_id,
            sequences=sorted(set(sequences)) or ["T1WI"],
            field_strength_t=field_strength,
            study_time=datetime.now(),
            dicom_dir=dicom_dir,
        )