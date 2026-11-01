"""生成 mock DICOM 序列与病灶掩码，便于无真实数据时在 PyCharm 中直接调试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"


def _make_volume(n_slices: int, size: int, spacing: float):
    zz, yy, xx = np.meshgrid(
        np.arange(n_slices), np.arange(size), np.arange(size), indexing="ij"
    )
    vol = np.zeros((n_slices, size, size), dtype=np.float32)
    brain = ((yy - size / 2) / (size * 0.4)) ** 2 + ((xx - size / 2) / (size * 0.4)) ** 2 <= 1.0
    vol[brain] = 100 + np.random.default_rng(42).normal(0, 10, brain.sum())

    # 一个 3.2mm 微小病灶（半径约 1.6mm，像素 1mm 时半径约 2 像素）
    lz, ly, lx = n_slices // 2, size // 2 + 10, size // 2 - 10
    lesion = ((zz - lz) ** 2 + (yy - ly) ** 2 + (xx - lx) ** 2) <= 2 ** 2
    vol[lesion] = 200.0
    return vol, lesion.astype(np.uint8)


def _write_dicom_series(out_dir: Path, vol: np.ndarray, spacing: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    n_slices, size, _ = vol.shape

    for i in range(n_slices):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = MR_IMAGE_STORAGE
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.PatientID = "PAT-0001"
        ds.PatientName = "Mock^Patient"
        ds.StudyID = "STU-MOCK-001"
        ds.SeriesDescription = "T1WI"
        ds.Modality = "MR"
        ds.MagneticFieldStrength = 3.0
        ds.SliceThickness = spacing
        ds.PixelSpacing = [spacing, spacing]
        ds.ImagePositionPatient = [0.0, 0.0, float(i * spacing)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.Rows = size
        ds.Columns = size
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = vol[i].astype(np.int16).tobytes()

        ds.save_as(str(out_dir / f"slice_{i:03d}.dcm"), enforce_file_format=True)


def ensure_mock_data(data_dir: Path) -> tuple[str, str]:
    """若 mock 数据不存在则生成，返回 (dicom_dir, mask_path)。"""
    data_dir = Path(data_dir)
    dicom_dir = data_dir / "mock_dicom"
    mask_path = data_dir / "mock_lesion_mask.nii.gz"

    has_dcm = dicom_dir.exists() and any(dicom_dir.glob("*.dcm"))
    if has_dcm and mask_path.exists():
        return str(dicom_dir), str(mask_path)

    spacing = 1.0
    vol, mask = _make_volume(n_slices=32, size=128, spacing=spacing)
    _write_dicom_series(dicom_dir, vol, spacing)

    mask_img = sitk.GetImageFromArray(mask)
    mask_img.SetSpacing((spacing, spacing, spacing))
    sitk.WriteImage(mask_img, str(mask_path))

    return str(dicom_dir), str(mask_path)