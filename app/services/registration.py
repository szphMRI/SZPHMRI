from __future__ import annotations

import SimpleITK as sitk


def resample_mask_to_reference(
    mask_image: sitk.Image,
    reference_image: sitk.Image,
) -> sitk.Image:
    """掩码空间对齐：必须用最近邻插值，保持二值性。"""
    return sitk.Resample(
        mask_image,
        reference_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask_image.GetPixelID(),
    )


def resample_intensity_to_reference(
    moving_image: sitk.Image,
    reference_image: sitk.Image,
) -> sitk.Image:
    return sitk.Resample(
        moving_image,
        reference_image,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )