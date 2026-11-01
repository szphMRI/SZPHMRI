import numpy as np
import SimpleITK as sitk

from app.services.registration import resample_mask_to_reference
from app.services.visualization import apply_window, overlay_mask_on_slice


def _make_image(array: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> sitk.Image:
    img = sitk.GetImageFromArray(array.astype(np.float32))
    img.SetSpacing(spacing)
    return img


def test_resample_mask_nearest_neighbor_keeps_binary():
    mask_arr = np.zeros((10, 20, 20), dtype=np.uint8)
    mask_arr[3:6, 5:8, 5:8] = 1
    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.SetSpacing((0.5, 0.5, 2.0))

    ref = _make_image(np.zeros((10, 20, 20)), spacing=(1.0, 1.0, 1.0))
    aligned = resample_mask_to_reference(mask_img, ref)
    aligned_arr = sitk.GetArrayFromImage(aligned)

    assert set(np.unique(aligned_arr).tolist()) <= {0, 1}


def test_apply_window_mr_percentile_returns_unit_range():
    vol = np.random.randn(5, 32, 32).astype(np.float32) * 100 + 50
    out = apply_window(vol[2], modality="MR")
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.shape == vol[2].shape


def test_overlay_mask_sets_red_on_mask_region():
    base = np.zeros((32, 32), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:15, 10:15] = 1

    rgb = overlay_mask_on_slice(base, mask, alpha=0.5)
    assert rgb.shape == (32, 32, 3)
    r, g, b = rgb[12, 12]
    assert r > g and r > b
    assert np.allclose(rgb[0, 0], rgb[0, 0, 0])