from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom


def apply_window(
    image: np.ndarray,
    window_center: float | None = None,
    window_width: float | None = None,
    modality: str = "MR",
) -> np.ndarray:
    img = image.astype(np.float32)

    if modality.upper() == "CT" and window_center is not None and window_width is not None:
        lo = window_center - window_width / 2
        hi = window_center + window_width / 2
        img = np.clip(img, lo, hi)
        return (img - lo) / max(hi - lo, 1e-6)

    lo = float(np.percentile(img, 1))
    hi = float(np.percentile(img, 99))
    if hi - lo < 1e-6:
        lo, hi = float(img.min()), float(img.max()) + 1e-6
    return np.clip((img - lo) / (hi - lo), 0, 1)


def dicom_window_from_dataset(ds: pydicom.Dataset) -> tuple[float | None, float | None]:
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is None or ww is None:
        return None, None
    wc = wc[0] if isinstance(wc, (list, tuple)) else wc
    ww = ww[0] if isinstance(ww, (list, tuple)) else ww
    return float(wc), float(ww)


def overlay_mask_on_slice(
    base_slice: np.ndarray,
    mask_slice: np.ndarray,
    alpha: float = 0.45,
    color: tuple[float, float, float] = (1.0, 0.15, 0.15),
    modality: str = "MR",
) -> np.ndarray:
    base = apply_window(base_slice, modality=modality)
    base_rgb = np.stack([base] * 3, axis=-1)

    mask = mask_slice.astype(bool)
    if mask.any():
        base_rgb[mask] = (1 - alpha) * base_rgb[mask] + alpha * np.array(color)
    return np.clip(base_rgb, 0, 1)


def _pick_slice_index(mask: np.ndarray, axis: int) -> int:
    if mask.sum() == 0:
        return mask.shape[axis] // 2
    projection = mask.sum(axis=tuple(i for i in range(mask.ndim) if i != axis))
    return int(np.argmax(projection))


def render_lesion_views(
    volume: np.ndarray,
    mask: np.ndarray,
    output_dir: str | Path,
    lesion_id: str,
    alpha: float = 0.45,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    views = {
        "axial": (_pick_slice_index(mask, 0), lambda a, i: a[i, :, :]),
        "coronal": (_pick_slice_index(mask, 1), lambda a, i: a[:, i, :]),
        "sagittal": (_pick_slice_index(mask, 2), lambda a, i: a[:, :, i]),
    }

    paths: list[Path] = []
    for name, (idx, slicer) in views.items():
        base_slice = slicer(volume, idx)
        mask_slice = slicer(mask, idx)
        rgb = overlay_mask_on_slice(base_slice, mask_slice, alpha=alpha)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"{lesion_id} · {name} · slice {idx}", fontsize=9)
        ax.axis("off")
        fig.tight_layout(pad=0.2)

        out_path = output_dir / f"{lesion_id}_{name}.png"
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        paths.append(out_path)

    return paths


def image_to_base64(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def render_views_to_markdown(
    volume: np.ndarray,
    mask: np.ndarray,
    lesion_id: str,
    output_dir: str | Path,
    alpha: float = 0.45,
) -> str:
    paths = render_lesion_views(volume, mask, output_dir, lesion_id, alpha=alpha)
    parts = [f"**病灶 {lesion_id} 三视图（axial / coronal / sagittal）**\n"]
    for p in paths:
        b64 = image_to_base64(p)
        parts.append(f"![{p.stem}](data:image/png;base64,{b64})\n")
    return "\n".join(parts)