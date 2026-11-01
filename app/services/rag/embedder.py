from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class BaseEmbedder(Protocol):
    dim: int

    def encode_one(self, text: str) -> np.ndarray: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """字符 n-gram 哈希嵌入，无需下载模型，便于本地调试。
    生产环境替换为医疗领域嵌入模型（保持同样接口即可）。
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _encode_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        text = (text or "").lower()
        for n in (1, 2, 3):
            for i in range(max(0, len(text) - n + 1)):
                gram = text[i : i + n]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def encode_one(self, text: str) -> np.ndarray:
        return self._encode_one(text)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._encode_one(t) for t in texts])