from __future__ import annotations

import numpy as np


def build_response_labels(B: np.ndarray) -> dict:
    mu = B.mean(axis=0, dtype=np.float64).astype(np.float32)
    R = (B - mu[None, :, :]).astype(np.float32)
    num_workloads, num_nodes, behavior_dim = R.shape
    S = np.transpose(R, (1, 0, 2)).reshape(num_nodes, num_workloads * behavior_dim)
    sensitivity = np.sum(np.square(R, dtype=np.float32), axis=(0, 2)).astype(np.float32)
    return {
        "B": B.astype(np.float32),
        "R": R,
        "S": S.astype(np.float32),
        "mu": mu,
        "sensitivity": sensitivity,
    }


def cosine_similarity_matrix(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(S, axis=1, keepdims=True)
    normalized = S / np.maximum(norm, eps)
    return normalized @ normalized.T

