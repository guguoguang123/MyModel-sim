from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    kind: str
    params: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def default_workloads() -> List[WorkloadSpec]:
    return [
        WorkloadSpec("uniform_p05", "independent", {"p": 0.5}),
        WorkloadSpec("low_bias_p02", "independent", {"p": 0.2}),
        WorkloadSpec("high_bias_p08", "independent", {"p": 0.8}),
        WorkloadSpec("random_beta_05_05", "beta_bias", {"alpha": 0.5, "beta": 0.5}),
        WorkloadSpec("low_toggle_markov", "markov_toggle", {"init_p": 0.5, "toggle_p": 0.05}),
        WorkloadSpec("high_toggle_markov", "markov_toggle", {"init_p": 0.5, "toggle_p": 0.45}),
        WorkloadSpec("group_correlation", "group_correlation", {"group_size": 8, "noise_p": 0.08, "latent_p": 0.5}),
        WorkloadSpec(
            "mixed_bias_temporal_corr",
            "mixed",
            {"alpha": 0.7, "beta": 0.7, "toggle_p": 0.18, "group_size": 8, "noise_p": 0.05},
        ),
    ]


def debug_workloads() -> List[WorkloadSpec]:
    return default_workloads()[:4]


def generate_pi_sequence(
    spec: WorkloadSpec,
    num_vectors: int,
    num_pis: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_pis == 0:
        return np.zeros((num_vectors, 0), dtype=np.bool_)

    if spec.kind == "independent":
        p = float(spec.params["p"])
        return rng.random((num_vectors, num_pis)) < p

    if spec.kind == "beta_bias":
        alpha = float(spec.params["alpha"])
        beta = float(spec.params["beta"])
        probs = rng.beta(alpha, beta, size=(num_pis,))
        return rng.random((num_vectors, num_pis)) < probs

    if spec.kind == "markov_toggle":
        init_p = float(spec.params["init_p"])
        toggle_p = float(spec.params["toggle_p"])
        seq = np.empty((num_vectors, num_pis), dtype=np.bool_)
        seq[0] = rng.random(num_pis) < init_p
        for t in range(1, num_vectors):
            flips = rng.random(num_pis) < toggle_p
            seq[t] = np.logical_xor(seq[t - 1], flips)
        return seq

    if spec.kind == "group_correlation":
        return _group_correlated_sequence(spec, num_vectors, num_pis, rng)

    if spec.kind == "mixed":
        return _mixed_sequence(spec, num_vectors, num_pis, rng)

    raise ValueError(f"Unknown workload kind: {spec.kind}")


def _group_correlated_sequence(
    spec: WorkloadSpec,
    num_vectors: int,
    num_pis: int,
    rng: np.random.Generator,
) -> np.ndarray:
    group_size = max(1, int(spec.params["group_size"]))
    noise_p = float(spec.params["noise_p"])
    latent_p = float(spec.params["latent_p"])
    num_groups = int(np.ceil(num_pis / group_size))
    latent = rng.random((num_vectors, num_groups)) < latent_p
    seq = np.empty((num_vectors, num_pis), dtype=np.bool_)
    for i in range(num_pis):
        seq[:, i] = latent[:, i // group_size]
    noise = rng.random((num_vectors, num_pis)) < noise_p
    return np.logical_xor(seq, noise)


def _mixed_sequence(
    spec: WorkloadSpec,
    num_vectors: int,
    num_pis: int,
    rng: np.random.Generator,
) -> np.ndarray:
    alpha = float(spec.params["alpha"])
    beta = float(spec.params["beta"])
    toggle_p = float(spec.params["toggle_p"])
    group_size = max(1, int(spec.params["group_size"]))
    noise_p = float(spec.params["noise_p"])

    probs = rng.beta(alpha, beta, size=(num_pis,))
    seq = np.empty((num_vectors, num_pis), dtype=np.bool_)
    seq[0] = rng.random(num_pis) < probs
    for t in range(1, num_vectors):
        flips = rng.random(num_pis) < toggle_p
        fresh = rng.random(num_pis) < probs
        seq[t] = np.where(flips, fresh, seq[t - 1])

    num_groups = int(np.ceil(num_pis / group_size))
    latent = rng.random((num_vectors, num_groups)) < 0.5
    corr = np.empty_like(seq)
    for i in range(num_pis):
        corr[:, i] = latent[:, i // group_size]
    use_corr = rng.random((num_vectors, num_pis)) > noise_p
    return np.where(use_corr, np.logical_xor(seq, corr), seq).astype(np.bool_)

