from __future__ import annotations

from typing import Tuple

import numpy as np


PI_GATE = 0
AND_GATE = 1
NOT_GATE = 2


def find_primary_inputs(x: np.ndarray) -> np.ndarray:
    return np.flatnonzero(x[:, 1].astype(np.int64) == PI_GATE)


def topological_order(x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    if x.shape[1] >= 3:
        return np.lexsort((x[:, 0], x[:, 2]))
    return np.arange(x.shape[0])


def simulate_aig(
    x: np.ndarray,
    edge_index: np.ndarray,
    pi_values: np.ndarray,
) -> np.ndarray:
    num_vectors = pi_values.shape[0]
    num_nodes = x.shape[0]
    gate_types = x[:, 1].astype(np.int64)
    values = np.zeros((num_vectors, num_nodes), dtype=np.bool_)

    pi_nodes = find_primary_inputs(x)
    if pi_values.shape[1] != len(pi_nodes):
        raise ValueError(f"PI value width {pi_values.shape[1]} != number of PIs {len(pi_nodes)}")
    values[:, pi_nodes] = pi_values

    fanins = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.astype(np.int64):
        fanins[int(dst)].append(int(src))

    for node in topological_order(x, edge_index):
        gate = int(gate_types[node])
        if gate == PI_GATE:
            continue
        node_fanins = fanins[int(node)]
        if gate == AND_GATE:
            if not node_fanins:
                values[:, node] = False
            else:
                acc = values[:, node_fanins[0]].copy()
                for src in node_fanins[1:]:
                    acc &= values[:, src]
                values[:, node] = acc
        elif gate == NOT_GATE:
            if len(node_fanins) != 1:
                raise ValueError(f"NOT node {node} has {len(node_fanins)} fanins")
            values[:, node] = ~values[:, node_fanins[0]]
        else:
            raise ValueError(f"Unsupported gate type {gate} at node {node}")

    return values


def behavior_from_values(values: np.ndarray) -> np.ndarray:
    prob_one = values.mean(axis=0, dtype=np.float64)
    if values.shape[0] <= 1:
        toggle = np.zeros(values.shape[1], dtype=np.float64)
    else:
        toggle = (values[1:] != values[:-1]).mean(axis=0, dtype=np.float64)
    return np.stack([prob_one, toggle], axis=-1).astype(np.float32)


def simulate_behavior(
    x: np.ndarray,
    edge_index: np.ndarray,
    pi_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    values = simulate_aig(x, edge_index, pi_values)
    return behavior_from_values(values), values

