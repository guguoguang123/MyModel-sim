from __future__ import annotations

import argparse
import time

import numpy as np

from build_behavior_labels import simulate_workload_average
from drgate_sim.io import load_config, load_graphs, save_json
from drgate_sim.labels import build_response_labels
from drgate_sim.simulator import find_primary_inputs
from drgate_sim.workloads import debug_workloads, default_workloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check behavior-label stability across stochastic workload realization counts."
    )
    parser.add_argument("--config", type=str, default="simProject/configs/default.json")
    parser.add_argument("--graphs", type=str, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--num-vectors", type=int, default=None)
    parser.add_argument("--realizations", type=str, default="4,8,16,32")
    parser.add_argument("--workload-mode", choices=["debug", "default"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def parse_realizations(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or values[0] < 1:
        raise ValueError("--realizations must contain positive integers")
    return values


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= eps:
        return float("nan")
    return float(np.dot(a, b) / denom)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_corr(rankdata(a), rankdata(b))


def topk_overlap(a: np.ndarray, b: np.ndarray, fraction: float = 0.05) -> float:
    k = max(1, int(round(len(a) * fraction)))
    top_a = set(np.argpartition(a, -k)[-k:].tolist())
    top_b = set(np.argpartition(b, -k)[-k:].tolist())
    return len(top_a & top_b) / k


def cosine_mean(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> tuple[float, int]:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denom > eps
    if not np.any(valid):
        return float("nan"), 0
    cosine = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return float(np.mean(cosine)), int(np.sum(valid))


def select_circuits(graphs: dict, start: int, end: int | None, limit: int | None) -> list[str]:
    names = list(graphs.keys())
    if end is None:
        names = names[start:]
    else:
        names = names[start:end]
    if limit is not None:
        names = names[:limit]
    return names


def compute_labels_for_m(
    graphs: dict,
    circuit_names: list[str],
    specs,
    num_vectors: int,
    seed: int,
    num_realizations: int,
) -> dict[str, dict]:
    labels = {}
    started = time.time()
    for idx, name in enumerate(circuit_names, start=1):
        graph = graphs[name]
        x = graph["x"]
        edge_index = graph["edge_index"]
        num_pis = len(find_primary_inputs(x))
        B = []
        for spec in specs:
            B.append(
                simulate_workload_average(
                    spec=spec,
                    x=x,
                    edge_index=edge_index,
                    num_vectors=num_vectors,
                    num_pis=num_pis,
                    global_seed=seed,
                    circuit_name=name,
                    num_realizations=num_realizations,
                )
            )
        item = build_response_labels(np.stack(B, axis=0))
        labels[name] = item
        elapsed = time.time() - started
        print(
            f"[M={num_realizations:>2}] [{idx:>4}/{len(circuit_names)}] "
            f"{name}: nodes={x.shape[0]} pis={num_pis} elapsed={elapsed:.1f}s",
            flush=True,
        )
    return labels


def stack_field(labels: dict[str, dict], circuit_names: list[str], field: str) -> np.ndarray:
    return np.concatenate([labels[name][field] for name in circuit_names], axis=0)


def compare_to_reference(
    labels_by_m: dict[int, dict[str, dict]],
    circuit_names: list[str],
    reference_m: int,
) -> list[dict[str, float | int]]:
    ref_s = stack_field(labels_by_m[reference_m], circuit_names, "S")
    ref_sens = stack_field(labels_by_m[reference_m], circuit_names, "sensitivity")
    ref_norm = float(np.linalg.norm(ref_s))
    rows = []
    for m in sorted(labels_by_m):
        cur_s = stack_field(labels_by_m[m], circuit_names, "S")
        cur_sens = stack_field(labels_by_m[m], circuit_names, "sensitivity")
        diff = cur_s - ref_s
        mean_cos, valid_cos_nodes = cosine_mean(cur_s, ref_s)
        rows.append(
            {
                "num_realizations": m,
                "reference_realizations": reference_m,
                "num_nodes": int(cur_s.shape[0]),
                "mean_abs_S_diff": float(np.mean(np.abs(diff))),
                "mean_node_l2_S_diff": float(np.mean(np.linalg.norm(diff, axis=1))),
                "relative_global_l2_S_diff": float(np.linalg.norm(diff) / max(ref_norm, 1e-12)),
                "mean_S_cosine": mean_cos,
                "valid_cosine_nodes": valid_cos_nodes,
                "sensitivity_spearman": spearman_corr(cur_sens, ref_sens),
                "sensitivity_top5pct_overlap": topk_overlap(cur_sens, ref_sens, fraction=0.05),
            }
        )
    return rows


def print_table(rows: list[dict[str, float | int]]) -> None:
    print()
    print("Stability relative to the largest M:")
    print(
        "M    rel_l2_S    mean_l2_S   mean_abs_S  mean_cos_S  sens_spear  top5_overlap"
    )
    for row in rows:
        print(
            f"{int(row['num_realizations']):<4} "
            f"{row['relative_global_l2_S_diff']:<11.6g} "
            f"{row['mean_node_l2_S_diff']:<11.6g} "
            f"{row['mean_abs_S_diff']:<11.6g} "
            f"{row['mean_S_cosine']:<11.6g} "
            f"{row['sensitivity_spearman']:<11.6g} "
            f"{row['sensitivity_top5pct_overlap']:<11.6g}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    graphs_path = args.graphs or cfg["graphs_path"]
    num_vectors = int(args.num_vectors or cfg["num_vectors"])
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 20260710))
    workload_mode = args.workload_mode or cfg.get("workload_mode", "default")
    start = int(args.start if args.start is not None else cfg.get("start", 0))
    end = args.end if args.end is not None else cfg.get("end")
    end = None if end is None else int(end)
    limit = args.limit if args.limit is not None else cfg.get("limit")
    limit = None if limit is None else int(limit)
    realization_values = parse_realizations(args.realizations)

    specs = debug_workloads() if workload_mode == "debug" else default_workloads()
    graphs = load_graphs(graphs_path)
    circuit_names = select_circuits(graphs, start=start, end=end, limit=limit)
    if not circuit_names:
        raise ValueError("No circuits selected")

    print(
        f"Checking {len(circuit_names)} circuits, workloads={len(specs)}, "
        f"num_vectors={num_vectors}, realizations={realization_values}",
        flush=True,
    )

    labels_by_m = {}
    for m in realization_values:
        labels_by_m[m] = compute_labels_for_m(
            graphs=graphs,
            circuit_names=circuit_names,
            specs=specs,
            num_vectors=num_vectors,
            seed=seed,
            num_realizations=m,
        )

    reference_m = realization_values[-1]
    rows = compare_to_reference(labels_by_m, circuit_names, reference_m)
    print_table(rows)

    result = {
        "graphs_path": str(graphs_path),
        "num_circuits": len(circuit_names),
        "circuit_names": circuit_names,
        "num_vectors": num_vectors,
        "seed": seed,
        "workload_mode": workload_mode,
        "num_workloads": len(specs),
        "realizations": realization_values,
        "reference_realizations": reference_m,
        "metrics": rows,
    }
    if args.output_json:
        save_json(args.output_json, result)
        print(f"[DONE] saved stability metrics: {args.output_json}")


if __name__ == "__main__":
    main()
