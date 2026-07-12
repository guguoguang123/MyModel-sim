from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PI_GATE = 0
AND_GATE = 1
NOT_GATE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate DRGate workload-response behavior labels."
    )
    parser.add_argument("--graphs", default="RelatedData/graphs.npz")
    parser.add_argument("--labels", default="simProject/outputs/behavior_labels.npz")
    parser.add_argument("--output-dir", default="simProject/outputs/label_validation")
    parser.add_argument("--zero-threshold", type=float, default=1e-6)
    parser.add_argument("--spearman-sample", type=int, default=200_000)
    parser.add_argument("--examples-per-band", type=int, default=3)
    return parser.parse_args()


def load_npz_dict(path: str | Path, key: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return data[key].item()


def load_label_file(path: str | Path) -> tuple[dict[str, dict], dict]:
    data = np.load(path, allow_pickle=True)
    labels = data["labels"].item()
    meta = data["meta"].item() if "meta" in data.files else {}
    return labels, meta


def as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [as_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return as_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(as_jsonable(data), f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantile_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q01": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q99": None,
            "max": None,
        }
    q01, q25, q50, q75, q99 = np.quantile(values, [0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "q01": float(q01),
        "q25": float(q25),
        "median": float(q50),
        "q75": float(q75),
        "q99": float(q99),
        "max": float(values.max()),
    }


def flatten_stats(prefix: str, stats: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


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


def gate_masks(x: np.ndarray) -> dict[str, np.ndarray]:
    gate_types = x[:, 1].astype(np.int64)
    return {
        "all": np.ones(x.shape[0], dtype=bool),
        "pi": gate_types == PI_GATE,
        "internal": gate_types != PI_GATE,
        "and": gate_types == AND_GATE,
        "not": gate_types == NOT_GATE,
    }


def get_workload_names(meta: dict, num_workloads: int) -> list[str]:
    workloads = meta.get("workloads") or []
    names = [str(item.get("name", f"workload_{idx}")) for idx, item in enumerate(workloads)]
    if len(names) != num_workloads:
        names = [f"workload_{idx}" for idx in range(num_workloads)]
    return names


def concentration(values: np.ndarray, fractions: tuple[float, ...] = (0.01, 0.05, 0.10)) -> dict:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    result = {}
    if values.size == 0 or total <= 0:
        for fraction in fractions:
            result[f"top_{int(fraction * 100)}pct_share"] = None
        return result
    sorted_values = np.sort(values)[::-1]
    for fraction in fractions:
        k = max(1, int(math.ceil(values.size * fraction)))
        result[f"top_{int(fraction * 100)}pct_share"] = float(sorted_values[:k].sum() / total)
    return result


def matrix_rows(
    names: list[str],
    matrix: np.ndarray,
    metric: str,
) -> list[dict[str, Any]]:
    rows = []
    for i, left in enumerate(names):
        row = {"metric": metric, "workload": left}
        for j, right in enumerate(names):
            row[right] = float(matrix[i, j])
        rows.append(row)
    return rows


def find_pair_extremes(
    names: list[str],
    matrix: np.ndarray,
    largest_is_most_similar: bool,
) -> dict[str, Any]:
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = float(matrix[i, j])
            pairs.append({"left": names[i], "right": names[j], "value": value})
    pairs.sort(key=lambda row: row["value"], reverse=largest_is_most_similar)
    return {"top": pairs[:5], "bottom": pairs[-5:]}


def select_node_examples(
    labels: dict[str, dict],
    graphs: dict[str, dict],
    median_sensitivity: float,
    examples_per_band: int,
) -> list[dict[str, Any]]:
    high: list[dict[str, Any]] = []
    low: list[dict[str, Any]] = []
    medium: list[dict[str, Any]] = []

    for circuit_name, item in labels.items():
        x = graphs[circuit_name]["x"]
        internal = x[:, 1].astype(np.int64) != PI_GATE
        sensitivity = np.asarray(item["sensitivity_var"], dtype=np.float64)
        internal_nodes = np.flatnonzero(internal)
        if internal_nodes.size == 0:
            continue
        for node in internal_nodes:
            value = float(sensitivity[node])
            record = {
                "circuit": circuit_name,
                "node": int(node),
                "gate_type": int(x[node, 1]),
                "level": int(x[node, 2]) if x.shape[1] >= 3 else None,
                "sensitivity": value,
            }
            high.append(record)
            high.sort(key=lambda row: row["sensitivity"], reverse=True)
            del high[examples_per_band:]

            low.append(record)
            low.sort(key=lambda row: row["sensitivity"])
            del low[examples_per_band:]

            medium_record = dict(record)
            medium_record["median_abs_diff"] = abs(value - median_sensitivity)
            medium.append(medium_record)
            medium.sort(key=lambda row: row["median_abs_diff"])
            del medium[examples_per_band:]

    for row in high:
        row["band"] = "high"
    for row in medium:
        row["band"] = "medium"
        row.pop("median_abs_diff", None)
    for row in low:
        row["band"] = "low"
    return high + medium + low


def try_make_plots(
    output_dir: Path,
    workload_names: list[str],
    pearson: np.ndarray,
    mae: np.ndarray,
    examples: list[dict[str, Any]],
    labels: dict[str, dict],
) -> list[str]:
    plot_paths: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[WARN] matplotlib unavailable, skipping plots: {exc}")
        return plot_paths

    def heatmap(matrix: np.ndarray, title: str, filename: str, fmt: str) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 7.2))
        im = ax.imshow(matrix, cmap="viridis")
        ax.set_xticks(np.arange(len(workload_names)), labels=workload_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(workload_names)), labels=workload_names)
        for i in range(len(workload_names)):
            for j in range(len(workload_names)):
                ax.text(j, i, format(float(matrix[i, j]), fmt), ha="center", va="center", fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(path))

    heatmap(pearson, "Workload Pearson correlation", "workload_pearson_heatmap.png", ".3f")
    heatmap(mae, "Workload mean absolute difference", "workload_mae_heatmap.png", ".4f")

    for idx, example in enumerate(examples):
        item = labels[example["circuit"]]
        node = int(example["node"])
        B = np.asarray(item["B"])[:, node, :]
        R = np.asarray(item["R"])[:, node, :]
        x_axis = np.arange(len(workload_names))

        fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True)
        axes[0].plot(x_axis, B[:, 0], marker="o", label="P(v=1)")
        axes[0].plot(x_axis, B[:, 1], marker="s", label="P(toggle)")
        axes[0].set_ylabel("B")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.25)

        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].plot(x_axis, R[:, 0], marker="o", label="R prob")
        axes[1].plot(x_axis, R[:, 1], marker="s", label="R toggle")
        axes[1].set_ylabel("Residual")
        axes[1].legend(loc="best")
        axes[1].grid(alpha=0.25)
        axes[1].set_xticks(x_axis, labels=workload_names, rotation=45, ha="right")

        title = (
            f"{example['band']} sensitivity node: {example['circuit']}:{node}, "
            f"sens={example['sensitivity']:.6g}"
        )
        fig.suptitle(title)
        fig.tight_layout()
        path = output_dir / f"node_response_{idx:02d}_{example['band']}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(path))

    return plot_paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] graphs: {args.graphs}", flush=True)
    graphs = load_npz_dict(args.graphs, "circuits")
    print(f"[LOAD] labels: {args.labels}", flush=True)
    labels, meta = load_label_file(args.labels)

    first_item = next(iter(labels.values()))
    num_workloads = int(first_item["B"].shape[0])
    behavior_dim = int(first_item["B"].shape[2])
    workload_names = get_workload_names(meta, num_workloads)
    if behavior_dim != 2:
        print(f"[WARN] expected behavior_dim=2, got {behavior_dim}")

    integrity = {
        "num_graphs": len(graphs),
        "num_labels": len(labels),
        "missing_in_labels": sorted(set(graphs) - set(labels))[:20],
        "missing_in_graphs": sorted(set(labels) - set(graphs))[:20],
        "num_missing_in_labels": len(set(graphs) - set(labels)),
        "num_missing_in_graphs": len(set(labels) - set(graphs)),
        "bad_circuits": [],
        "max_abs_R_mean": 0.0,
        "max_abs_R_minus_B_minus_mu": 0.0,
        "max_abs_S_minus_flat_R": 0.0,
        "max_abs_sensitivity_var": 0.0,
        "max_abs_sensitivity_alias": 0.0,
        "max_abs_sensitivity_sum": 0.0,
        "B_min": float("inf"),
        "B_max": float("-inf"),
        "B_out_of_range_count": 0,
        "nan_or_inf_count": 0,
    }

    counts = {
        "circuits": 0,
        "nodes": 0,
        "pi_nodes": 0,
        "internal_nodes": 0,
        "and_nodes": 0,
        "not_nodes": 0,
    }

    dist_values = {
        group: {
            "prob_one": [[] for _ in range(num_workloads)],
            "toggle": [[] for _ in range(num_workloads)],
        }
        for group in ("all", "pi", "internal")
    }
    sensitivity_values = {group: [] for group in ("all", "pi", "internal", "and", "not")}
    signature_norm_values = {group: [] for group in ("all", "pi", "internal", "and", "not")}
    level_rows_accumulator: dict[int, dict[str, float]] = {}

    sum_x = np.zeros(num_workloads, dtype=np.float64)
    sum_x2 = np.zeros(num_workloads, dtype=np.float64)
    cross = np.zeros((num_workloads, num_workloads), dtype=np.float64)
    abs_diff = np.zeros((num_workloads, num_workloads), dtype=np.float64)
    sq_diff = np.zeros((num_workloads, num_workloads), dtype=np.float64)
    similarity_count = 0
    spearman_samples: list[np.ndarray] = []
    per_circuit_sample = max(1, args.spearman_sample // max(1, len(labels)))
    rng = np.random.default_rng(20260712)

    for idx, (name, item) in enumerate(labels.items(), start=1):
        if name not in graphs:
            continue
        graph = graphs[name]
        x = np.asarray(graph["x"])
        num_nodes = x.shape[0]
        masks = gate_masks(x)
        B = np.asarray(item["B"])
        R = np.asarray(item["R"])
        S = np.asarray(item["S"])
        mu = np.asarray(item["mu"])
        sensitivity = np.asarray(item["sensitivity"])
        sensitivity_var = np.asarray(item["sensitivity_var"])
        sensitivity_sum = np.asarray(item["sensitivity_sum"])
        num_pis = int(np.asarray(item["num_pis"]).item())

        expected_shapes = {
            "B": (num_workloads, num_nodes, behavior_dim),
            "R": (num_workloads, num_nodes, behavior_dim),
            "S": (num_nodes, num_workloads * behavior_dim),
            "mu": (num_nodes, behavior_dim),
            "sensitivity": (num_nodes,),
            "sensitivity_var": (num_nodes,),
            "sensitivity_sum": (num_nodes,),
        }
        bad_shape = [
            f"{field}: expected {shape}, got {np.asarray(item[field]).shape}"
            for field, shape in expected_shapes.items()
            if np.asarray(item[field]).shape != shape
        ]
        true_num_pis = int(np.sum(masks["pi"]))
        if bad_shape or num_pis != true_num_pis:
            integrity["bad_circuits"].append(
                {
                    "circuit": name,
                    "issues": bad_shape
                    + ([f"num_pis: expected {true_num_pis}, got {num_pis}"] if num_pis != true_num_pis else []),
                }
            )
            continue

        finite_arrays = [B, R, S, mu, sensitivity, sensitivity_var, sensitivity_sum]
        integrity["nan_or_inf_count"] += int(
            sum(np.size(arr) - np.count_nonzero(np.isfinite(arr)) for arr in finite_arrays)
        )
        integrity["B_min"] = min(integrity["B_min"], float(np.nanmin(B)))
        integrity["B_max"] = max(integrity["B_max"], float(np.nanmax(B)))
        integrity["B_out_of_range_count"] += int(np.count_nonzero((B < -1e-6) | (B > 1.0 + 1e-6)))

        expected_R = B - mu[None, :, :]
        expected_S = np.transpose(R, (1, 0, 2)).reshape(num_nodes, num_workloads * behavior_dim)
        squared_response = np.square(R, dtype=np.float32)
        expected_sensitivity_var = np.mean(squared_response, axis=0).sum(axis=1).astype(np.float32)
        expected_sensitivity_sum = np.sum(squared_response, axis=(0, 2)).astype(np.float32)
        integrity["max_abs_R_mean"] = max(
            integrity["max_abs_R_mean"], float(np.max(np.abs(R.mean(axis=0))))
        )
        integrity["max_abs_R_minus_B_minus_mu"] = max(
            integrity["max_abs_R_minus_B_minus_mu"], float(np.max(np.abs(R - expected_R)))
        )
        integrity["max_abs_S_minus_flat_R"] = max(
            integrity["max_abs_S_minus_flat_R"], float(np.max(np.abs(S - expected_S)))
        )
        integrity["max_abs_sensitivity_var"] = max(
            integrity["max_abs_sensitivity_var"],
            float(np.max(np.abs(sensitivity_var - expected_sensitivity_var))),
        )
        integrity["max_abs_sensitivity_alias"] = max(
            integrity["max_abs_sensitivity_alias"],
            float(np.max(np.abs(sensitivity - sensitivity_var))),
        )
        integrity["max_abs_sensitivity_sum"] = max(
            integrity["max_abs_sensitivity_sum"],
            float(np.max(np.abs(sensitivity_sum - expected_sensitivity_sum))),
        )

        counts["circuits"] += 1
        counts["nodes"] += num_nodes
        counts["pi_nodes"] += int(np.sum(masks["pi"]))
        counts["internal_nodes"] += int(np.sum(masks["internal"]))
        counts["and_nodes"] += int(np.sum(masks["and"]))
        counts["not_nodes"] += int(np.sum(masks["not"]))

        for group in dist_values:
            mask = masks[group]
            if not np.any(mask):
                continue
            for workload_idx in range(num_workloads):
                dist_values[group]["prob_one"][workload_idx].append(B[workload_idx, mask, 0])
                dist_values[group]["toggle"][workload_idx].append(B[workload_idx, mask, 1])

        node_s_norm = np.linalg.norm(S.astype(np.float64), axis=1)
        for group in sensitivity_values:
            mask = masks[group]
            if np.any(mask):
                sensitivity_values[group].append(sensitivity_var[mask])
                signature_norm_values[group].append(node_s_norm[mask])

        if x.shape[1] >= 3:
            levels = x[:, 2].astype(np.int64)
            for level in np.unique(levels[masks["internal"]]):
                level_mask = masks["internal"] & (levels == level)
                row = level_rows_accumulator.setdefault(
                    int(level), {"count": 0.0, "sensitivity_sum": 0.0, "signature_norm_sum": 0.0}
                )
                row["count"] += float(np.sum(level_mask))
                row["sensitivity_sum"] += float(sensitivity_var[level_mask].sum())
                row["signature_norm_sum"] += float(node_s_norm[level_mask].sum())

        internal = masks["internal"]
        if np.any(internal):
            X = B[:, internal, :].reshape(num_workloads, -1).astype(np.float64)
            m = X.shape[1]
            similarity_count += m
            sum_x += X.sum(axis=1)
            sum_x2 += np.square(X).sum(axis=1)
            cross += X @ X.T
            for left in range(num_workloads):
                for right in range(num_workloads):
                    diff = X[left] - X[right]
                    abs_diff[left, right] += float(np.abs(diff).sum())
                    sq_diff[left, right] += float(np.square(diff).sum())

            take = min(per_circuit_sample, m)
            if take > 0:
                sample_idx = rng.choice(m, size=take, replace=False)
                spearman_samples.append(X[:, sample_idx].astype(np.float32))

        if idx == 1 or idx == len(labels) or idx % 1000 == 0:
            print(f"[SCAN] {idx}/{len(labels)} {name}", flush=True)

    distribution_rows: list[dict[str, Any]] = []
    distribution_summary: dict[str, Any] = {}
    for group, by_kind in dist_values.items():
        distribution_summary[group] = {}
        for workload_idx, workload_name in enumerate(workload_names):
            prob = np.concatenate(by_kind["prob_one"][workload_idx]) if by_kind["prob_one"][workload_idx] else np.array([])
            toggle = np.concatenate(by_kind["toggle"][workload_idx]) if by_kind["toggle"][workload_idx] else np.array([])
            prob_stats = quantile_stats(prob)
            toggle_stats = quantile_stats(toggle)
            distribution_summary[group][workload_name] = {
                "prob_one": prob_stats,
                "toggle": toggle_stats,
            }
            distribution_rows.append(
                {
                    "group": group,
                    "workload": workload_name,
                    **flatten_stats("prob_one", prob_stats),
                    **flatten_stats("toggle", toggle_stats),
                }
            )

    sensitivity_summary: dict[str, Any] = {}
    sensitivity_rows: list[dict[str, Any]] = []
    for group in sensitivity_values:
        sens = np.concatenate(sensitivity_values[group]) if sensitivity_values[group] else np.array([])
        norm = np.concatenate(signature_norm_values[group]) if signature_norm_values[group] else np.array([])
        sens_stats = quantile_stats(sens)
        norm_stats = quantile_stats(norm)
        zero_fracs = {
            f"signature_norm_le_{thr:g}": float(np.mean(norm <= thr)) if norm.size else None
            for thr in (1e-8, args.zero_threshold, 1e-4)
        }
        zero_fracs.update(
            {
                f"sensitivity_le_{thr:g}": float(np.mean(sens <= thr)) if sens.size else None
                for thr in (1e-12, args.zero_threshold, 1e-4)
            }
        )
        sensitivity_summary[group] = {
            "sensitivity": sens_stats,
            "signature_norm": norm_stats,
            "zero_fractions": zero_fracs,
            "concentration": concentration(sens),
        }
        sensitivity_rows.append(
            {
                "group": group,
                **flatten_stats("sensitivity", sens_stats),
                **flatten_stats("signature_norm", norm_stats),
                **zero_fracs,
                **concentration(sens),
            }
        )

    mean = sum_x / max(1, similarity_count)
    var = np.maximum(sum_x2 / max(1, similarity_count) - np.square(mean), 0.0)
    pearson = np.eye(num_workloads, dtype=np.float64)
    for left in range(num_workloads):
        for right in range(num_workloads):
            cov = cross[left, right] / max(1, similarity_count) - mean[left] * mean[right]
            denom = math.sqrt(var[left] * var[right])
            pearson[left, right] = cov / denom if denom > 1e-12 else float("nan")
    mae = abs_diff / max(1, similarity_count)
    rmse = np.sqrt(sq_diff / max(1, similarity_count))

    spearman = np.full((num_workloads, num_workloads), np.nan, dtype=np.float64)
    spearman_sample_count = 0
    if spearman_samples:
        sample = np.concatenate(spearman_samples, axis=1)
        if sample.shape[1] > args.spearman_sample:
            idx = rng.choice(sample.shape[1], size=args.spearman_sample, replace=False)
            sample = sample[:, idx]
        spearman_sample_count = int(sample.shape[1])
        for left in range(num_workloads):
            for right in range(num_workloads):
                spearman[left, right] = spearman_corr(sample[left], sample[right])

    similarity_summary = {
        "num_internal_behavior_values": int(similarity_count),
        "spearman_sample_values": spearman_sample_count,
        "pearson": pearson,
        "spearman_sample": spearman,
        "mae": mae,
        "rmse": rmse,
        "most_similar_by_pearson": find_pair_extremes(workload_names, pearson, True)["top"],
        "least_similar_by_pearson": find_pair_extremes(workload_names, pearson, True)["bottom"],
        "smallest_mae": find_pair_extremes(workload_names, mae, False)["top"],
        "largest_mae": find_pair_extremes(workload_names, mae, False)["bottom"],
    }

    internal_sensitivity = (
        np.concatenate(sensitivity_values["internal"]) if sensitivity_values["internal"] else np.array([])
    )
    median_sensitivity = float(np.median(internal_sensitivity)) if internal_sensitivity.size else 0.0
    examples = select_node_examples(labels, graphs, median_sensitivity, args.examples_per_band)
    example_rows: list[dict[str, Any]] = []
    for example in examples:
        item = labels[example["circuit"]]
        node = int(example["node"])
        B = np.asarray(item["B"])[:, node, :]
        R = np.asarray(item["R"])[:, node, :]
        for workload_idx, workload_name in enumerate(workload_names):
            example_rows.append(
                {
                    **example,
                    "workload": workload_name,
                    "prob_one": float(B[workload_idx, 0]),
                    "toggle": float(B[workload_idx, 1]),
                    "residual_prob_one": float(R[workload_idx, 0]),
                    "residual_toggle": float(R[workload_idx, 1]),
                }
            )

    level_rows = []
    for level in sorted(level_rows_accumulator):
        row = level_rows_accumulator[level]
        count = max(row["count"], 1.0)
        level_rows.append(
            {
                "level": level,
                "internal_node_count": int(row["count"]),
                "mean_sensitivity": row["sensitivity_sum"] / count,
                "mean_signature_norm": row["signature_norm_sum"] / count,
            }
        )

    plot_paths = try_make_plots(output_dir, workload_names, pearson, mae, examples, labels)

    integrity_pass = (
        integrity["num_missing_in_labels"] == 0
        and integrity["num_missing_in_graphs"] == 0
        and not integrity["bad_circuits"]
        and integrity["nan_or_inf_count"] == 0
        and integrity["B_out_of_range_count"] == 0
        and integrity["max_abs_R_mean"] <= 1e-6
        and integrity["max_abs_R_minus_B_minus_mu"] <= 1e-6
        and integrity["max_abs_S_minus_flat_R"] <= 1e-6
        and integrity["max_abs_sensitivity_var"] <= 1e-6
        and integrity["max_abs_sensitivity_alias"] <= 1e-12
        and integrity["max_abs_sensitivity_sum"] <= 1e-5
    )

    report = {
        "labels_path": str(args.labels),
        "graphs_path": str(args.graphs),
        "output_dir": str(output_dir),
        "workload_names": workload_names,
        "meta": {
            "num_vectors": meta.get("num_vectors"),
            "num_realizations": meta.get("num_realizations"),
            "seed": meta.get("seed"),
            "workload_mode": meta.get("workload_mode"),
        },
        "integrity_pass": integrity_pass,
        "integrity": integrity,
        "counts": counts,
        "distribution_summary": distribution_summary,
        "sensitivity_summary": sensitivity_summary,
        "similarity_summary": similarity_summary,
        "examples": examples,
        "plots": plot_paths,
    }

    write_json(output_dir / "validation_report.json", report)
    write_csv(output_dir / "workload_distribution.csv", distribution_rows)
    write_csv(output_dir / "sensitivity_summary.csv", sensitivity_rows)
    write_csv(output_dir / "workload_pearson.csv", matrix_rows(workload_names, pearson, "pearson"))
    write_csv(output_dir / "workload_spearman_sample.csv", matrix_rows(workload_names, spearman, "spearman_sample"))
    write_csv(output_dir / "workload_mae.csv", matrix_rows(workload_names, mae, "mae"))
    write_csv(output_dir / "workload_rmse.csv", matrix_rows(workload_names, rmse, "rmse"))
    write_csv(output_dir / "level_sensitivity.csv", level_rows)
    write_csv(output_dir / "node_response_examples.csv", example_rows)

    lines = [
        "# Behavior Label Validation",
        "",
        f"labels: {args.labels}",
        f"graphs: {args.graphs}",
        f"integrity_pass: {integrity_pass}",
        "",
        "## Counts",
        *(f"- {key}: {value}" for key, value in counts.items()),
        "",
        "## Integrity Max Errors",
        f"- missing graph/label keys: labels_missing={integrity['num_missing_in_labels']}, graphs_missing={integrity['num_missing_in_graphs']}",
        f"- bad circuits: {len(integrity['bad_circuits'])}",
        f"- NaN/Inf count: {integrity['nan_or_inf_count']}",
        f"- B range: [{integrity['B_min']:.8g}, {integrity['B_max']:.8g}], out_of_range={integrity['B_out_of_range_count']}",
        f"- max_abs_R_mean: {integrity['max_abs_R_mean']:.8g}",
        f"- max_abs_R_minus_B_minus_mu: {integrity['max_abs_R_minus_B_minus_mu']:.8g}",
        f"- max_abs_S_minus_flat_R: {integrity['max_abs_S_minus_flat_R']:.8g}",
        f"- max_abs_sensitivity_var: {integrity['max_abs_sensitivity_var']:.8g}",
        f"- max_abs_sensitivity_alias: {integrity['max_abs_sensitivity_alias']:.8g}",
        f"- max_abs_sensitivity_sum: {integrity['max_abs_sensitivity_sum']:.8g}",
        "",
        "## Internal Workload Means",
    ]
    for workload_name in workload_names:
        stats = distribution_summary["internal"][workload_name]
        lines.append(
            "- "
            f"{workload_name}: "
            f"P1_mean={stats['prob_one']['mean']:.6g}, "
            f"P1_median={stats['prob_one']['median']:.6g}, "
            f"T_mean={stats['toggle']['mean']:.6g}, "
            f"T_median={stats['toggle']['median']:.6g}"
        )
    lines.extend(["", "## Internal Sensitivity"])
    internal_summary = sensitivity_summary["internal"]
    lines.extend(
        [
            f"- S_norm median: {internal_summary['signature_norm']['median']:.8g}",
            f"- S_norm q99: {internal_summary['signature_norm']['q99']:.8g}",
            f"- fraction S_norm <= {args.zero_threshold:g}: {internal_summary['zero_fractions'][f'signature_norm_le_{args.zero_threshold:g}']:.6g}",
            f"- sensitivity median: {internal_summary['sensitivity']['median']:.8g}",
            f"- sensitivity q99: {internal_summary['sensitivity']['q99']:.8g}",
            f"- sensitivity top 1 pct share: {internal_summary['concentration']['top_1pct_share']:.6g}",
            f"- sensitivity top 10 pct share: {internal_summary['concentration']['top_10pct_share']:.6g}",
            "",
            "## Workload Similarity",
            f"- internal behavior values compared: {similarity_count}",
            f"- spearman sample values: {spearman_sample_count}",
        ]
    )
    for pair in similarity_summary["most_similar_by_pearson"]:
        lines.append(f"- high pearson {pair['left']} vs {pair['right']}: {pair['value']:.8g}")
    for pair in similarity_summary["smallest_mae"]:
        lines.append(f"- low MAE {pair['left']} vs {pair['right']}: {pair['value']:.8g}")
    lines.extend(["", "## Outputs"])
    for rel in [
        "validation_report.json",
        "workload_distribution.csv",
        "sensitivity_summary.csv",
        "workload_pearson.csv",
        "workload_spearman_sample.csv",
        "workload_mae.csv",
        "workload_rmse.csv",
        "level_sensitivity.csv",
        "node_response_examples.csv",
    ]:
        lines.append(f"- {output_dir / rel}")
    for path in plot_paths:
        lines.append(f"- {path}")

    (output_dir / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] wrote validation report to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
