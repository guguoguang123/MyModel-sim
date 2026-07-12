from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np

from drgate_sim.io import load_config, load_graphs, save_behavior_labels, save_json
from drgate_sim.labels import build_response_labels
from drgate_sim.simulator import find_primary_inputs, simulate_behavior
from drgate_sim.workloads import (
    debug_workloads,
    default_workloads,
    generate_pi_sequence,
    needs_realization_average,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DRGate workload-response behavior labels.")
    parser.add_argument("--config", type=str, default="simProject/configs/debug.json")
    parser.add_argument("--graphs", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--meta-output", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--num-vectors", type=int, default=None)
    parser.add_argument("--num-realizations", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--workload-mode", choices=["debug", "default"], default=None)
    return parser.parse_args()


def stable_seed(global_seed: int, *parts: object) -> int:
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(str(global_seed).encode("utf-8"))
    for part in parts:
        hasher.update(b"\0")
        hasher.update(str(part).encode("utf-8"))
    return int.from_bytes(hasher.digest(), byteorder="little", signed=False)


def realization_count_for(spec, num_realizations: int) -> int:
    if needs_realization_average(spec):
        return max(1, int(num_realizations))
    return 1


def workload_metadata(specs, num_realizations: int) -> list[dict]:
    meta = []
    for spec in specs:
        item = spec.to_dict()
        item["realization_count"] = realization_count_for(spec, num_realizations)
        item["averaged_realizations"] = needs_realization_average(spec)
        meta.append(item)
    return meta


def simulate_workload_average(
    spec,
    x: np.ndarray,
    edge_index: np.ndarray,
    num_vectors: int,
    num_pis: int,
    global_seed: int,
    circuit_name: str,
    num_realizations: int,
) -> np.ndarray:
    realization_count = realization_count_for(spec, num_realizations)
    behavior_sum = None
    for realization_idx in range(realization_count):
        rng = np.random.default_rng(
            stable_seed(global_seed, circuit_name, spec.name, realization_idx)
        )
        pi_values = generate_pi_sequence(spec, num_vectors, num_pis, rng)
        behavior, _ = simulate_behavior(x, edge_index, pi_values)
        if behavior_sum is None:
            behavior_sum = behavior.astype(np.float64)
        else:
            behavior_sum += behavior
    return (behavior_sum / realization_count).astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    graphs_path = args.graphs or cfg["graphs_path"]
    output_path = args.output or cfg["output_path"]
    meta_output_path = args.meta_output or cfg["meta_output_path"]
    num_vectors = int(args.num_vectors or cfg["num_vectors"])
    num_realizations = int(args.num_realizations or cfg.get("num_realizations", 1))
    if num_realizations < 1:
        raise ValueError("num_realizations must be >= 1")
    limit = args.limit if args.limit is not None else cfg.get("limit")
    start = args.start if args.start is not None else int(cfg.get("start", 0))
    end = args.end if args.end is not None else cfg.get("end")
    seed = int(cfg.get("seed", 20260710))
    log_every = int(args.log_every or cfg.get("log_every", 1))
    if log_every < 1:
        raise ValueError("log_every must be >= 1")
    workload_mode = args.workload_mode or cfg.get("workload_mode", "default")

    specs = debug_workloads() if workload_mode == "debug" else default_workloads()
    workload_meta = workload_metadata(specs, num_realizations)
    graphs = load_graphs(graphs_path)
    circuit_names = list(graphs.keys())
    if end is None:
        circuit_names = circuit_names[start:]
    else:
        circuit_names = circuit_names[start : int(end)]
    if limit is not None:
        circuit_names = circuit_names[: int(limit)]

    labels = {}
    started = time.time()
    max_realizations = max(item["realization_count"] for item in workload_meta)
    for idx, name in enumerate(circuit_names, start=1):
        graph = graphs[name]
        x = graph["x"]
        edge_index = graph["edge_index"]
        pi_nodes = find_primary_inputs(x)
        B_list = []
        for spec in specs:
            behavior = simulate_workload_average(
                spec=spec,
                x=x,
                edge_index=edge_index,
                num_vectors=num_vectors,
                num_pis=len(pi_nodes),
                global_seed=seed,
                circuit_name=name,
                num_realizations=num_realizations,
            )
            B_list.append(behavior)

        B = np.stack(B_list, axis=0)
        item = build_response_labels(B)
        item["num_pis"] = np.array(len(pi_nodes), dtype=np.int32)
        labels[name] = item

        if idx == 1 or idx == len(circuit_names) or idx % log_every == 0:
            elapsed = time.time() - started
            print(
                f"[{idx:>5}/{len(circuit_names)}] {name}: "
                f"nodes={x.shape[0]} pis={len(pi_nodes)} workloads={len(specs)} "
                f"max_realizations={max_realizations} elapsed={elapsed:.1f}s",
                flush=True,
            )

    meta = {
        "simulation_type": "AIG-level stable Boolean simulation",
        "label_objective": "behavior-response embedding h_b over workload probes",
        "model_input_assumption": "G and node v only; S_v is a supervision target, not a model input",
        "behavior_definition": "B_v(D_k) = [P(v=1 | D_k), P(toggle_v | D_k)]",
        "response_signature_definition": "mu_v = mean_k B_v(D_k); R_v(D_k) = B_v(D_k) - mu_v; S_v = concat_k R_v(D_k)",
        "stochastic_probe_policy": "stochastic workload probes are averaged over independent realizations before B/R/S construction",
        "graphs_path": str(graphs_path),
        "num_circuits": len(circuit_names),
        "start": start,
        "end": end,
        "limit": limit,
        "num_vectors": num_vectors,
        "num_realizations": num_realizations,
        "log_every": log_every,
        "num_workloads": len(specs),
        "behavior_dim": 2,
        "seed": seed,
        "seed_strategy": "blake2b64(global_seed, circuit_name, workload_name, realization_index)",
        "workload_mode": workload_mode,
        "workloads": workload_meta,
        "label_fields": {
            "B": "[K, num_nodes, 2], columns are P(v=1) and toggle probability; stochastic probes are averaged over realizations",
            "R": "[K, num_nodes, 2], workload residual B - mean_k(B)",
            "S": "[num_nodes, K*2], flattened workload response signature and main h_b supervision target",
            "mu": "[num_nodes, 2], mean behavior over workloads",
            "sensitivity": "[num_nodes], alias of sensitivity_var",
            "sensitivity_var": "[num_nodes], sum_behavior_dim mean_k R_v(D_k)^2",
            "sensitivity_sum": "[num_nodes], sum_k ||R_v(D_k)||_2^2, kept for comparison with older labels",
            "num_pis": "scalar number of primary inputs",
        },
    }

    save_behavior_labels(output_path, labels, meta)
    save_json(meta_output_path, meta)
    print(f"[DONE] saved labels: {output_path}")
    print(f"[DONE] saved meta:   {meta_output_path}")


if __name__ == "__main__":
    main()
