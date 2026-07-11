from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from drgate_sim.io import load_config, load_graphs, save_behavior_labels, save_json
from drgate_sim.labels import build_response_labels
from drgate_sim.simulator import find_primary_inputs, simulate_behavior
from drgate_sim.workloads import debug_workloads, default_workloads, generate_pi_sequence


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
    parser.add_argument("--workload-mode", choices=["debug", "default"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    graphs_path = args.graphs or cfg["graphs_path"]
    output_path = args.output or cfg["output_path"]
    meta_output_path = args.meta_output or cfg["meta_output_path"]
    num_vectors = int(args.num_vectors or cfg["num_vectors"])
    limit = args.limit if args.limit is not None else cfg.get("limit")
    start = args.start if args.start is not None else int(cfg.get("start", 0))
    end = args.end if args.end is not None else cfg.get("end")
    seed = int(cfg.get("seed", 20260710))
    workload_mode = args.workload_mode or cfg.get("workload_mode", "default")

    specs = debug_workloads() if workload_mode == "debug" else default_workloads()
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
    for idx, name in enumerate(circuit_names, start=1):
        graph = graphs[name]
        x = graph["x"]
        edge_index = graph["edge_index"]
        pi_nodes = find_primary_inputs(x)
        B_list = []
        circuit_seed = seed + idx * 1009
        for wk_idx, spec in enumerate(specs):
            rng = np.random.default_rng(circuit_seed + wk_idx)
            pi_values = generate_pi_sequence(spec, num_vectors, len(pi_nodes), rng)
            behavior, _ = simulate_behavior(x, edge_index, pi_values)
            B_list.append(behavior)

        B = np.stack(B_list, axis=0)
        item = build_response_labels(B)
        item["num_pis"] = np.array(len(pi_nodes), dtype=np.int32)
        labels[name] = item

        elapsed = time.time() - started
        print(
            f"[{idx:>5}/{len(circuit_names)}] {name}: "
            f"nodes={x.shape[0]} pis={len(pi_nodes)} workloads={len(specs)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    meta = {
        "simulation_type": "AIG-level stable Boolean simulation",
        "graphs_path": str(graphs_path),
        "num_circuits": len(circuit_names),
        "start": start,
        "end": end,
        "limit": limit,
        "num_vectors": num_vectors,
        "num_workloads": len(specs),
        "behavior_dim": 2,
        "seed": seed,
        "workload_mode": workload_mode,
        "workloads": [spec.to_dict() for spec in specs],
        "label_fields": {
            "B": "[K, num_nodes, 2], columns are P(v=1) and toggle probability",
            "R": "[K, num_nodes, 2], workload residual B - mean_k(B)",
            "S": "[num_nodes, K*2], flattened response signature",
            "mu": "[num_nodes, 2], mean behavior over workloads",
            "sensitivity": "[num_nodes], sum_k ||R_v(D_k)||_2^2",
            "num_pis": "scalar number of primary inputs",
        },
    }

    save_behavior_labels(output_path, labels, meta)
    save_json(meta_output_path, meta)
    print(f"[DONE] saved labels: {output_path}")
    print(f"[DONE] saved meta:   {meta_output_path}")


if __name__ == "__main__":
    main()

