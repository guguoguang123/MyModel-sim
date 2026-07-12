from __future__ import annotations

import argparse

import numpy as np

from drgate_sim.io import save_behavior_labels, save_json


COMMON_META_KEYS = (
    "label_objective",
    "model_input_assumption",
    "behavior_definition",
    "response_signature_definition",
    "stochastic_probe_policy",
    "graphs_path",
    "num_vectors",
    "num_realizations",
    "num_workloads",
    "behavior_dim",
    "seed",
    "seed_strategy",
    "workload_mode",
    "workloads",
    "label_fields",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge DRGate behavior label shards.")
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-output", required=True)
    return parser.parse_args()


def common_meta(raw_metas: list[dict]) -> dict:
    if not raw_metas:
        return {}
    common = {}
    for key in COMMON_META_KEYS:
        first = raw_metas[0].get(key)
        if all(meta.get(key) == first for meta in raw_metas):
            common[key] = first
    return common


def main() -> None:
    args = parse_args()
    merged = {}
    shard_metas = []
    raw_metas = []
    for shard in args.shards:
        data = np.load(shard, allow_pickle=True)
        labels = data["labels"].item()
        meta = data["meta"].item()
        overlap = sorted(set(merged).intersection(labels))
        if overlap:
            raise ValueError(f"Shard {shard} overlaps existing circuits, first overlap: {overlap[0]}")
        merged.update(labels)
        raw_metas.append(meta)
        shard_metas.append({"path": shard, "num_circuits": len(labels), "meta": meta})
        print(f"[MERGE] {shard}: {len(labels)} circuits")

    meta = {
        "simulation_type": raw_metas[0].get("simulation_type", "AIG-level stable Boolean simulation")
        if raw_metas
        else "AIG-level stable Boolean simulation",
        "num_circuits": len(merged),
        "num_shards": len(args.shards),
        **common_meta(raw_metas),
        "shards": shard_metas,
    }
    save_behavior_labels(args.output, merged, meta)
    save_json(args.meta_output, meta)
    print(f"[DONE] saved merged labels: {args.output}")
    print(f"[DONE] saved merged meta:   {args.meta_output}")


if __name__ == "__main__":
    main()
