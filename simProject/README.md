# DRGate-Sim

This project generates workload-response behavior labels for DRGate.

It uses AIG-level stable Boolean simulation over `graphs.npz`. It does not use
DeepGate2 `labels.npz` to create behavior labels.

## Workloads

The default setup uses eight workload probes:

1. `uniform_p05`
2. `low_bias_p02`
3. `high_bias_p08`
4. `random_beta_05_05`
5. `low_toggle_markov`
6. `high_toggle_markov`
7. `group_correlation`
8. `mixed_bias_temporal_corr`

## Output

`behavior_labels.npz` contains:

- `labels`: circuit-name keyed dictionary
- `meta`: configuration and schema metadata

For each circuit:

- `B`: `[K, num_nodes, 2]`, columns are logic-1 probability and toggle probability
- `R`: `[K, num_nodes, 2]`, residual behavior `B - mean_k(B)`
- `S`: `[num_nodes, K * 2]`, flattened response signature
- `mu`: `[num_nodes, 2]`
- `sensitivity`: `[num_nodes]`
- `num_pis`: scalar

## Run

Debug run:

```bash
PYTHONPATH=simProject python simProject/scripts/build_behavior_labels.py --config simProject/configs/debug.json
```

Full run:

```bash
PYTHONPATH=simProject python simProject/scripts/build_behavior_labels.py --config simProject/configs/default.json
```

## Sharded full generation

Generate shards by circuit index range:

```bash
PYTHONPATH=simProject python simProject/scripts/build_behavior_labels.py \
  --config simProject/configs/default.json \
  --start 0 --end 1000 \
  --output simProject/outputs/behavior_labels_shard_00000_01000.npz \
  --meta-output simProject/outputs/label_meta_shard_00000_01000.json
```

Merge shards after all ranges finish:

```bash
PYTHONPATH=simProject python simProject/scripts/merge_behavior_shards.py \
  --shards simProject/outputs/behavior_labels_shard_*.npz \
  --output simProject/outputs/behavior_labels.npz \
  --meta-output simProject/outputs/label_meta.json
```

DeepGate2 `labels.npz` is not used to generate these behavior labels. It can be
used later on the training machine for the original DeepGate2 losses or for
hard-negative mining with truth-table similarity.
