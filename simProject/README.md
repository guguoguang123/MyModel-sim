# DRGate-Sim

This project generates workload-response behavior labels for DRGate. The labels
are designed to supervise a behavior-response embedding `h_b` that is distinct
from a static functional embedding `h_f`: `h_b` should learn how a node responds
across a set of workload probes, instead of predicting logic probability or
toggle probability under one single workload.

It uses AIG-level stable Boolean simulation over `graphs.npz`. It does not use
DeepGate2 `labels.npz` to create behavior labels.

## Label Definition

For each node `v` and workload probe `D_k`, the simulator first estimates:

```text
B_v(D_k) = [P(v=1 | D_k), P(toggle_v | D_k)]
```

Then it removes the node's own average behavior across probes:

```text
mu_v      = mean_k B_v(D_k)
R_v(D_k) = B_v(D_k) - mu_v
S_v      = concat_k R_v(D_k)
```

`S_v` is the workload response signature used to supervise `h_b`. It is not a
model input feature.

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

Fixed probes are simulated once per circuit. Stochastic probes
(`random_beta_05_05`, `group_correlation`, and `mixed_bias_temporal_corr`) are
estimated by averaging `num_realizations` independent realizations before
constructing `R` and `S`. The default full configuration uses
`num_realizations = 16`. This stabilizes each stochastic probe so that `S_v`
reflects a reproducible response pattern rather than one hidden random draw.

Random seeds are derived from `global_seed`, circuit name, workload name, and
realization index, so labels are stable across shard boundaries.

## Output

`behavior_labels.npz` contains:

- `labels`: circuit-name keyed dictionary
- `meta`: configuration and schema metadata

For each circuit:

- `B`: `[K, num_nodes, 2]`, columns are logic-1 probability and toggle probability
- `R`: `[K, num_nodes, 2]`, residual behavior `B - mean_k(B)`
- `S`: `[num_nodes, K * 2]`, flattened response signature and main `h_b` supervision target
- `mu`: `[num_nodes, 2]`
- `sensitivity`: `[num_nodes]`, alias of `sensitivity_var`
- `sensitivity_var`: `[num_nodes]`, `sum_behavior_dim mean_k R_v(D_k)^2`
- `sensitivity_sum`: `[num_nodes]`, old-style `sum_k ||R_v(D_k)||_2^2` for comparison
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

Override the stochastic workload averaging count:

```bash
PYTHONPATH=simProject python simProject/scripts/build_behavior_labels.py \
  --config simProject/configs/default.json \
  --num-realizations 16
```

Check whether the realization count is stable enough on a subset:

```bash
PYTHONPATH=simProject python simProject/scripts/check_realization_stability.py \
  --config simProject/configs/default.json \
  --limit 20 \
  --num-vectors 4096 \
  --realizations 8,16 \
  --output-json /tmp/realization_stability_4096_m8_m16.json
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
