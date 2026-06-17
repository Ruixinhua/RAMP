# Runnable Demo

This demo is a tiny synthetic dataset for checking that the RAMP training path runs end to end. It is not intended to reproduce the paper's reported metrics.

## What It Covers

The demo includes:

- complete samples with `is_personalization = 1`
- feature-constrained samples with `is_personalization = 2`
- three personalized fields: `user_id`, `device_id`, and `age_bucket`
- non-personalized fields: `context_id`, `item_id`, `price`, and `hour`
- a small PNN/PNN `DualTowerCL` configuration

## Run

From the repository root:

```bash
PYTHONPATH=. python experiment/run_expid.py \
  --config config/demo \
  --expid ramp_demo_pnn \
  --gpu -1 \
  --profile 0
```

The script changes into the `experiment` directory internally, so `--config` is intentionally relative to `experiment/`.

Generated feature maps, parquet files, logs, and checkpoints are written under `demo_runs/`, which is ignored by Git.

## Expected Result

The run should:

1. read the raw CSV files from `demo_data/raw/ramp_demo/`,
2. build a FuxiCTR feature map and parquet splits,
3. train the small RAMP model for two epochs on CPU,
4. print validation and test metrics.

On such a tiny synthetic dataset, metric values are only a smoke-test signal. Use the public benchmark configs and the paper appendix settings for reproduction experiments.
