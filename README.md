# RAMP

This repository contains the code release for:

**RAMP: Robust Ad Recommendation Under Limited Personalized-Feature Availability via Masking and Alignment Pathways**

RAMP is implemented on top of FuxiCTR. The released code focuses on the RAMP/DTCN implementation, the masking-and-merge data preparation utility, and experiment configurations used for the paper's public benchmark reproduction.

## Repository Structure

```text
fuxictr/                     Core FuxiCTR runtime used by the experiments
model_zoo/DTCN/              RAMP dual-tower masking and alignment pathway implementation
model_zoo/CL/                Alignment and distillation loss components
model_zoo/PNN/               PNN baseline/backbone
model_zoo/DCNv3/             FCN-style backbone used by RAMP configs
model_zoo/FinalNet/          FINAL/FinalNet baseline/backbone
experiment/run_expid.py      Main experiment entry point
experiment/config/           Paper experiment and baseline configurations
scripts/common/              Dataset masking and result aggregation helpers
```

## Installation

```bash
git clone https://github.com/Ruixinhua/RAMP.git
cd RAMP
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Install a PyTorch build matching your CUDA environment separately if needed.

## Data Preparation

RAMP expects FuxiCTR-style processed CTR/CVR datasets with `feature_map.json`, `feature_vocab.json`, and train/valid/test parquet files.

For a processed dataset, create the masked-merge version with:

```bash
python scripts/common/process_mask_merge_dataset.py \
  /path/to/processed_dataset \
  --mask_features user_id,feature_a,feature_b \
  --tag 050_050_maskmerge
```

The script appends an `is_personalization` feature:

- `1`: original sample with personalized features available
- `2`: masked sample with selected personalized features unavailable

The public experiment configs list the feature fields used for Avazu, TaobaoAd, and CriteoPrivateAd. Before running, update each config's `data_root`, `train_data`, `valid_data`, `test_data`, and `model_root` paths for your local environment.

## Running Experiments

Use `experiment/run_expid.py` as the main entry point.

Example:

```bash
python experiment/run_expid.py \
  --config experiment/config/avazu_wisteria \
  --expid FINAL_FINAL_DT_avazu_x4_050_050_maskmerge_CL_hyper \
  --gpu 0
```

Useful config groups:

- `experiment/config/avazu_wisteria/`
- `experiment/config/criteo_wisteria/`
- `experiment/config/taobaoad_sonic/`
- `experiment/config/tuner_config/`

Baseline PNN masking configs are also provided directly under `experiment/config/`.

## Implementation Notes

The core model class is `model_zoo.DTCN.src.DualTowerCL`. It combines:

- a personalized pathway trained with the configured personalized feature availability,
- a non-personalized pathway trained on masked features,
- alignment/distillation losses implemented in `model_zoo.CL.src.base`.

The backbone choices exposed in this release are `PNN`, `DCNv3`, and `FinalNet`.

## Citation

Publication metadata:

- DOI: [10.1145/3805713.3820399](https://doi.org/10.1145/3805713.3820399)
- Conference: Proceedings of the 2026 International ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR)
- Short name: ICTIR '26
- Date and location: July 25, 2026, Melbourne, VIC, Australia
- ISBN: 979-8-4007-2600-2/2026/07
- Copyright year: 2026
- License in proceedings: CC BY

BibTeX template:

```bibtex
@inproceedings{ramp2026,
  title = {RAMP: Robust Ad Recommendation Under Limited Personalized-Feature Availability via Masking and Alignment Pathways},
  booktitle = {Proceedings of the 2026 International ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR)},
  year = {2026},
  address = {Melbourne, VIC, Australia},
  doi = {10.1145/3805713.3820399},
  isbn = {979-8-4007-2600-2/2026/07}
}
```

## License

This code release is based on FuxiCTR and follows the Apache License 2.0 used by the original library.
