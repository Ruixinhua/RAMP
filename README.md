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
demo_data/raw/ramp_demo/     Tiny synthetic data for smoke-test reproduction
docs/                        Appendix source and demo instructions
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

Runnable demo:

```bash
PYTHONPATH=. python experiment/run_expid.py \
  --config config/demo \
  --expid ramp_demo_pnn \
  --gpu -1 \
  --profile 0
```

The demo uses a tiny synthetic dataset and is intended to verify the RAMP training path, not to reproduce paper metrics. See `docs/demo.md` for details.

Paper config example:

```bash
PYTHONPATH=. python experiment/run_expid.py \
  --config config/avazu_wisteria \
  --expid FINAL_FINAL_DT_avazu_x4_050_050_maskmerge_CL_hyper \
  --gpu 0
```

`experiment/run_expid.py` changes into the `experiment` directory internally, so `--config` paths are relative to `experiment/`.

Useful config groups:

- `experiment/config/avazu_wisteria/`
- `experiment/config/criteo_wisteria/`
- `experiment/config/taobaoad_sonic/`
- `experiment/config/tuner_config/`

Baseline PNN masking configs are also provided directly under `experiment/config/`.

## Appendix

The Overleaf LaTeX source is also kept in `docs/appendix.tex`.

### Feature Importance Analysis

To complement the robustness analysis, we conduct a feature importance analysis to identify the most critical features for prediction performance across datasets. We use a leave-one-out ablation approach with PNN as the backbone model. For each feature, the model is trained with that feature removed while all other features remain intact, and the resulting test-set performance degradation is measured.

The results show different dataset patterns. For Avazu, the most impactful features are personalized device-related identifiers: `device_ip`, `device_model`, and `device_id`. TaobaoAd has the strongest dependence on personalized features, where `userid` alone contributes 3.84% AUC, substantially more than other fields. CriteoPrivateAd shows a more balanced feature importance distribution, but personalized identifiers still appear near the top. These observations motivate the feature elimination strategy used in the robustness evaluation.

Personalized feature fields are marked with `(personalized)`.

| Dataset | Feature Field Name | Delta AUC (%) |
|---|---|---:|
| Avazu | `device_ip` (personalized) | -2.17 |
| Avazu | `device_model` (personalized) | -0.54 |
| Avazu | `device_id` (personalized) | -0.39 |
| Avazu | `app_id` | -0.19 |
| Avazu | `C14` | -0.13 |
| TaobaoAd | `userid` (personalized) | -3.84 |
| TaobaoAd | `cate_id` | -0.41 |
| TaobaoAd | `cate_his` (personalized) | -0.28 |
| TaobaoAd | `adgroup_id` | -0.17 |
| TaobaoAd | `price` | -0.12 |
| CriteoPrivateAd | `campaign_id` | -0.58 |
| CriteoPrivateAd | `features_ctx_not_constrained_6` | -0.45 |
| CriteoPrivateAd | `features_ctx_not_constrained_4` | -0.18 |
| CriteoPrivateAd | `features_kv_bits_constrained_30` (personalized) | -0.09 |
| CriteoPrivateAd | `features_browser_bits_constrained_10` (personalized) | -0.07 |

### Training Efficiency Analysis

We profile RAMP against backbone models (PNN, FINAL) and three knowledge distillation baselines (KD, PFD, HAPKD) across three public datasets. We report trainable parameters, mean wall-clock time per epoch, training throughput, peak GPU memory during training, and mean inference latency per batch. Measurements were collected on a single NVIDIA Tesla H100 GPU with batch size 10,000.

| Dataset | Model | Params (M) | Epoch Time (s) | Throughput (K/s) | GPU Mem (MB) | Inf. Lat. (ms) |
|---|---|---:|---:|---:|---:|---:|
| Avazu | PNN | 120.8 | 270.9 | 238.8 | 2,368 | 2.20 |
| Avazu | FINAL | 122.0 | 290.5 | 222.7 | 2,599 | 2.85 |
| Avazu | KD | 122.0 | 312.1 | 207.3 | 3,439 | 4.81 |
| Avazu | PFD | 243.9 | 365.4 | 177.1 | 4,726 | 4.79 |
| Avazu | HAPKD | 243.9 | 365.7 | 176.9 | 4,757 | 4.77 |
| Avazu | RAMP (Ours) | 123.9 | 330.9 | 195.5 | 3,319 | 4.61 |
| TaobaoAd | PNN | 43.5 | 316.8 | 146.8 | 934 | 4.73 |
| TaobaoAd | FINAL | 43.9 | 320.9 | 144.9 | 1,058 | 4.84 |
| TaobaoAd | KD | 43.9 | 340.7 | 136.5 | 1,364 | 7.34 |
| TaobaoAd | PFD | 87.9 | 384.2 | 121.0 | 1,941 | 7.33 |
| TaobaoAd | HAPKD | 87.9 | 390.0 | 119.2 | 1,998 | 7.33 |
| TaobaoAd | RAMP (Ours) | 45.0 | 387.1 | 120.1 | 1,334 | 7.16 |
| CriteoPrivateAd | PNN | 121.9 | 307.2 | 199.9 | 2,583 | 5.02 |
| CriteoPrivateAd | FINAL | 123.1 | 310.9 | 197.6 | 2,495 | 4.94 |
| CriteoPrivateAd | KD | 123.1 | 355.7 | 172.9 | 3,441 | 8.72 |
| CriteoPrivateAd | PFD | 246.1 | 501.5 | 122.5 | 4,804 | 8.71 |
| CriteoPrivateAd | HAPKD | 246.1 | 501.5 | 122.5 | 4,868 | 8.79 |
| CriteoPrivateAd | RAMP (Ours) | 125.7 | 466.8 | 131.6 | 2,951 | 8.56 |

RAMP introduces moderate training overhead compared with a single-model backbone. On Avazu, RAMP increases epoch time by about 22% relative to PNN (330.9 s vs. 270.9 s) and about 14% relative to FINAL (330.9 s vs. 290.5 s), while keeping comparable parameter counts (123.9M vs. 120.8M/122.0M). Peak GPU memory is higher than a single backbone because the dual-tower architecture and NP pathway are active during training. However, PFD and HAPKD require nearly double the parameters and the highest GPU memory because they maintain both teacher and student models. KD also uses a teacher-student pair during training, though only the student parameters are updated. RAMP has lower overhead than the KD baselines in both parameter count and GPU memory while improving non-personalized prediction accuracy. At inference time, only the dual-tower component is used; the NP pathway is discarded.

### Necessity of the NP-Only Pathway

A natural question is whether the dedicated NP pathway is necessary, or whether a simpler consistency regularizer, such as an L2 or KL alignment loss between the two existing towers, could achieve comparable gains. The PP results in the paper already confirm that the dual-tower design of the personalized pathway provides meaningful improvements over single-tower baselines.

RAMP consistently surpasses PP across all backbone-dataset combinations, with gains ranging from +0.06% to +0.75% AUC. This gap indicates that the dedicated NP pathway, which maintains its own feature interaction parameters and is trained exclusively on non-personalized samples, captures complementary knowledge that a simple inter-tower regularizer cannot. The NP pathway creates an independent capacity bottleneck optimized for the restricted-feature regime, forcing it to learn compact representations from non-personalized features. A regularizer applied directly to the existing towers would instead constrain their specialization and may degrade personalized-tower performance. The consistent improvement across PNN, FCN, and FINAL suggests that this benefit is not backbone-specific.

### Guidelines for Reproduction

RAMP is evaluated on CTR datasets including Avazu and TaobaoAd, and CVR datasets including CriteoPrivateAd and a private industry dataset. For the private industry dataset, a binary variable named `is_personalization` is available. Features with null values when `is_personalization = 0` are treated as personalized features, accounting for 30 out of 60 feature fields.

For the three public datasets, the feature fields used in the experiments are listed below. Non-personalized fields are shown in bold.

| Dataset | Feature Fields |
|---|---|
| Avazu | **`banner_pos`**, **`site_id`**, **`site_domain`**, **`site_category`**, **`app_id`**, **`app_domain`**, **`app_category`**, **`hour`**, **`weekday`**, **`weekend`**, **`C1`**, **`C14`**, **`C15`**, **`C16`**, **`C17`**, **`C18`**, **`C19`**, **`C20`**, **`C21`**, `device_ip`, `device_id`, `device_model`, `device_type`, `device_conn_type` |
| TaobaoAd | **`adgroup_id`**, **`cate_id`**, **`campaign_id`**, **`customer`**, **`brand`**, **`pid`**, **`btag`**, **`price`**, `cms_segid`, `cms_group_id`, `userid`, `final_gender_code`, `age_level`, `pvalue_level`, `shopping_level`, `occupation`, `new_user_class_level`, `cate_his`, `brand_his`, `btag_his` |
| CriteoPrivateAd | **`campaign_id`**, **`display_order`**, **`publisher_id`**, **`features_kv_not_constrained_1~7`**, **`features_ctx_not_constrained_0~7`**, `user_id`, `features_kv_bits_constrained_0~24`, `features_browser_bits_constrained_0~6` |

The hyperparameters below are the best configurations found through grid search.

| Dataset | Backbone | Beta | Emb. Reg. | Model-Specific Parameters |
|---|---|---:|---:|---|
| Avazu | FINAL | 40 | 1e-05 | `embedding_dim=32`<br>`FI_A` personalized tower: FINAL, `block_type=2B`, `block1_hidden_units=[800]`, `block1_dropout=0.2`, `block2_hidden_units=[800,800]`, `block2_dropout=0.3`<br>`FI_B`/`FI_NP` non-personalized tower: FINAL, `block_type=1B`, `block1_hidden_units=[800]`, `block1_dropout=0.2` |
| TaobaoAd | PNN | 150 | 0.05 | `embedding_dim=16`<br>`FI_A` personalized tower: PNN, `hidden_units=[512,256]`<br>`FI_B`/`FI_NP` non-personalized tower: PNN, `hidden_units=[512,256]` |
| TaobaoAd | FCN | 150 | 0.05 | `embedding_dim=32`<br>`FI_A` personalized tower: FCN, `num_heads=1`, `num_deep_cross_layers=4`, `num_shallow_cross_layers=4`<br>`FI_B`/`FI_NP` non-personalized tower: FCN, `num_heads=1`, `num_deep_cross_layers=4`, `num_shallow_cross_layers=4` |
| CriteoPrivateAd | PNN | 80 | 1e-05 | `embedding_dim=16`<br>`FI_A` personalized tower: PNN, `hidden_units=[1000,1000]`<br>`FI_B`/`FI_NP` non-personalized tower: PNN, `hidden_units=[1000,1000]` |

Experiments were run on one NVIDIA Tesla A100 GPU to reduce hardware-related variance. Random seeds are fixed for training, validation, and testing.

### Production A/B Test Configuration

We conducted an online A/B test on CVR prediction to evaluate RAMP in production. RAMP was deployed to 15% of production traffic. In the treatment group, RAMP was applied to non-personalized traffic, while personalized traffic continued to be served by the baseline model, MaskNet. The experiment ran for two weeks under stable production conditions.

The primary evaluation metric is total advertising value (TAV), defined as the number of conversions multiplied by the average bid per conversion. Compared with the baseline, RAMP achieved a +3.5% improvement in TAV, demonstrating its effectiveness in enhancing monetization performance under feature-constrained settings.

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
