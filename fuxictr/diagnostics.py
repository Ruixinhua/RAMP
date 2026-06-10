from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from fuxictr.metrics import MRR, NDCG, avgAUC, gAUC


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _summary(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return OrderedDict()
    return OrderedDict([
        (f"{prefix}_min", float(np.min(values))),
        (f"{prefix}_p50", float(np.quantile(values, 0.5))),
        (f"{prefix}_mean", float(np.mean(values))),
        (f"{prefix}_p90", float(np.quantile(values, 0.9))),
        (f"{prefix}_max", float(np.max(values))),
    ])


def group_rank_diagnostics(y_true, y_pred, group_id):
    df = pd.DataFrame({
        "group_id": group_id,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    ranks = []
    group_sizes = []
    group_positive_counts = []
    for _, group in df.groupby("group_id", sort=False):
        group_sizes.append(len(group))
        positives = int(group["y_true"].sum())
        group_positive_counts.append(positives)
        if positives <= 0:
            continue
        ordered = group.sort_values("y_pred", ascending=False).reset_index(drop=True)
        positive_positions = np.flatnonzero(ordered["y_true"].to_numpy() > 0)
        if positive_positions.size > 0:
            ranks.append(int(positive_positions[0] + 1))
    output = OrderedDict()
    output["groups"] = int(df["group_id"].nunique())
    output["groups_with_one_positive"] = int(np.sum(np.asarray(group_positive_counts) == 1))
    output["groups_without_positive"] = int(np.sum(np.asarray(group_positive_counts) == 0))
    output.update(_summary(group_sizes, "group_size"))
    output.update(_summary(ranks, "positive_rank"))
    return output


def prediction_diagnostics(y_true, y_pred, metrics=None, group_id=None, seed=2019):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    output = OrderedDict([
        ("pred_mean", float(np.mean(y_pred)) if y_pred.size else 0.0),
        ("pred_std", float(np.std(y_pred)) if y_pred.size else 0.0),
        ("pred_min", float(np.min(y_pred)) if y_pred.size else 0.0),
        ("pred_max", float(np.max(y_pred)) if y_pred.size else 0.0),
    ])
    if group_id is not None:
        output.update(group_rank_diagnostics(y_true, y_pred, group_id))

    if metrics:
        baseline_metrics = [m for m in metrics if m in {"AUC", "logloss", "binary_crossentropy", "gAUC", "avgAUC", "MRR"} or str(m).startswith("NDCG")]
        if baseline_metrics:
            rng = np.random.default_rng(seed)
            baselines = OrderedDict([
                ("constant", np.full_like(y_pred, 0.5, dtype=np.float64)),
                ("random", rng.random(y_pred.shape[0])),
            ])
            for baseline_name, baseline_pred in baselines.items():
                try:
                    baseline_result = evaluate_metrics_inprocess(
                        y_true, baseline_pred, baseline_metrics, group_id=group_id
                    )
                    for key, value in baseline_result.items():
                        output[f"{baseline_name}_{key}"] = _safe_float(value)
                except Exception as err:
                    output[f"{baseline_name}_error"] = str(err)
    return output


def evaluate_metrics_inprocess(y_true, y_pred, metrics, group_id=None):
    results = OrderedDict()
    group_metrics = []
    for metric in metrics:
        if metric in ["logloss", "binary_crossentropy"]:
            results[metric] = float(log_loss(y_true, y_pred))
        elif metric == "AUC":
            results[metric] = float(roc_auc_score(y_true, y_pred))
        elif metric in ["gAUC", "avgAUC", "MRR"] or str(metric).startswith("NDCG"):
            group_metrics.append(metric)
    if group_metrics:
        if group_id is None:
            raise ValueError("group_id is required for group metrics")
        df = pd.DataFrame({"group_id": group_id, "y_true": y_true, "y_pred": y_pred})
        sums = {metric: [0.0, 0.0] for metric in group_metrics}
        for _, group in df.groupby("group_id", sort=False):
            true_values = group["y_true"].to_numpy()
            pred_values = group["y_pred"].to_numpy()
            for metric in group_metrics:
                value = _group_metric(metric, true_values, pred_values)
                if isinstance(value, tuple):
                    num, denom = value
                else:
                    num, denom = value, 1
                sums[metric][0] += num
                sums[metric][1] += denom
        for metric, (num, denom) in sums.items():
            results[metric] = float(num / denom) if denom else 0.0
    return results


def _group_metric(metric, y_true, y_pred):
    if metric == "gAUC":
        return gAUC(y_true, y_pred)
    if metric == "avgAUC":
        return avgAUC(y_true, y_pred)
    if metric == "MRR":
        return MRR(y_true, y_pred)
    if str(metric).startswith("NDCG"):
        topk = int(str(metric).split("(")[1].split(")")[0])
        return NDCG(topk)(y_true, y_pred)
    raise ValueError(f"Unsupported group metric: {metric}")


def candidate_dataframe_diagnostics(df, group_col="taac_group_id", label_col="label", item_col="item_id"):
    output = OrderedDict()
    output["rows"] = int(len(df))
    output["label_counts"] = {
        str(k): int(v) for k, v in df[label_col].value_counts(dropna=False).sort_index().items()
    } if label_col in df else {}
    if group_col not in df:
        return output

    group_sizes = df.groupby(group_col).size()
    output["groups"] = int(group_sizes.shape[0])
    output["group_size_min"] = int(group_sizes.min()) if len(group_sizes) else 0
    output["group_size_max"] = int(group_sizes.max()) if len(group_sizes) else 0
    if label_col in df:
        positive_counts = df.groupby(group_col)[label_col].sum()
        output["groups_with_one_positive"] = int((positive_counts == 1).sum())
        output["groups_without_positive"] = int((positive_counts == 0).sum())
    if item_col in df:
        item_counts = df.groupby(group_col)[item_col].agg(["count", "nunique"])
        output["groups_with_duplicate_items"] = int((item_counts["count"] != item_counts["nunique"]).sum())
        output["candidate_unique_item_ratio"] = float(item_counts["nunique"].sum() / item_counts["count"].sum())
        if label_col in df:
            overlap = 0
            for _, group in df.groupby(group_col, sort=False):
                positive_items = set(group.loc[group[label_col] > 0, item_col].tolist())
                negative_items = set(group.loc[group[label_col] <= 0, item_col].tolist())
                if positive_items & negative_items:
                    overlap += 1
            output["groups_negative_overlaps_positive_item"] = int(overlap)
    return output
