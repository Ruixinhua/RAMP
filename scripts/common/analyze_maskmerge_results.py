import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class FileMeta:
    dataset: str
    base_model: str
    improvement: str  # one of: "base", "DT", "DT+CL"
    filename: str
    stem: str


DATASET_ALIASES = {
    "avazu": "Avazu",
    "taobaoad": "TaobaoAd",
    "criteoprivate_v4": "CriteoPrivateAdV4",
    "criteoprivate": "CriteoPrivateAdV4",  # fallback
}

GROUP_NAME_MAP: Dict[str, str] = {
    "1.0": "personalized",
    "2.0": "non_personalized",
    "all": "all",
}


def parse_filename(path: str) -> Optional[FileMeta]:
    filename = os.path.basename(path)
    stem = os.path.splitext(filename)[0]

    lowered = filename.lower()

    dataset = None
    for key in DATASET_ALIASES.keys():
        if key in lowered:
            dataset = DATASET_ALIASES.get(key, key)
            break

    # Base model: capture the token right after dataset name
    # Examples:
    #  - avazu_FCN_maskmerge_hyper.csv -> base_model = FCN
    #  - avazu_FCN_FCN_DT_maskmerge_CL_hyper.csv -> base_model = FCN
    base_model = ""
    m = re.search(r"(?:avazu|taobaoad|criteoprivate_v4|criteoprivate)_([A-Za-z0-9]+)", lowered)
    if m:
        base_model = m.group(1).upper()

    has_dt = "_dt_" in lowered or lowered.endswith("_dt.csv") or lowered.endswith("_dt_hyper.csv") or lowered.startswith("dt_") or lowered.endswith("_dt")
    has_cl = "_cl_" in lowered or lowered.endswith("_cl.csv") or lowered.endswith("_cl_hyper.csv") or lowered.startswith("cl_") or lowered.endswith("_cl")

    if has_dt and has_cl:
        improvement = "DT+CL"
    elif has_dt:
        improvement = "DT"
    else:
        improvement = "base"

    if dataset is None:
        return None

    return FileMeta(dataset=dataset, base_model=base_model, improvement=improvement, filename=filename, stem=stem)


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def list_input_files(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        files = []
        for name in os.listdir(input_path):
            if name.lower().endswith(".csv"):
                files.append(os.path.join(input_path, name))
        return sorted(files)
    if os.path.isfile(input_path):
        return [input_path]
    return []


def normalize_group_value(value) -> str:
    if pd.isna(value):
        return "all"
    try:
        # Keep exact textual 'all'
        s = str(value).strip().lower()
        if s == "all":
            return "all"
        # Normalize numeric forms like 1, 1.0, 2.0
        f = float(s)
        if abs(f - 1.0) < 1e-8:
            return "1.0"
        if abs(f - 2.0) < 1e-8:
            return "2.0"
        # Fallback to raw string when unexpected
        return s
    except Exception:
        return str(value).strip()


def split_and_save_by_group(df: pd.DataFrame, meta: FileMeta, output_dir: str) -> List[Tuple[str, str]]:
    outputs: List[Tuple[str, str]] = []
    if "group_id" not in df.columns:
        # Save as "all" when no group info
        group_name = GROUP_NAME_MAP["all"]
        group_dir = os.path.join(output_dir, group_name, meta.dataset)
        ensure_dir(group_dir)
        out_file = os.path.join(group_dir, f"{meta.stem}.csv")
        df_to_save = df.copy()
        # format float columns to 4 significant digits
        for c in df_to_save.columns:
            if pd.api.types.is_float_dtype(df_to_save[c]):
                df_to_save[c] = df_to_save[c].apply(lambda v: format_significant_fixed(v, 4))
        df_to_save.to_csv(out_file, index=False)
        outputs.append((group_name, out_file))
        return outputs

    # Normalize group labels
    df = df.copy()
    df["__group_norm__"] = df["group_id"].map(normalize_group_value)

    for group_value, df_g in df.groupby("__group_norm__", dropna=False):
        key = group_value if group_value in GROUP_NAME_MAP else str(group_value)
        mapped = GROUP_NAME_MAP.get(key, key)
        group_dir = os.path.join(output_dir, mapped, meta.dataset)
        ensure_dir(group_dir)
        out_file = os.path.join(group_dir, f"{meta.stem}.csv")
        df_g = df_g.drop(columns=["__group_norm__"], errors="ignore")
        # format float columns to 4 significant digits
        for c in df_g.columns:
            if pd.api.types.is_float_dtype(df_g[c]):
                df_g[c] = df_g[c].apply(lambda v: format_significant_fixed(v, 4))
        df_g.to_csv(out_file, index=False)
        outputs.append((mapped, out_file))

    return outputs


def find_hparam_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    if "test_logloss" not in df.columns:
        # If missing, consider all columns after any known metric-like columns fallbacks
        metric_cols: List[str] = [c for c in df.columns if c.startswith("test_") or c.startswith("val_") or c.startswith("metric_")]
        hparam_cols = [c for c in df.columns if c not in metric_cols]
        return metric_cols, hparam_cols

    idx = list(df.columns).index("test_logloss")
    metric_cols = list(df.columns[: idx + 1])
    hparam_cols = list(df.columns[idx + 1 :])
    return metric_cols, hparam_cols


def aggregate_hparams(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    metric_cols, hparam_cols = find_hparam_columns(df)
    # limit metrics to only the specified set if present
    wanted_metrics = ["test_auc", "test_logloss", "val_auc", "val_logloss"]
    metric_cols = [c for c in metric_cols if c in wanted_metrics]
    # If none matched (e.g., missing wanted names), keep fallback to any of the wanted existing columns
    if not metric_cols:
        metric_cols = [c for c in df.columns if c in wanted_metrics]

    # Remove non-hparam columns from hparam_cols
    drop_cols = {"seed", "fold", "trial", "run_id"}
    hparam_cols = [c for c in hparam_cols if c not in drop_cols]

    if not hparam_cols:
        # Nothing to group, just compute overall stats on metrics
        out = {}
        for col in metric_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                out[f"{col}_mean"] = [df[col].mean()]
                out[f"{col}_std"] = [df[col].std(ddof=1) if len(df[col].dropna()) > 1 else 0.0]
        # n_trials overall
        out["n_trials"] = [int(len(df))]
        return pd.DataFrame(out)

    # Group by all hparam columns
    grouped = df.groupby(hparam_cols, dropna=False)

    agg_dict = {}
    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[col] = ["mean", "std"]

    result = grouped.agg(agg_dict)
    # Flatten MultiIndex columns
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    result = result.reset_index()

    # When std is NaN (single sample), set to 0.0
    for col in list(result.columns):
        if col.endswith("_std"):
            result[col] = result[col].fillna(0.0)

    # Add n_trials as overall count of rows per combo (even if metrics are NaN)
    result["n_trials"] = grouped.size().values

    # Reorder columns: metric results first, then hparams
    metric_stat_cols = [c for c in result.columns if c.endswith("_mean") or c.endswith("_std")] + ["n_trials"]
    hparam_cols = [c for c in result.columns if c not in metric_stat_cols]
    ordered = metric_stat_cols + hparam_cols
    result = result[ordered]
    return result


def format_significant_fixed(x: float, sig: int = 4) -> str:
    if pd.isna(x):
        return ""
    try:
        if x == 0:
            return "0." + ("0" * (sig))  # e.g., 0.0000 for 4 significant digits
        ax = abs(x)
        import math
        k = math.floor(math.log10(ax))
        decimals = max(0, sig - (k + 1))
        # Cap decimals to a reasonable number to avoid overly long tails
        decimals = min(decimals, 10)
        fmt = f"{{:.{decimals}f}}"
        return fmt.format(x)
    except Exception:
        return str(x)


def process_file(path: str, output_dir: str, save_group_files: bool = True, save_hparam_stats: bool = True) -> None:
    meta = parse_filename(path)
    if meta is None:
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return

    # Split by group and save
    group_outputs = split_and_save_by_group(df, meta, output_dir) if save_group_files else []

    # Aggregate per group
    if save_hparam_stats:
        for group_name, group_file in (group_outputs or [("all", path)]):
            try:
                df_g = pd.read_csv(group_file) if save_group_files else df.copy()
            except Exception:
                df_g = df.copy()
            stats_df = aggregate_hparams(df_g)
            stats_dir = os.path.join(output_dir, group_name, meta.dataset, "hparam_stats")
            ensure_dir(stats_dir)
            stats_file = os.path.join(stats_dir, f"{meta.stem}.csv")
            # Format numeric metric columns to 4 significant digits (keep counts as integers)
            metric_like_cols = [c for c in stats_df.columns if c.endswith("_mean") or c.endswith("_std")]
            formatted = stats_df.copy()
            for col in metric_like_cols:
                if pd.api.types.is_numeric_dtype(formatted[col]):
                    formatted[col] = formatted[col].apply(lambda v: format_significant_fixed(v, 4))
            formatted.to_csv(stats_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Analyze maskmerge experiment results: split by group_id and aggregate hparam performance.")
    parser.add_argument("input", help="输入的CSV文件或目录。若为目录，则批量处理目录内的*.csv 文件。")
    parser.add_argument("--output", "-o", default=None, help="输出目录（默认：与输入同级创建 maskmerge_analysis 目录）")
    parser.add_argument("--no-save-groups", action="store_true", help="不保存按 group 拆分后的原始记录，仅计算超参统计结果。")
    parser.add_argument("--no-save-hparams", action="store_true", help="不保存超参聚合结果，仅输出拆分后的原始记录。")

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"[ERROR] 输入路径不存在: {input_path}")
        sys.exit(1)

    # Default output directory
    if args.output:
        output_dir = os.path.abspath(args.output)
    else:
        parent = os.path.dirname(input_path) if os.path.isfile(input_path) else input_path
        output_dir = os.path.join(parent, "maskmerge_analysis")

    ensure_dir(output_dir)

    files = list_input_files(input_path)
    if not files:
        print(f"[WARN] 未找到可处理的CSV文件: {input_path}")
        sys.exit(0)

    for f in files:
        meta = parse_filename(f)
        if meta is None:
            print(f"[INFO] 跳过无法识别的数据集文件: {f}")
            continue
        print(f"[INFO] 处理: dataset={meta.dataset}, base={meta.base_model}, imp={meta.improvement}, file={meta.filename}")
        process_file(
            f,
            output_dir=output_dir,
            save_group_files=not args.no_save_groups,
            save_hparam_stats=not args.no_save_hparams,
        )


if __name__ == "__main__":
    main()


