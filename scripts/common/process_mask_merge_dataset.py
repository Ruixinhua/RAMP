#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按传入的 mask 特征对数据进行全量 mask，并将原始数据与 masked 数据在
train/valid/test 三个 split 上分别进行纵向合并，生成新的数据集。

使用方式：
    python scripts/process_mask_merge_dataset.py \
        /path/to/dataset_dir \
        --mask_features user_id,brand_id,cate_his \
        --config_path /path/to/datasets_config.yaml \
        --tag maskmerge

说明：
1. 不依赖原始数据中的 is_personalization 字段，也不使用 processing_results.json。
2. 仅需要 `feature_map.json` 与 `feature_vocab.json`（用以获取 PAD 值）。
3. 对指定的 mask_features 进行 100% mask：
   - 非序列特征：替换为该特征在 vocab 中的 `__PAD__` 值（若存在）。
   - 序列特征：保留序列长度，用 0 进行填充（与现有脚本保持一致）。
4. 输出数据集中，每个 split 为 原始数据 与 masked 数据 的按行 concat 结果；
   额外添加列 `is_personalization`：原始数据=1，masked 数据=2。
5. 在输出目录的 `feature_map.json` 中确保包含 `is_personalization` 特征定义；
   若提供 `--config_path`，会按文本方式复制原数据集配置生成新数据集配置条目。
"""

import os
import json
import shutil
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_feature_configs(data_path: str) -> Tuple[Dict, Dict]:
    """加载必要的特征配置：feature_map.json 与 feature_vocab.json"""
    with open(os.path.join(data_path, 'feature_map.json'), 'r') as f:
        feature_map = json.load(f)

    with open(os.path.join(data_path, 'feature_vocab.json'), 'r') as f:
        feature_vocab = json.load(f)

    return feature_map, feature_vocab


def load_data_files(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载三分数据文件（parquet）"""
    train_df = pd.read_parquet(os.path.join(data_path, 'train.parquet'))
    valid_df = pd.read_parquet(os.path.join(data_path, 'valid.parquet'))
    test_df = pd.read_parquet(os.path.join(data_path, 'test.parquet'))
    return train_df, valid_df, test_df


def ensure_output_dir(base_path: str, tag: str) -> str:
    """创建输出目录，命名为 <base>_<tag>"""
    output_path = os.path.join(os.path.dirname(base_path), f"{os.path.basename(base_path)}_{tag}")
    os.makedirs(output_path, exist_ok=True)
    return output_path


def _ensure_is_personalization_feature(feature_map: Dict):
    """确保 feature_map['features'] 中包含 is_personalization 特征。

    - 若不存在，则追加结构：
        {"is_personalization": {"source": "", "type": "categorical", "padding_idx": 0, "vocab_size": 4}}
    - 同时尽量维护 `num_fields` 与 `input_length` (+1)。
    - 不强行修改 `total_features`，避免统计口径差异引入错误。
    """
    features_list = feature_map.get('features', [])
    exists = False
    for item in features_list:
        if isinstance(item, dict) and 'is_personalization' in item:
            exists = True
            break
    if not exists:
        features_list.append({
            'is_personalization': {
                'source': '',
                'type': 'categorical',
                'padding_idx': 0,
                'vocab_size': 4
            }
        })
        feature_map['features'] = features_list
        # 尽量维护字段计数
        if isinstance(feature_map.get('num_fields'), int):
            feature_map['num_fields'] = feature_map['num_fields'] + 1
        if isinstance(feature_map.get('input_length'), int):
            feature_map['input_length'] = feature_map['input_length'] + 1


def copy_side_configs(source_path: str, target_path: str, new_dataset_id: str, 
                      original_dataset_id: str = None, config_path: str = None):
    """复制必要的侧文件并更新 feature_map；可选写入 datasets_config.yaml。"""
    # feature_processor.pkl 可选存在，若有则拷贝
    fp_path = os.path.join(source_path, 'feature_processor.pkl')
    if os.path.exists(fp_path):
        shutil.copy2(fp_path, os.path.join(target_path, 'feature_processor.pkl'))

    # feature_vocab.json 必需
    shutil.copy2(os.path.join(source_path, 'feature_vocab.json'), os.path.join(target_path, 'feature_vocab.json'))

    # feature_map.json 更新 dataset_id
    with open(os.path.join(source_path, 'feature_map.json'), 'r') as f:
        feature_map = json.load(f)
    feature_map['dataset_id'] = new_dataset_id
    _ensure_is_personalization_feature(feature_map)
    with open(os.path.join(target_path, 'feature_map.json'), 'w') as f:
        json.dump(feature_map, f, indent=4)

    # 若提供 config_path，则在 datasets_config.yaml 里创建新条目
    if config_path and original_dataset_id:
        if os.path.exists(config_path):
            _create_dataset_config_entry_text_based(config_path, original_dataset_id, new_dataset_id)


def _create_dataset_config_entry_text_based(config_path: str, original_dataset_id: str, new_dataset_id: str):
    """文本方式复制原数据集配置为新数据集配置，保持原 YAML 格式风格。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if f"{new_dataset_id}:" in content:
        print(f"数据集配置 {new_dataset_id} 已存在，跳过创建")
        return

    start_pattern = f"{original_dataset_id}:"
    start_idx = content.find(start_pattern)
    if start_idx == -1:
        print(f"警告: 原始数据集配置 {original_dataset_id} 不存在，无法创建新配置")
        return

    lines = content[start_idx:].split('\n')
    config_lines = [lines[0]]
    first_line_indent = 0

    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == '':
            config_lines.append(line)
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent == first_line_indent and line.strip().endswith(':') and not line.strip().startswith('-'):
            break
        config_lines.append(line)

    original_config_text = '\n'.join(config_lines)
    new_config_text = original_config_text.replace(f"{original_dataset_id}:", f"{new_dataset_id}:")

    with open(config_path, 'a', encoding='utf-8') as f:
        f.write('\n\n' + new_config_text)
    print(f"已为数据集 {new_dataset_id} 创建配置条目")


def save_split(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, output_path: str):
    train_df.to_parquet(os.path.join(output_path, 'train.parquet'), index=False)
    valid_df.to_parquet(os.path.join(output_path, 'valid.parquet'), index=False)
    test_df.to_parquet(os.path.join(output_path, 'test.parquet'), index=False)


def mask_features(df: pd.DataFrame, mask_feature_names: List[str], feature_vocab: Dict) -> pd.DataFrame:
    """对给定特征进行 100% mask。

    - 序列特征：使用 0 进行全序列填充，保持长度和 dtype。
    - 非序列特征：若在 vocab 中存在 `__PAD__`，替换为该值；否则跳过。
    """
    df_masked = df.copy()

    # 与现有脚本一致的序列特征集合
    sequence_features = {"cate_his", "brand_his", "btag_his"}

    # 预取非序列特征的 PAD 值
    pad_values = {}
    for name in mask_feature_names:
        if name in feature_vocab and '__PAD__' in feature_vocab[name]:
            pad_values[name] = feature_vocab[name]['__PAD__']

    def mask_sequence_series(series, pad_value: int = 0):
        def to_padded(seq):
            if isinstance(seq, np.ndarray):
                return np.full_like(seq, pad_value, dtype=seq.dtype)
            arr = np.array(seq)
            return np.full_like(arr, pad_value, dtype=arr.dtype)

        return series.apply(to_padded)

    for feat in mask_feature_names:
        if feat not in df_masked.columns:
            continue
        if feat in sequence_features:
            df_masked[feat] = mask_sequence_series(df_masked[feat], pad_value=0)
        elif feat in pad_values:
            df_masked[feat] = pad_values[feat]
        # 若既不是序列也没有 PAD 值，保持原状（跳过）

    return df_masked


def process(data_path: str, mask_features_arg: str, tag: str = 'maskmerge', config_path: str = None, add_train_mask: bool = True):
    print(f"开始处理：{data_path}")

    feature_map, feature_vocab = load_feature_configs(data_path)
    train_df, valid_df, test_df = load_data_files(data_path)

    # 解析待 mask 的特征列表
    mask_feature_names = [x.strip() for x in mask_features_arg.split(',') if x.strip()]
    if not mask_feature_names:
        raise ValueError("--mask_features 不能为空，例如：--mask_features user_id,brand_id")

    print(f"将进行 mask 的特征: {mask_feature_names}")

    # 生成 masked 版本
    train_masked = mask_features(train_df, mask_feature_names, feature_vocab)
    valid_masked = mask_features(valid_df, mask_feature_names, feature_vocab)
    test_masked = mask_features(test_df, mask_feature_names, feature_vocab)

    # 添加 is_personalization：原始=1，masked=2（若已存在则覆盖）
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()
    train_df['is_personalization'] = 1
    valid_df['is_personalization'] = 1
    test_df['is_personalization'] = 1
    train_masked['is_personalization'] = 2
    valid_masked['is_personalization'] = 2
    test_masked['is_personalization'] = 2

    # 合并：原始 + masked（行拼接）
    if add_train_mask:
        merged_train = pd.concat([train_df, train_masked], ignore_index=True)
    else:
        merged_train = train_df
    merged_valid = pd.concat([valid_df, valid_masked], ignore_index=True)
    merged_test = pd.concat([test_df, test_masked], ignore_index=True)

    # 输出
    output_path = ensure_output_dir(data_path, tag)
    copy_side_configs(
        data_path,
        output_path,
        os.path.basename(output_path),
        original_dataset_id=os.path.basename(data_path),
        config_path=config_path,
    )
    save_split(merged_train, merged_valid, merged_test, output_path)

    print("数据处理完成！输出路径：", output_path)
    print(
        f"train: 原始 {len(train_df)} + masked {len(train_masked)} => 合并 {len(merged_train)}"
    )
    print(
        f"valid: 原始 {len(valid_df)} + masked {len(valid_masked)} => 合并 {len(merged_valid)}"
    )
    print(
        f"test:  原始 {len(test_df)} + masked {len(test_masked)} => 合并 {len(merged_test)}"
    )


def main():
    parser = argparse.ArgumentParser(description='根据指定特征进行全量mask并与原数据合并')
    parser.add_argument('data_path', type=str, help='数据集目录，需包含 train/valid/test parquet 与特征配置')
    parser.add_argument('--mask_features', type=str, required=True, help='以逗号分隔的待mask特征名列表')
    parser.add_argument('--tag', type=str, default='maskmerge', help='输出目录后缀标识，默认 maskmerge')
    parser.add_argument('--config_path', type=str, default=None, help='数据集配置 YAML 路径，若提供则生成对应配置条目')
    parser.add_argument('--add_train_mask', type=bool, default=False, help='是否对训练集进行mask，默认True')
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"错误：数据路径不存在：{args.data_path}")
        return

    # 关键输入文件检查
    for required_file in ['train.parquet', 'valid.parquet', 'test.parquet', 'feature_map.json', 'feature_vocab.json']:
        if not os.path.exists(os.path.join(args.data_path, required_file)):
            print(f"错误：缺少必要文件：{required_file}")
            return

    process(args.data_path, args.mask_features, args.tag, args.config_path, args.add_train_mask)


if __name__ == '__main__':
    main()


