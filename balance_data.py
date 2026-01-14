#!/usr/bin/env python3
"""
数据平衡脚本 - 对少数类进行过采样

使用方法:
    python balance_data.py --input_dir ./filter --output_dir ./filter_balanced

功能:
    1. 过采样: 复制少数类样本使各类别数量接近平衡
    2. 支持添加轻微噪声的过采样（数据增强）
"""

import os
import json
import random
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def load_num_files(input_dir):
    """
    加载所有.num文件
    
    Returns:
        data: dict, {类别名: [流记录列表]}
    """
    data = {}
    for fname in os.listdir(input_dir):
        if fname.endswith('.num'):
            class_name = fname.replace('.num', '')
            filepath = os.path.join(input_dir, fname)
            with open(filepath, 'r') as f:
                records = [line.strip() for line in f if line.strip()]
            data[class_name] = records
            print(f"  {class_name}: {len(records)} 条记录")
    return data


def augment_record(record, noise_level=0.1):
    """
    对记录添加轻微噪声进行数据增强
    
    Args:
        record: 原始记录 "状态序列;包长度序列"
        noise_level: 噪声比例 (0-1)
    
    Returns:
        增强后的记录
    """
    parts = record.split(';')
    if len(parts) != 2:
        return record
    
    status_seq = parts[0].split('\t')
    length_seq = parts[1].split('\t')
    
    # 对包长度添加噪声 (±noise_level%)
    new_length_seq = []
    for length in length_seq:
        try:
            l = int(length)
            # 添加随机噪声
            noise = random.uniform(-noise_level, noise_level)
            new_l = max(1, int(l * (1 + noise)))
            new_length_seq.append(str(new_l))
        except ValueError:
            new_length_seq.append(length)
    
    # 状态序列保持不变（时间间隔编码是离散的）
    return f"{parts[0]};{chr(9).join(new_length_seq)}"


def oversample_data(data, target_count=None, augment=True, noise_level=0.1):
    """
    过采样少数类
    
    Args:
        data: dict, {类别名: [记录列表]}
        target_count: 目标样本数，默认为最大类的样本数
        augment: 是否在过采样时添加噪声
        noise_level: 噪声级别
    
    Returns:
        balanced_data: 平衡后的数据
    """
    # 确定目标数量
    counts = {k: len(v) for k, v in data.items()}
    if target_count is None:
        target_count = max(counts.values())
    
    print(f"\n目标样本数: {target_count}")
    
    balanced_data = {}
    for class_name, records in data.items():
        current_count = len(records)
        
        if current_count >= target_count:
            # 如果样本数已经足够，随机采样到目标数量
            balanced_data[class_name] = random.sample(records, target_count)
            print(f"  {class_name}: {current_count} -> {target_count} (下采样)")
        else:
            # 需要过采样
            new_records = records.copy()
            samples_needed = target_count - current_count
            
            # 循环采样并可选地添加噪声
            for i in range(samples_needed):
                original = random.choice(records)
                if augment:
                    augmented = augment_record(original, noise_level)
                    new_records.append(augmented)
                else:
                    new_records.append(original)
            
            balanced_data[class_name] = new_records
            aug_str = "(带噪声增强)" if augment else "(直接复制)"
            print(f"  {class_name}: {current_count} -> {target_count} (过采样 {aug_str})")
    
    return balanced_data


def save_num_files(data, output_dir):
    """
    保存平衡后的数据
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name, records in data.items():
        filepath = os.path.join(output_dir, f"{class_name}.num")
        with open(filepath, 'w') as f:
            for record in records:
                f.write(record + '\n')
        print(f"  已保存: {filepath} ({len(records)} 条)")


def analyze_flow_lengths(data):
    """
    分析各类别的流长度分布
    """
    print("\n流长度分析:")
    for class_name, records in data.items():
        lengths = []
        for record in records:
            parts = record.split(';')
            if len(parts) == 2:
                length_seq = parts[1].split('\t')
                lengths.append(len(length_seq))
        
        if lengths:
            print(f"  {class_name}:")
            print(f"    流数量: {len(lengths)}")
            print(f"    包数量 - 最小: {min(lengths)}, 最大: {max(lengths)}, "
                  f"平均: {np.mean(lengths):.1f}, 中位数: {np.median(lengths):.0f}")


def main():
    parser = argparse.ArgumentParser(description='数据平衡 - 过采样少数类')
    parser.add_argument('--input_dir', type=str, default='./filter',
                        help='输入目录（包含.num文件）')
    parser.add_argument('--output_dir', type=str, default='./filter_balanced',
                        help='输出目录')
    parser.add_argument('--target', type=int, default=None,
                        help='目标样本数（默认为最大类的数量）')
    parser.add_argument('--no_augment', action='store_true',
                        help='禁用噪声增强，直接复制')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='噪声级别 (默认: 0.1, 即±10%%)')
    parser.add_argument('--analyze', action='store_true',
                        help='只分析数据，不进行过采样')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("加载数据...")
    print("=" * 50)
    data = load_num_files(args.input_dir)
    
    if not data:
        print("错误: 没有找到.num文件")
        return
    
    # 分析流长度
    analyze_flow_lengths(data)
    
    if args.analyze:
        return
    
    print("\n" + "=" * 50)
    print("过采样处理...")
    print("=" * 50)
    balanced_data = oversample_data(
        data, 
        target_count=args.target,
        augment=not args.no_augment,
        noise_level=args.noise
    )
    
    print("\n" + "=" * 50)
    print("保存数据...")
    print("=" * 50)
    save_num_files(balanced_data, args.output_dir)
    
    print("\n" + "=" * 50)
    print("完成!")
    print("=" * 50)
    print(f"\n下一步: 使用平衡后的数据进行训练")
    print(f"  python main.py --mode=prepro --class_num=3 --data_dir={args.output_dir}")
    print(f"  python main.py --mode=train --class_num=3")


if __name__ == '__main__':
    main()
