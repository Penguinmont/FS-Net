"""
预处理脚本 - 修复版
修复标签映射顺序问题：确保 mid=0, low=1, high=2
"""

import tqdm
import numpy as np
import os
import sys
import json


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# 固定标签映射
LABEL_MAP = {
    'mid': 0,
    'low': 1,
    'high': 2
}


def load_origin_data(data_dir, app_num):
    """加载数据，使用固定的标签映射"""
    datas = [[] for _ in range(app_num)]
    
    # 获取所有.num文件
    filenames = [f for f in os.listdir(data_dir) 
                 if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.num')]
    
    print(f"找到文件: {filenames}")
    print(f"标签映射: {LABEL_MAP}")
    
    for filename in tqdm.tqdm(filenames, ascii=True, desc='[Load Data]'):
        # 从文件名获取类别
        class_name = filename.replace('.num', '')
        
        if class_name not in LABEL_MAP:
            print(f"警告: 未知类别 '{class_name}'，跳过")
            continue
        
        label = LABEL_MAP[class_name]
        
        filepath = os.path.join(data_dir, filename)
        with open(filepath) as fp:
            for line in fp:
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    length = parts[1].strip().split('\t')
                    length = [int(ix) for ix in length if ix]
                    if length:  # 确保不为空
                        datas[label].append({'label': label, 'flow': length, 'lo': length.copy()})
        
        print(f"  {filename} -> label={label}, 样本数={len(datas[label])}")
    
    return datas


def _transform(datas, block, limit, max_packet):
    data_trans = [[] for _ in range(len(datas))]
    for app in tqdm.tqdm(range(len(datas)), ascii=True, desc='[Transform]'):
        app_data = datas[app]
        for idx, example in enumerate(app_data):
            flow = example['flow']
            if len(flow) < limit:
                continue
            flow = [ix if ix <= max_packet else max_packet for ix in flow]
            flow = [ix // block + 3 for ix in flow]
            data_trans[app].append(
                {'label': example['label'], 'flow': flow, 'lo': example['lo'], 'id': str(app) + '-' + str(idx)}
            )
    return data_trans


def split_train_and_dev(datas, ratio=0.8, keep_ratio=1):
    train, dev = [], []
    for app_data in tqdm.tqdm(datas, ascii=True, desc='[Split]'):
        is_keep = np.random.rand(len(app_data)) <= keep_ratio
        is_train = np.random.rand(len(app_data)) <= ratio
        for example, kp, tr in zip(app_data, is_keep, is_train):
            if kp and tr:
                train.append(example)
            elif kp and not tr:
                dev.append(example)
    np.random.shuffle(train)
    np.random.shuffle(dev)
    return train, dev


def preprocess(config):
    eprint('Generate train and test.')
    print(f"数据目录: {config.data_dir}")
    print(f"类别数量: {config.class_num}")
    
    origin = load_origin_data(config.data_dir, config.class_num)
    
    # 打印各类别样本数
    print("\n各类别样本数:")
    for label, class_name in enumerate(['mid', 'low', 'high']):
        print(f"  {class_name} (label={label}): {len(origin[label])}")
    
    length = _transform(origin, config.length_block, config.min_length, config.max_packet_length)
    train, test = split_train_and_dev(length, config.split_ratio, config.keep_ratio)
    
    # 统计训练集和测试集的类别分布
    train_dist = {}
    test_dist = {}
    for item in train:
        train_dist[item['label']] = train_dist.get(item['label'], 0) + 1
    for item in test:
        test_dist[item['label']] = test_dist.get(item['label'], 0) + 1
    
    print(f"\n训练集: {len(train)} 样本")
    for label in sorted(train_dist.keys()):
        print(f"  label={label}: {train_dist[label]}")
    
    print(f"\n测试集: {len(test)} 样本")
    for label in sorted(test_dist.keys()):
        print(f"  label={label}: {test_dist[label]}")
    
    with open(config.train_json, 'w') as fp:
        json.dump(train, fp, indent=1)
    with open(config.test_json, 'w') as fp:
        json.dump(test, fp, indent=1)
    with open(config.train_meta, 'w') as fp:
        fp.write(str(len(train)))
    with open(config.test_meta, 'w') as fp:
        fp.write(str(len(test)))
    
    print(f"\n保存完成:")
    print(f"  训练集: {config.train_json}")
    print(f"  测试集: {config.test_json}")
