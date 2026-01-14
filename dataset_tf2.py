"""
数据集处理 - TensorFlow 2.x 兼容版本
"""

import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm


PAD_KEY = 0
START_KEY = 1
END_KEY = 2


def load_data_from_json(filename, max_len):
    """从JSON文件加载数据"""
    with open(filename, 'r') as fp:
        data = json.load(fp)
    
    ids_list = []
    labels_list = []
    flows_list = []
    
    for exp in data:
        flow_length = len(exp['flow'])
        if flow_length <= max_len:
            flow = [START_KEY] + exp['flow'] + [END_KEY] + [PAD_KEY] * (max_len - flow_length)
            ids_list.append(exp['id'])
            labels_list.append(exp['label'])
            flows_list.append(flow)
    
    return ids_list, np.array(labels_list, dtype=np.int32), np.array(flows_list, dtype=np.int32)


def create_dataset(filename, config, max_len, shuffle=True):
    """创建TensorFlow数据集"""
    ids_list, labels, flows = load_data_from_json(filename, max_len)
    
    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices({
        'id': ids_list,
        'label': labels,
        'flow': flows
    })
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=config.capacity)
    
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(ids_list)


def compute_accuracy(labels, predictions):
    """计算准确率"""
    correct = np.sum(labels == predictions)
    total = len(labels)
    return correct / total if total > 0 else 0.0


def compute_metrics(labels, predictions, num_classes):
    """计算每个类别的指标"""
    metrics = {}
    
    for c in range(num_classes):
        true_positives = np.sum((labels == c) & (predictions == c))
        false_positives = np.sum((labels != c) & (predictions == c))
        false_negatives = np.sum((labels == c) & (predictions != c))
        true_negatives = np.sum((labels != c) & (predictions != c))
        
        # TPR (Recall)
        tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        # FPR
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        # Precision
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        # F1
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
        
        metrics[f'class_{c}'] = {
            'TPR': tpr,
            'FPR': fpr,
            'Precision': precision,
            'F1': f1,
            'Support': int(np.sum(labels == c))
        }
    
    # Overall accuracy
    metrics['accuracy'] = compute_accuracy(labels, predictions)
    
    return metrics
