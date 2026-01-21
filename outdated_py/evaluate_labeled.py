#!/usr/bin/env python3
"""
评估有标签的分类流量
基于 IP DSCP 字段划分的标签
"""

import os
import sys
import json
import pickle
import numpy as np
from collections import defaultdict
from scapy.all import IP, TCP, UDP, PcapReader

# 添加 FS-Net 路径
sys.path.insert(0, '/home/cs1204.UNT/project/FS/fs-net')

# 特殊标记 (与 dataset_tf2.py 一致)
PAD_KEY = 0
START_KEY = 1
END_KEY = 2


def extract_flows_from_pcap(pcap_path, max_packets_per_flow=200, min_packets=3):
    """从 PCAP 文件中提取流"""
    flows = defaultdict(list)
    
    pkt_count = 0
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            pkt_count += 1
            
            if not pkt.haslayer(IP):
                continue
            
            ip = pkt[IP]
            
            if pkt.haslayer(TCP):
                proto = 'TCP'
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
            elif pkt.haslayer(UDP):
                proto = 'UDP'
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport
            else:
                continue
            
            # 双向流 key
            if (ip.src, sport) < (ip.dst, dport):
                flow_key = (proto, ip.src, sport, ip.dst, dport)
            else:
                flow_key = (proto, ip.dst, dport, ip.src, sport)
            
            # 包长度（绝对值）
            pkt_len = len(pkt)
            flows[flow_key].append(pkt_len)
    
    # 过滤有效流
    valid_flows = []
    for flow_key, packets in flows.items():
        if len(packets) >= min_packets:
            packets = packets[:max_packets_per_flow]
            valid_flows.append({
                'flow_key': flow_key,
                'packets': packets,
                'num_packets': len(packets)
            })
    
    return valid_flows, pkt_count


def prepare_flow_for_model(flow_packets, max_len=200, max_packet_length=5000, length_block=1):
    """与训练数据处理一致"""
    processed = [min(pkt_len, max_packet_length) for pkt_len in flow_packets]
    processed = [pkt_len // length_block + 3 for pkt_len in processed]
    processed = [START_KEY] + processed + [END_KEY]
    
    target_len = max_len + 2
    if len(processed) < target_len:
        processed = processed + [PAD_KEY] * (target_len - len(processed))
    else:
        processed = processed[:target_len]
    
    return processed


def load_model(model_dir):
    """加载模型"""
    import tensorflow as tf
    from model_tf2 import create_model
    
    # 加载配置
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    
    max_len = saved_config.get('max_len', 200)
    length_num = saved_config.get('length_num', 5004)
    max_packet_length = (length_num - 4) * 1
    
    class Config:
        pass
    config = Config()
    for k, v in saved_config.items():
        setattr(config, k, v)
    
    model = create_model(config)
    
    # 构建模型
    seq_len = max_len + 2
    dummy_input = tf.zeros((1, seq_len), dtype=tf.int32)
    dummy_mask = tf.cast(dummy_input, tf.bool)
    _ = model((dummy_input, dummy_mask), training=False)
    
    # 加载权重
    weights_path = os.path.join(model_dir, 'best_model.pkl')
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    
    return model, max_len, max_packet_length


def predict_flows(model, flows, max_len, max_packet_length):
    """预测流"""
    import tensorflow as tf
    
    batch_size = 64
    all_predictions = []
    all_confidences = []
    
    for i in range(0, len(flows), batch_size):
        batch_flows = flows[i:i+batch_size]
        
        batch_data = []
        for flow in batch_flows:
            processed = prepare_flow_for_model(
                flow['packets'], 
                max_len=max_len,
                max_packet_length=max_packet_length
            )
            batch_data.append(processed)
        
        batch_tensor = tf.constant(batch_data, dtype=tf.int32)
        batch_mask = tf.cast(batch_tensor, tf.bool)
        
        logits, pred, _ = model((batch_tensor, batch_mask), training=False)
        probs = tf.nn.softmax(logits, axis=-1)
        confidence = tf.reduce_max(probs, axis=-1)
        
        all_predictions.extend(pred.numpy().tolist())
        all_confidences.extend(confidence.numpy().tolist())
    
    return all_predictions, all_confidences


def compute_metrics(y_true, y_pred, num_classes=3):
    """计算评估指标"""
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    
    return accuracy, cm, report


def main():
    print("=" * 70)
    print("有标签流量分类评估")
    print("=" * 70)
    
    # DSCP 到标签的映射
    # 根据你的划分:
    #   cap_high.pcap (0x90 = EF) -> high (2) = VoIP
    #   cap_mid.pcap (0x00 = BE) -> mid (0) = Chat
    #   cap_low.pcap (0x20 = CS1) -> low (1) = Streaming
    #   cap_af21.pcap (AF21) -> ? (需要确认)
    
    labeled_files = {
        # 'pcap文件': 真实标签
        'cap_high.pcap': 2,    # high -> VoIP
        'cap_mid.pcap': 0,     # mid -> Chat  
        'cap_low.pcap': 1,     # low -> Streaming
        # 'cap_af21.pcap': ?,  # 需要确认映射到哪个类别
    }
    
    class_names = {0: 'mid (Chat)', 1: 'low (Streaming)', 2: 'high (VoIP)'}
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("\n用法: python evaluate_labeled.py <pcap_dir> [model_dir]")
        print("示例: python evaluate_labeled.py ./labeled_pcaps ./log")
        print("\n预期文件:")
        for f, label in labeled_files.items():
            print(f"  {f} -> {class_names[label]}")
        sys.exit(1)
    
    pcap_dir = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else './log'
    
    # 检查文件
    print(f"\n[1] 检查标签文件...")
    print(f"    目录: {pcap_dir}")
    
    available_files = {}
    for filename, label in labeled_files.items():
        filepath = os.path.join(pcap_dir, filename)
        if os.path.exists(filepath):
            available_files[filepath] = label
            print(f"    ✓ {filename} -> {class_names[label]}")
        else:
            print(f"    ✗ {filename} (未找到)")
    
    # 检查是否有 af21 文件
    af21_path = os.path.join(pcap_dir, 'cap_af21.pcap')
    if os.path.exists(af21_path):
        print(f"\n    发现 cap_af21.pcap，请指定其标签:")
        print(f"      0 = mid (Chat)")
        print(f"      1 = low (Streaming)")
        print(f"      2 = high (VoIP)")
        try:
            af21_label = int(input("    输入标签 (0/1/2): ").strip())
            if af21_label in [0, 1, 2]:
                available_files[af21_path] = af21_label
                print(f"    ✓ cap_af21.pcap -> {class_names[af21_label]}")
        except:
            print(f"    跳过 cap_af21.pcap")
    
    if not available_files:
        print("\n错误: 没有找到任何标签文件")
        sys.exit(1)
    
    # 加载模型
    print(f"\n[2] 加载模型...")
    model, max_len, max_packet_length = load_model(model_dir)
    print(f"    max_len: {max_len}")
    print(f"    max_packet_length: {max_packet_length}")
    
    # 处理每个文件
    print(f"\n[3] 提取流并预测...")
    
    all_true_labels = []
    all_pred_labels = []
    all_confidences = []
    
    results_by_class = defaultdict(lambda: {'total': 0, 'correct': 0, 'flows': []})
    
    for filepath, true_label in available_files.items():
        filename = os.path.basename(filepath)
        print(f"\n    处理: {filename}")
        
        flows, pkt_count = extract_flows_from_pcap(filepath, max_packets_per_flow=max_len)
        print(f"      包数: {pkt_count}, 流数: {len(flows)}")
        
        if not flows:
            print(f"      跳过 (无有效流)")
            continue
        
        predictions, confidences = predict_flows(model, flows, max_len, max_packet_length)
        
        # 统计
        correct = sum(1 for p in predictions if p == true_label)
        acc = correct / len(predictions) * 100
        print(f"      准确率: {correct}/{len(predictions)} = {acc:.1f}%")
        
        # 预测分布
        pred_dist = defaultdict(int)
        for p in predictions:
            pred_dist[p] += 1
        print(f"      预测分布: {dict(pred_dist)}")
        
        # 收集结果
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            all_true_labels.append(true_label)
            all_pred_labels.append(pred)
            all_confidences.append(conf)
            
            results_by_class[true_label]['total'] += 1
            if pred == true_label:
                results_by_class[true_label]['correct'] += 1
    
    # 总体评估
    print("\n" + "=" * 70)
    print("总体评估结果")
    print("=" * 70)
    
    if not all_true_labels:
        print("错误: 没有有效的流进行评估")
        sys.exit(1)
    
    accuracy, cm, report = compute_metrics(all_true_labels, all_pred_labels)
    
    print(f"\n总流数: {len(all_true_labels)}")
    print(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\n各类别准确率:")
    for label in sorted(results_by_class.keys()):
        data = results_by_class[label]
        if data['total'] > 0:
            acc = data['correct'] / data['total']
            print(f"  {class_names[label]}: {data['correct']}/{data['total']} = {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\n混淆矩阵:")
    print(f"           预测")
    print(f"真实     mid   low  high")
    for i, row in enumerate(cm):
        name = ['mid ', 'low ', 'high'][i] if i < 3 else f'{i}   '
        print(f"  {name}  {row[0]:4d}  {row[1]:4d}  {row[2]:4d}")
    
    print(f"\n详细分类报告:")
    print(report)
    
    print(f"\n置信度统计:")
    conf_array = np.array(all_confidences)
    print(f"  平均: {conf_array.mean():.4f}")
    print(f"  最低: {conf_array.min():.4f}")
    print(f"  最高: {conf_array.max():.4f}")
    
    # 错误分析
    print(f"\n" + "=" * 70)
    print("错误分析")
    print("=" * 70)
    
    errors = [(i, all_true_labels[i], all_pred_labels[i], all_confidences[i]) 
              for i in range(len(all_true_labels)) 
              if all_true_labels[i] != all_pred_labels[i]]
    
    print(f"\n错误数: {len(errors)} ({len(errors)/len(all_true_labels)*100:.1f}%)")
    
    if errors:
        # 按置信度排序
        errors_sorted = sorted(errors, key=lambda x: x[3], reverse=True)
        print(f"\n高置信度错误 (Top 10):")
        for idx, true_l, pred_l, conf in errors_sorted[:10]:
            print(f"  流{idx}: 真实={class_names[true_l]}, 预测={class_names[pred_l]}, 置信度={conf:.3f}")
    
    # 保存结果
    output_path = os.path.join(pcap_dir, 'evaluation_results.json')
    results = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'class_results': {
            class_names[k]: {
                'total': v['total'],
                'correct': v['correct'],
                'accuracy': v['correct']/v['total'] if v['total'] > 0 else 0
            }
            for k, v in results_by_class.items()
        },
        'total_flows': len(all_true_labels),
        'total_errors': len(errors)
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
