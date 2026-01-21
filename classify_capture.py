#!/usr/bin/env python3
"""
对捕获的混合流量进行分类
"""

import os
import sys
import json
import pickle
import numpy as np
from collections import defaultdict
from scapy.all import rdpcap, IP, TCP, UDP, PcapReader

# 添加 FS-Net 路径
sys.path.insert(0, '/home/cs1204.UNT/project/FS/fs-net')


def extract_flows_from_pcap(pcap_path, max_packets_per_flow=100, min_packets=3):
    """从 PCAP 文件中提取流"""
    print(f"\n[1] 从 {pcap_path} 提取流...")
    
    flows = defaultdict(list)  # flow_key -> [(pkt_len, direction), ...]
    
    # 使用流式读取处理大文件
    pkt_count = 0
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            pkt_count += 1
            if pkt_count % 500000 == 0:
                print(f"    已处理 {pkt_count} 个包, 当前流数: {len(flows)}")
            
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
            
            # 创建双向流 key（排序 IP 和端口）
            if (ip.src, sport) < (ip.dst, dport):
                flow_key = (proto, ip.src, sport, ip.dst, dport)
                direction = 1  # 正向
            else:
                flow_key = (proto, ip.dst, dport, ip.src, sport)
                direction = -1  # 反向
            
            # 记录包长度（带方向）
            pkt_len = len(pkt)
            flows[flow_key].append(pkt_len * direction)
    
    print(f"    总包数: {pkt_count}")
    print(f"    提取流数: {len(flows)}")
    
    # 过滤并转换
    valid_flows = []
    for flow_key, packets in flows.items():
        if len(packets) >= min_packets:
            # 截断到 max_packets_per_flow
            packets = packets[:max_packets_per_flow]
            valid_flows.append({
                'flow_key': flow_key,
                'packets': packets,
                'num_packets': len(packets)
            })
    
    print(f"    有效流数 (>={min_packets} 包): {len(valid_flows)}")
    
    return valid_flows


def prepare_flow_for_model(flow_packets, max_len=100, length_num=5004):
    """将流数据转换为模型输入格式"""
    # 与 dataset_tf2.py 中的处理保持一致
    processed = []
    for pkt_len in flow_packets:
        # 处理方向和长度
        if pkt_len > 0:
            # 正向: 1-2500
            val = min(pkt_len, 2500)
        else:
            # 反向: 2501-5000
            val = min(abs(pkt_len), 2500) + 2500
        processed.append(val)
    
    # 填充或截断
    if len(processed) < max_len:
        processed = processed + [0] * (max_len - len(processed))
    else:
        processed = processed[:max_len]
    
    return processed


def classify_flows(flows, model_dir, max_len=100):
    """使用 FS-Net 模型分类流"""
    import tensorflow as tf
    from model_tf2 import create_model
    
    print(f"\n[2] 加载模型...")
    
    # 加载配置
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    print(f"    配置: {saved_config}")
    
    # 创建配置对象
    class Config:
        pass
    config = Config()
    for k, v in saved_config.items():
        setattr(config, k, v)
    
    # 创建模型
    model = create_model(config)
    
    # 构建模型
    dummy_input = tf.zeros((1, max_len), dtype=tf.int32)
    dummy_mask = tf.cast(dummy_input, tf.bool)
    _ = model((dummy_input, dummy_mask), training=False)
    
    # 加载权重
    weights_path = os.path.join(model_dir, 'best_model.pkl')
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    print(f"    已加载权重")
    
    # 准备数据
    print(f"\n[3] 准备数据并分类...")
    
    batch_size = 64
    all_predictions = []
    all_confidences = []
    
    for i in range(0, len(flows), batch_size):
        batch_flows = flows[i:i+batch_size]
        
        # 转换为模型输入
        batch_data = []
        for flow in batch_flows:
            processed = prepare_flow_for_model(flow['packets'], max_len, config.length_num)
            batch_data.append(processed)
        
        batch_tensor = tf.constant(batch_data, dtype=tf.int32)
        batch_mask = tf.cast(batch_tensor, tf.bool)
        
        # 预测
        logits, pred, _ = model((batch_tensor, batch_mask), training=False)
        
        # 计算置信度
        probs = tf.nn.softmax(logits, axis=-1)
        confidence = tf.reduce_max(probs, axis=-1)
        
        all_predictions.extend(pred.numpy().tolist())
        all_confidences.extend(confidence.numpy().tolist())
        
        if (i + batch_size) % 1000 == 0 or i + batch_size >= len(flows):
            print(f"    已分类 {min(i+batch_size, len(flows))}/{len(flows)} 流")
    
    return all_predictions, all_confidences


def main():
    if len(sys.argv) < 2:
        print("用法: python classify_capture.py <pcap_file> [model_dir]")
        print("示例: python classify_capture.py capture.pcap ./log")
        sys.exit(1)
    
    pcap_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else './log'
    
    print("=" * 60)
    print("混合流量分类")
    print("=" * 60)
    
    # 提取流
    flows = extract_flows_from_pcap(pcap_path, max_packets_per_flow=100, min_packets=3)
    
    if not flows:
        print("错误: 没有提取到有效流")
        sys.exit(1)
    
    # 分类
    predictions, confidences = classify_flows(flows, model_dir)
    
    # 统计结果
    print("\n" + "=" * 60)
    print("分类结果统计")
    print("=" * 60)
    
    class_names = {0: 'mid (Chat)', 1: 'low (Streaming)', 2: 'high (VoIP)'}
    
    # 按类别统计
    class_counts = defaultdict(int)
    class_flows = defaultdict(list)
    
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        class_counts[pred] += 1
        class_flows[pred].append((flows[i], conf))
    
    print(f"\n总流数: {len(flows)}")
    print("\n按类别分布:")
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        pct = count / len(flows) * 100
        name = class_names.get(cls, f'class_{cls}')
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # 按置信度统计
    print("\n置信度分布:")
    conf_array = np.array(confidences)
    print(f"  平均置信度: {conf_array.mean():.3f}")
    print(f"  最低置信度: {conf_array.min():.3f}")
    print(f"  最高置信度: {conf_array.max():.3f}")
    
    high_conf = (conf_array >= 0.9).sum()
    mid_conf = ((conf_array >= 0.7) & (conf_array < 0.9)).sum()
    low_conf = (conf_array < 0.7).sum()
    print(f"  高置信度 (>=0.9): {high_conf} ({high_conf/len(flows)*100:.1f}%)")
    print(f"  中置信度 (0.7-0.9): {mid_conf} ({mid_conf/len(flows)*100:.1f}%)")
    print(f"  低置信度 (<0.7): {low_conf} ({low_conf/len(flows)*100:.1f}%)")
    
    # 显示每个类别的示例流
    print("\n" + "=" * 60)
    print("各类别示例流 (按置信度排序)")
    print("=" * 60)
    
    for cls in sorted(class_counts.keys()):
        name = class_names.get(cls, f'class_{cls}')
        print(f"\n{name}:")
        
        # 按置信度排序
        sorted_flows = sorted(class_flows[cls], key=lambda x: x[1], reverse=True)
        
        # 显示前3个高置信度流
        for j, (flow, conf) in enumerate(sorted_flows[:3]):
            key = flow['flow_key']
            print(f"  [{j+1}] {key[0]} {key[1]}:{key[2]} <-> {key[3]}:{key[4]}")
            print(f"      包数: {flow['num_packets']}, 置信度: {conf:.3f}")
    
    # 保存详细结果
    output_path = pcap_path.replace('.pcap', '_classification.json')
    results = []
    for i, (flow, pred, conf) in enumerate(zip(flows, predictions, confidences)):
        results.append({
            'flow_id': i,
            'flow_key': flow['flow_key'],
            'num_packets': flow['num_packets'],
            'prediction': int(pred),
            'class_name': class_names.get(pred, f'class_{pred}'),
            'confidence': float(conf)
        })
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n详细结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
