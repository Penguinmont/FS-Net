#!/usr/bin/env python3
"""
分析高置信度错误流的详细信息
"""

import os
import sys
import json
import pickle
import numpy as np
from collections import defaultdict
from scapy.all import IP, TCP, UDP, PcapReader

sys.path.insert(0, '/home/cs1204.UNT/project/FS/fs-net')

PAD_KEY = 0
START_KEY = 1
END_KEY = 2


def extract_flows_with_details(pcap_path, max_packets_per_flow=200, min_packets=3):
    """提取流并保留详细信息"""
    flows = defaultdict(lambda: {'packets': [], 'pkt_sizes': []})
    
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
            
            if (ip.src, sport) < (ip.dst, dport):
                flow_key = (proto, ip.src, sport, ip.dst, dport)
            else:
                flow_key = (proto, ip.dst, dport, ip.src, sport)
            
            pkt_len = len(pkt)
            flows[flow_key]['packets'].append(pkt_len)
            flows[flow_key]['pkt_sizes'].append(pkt_len)
    
    valid_flows = []
    for flow_key, data in flows.items():
        if len(data['packets']) >= min_packets:
            packets = data['packets'][:max_packets_per_flow]
            all_sizes = data['pkt_sizes']
            valid_flows.append({
                'flow_key': flow_key,
                'packets': packets,
                'num_packets': len(data['pkt_sizes']),  # 总包数
                'used_packets': len(packets),  # 用于分类的包数
                'min_size': min(all_sizes),
                'max_size': max(all_sizes),
                'avg_size': sum(all_sizes) / len(all_sizes),
                'total_bytes': sum(all_sizes)
            })
    
    return valid_flows, pkt_count


def prepare_flow_for_model(flow_packets, max_len=200, max_packet_length=5000, length_block=1):
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
    import tensorflow as tf
    from model_tf2 import create_model
    
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
    
    seq_len = max_len + 2
    dummy_input = tf.zeros((1, seq_len), dtype=tf.int32)
    dummy_mask = tf.cast(dummy_input, tf.bool)
    _ = model((dummy_input, dummy_mask), training=False)
    
    weights_path = os.path.join(model_dir, 'best_model.pkl')
    with open(weights_path, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    
    return model, max_len, max_packet_length


def predict_flows(model, flows, max_len, max_packet_length):
    import tensorflow as tf
    
    batch_size = 64
    all_predictions = []
    all_confidences = []
    all_probs = []
    
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
        all_probs.extend(probs.numpy().tolist())
    
    return all_predictions, all_confidences, all_probs


def main():
    print("=" * 80)
    print("高置信度错误流详细分析")
    print("=" * 80)
    
    labeled_files = {
        'cap_high.pcap': 2,
        'cap_mid.pcap': 0,
        'cap_low.pcap': 1,
    }
    
    class_names = {0: 'mid (Chat)', 1: 'low (Streaming)', 2: 'high (VoIP)'}
    
    if len(sys.argv) < 2:
        print("\n用法: python analyze_errors.py <pcap_dir> [model_dir]")
        sys.exit(1)
    
    pcap_dir = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else './log'
    
    # 加载模型
    print("\n[1] 加载模型...")
    model, max_len, max_packet_length = load_model(model_dir)
    
    # 收集所有流和错误
    all_flows = []
    all_results = []
    
    print("\n[2] 处理文件...")
    for filename, true_label in labeled_files.items():
        filepath = os.path.join(pcap_dir, filename)
        if not os.path.exists(filepath):
            continue
        
        print(f"\n    {filename}...")
        flows, _ = extract_flows_with_details(filepath, max_packets_per_flow=max_len)
        predictions, confidences, probs = predict_flows(model, flows, max_len, max_packet_length)
        
        for i, (flow, pred, conf, prob) in enumerate(zip(flows, predictions, confidences, probs)):
            all_flows.append(flow)
            all_results.append({
                'source_file': filename,
                'true_label': true_label,
                'pred_label': pred,
                'confidence': conf,
                'probs': prob,
                'is_error': pred != true_label
            })
    
    # 找出高置信度错误
    errors = []
    for i, (flow, result) in enumerate(zip(all_flows, all_results)):
        if result['is_error']:
            errors.append({
                'index': i,
                'flow': flow,
                'result': result
            })
    
    # 按置信度排序
    errors.sort(key=lambda x: x['result']['confidence'], reverse=True)
    
    print("\n" + "=" * 80)
    print(f"高置信度错误流详情 (共 {len(errors)} 个错误)")
    print("=" * 80)
    
    # 输出前20个高置信度错误
    print("\n" + "-" * 80)
    for rank, error in enumerate(errors[:20], 1):
        flow = error['flow']
        result = error['result']
        
        proto, src_ip, src_port, dst_ip, dst_port = flow['flow_key']
        
        print(f"\n【错误 #{rank}】")
        print(f"  来源文件: {result['source_file']}")
        print(f"  五元组: {proto} {src_ip}:{src_port} <-> {dst_ip}:{dst_port}")
        print(f"  真实标签: {class_names[result['true_label']]}")
        print(f"  预测标签: {class_names[result['pred_label']]}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  各类概率: mid={result['probs'][0]:.4f}, low={result['probs'][1]:.4f}, high={result['probs'][2]:.4f}")
        print(f"  总包数: {flow['num_packets']}")
        print(f"  用于分类: {flow['used_packets']} 包")
        print(f"  包大小: min={flow['min_size']}, max={flow['max_size']}, avg={flow['avg_size']:.1f}")
        print(f"  总字节: {flow['total_bytes']}")
        print(f"  前10个包大小: {flow['packets'][:10]}")
        print("-" * 80)
    
    # 生成 Wireshark 过滤器
    print("\n" + "=" * 80)
    print("Wireshark 过滤器 (用于定位这些流)")
    print("=" * 80)
    
    for rank, error in enumerate(errors[:10], 1):
        flow = error['flow']
        proto, src_ip, src_port, dst_ip, dst_port = flow['flow_key']
        
        if proto == 'TCP':
            filter_str = f"tcp && ((ip.src=={src_ip} && tcp.srcport=={src_port} && ip.dst=={dst_ip} && tcp.dstport=={dst_port}) || (ip.src=={dst_ip} && tcp.srcport=={dst_port} && ip.dst=={src_ip} && tcp.dstport=={src_port}))"
        else:
            filter_str = f"udp && ((ip.src=={src_ip} && udp.srcport=={src_port} && ip.dst=={dst_ip} && udp.dstport=={dst_port}) || (ip.src=={dst_ip} && udp.srcport=={dst_port} && ip.dst=={src_ip} && udp.dstport=={src_port}))"
        
        print(f"\n错误 #{rank} ({result['source_file']}):")
        print(f"  {filter_str}")
    
    # 保存详细结果到 JSON
    output_path = os.path.join(pcap_dir, 'error_analysis.json')
    error_details = []
    for error in errors:
        flow = error['flow']
        result = error['result']
        error_details.append({
            'source_file': result['source_file'],
            'flow_key': {
                'proto': flow['flow_key'][0],
                'src_ip': flow['flow_key'][1],
                'src_port': flow['flow_key'][2],
                'dst_ip': flow['flow_key'][3],
                'dst_port': flow['flow_key'][4]
            },
            'true_label': result['true_label'],
            'true_class': class_names[result['true_label']],
            'pred_label': result['pred_label'],
            'pred_class': class_names[result['pred_label']],
            'confidence': float(result['confidence']),
            'probs': {
                'mid': float(result['probs'][0]),
                'low': float(result['probs'][1]),
                'high': float(result['probs'][2])
            },
            'num_packets': flow['num_packets'],
            'min_size': flow['min_size'],
            'max_size': flow['max_size'],
            'avg_size': float(flow['avg_size']),
            'total_bytes': flow['total_bytes'],
            'first_10_packets': flow['packets'][:10]
        })
    
    with open(output_path, 'w') as f:
        json.dump(error_details, f, indent=2)
    print(f"\n\n详细结果已保存到: {output_path}")
    
    # 错误模式统计
    print("\n" + "=" * 80)
    print("错误模式统计")
    print("=" * 80)
    
    error_patterns = defaultdict(list)
    for error in errors:
        pattern = (error['result']['true_label'], error['result']['pred_label'])
        error_patterns[pattern].append(error)
    
    print("\n按错误类型分组:")
    for (true_l, pred_l), err_list in sorted(error_patterns.items(), key=lambda x: -len(x[1])):
        print(f"\n  {class_names[true_l]} -> {class_names[pred_l]}: {len(err_list)} 个")
        
        # 统计这类错误的特征
        sizes = [e['flow']['avg_size'] for e in err_list]
        pkts = [e['flow']['num_packets'] for e in err_list]
        confs = [e['result']['confidence'] for e in err_list]
        
        print(f"    平均包大小: min={min(sizes):.0f}, max={max(sizes):.0f}, avg={np.mean(sizes):.0f}")
        print(f"    包数量: min={min(pkts)}, max={max(pkts)}, avg={np.mean(pkts):.0f}")
        print(f"    置信度: min={min(confs):.3f}, max={max(confs):.3f}, avg={np.mean(confs):.3f}")


if __name__ == '__main__':
    main()
