#!/usr/bin/env python3
"""
综合流量分类评估脚本
功能：
  1. 整体性能指标（准确率、Precision、Recall、F1、混淆矩阵）
  2. 各文件/类别详细统计
  3. 错误流详细信息（五元组、包大小、Wireshark过滤器）
  4. 错误模式分析
  5. 流特征分布对比
  6. 导出多种格式报告
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from collections import defaultdict
from datetime import datetime
from scapy.all import IP, TCP, UDP, PcapReader

sys.path.insert(0, '/home/cs1204.UNT/project/FS/fs-net')

# 特殊标记
PAD_KEY = 0
START_KEY = 1
END_KEY = 2

# 类别名称
CLASS_NAMES = {0: 'mid (Chat)', 1: 'low (Streaming)', 2: 'high (VoIP)'}
CLASS_SHORT = {0: 'mid', 1: 'low', 2: 'high'}


def extract_flows_with_details(pcap_path, max_packets_per_flow=200, min_packets=3):
    """提取流并保留详细信息"""
    flows = defaultdict(lambda: {
        'packets': [], 
        'all_sizes': [],
        'directions': [],  # 1=正向, -1=反向
    })
    
    pkt_count = 0
    first_ts = None
    last_ts = None
    
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            pkt_count += 1
            
            # 记录时间戳
            if hasattr(pkt, 'time'):
                if first_ts is None:
                    first_ts = float(pkt.time)
                last_ts = float(pkt.time)
            
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
            
            # 双向流 key 和方向
            if (ip.src, sport) < (ip.dst, dport):
                flow_key = (proto, ip.src, sport, ip.dst, dport)
                direction = 1
            else:
                flow_key = (proto, ip.dst, dport, ip.src, sport)
                direction = -1
            
            pkt_len = len(pkt)
            flows[flow_key]['packets'].append(pkt_len)
            flows[flow_key]['all_sizes'].append(pkt_len)
            flows[flow_key]['directions'].append(direction)
    
    # 转换为列表
    valid_flows = []
    for flow_key, data in flows.items():
        if len(data['packets']) >= min_packets:
            all_sizes = data['all_sizes']
            directions = data['directions']
            packets = data['packets'][:max_packets_per_flow]
            
            # 计算方向统计
            fwd_count = sum(1 for d in directions if d == 1)
            bwd_count = sum(1 for d in directions if d == -1)
            fwd_bytes = sum(s for s, d in zip(all_sizes, directions) if d == 1)
            bwd_bytes = sum(s for s, d in zip(all_sizes, directions) if d == -1)
            
            valid_flows.append({
                'flow_key': flow_key,
                'packets': packets,
                'num_packets': len(all_sizes),
                'used_packets': len(packets),
                'min_size': min(all_sizes),
                'max_size': max(all_sizes),
                'avg_size': sum(all_sizes) / len(all_sizes),
                'std_size': np.std(all_sizes) if len(all_sizes) > 1 else 0,
                'total_bytes': sum(all_sizes),
                'fwd_packets': fwd_count,
                'bwd_packets': bwd_count,
                'fwd_bytes': fwd_bytes,
                'bwd_bytes': bwd_bytes,
                'fwd_ratio': fwd_count / len(all_sizes),
            })
    
    duration = (last_ts - first_ts) if first_ts and last_ts else 0
    
    return valid_flows, pkt_count, duration


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
    
    return model, max_len, max_packet_length, saved_config


def predict_flows(model, flows, max_len, max_packet_length):
    """预测流"""
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


def compute_metrics(y_true, y_pred, num_classes=3):
    """计算评估指标"""
    from sklearn.metrics import (
        confusion_matrix, classification_report, accuracy_score,
        precision_recall_fscore_support
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(y_true, y_pred, digits=4, zero_division=0,
                                   target_names=[CLASS_SHORT[i] for i in range(num_classes)])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'report': report,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }


def generate_wireshark_filter(flow_key):
    """生成 Wireshark 过滤器"""
    proto, src_ip, src_port, dst_ip, dst_port = flow_key
    proto_lower = proto.lower()
    
    filter_str = (
        f"{proto_lower} && ("
        f"(ip.src=={src_ip} && {proto_lower}.srcport=={src_port} && "
        f"ip.dst=={dst_ip} && {proto_lower}.dstport=={dst_port}) || "
        f"(ip.src=={dst_ip} && {proto_lower}.srcport=={dst_port} && "
        f"ip.dst=={src_ip} && {proto_lower}.dstport=={src_port}))"
    )
    return filter_str


def print_section(title, char='='):
    """打印分节标题"""
    print(f"\n{char * 80}")
    print(f" {title}")
    print(f"{char * 80}")


def print_subsection(title, char='-'):
    """打印子节标题"""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def main():
    parser = argparse.ArgumentParser(description='综合流量分类评估')
    parser.add_argument('pcap_dir', help='包含标签 PCAP 文件的目录')
    parser.add_argument('model_dir', nargs='?', default='./log', help='模型目录')
    parser.add_argument('--top-errors', type=int, default=20, help='显示前N个高置信度错误')
    parser.add_argument('--min-packets', type=int, default=3, help='最小包数过滤')
    parser.add_argument('--output', type=str, default=None, help='输出报告路径')
    parser.add_argument('--no-wireshark', action='store_true', help='不生成Wireshark过滤器')
    args = parser.parse_args()
    
    # 标签文件映射
    labeled_files = {
        'cap_high.pcap': 2,
        'cap_mid.pcap': 0,
        'cap_low.pcap': 1,
    }
    
    print_section("综合流量分类评估报告")
    print(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PCAP 目录: {args.pcap_dir}")
    print(f"模型目录: {args.model_dir}")
    
    # ==================== 1. 检查文件 ====================
    print_section("1. 文件检查")
    
    available_files = {}
    for filename, label in labeled_files.items():
        filepath = os.path.join(args.pcap_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024 / 1024
            available_files[filepath] = label
            print(f"  ✓ {filename:25s} -> {CLASS_NAMES[label]:20s} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {filename:25s} (未找到)")
    
    # 检查 af21
    af21_path = os.path.join(args.pcap_dir, 'cap_af21.pcap')
    if os.path.exists(af21_path):
        print(f"\n  发现 cap_af21.pcap，请指定标签 (0=mid, 1=low, 2=high):")
        try:
            af21_label = int(input("  输入标签: ").strip())
            if af21_label in [0, 1, 2]:
                available_files[af21_path] = af21_label
                print(f"  ✓ cap_af21.pcap -> {CLASS_NAMES[af21_label]}")
        except:
            print(f"  跳过 cap_af21.pcap")
    
    if not available_files:
        print("\n错误: 没有找到任何标签文件")
        sys.exit(1)
    
    # ==================== 2. 加载模型 ====================
    print_section("2. 模型信息")
    
    model, max_len, max_packet_length, model_config = load_model(args.model_dir)
    
    print(f"  max_len: {max_len}")
    print(f"  max_packet_length: {max_packet_length}")
    print(f"  length_num: {model_config.get('length_num', 'N/A')}")
    print(f"  hidden: {model_config.get('hidden', 'N/A')}")
    print(f"  layer: {model_config.get('layer', 'N/A')}")
    
    # ==================== 3. 提取和预测 ====================
    print_section("3. 流提取与预测")
    
    all_flows = []
    all_results = []
    file_stats = {}
    
    for filepath, true_label in available_files.items():
        filename = os.path.basename(filepath)
        print(f"\n  处理: {filename}")
        
        flows, pkt_count, duration = extract_flows_with_details(
            filepath, max_packets_per_flow=max_len, min_packets=args.min_packets
        )
        
        print(f"    包数: {pkt_count:,}")
        print(f"    流数: {len(flows)}")
        print(f"    持续时间: {duration:.1f} 秒")
        
        if not flows:
            print(f"    ⚠ 无有效流，跳过")
            continue
        
        predictions, confidences, probs = predict_flows(model, flows, max_len, max_packet_length)
        
        # 统计
        correct = sum(1 for p in predictions if p == true_label)
        acc = correct / len(predictions) * 100
        
        pred_dist = defaultdict(int)
        for p in predictions:
            pred_dist[p] += 1
        
        print(f"    准确率: {correct}/{len(predictions)} = {acc:.1f}%")
        print(f"    预测分布: " + ", ".join(f"{CLASS_SHORT[k]}={v}" for k, v in sorted(pred_dist.items())))
        
        # 保存统计
        file_stats[filename] = {
            'true_label': true_label,
            'pkt_count': pkt_count,
            'flow_count': len(flows),
            'duration': duration,
            'accuracy': acc,
            'pred_dist': dict(pred_dist)
        }
        
        # 收集结果
        for i, (flow, pred, conf, prob) in enumerate(zip(flows, predictions, confidences, probs)):
            all_flows.append(flow)
            all_results.append({
                'source_file': filename,
                'true_label': true_label,
                'pred_label': pred,
                'confidence': conf,
                'probs': prob,
                'is_error': pred != true_label,
                'is_correct': pred == true_label
            })
    
    # ==================== 4. 总体指标 ====================
    print_section("4. 总体评估指标")
    
    y_true = [r['true_label'] for r in all_results]
    y_pred = [r['pred_label'] for r in all_results]
    
    metrics = compute_metrics(y_true, y_pred)
    
    print(f"\n  总流数: {len(all_results):,}")
    print(f"  总体准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print_subsection("各类别指标")
    print(f"  {'类别':<20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print(f"  {'-'*60}")
    for i in range(3):
        print(f"  {CLASS_NAMES[i]:<20s} {metrics['precision'][i]:>10.4f} {metrics['recall'][i]:>10.4f} "
              f"{metrics['f1'][i]:>10.4f} {metrics['support'][i]:>10d}")
    
    print_subsection("混淆矩阵")
    cm = metrics['confusion_matrix']
    print(f"\n  {'':>15s} | {'预测':^30s}")
    print(f"  {'真实':<15s} | {'mid':>10s} {'low':>10s} {'high':>10s}")
    print(f"  {'-'*50}")
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:<15s} | {row[0]:>10d} {row[1]:>10d} {row[2]:>10d}")
    
    print_subsection("置信度统计")
    conf_array = np.array([r['confidence'] for r in all_results])
    correct_conf = np.array([r['confidence'] for r in all_results if r['is_correct']])
    error_conf = np.array([r['confidence'] for r in all_results if r['is_error']])
    
    print(f"\n  {'指标':<20s} {'全部':>12s} {'正确':>12s} {'错误':>12s}")
    print(f"  {'-'*56}")
    print(f"  {'数量':<20s} {len(conf_array):>12d} {len(correct_conf):>12d} {len(error_conf):>12d}")
    print(f"  {'平均置信度':<20s} {conf_array.mean():>12.4f} {correct_conf.mean() if len(correct_conf) else 0:>12.4f} "
          f"{error_conf.mean() if len(error_conf) else 0:>12.4f}")
    print(f"  {'最低置信度':<20s} {conf_array.min():>12.4f} {correct_conf.min() if len(correct_conf) else 0:>12.4f} "
          f"{error_conf.min() if len(error_conf) else 0:>12.4f}")
    print(f"  {'最高置信度':<20s} {conf_array.max():>12.4f} {correct_conf.max() if len(correct_conf) else 0:>12.4f} "
          f"{error_conf.max() if len(error_conf) else 0:>12.4f}")
    
    # 置信度分布
    print(f"\n  置信度分布:")
    for threshold in [0.5, 0.7, 0.9, 0.95, 0.99]:
        count = (conf_array >= threshold).sum()
        print(f"    >= {threshold}: {count:5d} ({count/len(conf_array)*100:5.1f}%)")
    
    # ==================== 5. 错误分析 ====================
    print_section("5. 错误分析")
    
    errors = [
        {'index': i, 'flow': all_flows[i], 'result': all_results[i]}
        for i in range(len(all_results))
        if all_results[i]['is_error']
    ]
    errors.sort(key=lambda x: x['result']['confidence'], reverse=True)
    
    print(f"\n  总错误数: {len(errors)} ({len(errors)/len(all_results)*100:.1f}%)")
    
    # 错误模式统计
    print_subsection("错误模式分布")
    error_patterns = defaultdict(list)
    for error in errors:
        pattern = (error['result']['true_label'], error['result']['pred_label'])
        error_patterns[pattern].append(error)
    
    print(f"\n  {'真实类别':<20s} -> {'预测类别':<20s} {'数量':>8s} {'占比':>8s}")
    print(f"  {'-'*60}")
    for (true_l, pred_l), err_list in sorted(error_patterns.items(), key=lambda x: -len(x[1])):
        pct = len(err_list) / len(errors) * 100 if errors else 0
        print(f"  {CLASS_NAMES[true_l]:<20s} -> {CLASS_NAMES[pred_l]:<20s} {len(err_list):>8d} {pct:>7.1f}%")
    
    # 各错误类型的特征统计
    print_subsection("错误类型特征分析")
    for (true_l, pred_l), err_list in sorted(error_patterns.items(), key=lambda x: -len(x[1])):
        if len(err_list) < 3:
            continue
        
        print(f"\n  {CLASS_NAMES[true_l]} -> {CLASS_NAMES[pred_l]} ({len(err_list)} 个):")
        
        sizes = [e['flow']['avg_size'] for e in err_list]
        pkts = [e['flow']['num_packets'] for e in err_list]
        confs = [e['result']['confidence'] for e in err_list]
        fwd_ratios = [e['flow']['fwd_ratio'] for e in err_list]
        
        print(f"    平均包大小:  min={min(sizes):.0f}, max={max(sizes):.0f}, avg={np.mean(sizes):.0f}, std={np.std(sizes):.0f}")
        print(f"    总包数量:    min={min(pkts)}, max={max(pkts)}, avg={np.mean(pkts):.0f}")
        print(f"    正向包比例:  min={min(fwd_ratios):.2f}, max={max(fwd_ratios):.2f}, avg={np.mean(fwd_ratios):.2f}")
        print(f"    置信度:      min={min(confs):.3f}, max={max(confs):.3f}, avg={np.mean(confs):.3f}")
    
    # ==================== 6. 高置信度错误详情 ====================
    print_section(f"6. 高置信度错误详情 (Top {args.top_errors})")
    
    for rank, error in enumerate(errors[:args.top_errors], 1):
        flow = error['flow']
        result = error['result']
        proto, src_ip, src_port, dst_ip, dst_port = flow['flow_key']
        
        print(f"\n  ┌{'─' * 76}┐")
        print(f"  │ 错误 #{rank:<3d}                                                                  │")
        print(f"  ├{'─' * 76}┤")
        print(f"  │ 来源文件:   {result['source_file']:<62s} │")
        print(f"  │ 五元组:     {proto} {src_ip}:{src_port} <-> {dst_ip}:{dst_port:<30} │")
        print(f"  │ 真实标签:   {CLASS_NAMES[result['true_label']]:<62s} │")
        print(f"  │ 预测标签:   {CLASS_NAMES[result['pred_label']]:<62s} │")
        print(f"  │ 置信度:     {result['confidence']:.4f}                                                      │")
        print(f"  │ 各类概率:   mid={result['probs'][0]:.4f}, low={result['probs'][1]:.4f}, high={result['probs'][2]:.4f}                       │")
        print(f"  ├{'─' * 76}┤")
        print(f"  │ 总包数:     {flow['num_packets']:<10d}  用于分类: {flow['used_packets']:<10d}                        │")
        print(f"  │ 包大小:     min={flow['min_size']:<5d}  max={flow['max_size']:<5d}  avg={flow['avg_size']:<7.1f}  std={flow['std_size']:<7.1f}   │")
        print(f"  │ 字节数:     总计={flow['total_bytes']:<10d}  正向={flow['fwd_bytes']:<10d}  反向={flow['bwd_bytes']:<10d}   │")
        print(f"  │ 包方向:     正向={flow['fwd_packets']:<5d}  反向={flow['bwd_packets']:<5d}  比例={flow['fwd_ratio']:.2f}                   │")
        print(f"  ├{'─' * 76}┤")
        pkt_str = str(flow['packets'][:15])
        if len(pkt_str) > 70:
            pkt_str = pkt_str[:67] + "..."
        print(f"  │ 前15包大小: {pkt_str:<63s}│")
        print(f"  └{'─' * 76}┘")
    
    # ==================== 7. Wireshark 过滤器 ====================
    if not args.no_wireshark:
        print_section("7. Wireshark 过滤器")
        print("\n  复制以下过滤器到 Wireshark 可定位对应流:\n")
        
        for rank, error in enumerate(errors[:10], 1):
            flow = error['flow']
            result = error['result']
            filter_str = generate_wireshark_filter(flow['flow_key'])
            
            print(f"  # 错误 #{rank} ({result['source_file']}, {CLASS_NAMES[result['true_label']]} -> {CLASS_NAMES[result['pred_label']]})")
            print(f"  {filter_str}\n")
    
    # ==================== 8. 正确分类的流特征对比 ====================
    print_section("8. 正确 vs 错误分类的流特征对比")
    
    correct_flows = [all_flows[i] for i in range(len(all_results)) if all_results[i]['is_correct']]
    error_flows = [all_flows[i] for i in range(len(all_results)) if all_results[i]['is_error']]
    
    if correct_flows and error_flows:
        print(f"\n  {'特征':<20s} {'正确分类':>15s} {'错误分类':>15s} {'差异':>15s}")
        print(f"  {'-'*65}")
        
        features = [
            ('平均包大小', 'avg_size'),
            ('最大包大小', 'max_size'),
            ('最小包大小', 'min_size'),
            ('总包数', 'num_packets'),
            ('总字节数', 'total_bytes'),
            ('正向包比例', 'fwd_ratio'),
        ]
        
        for name, key in features:
            correct_vals = [f[key] for f in correct_flows]
            error_vals = [f[key] for f in error_flows]
            correct_mean = np.mean(correct_vals)
            error_mean = np.mean(error_vals)
            diff = error_mean - correct_mean
            diff_pct = diff / correct_mean * 100 if correct_mean != 0 else 0
            print(f"  {name:<20s} {correct_mean:>15.1f} {error_mean:>15.1f} {diff_pct:>+14.1f}%")
    
    # ==================== 9. 保存报告 ====================
    print_section("9. 保存报告")
    
    output_dir = args.output if args.output else args.pcap_dir
    
    # JSON 详细报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'pcap_dir': args.pcap_dir,
            'model_dir': args.model_dir,
            'min_packets': args.min_packets,
        },
        'summary': {
            'total_flows': len(all_results),
            'accuracy': float(metrics['accuracy']),
            'total_errors': len(errors),
        },
        'class_metrics': {
            CLASS_SHORT[i]: {
                'precision': float(metrics['precision'][i]),
                'recall': float(metrics['recall'][i]),
                'f1': float(metrics['f1'][i]),
                'support': int(metrics['support'][i]),
            }
            for i in range(3)
        },
        'confusion_matrix': cm.tolist(),
        'file_stats': file_stats,
        'error_patterns': {
            f"{CLASS_SHORT[k[0]]}->{CLASS_SHORT[k[1]]}": len(v)
            for k, v in error_patterns.items()
        },
        'errors': [
            {
                'rank': rank,
                'source_file': error['result']['source_file'],
                'flow_key': {
                    'proto': error['flow']['flow_key'][0],
                    'src_ip': error['flow']['flow_key'][1],
                    'src_port': error['flow']['flow_key'][2],
                    'dst_ip': error['flow']['flow_key'][3],
                    'dst_port': error['flow']['flow_key'][4],
                },
                'true_label': error['result']['true_label'],
                'true_class': CLASS_NAMES[error['result']['true_label']],
                'pred_label': error['result']['pred_label'],
                'pred_class': CLASS_NAMES[error['result']['pred_label']],
                'confidence': float(error['result']['confidence']),
                'probs': {
                    'mid': float(error['result']['probs'][0]),
                    'low': float(error['result']['probs'][1]),
                    'high': float(error['result']['probs'][2]),
                },
                'flow_stats': {
                    'num_packets': error['flow']['num_packets'],
                    'min_size': error['flow']['min_size'],
                    'max_size': error['flow']['max_size'],
                    'avg_size': float(error['flow']['avg_size']),
                    'total_bytes': error['flow']['total_bytes'],
                    'fwd_packets': error['flow']['fwd_packets'],
                    'bwd_packets': error['flow']['bwd_packets'],
                    'fwd_ratio': float(error['flow']['fwd_ratio']),
                },
                'first_15_packets': error['flow']['packets'][:15],
                'wireshark_filter': generate_wireshark_filter(error['flow']['flow_key']),
            }
            for rank, error in enumerate(errors, 1)
        ]
    }
    
    json_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ JSON 报告: {json_path}")
    
    # CSV 错误列表
    csv_path = os.path.join(output_dir, 'error_flows.csv')
    with open(csv_path, 'w') as f:
        f.write("rank,source_file,proto,src_ip,src_port,dst_ip,dst_port,true_class,pred_class,"
                "confidence,num_packets,avg_size,total_bytes,fwd_ratio\n")
        for rank, error in enumerate(errors, 1):
            fk = error['flow']['flow_key']
            f.write(f"{rank},{error['result']['source_file']},{fk[0]},{fk[1]},{fk[2]},{fk[3]},{fk[4]},"
                    f"{CLASS_SHORT[error['result']['true_label']]},{CLASS_SHORT[error['result']['pred_label']]},"
                    f"{error['result']['confidence']:.4f},{error['flow']['num_packets']},"
                    f"{error['flow']['avg_size']:.1f},{error['flow']['total_bytes']},"
                    f"{error['flow']['fwd_ratio']:.2f}\n")
    print(f"  ✓ CSV 错误列表: {csv_path}")
    
    # Wireshark 过滤器文件
    filter_path = os.path.join(output_dir, 'wireshark_filters.txt')
    with open(filter_path, 'w') as f:
        for rank, error in enumerate(errors[:50], 1):
            result = error['result']
            filter_str = generate_wireshark_filter(error['flow']['flow_key'])
            f.write(f"# 错误 #{rank}: {result['source_file']}, "
                    f"{CLASS_SHORT[result['true_label']]}->{CLASS_SHORT[result['pred_label']]}, "
                    f"conf={result['confidence']:.3f}\n")
            f.write(f"{filter_str}\n\n")
    print(f"  ✓ Wireshark 过滤器: {filter_path}")
    
    print_section("评估完成")
    print(f"\n  总体准确率: {metrics['accuracy']*100:.2f}%")
    print(f"  错误数: {len(errors)}/{len(all_results)} ({len(errors)/len(all_results)*100:.1f}%)")


if __name__ == '__main__':
    main()
