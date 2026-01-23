#!/usr/bin/env python3
"""
DSCP 变化点检测 - 找到 qosmate DPI 识别完成的位置

原理：
    qosmate 流程: 新流 → 默认标记 → DPI识别 → 重新分类
    变化点就是"DPI识别完成"的位置，之后的包代表最终分类

用法:
    python detect_dscp_changepoint.py capture.pcap --dport 43533,42190
    python detect_dscp_changepoint.py capture.pcap --analyze_all
"""

import sys
import argparse
from collections import defaultdict, Counter
from scapy.all import PcapReader, IP, TCP, UDP
import numpy as np

DSCP_TO_CLASS = {0x00: 'mid', 0x20: 'low', 0x48: 'low', 0x90: 'high', 0xb8: 'high'}


def get_flow_key(pkt):
    if not pkt.haslayer(IP):
        return None, None
    ip = pkt[IP]
    if pkt.haslayer(TCP):
        proto, sport, dport = 'TCP', pkt[TCP].sport, pkt[TCP].dport
    elif pkt.haslayer(UDP):
        proto, sport, dport = 'UDP', pkt[UDP].sport, pkt[UDP].dport
    else:
        return None, None
    
    if (ip.src, sport) < (ip.dst, dport):
        return (proto, ip.src, sport, ip.dst, dport), (ip.tos & 0xfc)
    return (proto, ip.dst, dport, ip.src, sport), (ip.tos & 0xfc)


def detect_changepoint(dscp_list, window_size=50, stability_threshold=0.8):
    """
    检测 DSCP 稳定区间，确定最终分类
    
    算法（简化版，更可靠）：
    1. 使用"后 30% 排除最后 5%"的区间
    2. 计算该区间的 majority DSCP
    3. 这代表 qosmate DPI 识别后的最终决定
    
    参数:
        dscp_list: DSCP值列表（按时间顺序）
        window_size: 未使用（保留兼容性）
        stability_threshold: 未使用（保留兼容性）
    
    返回:
        stable_start: 稳定区间开始位置
        stable_end: 稳定区间结束位置
        final_class: 最终类别
    """
    n = len(dscp_list)
    
    if n < 100:
        # 短流：直接用 majority
        counts = Counter(dscp_list)
        majority_dscp = counts.most_common(1)[0][0]
        return {
            'changepoint': 0,
            'stable_start': 0,
            'stable_end': n,
            'final_class': DSCP_TO_CLASS.get(majority_dscp, '?'),
            'final_dscp': majority_dscp,
            'method': 'short_flow'
        }
    
    # 策略：后 30% 排除最后 5%
    # 这避开了：1) 开头的默认标记 2) 结尾的 TCP 控制包
    stable_start = int(n * 0.70)  # 从 70% 位置开始
    stable_end = int(n * 0.95)    # 到 95% 位置结束
    
    # 确保区间有效
    if stable_end <= stable_start:
        stable_end = n
    
    stable_dscp = dscp_list[stable_start:stable_end]
    stable_counts = Counter(stable_dscp)
    final_dscp = stable_counts.most_common(1)[0][0]
    final_class = DSCP_TO_CLASS.get(final_dscp, '?')
    
    # 统计变化点数量（用于诊断）
    change_count = sum(1 for i in range(1, n) if dscp_list[i] != dscp_list[i-1])
    
    return {
        'changepoint': stable_start,  # 近似变化点
        'stable_start': stable_start,
        'stable_end': stable_end,
        'stable_length': stable_end - stable_start,
        'final_class': final_class,
        'final_dscp': final_dscp,
        'change_count': change_count,
        'method': 'tail_segment',
        'segment_distribution': dict(stable_counts)
    }


def extract_stable_segment(dscp_list, pkt_list, max_packets=None):
    """
    提取稳定区间的包
    
    返回:
        stable_packets: 稳定区间的包列表
        info: 变化点检测信息
    """
    info = detect_changepoint(dscp_list)
    
    start = info['stable_start']
    end = info['stable_end']
    
    # 如果指定了 max_packets，从稳定区间中间取
    if max_packets and (end - start) > max_packets:
        # 从稳定区间中间取 max_packets 个包
        mid = (start + end) // 2
        half = max_packets // 2
        start = max(info['stable_start'], mid - half)
        end = min(info['stable_end'], start + max_packets)
    
    stable_packets = pkt_list[start:end]
    info['extracted_start'] = start
    info['extracted_end'] = end
    info['extracted_length'] = len(stable_packets)
    
    return stable_packets, info


def analyze_flow(dscp_list, port=None):
    """分析单个流并打印结果"""
    n = len(dscp_list)
    info = detect_changepoint(dscp_list)
    
    print(f"\n{'='*60}")
    if port:
        print(f"端口 {port}: {n:,} 包")
    else:
        print(f"流: {n:,} 包")
    print(f"{'='*60}")
    
    # 总体分布
    total_counts = Counter(dscp_list)
    print(f"\nDSCP 总体分布:")
    for dscp, cnt in total_counts.most_common():
        cls = DSCP_TO_CLASS.get(dscp, '?')
        print(f"  0x{dscp:02x} ({cls:>4s}): {cnt:>8,} ({cnt/n*100:>5.1f}%)")
    
    # 分段分布（诊断用）
    segments = [
        ("前 25%", 0, int(n * 0.25)),
        ("中间 50%", int(n * 0.25), int(n * 0.75)),
        ("后 25%", int(n * 0.75), n),
        ("稳定区间 (70%-95%)", int(n * 0.70), int(n * 0.95)),
    ]
    
    print(f"\n分段分布:")
    for name, start, end in segments:
        if start >= end:
            continue
        seg_dscp = dscp_list[start:end]
        seg_counts = Counter(seg_dscp)
        seg_majority = seg_counts.most_common(1)[0][0]
        seg_cls = DSCP_TO_CLASS.get(seg_majority, '?')
        
        dist_str = ", ".join(f"0x{d:02x}:{c}" for d, c in seg_counts.most_common()[:3])
        print(f"  {name:<25s}: {dist_str} → {seg_cls}")
    
    # 检测结果
    print(f"\n稳定区间检测:")
    print(f"  方法: {info['method']}")
    print(f"  稳定区间: [{info['stable_start']:,}, {info['stable_end']:,}] ({info.get('stable_length', 0):,} 包)")
    print(f"  最终类别: {info['final_class']} (0x{info['final_dscp']:02x})")
    if 'segment_distribution' in info:
        print(f"  区间分布: {info['segment_distribution']}")
    
    # 对比 majority 方法
    majority_dscp = total_counts.most_common(1)[0][0]
    majority_class = DSCP_TO_CLASS.get(majority_dscp, '?')
    
    print(f"\n方法对比:")
    print(f"  majority:     {majority_class} (0x{majority_dscp:02x})")
    print(f"  tail_segment: {info['final_class']} (0x{info['final_dscp']:02x})")
    
    if majority_class != info['final_class']:
        print(f"  ⚠️  结果不同！tail_segment 更准确（代表 qosmate 最终决定）")
    else:
        print(f"  ✓ 结果一致")
    
    return info


def main():
    parser = argparse.ArgumentParser(description='DSCP 变化点检测')
    parser.add_argument('pcap', help='PCAP 文件')
    parser.add_argument('--dport', help='分析指定端口 (逗号分隔)')
    parser.add_argument('--analyze_all', action='store_true', help='分析所有混合 DSCP 流')
    parser.add_argument('--top', type=int, default=10, help='显示前 N 个差异最大的流')
    args = parser.parse_args()
    
    # 读取 PCAP
    print(f"读取 {args.pcap}...")
    flows = defaultdict(list)
    
    pkt_count = 0
    with PcapReader(args.pcap) as reader:
        for pkt in reader:
            pkt_count += 1
            if pkt_count % 500000 == 0:
                print(f"  ...{pkt_count//1000}k 包")
            
            flow_key, dscp = get_flow_key(pkt)
            if flow_key and dscp is not None:
                flows[flow_key].append(dscp)
    
    print(f"总包数: {pkt_count:,}, 总流数: {len(flows):,}")
    
    if args.dport:
        # 分析指定端口
        target_ports = set(int(p) for p in args.dport.split(','))
        
        for flow_key, dscp_list in flows.items():
            proto, src_ip, sport, dst_ip, dport = flow_key
            if sport in target_ports or dport in target_ports:
                port = sport if sport in target_ports else dport
                analyze_flow(dscp_list, port)
    
    elif args.analyze_all:
        # 找出所有 majority 和 changepoint 结果不同的流
        differences = []
        
        for flow_key, dscp_list in flows.items():
            if len(dscp_list) < 100:  # 跳过短流
                continue
            
            total_counts = Counter(dscp_list)
            if len(total_counts) < 2:  # 跳过单一 DSCP 流
                continue
            
            majority_dscp = total_counts.most_common(1)[0][0]
            majority_class = DSCP_TO_CLASS.get(majority_dscp, '?')
            
            info = detect_changepoint(dscp_list)
            
            if majority_class != info['final_class']:
                differences.append({
                    'flow_key': flow_key,
                    'length': len(dscp_list),
                    'majority_class': majority_class,
                    'changepoint_class': info['final_class'],
                    'changepoint_pos': info['changepoint'],
                    'stable_length': info['stable_end'] - info['stable_start']
                })
        
        print(f"\n{'='*60}")
        print(f"majority 与 changepoint 方法结果不同的流: {len(differences)} 个")
        print(f"{'='*60}")
        
        # 按流长度排序
        differences.sort(key=lambda x: x['length'], reverse=True)
        
        print(f"\n前 {args.top} 个 (按流长度):")
        print(f"{'流':<45s} {'包数':>10s} {'majority':>10s} {'changepoint':>12s}")
        print(f"{'-'*77}")
        
        for d in differences[:args.top]:
            fk = d['flow_key']
            flow_str = f"{fk[0]} {fk[2]}<->{fk[4]}"
            print(f"{flow_str:<45s} {d['length']:>10,} {d['majority_class']:>10s} {d['changepoint_class']:>12s}")
        
        # 统计
        print(f"\n类别变化统计:")
        changes = Counter((d['majority_class'], d['changepoint_class']) for d in differences)
        for (from_cls, to_cls), cnt in changes.most_common():
            print(f"  {from_cls} → {to_cls}: {cnt} 流")


if __name__ == '__main__':
    main()