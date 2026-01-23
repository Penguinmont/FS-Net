#!/usr/bin/env python3
"""
分析 PCAP 文件中流的包数量分布

使用方法:
    python analyze_flow_lengths.py capture.pcap
    python analyze_flow_lengths.py ./flow_split/
"""

import os
import sys
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
        flow_key = (proto, ip.src, sport, ip.dst, dport)
    else:
        flow_key = (proto, ip.dst, dport, ip.src, sport)
    return flow_key, (ip.tos & 0xfc)


def analyze_pcap(pcap_path, class_label=None):
    flows = defaultdict(lambda: {'count': 0, 'dscp': Counter()})
    
    pkt_count = 0
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            pkt_count += 1
            if pkt_count % 500000 == 0:
                print(f"  ...{pkt_count//1000}k 包", flush=True)
            flow_key, dscp = get_flow_key(pkt)
            if flow_key:
                flows[flow_key]['count'] += 1
                flows[flow_key]['dscp'][dscp] += 1
    
    result = []
    for fk, data in flows.items():
        if class_label:
            cls = class_label
        else:
            majority_dscp = data['dscp'].most_common(1)[0][0]
            cls = DSCP_TO_CLASS.get(majority_dscp, '?')
        result.append({'length': data['count'], 'class': cls})
    
    return result


def print_stats(flows, title=""):
    if not flows:
        return
    
    lengths = sorted([f['length'] for f in flows])
    
    print(f"\n[{title}]" if title else "\n[流统计]")
    print(f"流数: {len(lengths):,}  包数: {sum(lengths):,}  范围: {min(lengths)}-{max(lengths):,}  中位数: {np.median(lengths):.0f}")
    
    bins = [(1,2), (3,9), (10,49), (50,99), (100,199), (200,499), (500,999), (1000,9999), (10000,float('inf'))]
    labels = ['1-2', '3-9', '10-49', '50-99', '100-199', '200-499', '500-999', '1k-10k', '10k+']
    
    dist = []
    for (lo, hi), label in zip(bins, labels):
        cnt = sum(1 for l in lengths if lo <= l <= hi)
        if cnt > 0:
            dist.append(f"{label}:{cnt}")
    print(f"分布: {', '.join(dist)}")
    
    # max_len=200 影响
    kept = sum(1 for l in lengths if l <= 200)
    print(f"max_len=200: 保留 {kept}/{len(lengths)} ({kept/len(lengths)*100:.1f}%)")


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_flow_lengths.py <pcap_or_dir>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        by_class = defaultdict(list)
        
        for fname in sorted(os.listdir(path)):
            if not fname.endswith('.pcap'):
                continue
            
            cls = None
            for c in ['high', 'low', 'mid']:
                if c in fname.lower():
                    cls = c
                    break
            
            flows = analyze_pcap(os.path.join(path, fname), cls)
            if cls:
                by_class[cls].extend(flows)
        
        for cls in ['mid', 'low', 'high']:
            if cls in by_class:
                print_stats(by_class[cls], cls)
    else:
        flows = analyze_pcap(path)
        print_stats(flows, os.path.basename(path))


if __name__ == '__main__':
    main()