#!/usr/bin/env python3
"""
分析特定流的 DSCP 分布（按时间顺序）

用法:
    python analyze_flow_dscp.py capture.pcap --dport 43533
    python analyze_flow_dscp.py capture.pcap --dport 43533,42190
"""

import sys
from collections import Counter
from scapy.all import PcapReader, IP, TCP, UDP

DSCP_TO_CLASS = {0x00: 'mid', 0x20: 'low', 0x48: 'low', 0x90: 'high', 0xb8: 'high'}


def analyze_flow(pcap_path, target_ports):
    """分析指定端口的流"""
    flows = {}  # port -> list of (pkt_idx, dscp)
    
    pkt_idx = 0
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            pkt_idx += 1
            if not pkt.haslayer(IP) or not pkt.haslayer(TCP):
                continue
            
            tcp = pkt[TCP]
            sport, dport = tcp.sport, tcp.dport
            
            # 检查是否是目标端口
            port = None
            if sport in target_ports:
                port = sport
            elif dport in target_ports:
                port = dport
            
            if port is None:
                continue
            
            dscp = pkt[IP].tos & 0xfc
            
            if port not in flows:
                flows[port] = []
            flows[port].append((pkt_idx, dscp))
    
    return flows


def print_flow_analysis(flows):
    """打印流分析结果"""
    for port, packets in sorted(flows.items()):
        print(f"\n{'='*70}")
        print(f"端口 {port}: 共 {len(packets)} 包")
        print(f"{'='*70}")
        
        # DSCP 总体分布
        dscp_counts = Counter(dscp for _, dscp in packets)
        print(f"\nDSCP 总体分布:")
        for dscp, cnt in dscp_counts.most_common():
            cls = DSCP_TO_CLASS.get(dscp, '?')
            pct = cnt / len(packets) * 100
            print(f"  0x{dscp:02x} ({cls:>4s}): {cnt:>6,} ({pct:>5.1f}%)")
        
        # majority 结果
        majority_dscp = dscp_counts.most_common(1)[0][0]
        majority_cls = DSCP_TO_CLASS.get(majority_dscp, '?')
        print(f"\nmajority 方法 → {majority_cls} (0x{majority_dscp:02x})")
        
        # 分段分析
        n = len(packets)
        segments = [
            ("前 10%", 0, int(n * 0.1)),
            ("前 25%", 0, int(n * 0.25)),
            ("中间 50%", int(n * 0.25), int(n * 0.75)),
            ("后 25%", int(n * 0.75), n),
            ("后 10%", int(n * 0.9), n),
            ("最后 50 包", max(0, n-50), n),
            ("最后 20 包", max(0, n-20), n),
        ]
        
        print(f"\n分段 DSCP 分布:")
        for name, start, end in segments:
            if start >= end:
                continue
            seg_packets = packets[start:end]
            seg_counts = Counter(dscp for _, dscp in seg_packets)
            
            dist_str = ", ".join(f"0x{d:02x}:{c}" for d, c in seg_counts.most_common())
            seg_majority = seg_counts.most_common(1)[0][0]
            seg_cls = DSCP_TO_CLASS.get(seg_majority, '?')
            
            print(f"  {name:<12s} [{start:>6,}-{end:>6,}]: {dist_str} → {seg_cls}")
        
        # 显示 DSCP 变化点
        print(f"\nDSCP 变化点 (前20个):")
        prev_dscp = None
        changes = []
        for i, (pkt_idx, dscp) in enumerate(packets):
            if dscp != prev_dscp:
                changes.append((i, pkt_idx, prev_dscp, dscp))
                prev_dscp = dscp
        
        for i, (flow_idx, pkt_idx, old, new) in enumerate(changes[:20]):
            old_str = f"0x{old:02x}" if old is not None else "None"
            print(f"  包 #{flow_idx:>6,} (全局 #{pkt_idx:>7,}): {old_str} → 0x{new:02x} ({DSCP_TO_CLASS.get(new, '?')})")
        
        if len(changes) > 20:
            print(f"  ... 还有 {len(changes) - 20} 次变化")
        
        print(f"  总变化次数: {len(changes) - 1}")


def main():
    if len(sys.argv) < 3 or '--dport' not in sys.argv:
        print("用法: python analyze_flow_dscp.py <pcap> --dport <port1,port2,...>")
        print("示例: python analyze_flow_dscp.py capture.pcap --dport 43533,42190")
        sys.exit(1)
    
    pcap_path = sys.argv[1]
    dport_idx = sys.argv.index('--dport')
    ports_str = sys.argv[dport_idx + 1]
    target_ports = set(int(p) for p in ports_str.split(','))
    
    print(f"分析 {pcap_path} 中端口 {target_ports} 的流...")
    
    flows = analyze_flow(pcap_path, target_ports)
    
    if not flows:
        print("未找到匹配的流")
        sys.exit(1)
    
    print_flow_analysis(flows)


if __name__ == '__main__':
    main()