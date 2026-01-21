#!/usr/bin/env python3
"""
按流（五元组）划分 PCAP，而不是按包划分

问题：按单个包的 DSCP 划分会导致同一流被拆分到不同类别
解决：先识别所有流，然后根据流的主要 DSCP 值或第一个包的 DSCP 决定类别
"""

import os
import sys
import argparse
from collections import defaultdict, Counter
from scapy.all import PcapReader, PcapWriter, IP, TCP, UDP, ICMP, Ether

# DSCP 到类别的映射
DSCP_TO_CLASS = {
    0x00: 'mid',    # CS0 / Best Effort
    0x20: 'low',    # CS1
    0x48: 'low',    # AF21 (0x48 >> 2 = 18)
    0x90: 'high',   # EF (0x90 >> 2 = 36) 或 AF42
    0xb8: 'high',   # EF (0xb8 >> 2 = 46)
    0xc0: 'control', # CS6 - 网络控制，应该排除
}

def get_dscp(tos_field):
    """从 TOS 字段提取 DSCP (前6位)"""
    return tos_field & 0xfc  # 保留前6位

def get_flow_key(pkt):
    """获取流的五元组 key（双向）"""
    if not pkt.haslayer(IP):
        return None, None
    
    ip = pkt[IP]
    proto = ip.proto
    
    if pkt.haslayer(TCP):
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
        proto_name = 'TCP'
    elif pkt.haslayer(UDP):
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
        proto_name = 'UDP'
    elif pkt.haslayer(ICMP):
        # ICMP 没有端口，用 type/code 代替
        sport = pkt[ICMP].type
        dport = pkt[ICMP].code
        proto_name = 'ICMP'
    else:
        return None, None
    
    # 双向流 key（排序确保双向一致）
    if (ip.src, sport) < (ip.dst, dport):
        flow_key = (proto_name, ip.src, sport, ip.dst, dport)
    else:
        flow_key = (proto_name, ip.dst, dport, ip.src, sport)
    
    dscp = get_dscp(ip.tos)
    return flow_key, dscp


def analyze_pcap(pcap_path):
    """分析 PCAP 中的流和 DSCP 分布"""
    print(f"\n分析文件: {pcap_path}")
    
    flows = defaultdict(lambda: {'dscp_counts': Counter(), 'packets': [], 'first_dscp': None})
    
    pkt_count = 0
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            pkt_count += 1
            if pkt_count % 500000 == 0:
                print(f"  已处理 {pkt_count} 包...")
            
            flow_key, dscp = get_flow_key(pkt)
            if flow_key is None:
                continue
            
            flows[flow_key]['dscp_counts'][dscp] += 1
            flows[flow_key]['packets'].append(pkt)
            if flows[flow_key]['first_dscp'] is None:
                flows[flow_key]['first_dscp'] = dscp
    
    print(f"  总包数: {pkt_count}")
    print(f"  总流数: {len(flows)}")
    
    return flows


def classify_flow(flow_data, method='majority'):
    """
    确定流的类别
    
    method:
        'majority': 使用最多的 DSCP 值
        'first': 使用第一个包的 DSCP 值
        'strict': 如果有多个 DSCP 值，标记为 'mixed'
    """
    dscp_counts = flow_data['dscp_counts']
    first_dscp = flow_data['first_dscp']
    
    if method == 'first':
        dscp = first_dscp
    elif method == 'majority':
        dscp = dscp_counts.most_common(1)[0][0]
    elif method == 'strict':
        if len(dscp_counts) > 1:
            return 'mixed', dscp_counts
        dscp = list(dscp_counts.keys())[0]
    else:
        dscp = first_dscp
    
    # 映射到类别
    if dscp in DSCP_TO_CLASS:
        return DSCP_TO_CLASS[dscp], dscp_counts
    else:
        # 未知 DSCP，根据值范围猜测
        dscp_value = dscp >> 2
        if dscp_value >= 40:
            return 'high', dscp_counts
        elif dscp_value >= 16:
            return 'low', dscp_counts
        else:
            return 'mid', dscp_counts


def split_by_flow(pcap_path, output_dir, method='majority', exclude_icmp=True, exclude_control=True):
    """按流划分 PCAP"""
    
    # 分析流
    flows = analyze_pcap(pcap_path)
    
    # 分类每个流
    flow_classes = defaultdict(list)  # class -> [flow_key, ...]
    mixed_flows = []
    excluded_flows = []
    
    print(f"\n分类流...")
    for flow_key, flow_data in flows.items():
        proto = flow_key[0]
        
        # 排除 ICMP
        if exclude_icmp and proto == 'ICMP':
            excluded_flows.append((flow_key, 'ICMP'))
            continue
        
        # 分类
        flow_class, dscp_counts = classify_flow(flow_data, method)
        
        # 排除控制流量
        if exclude_control and flow_class == 'control':
            excluded_flows.append((flow_key, 'control'))
            continue
        
        # 检查是否混合 DSCP
        if len(dscp_counts) > 1:
            mixed_flows.append((flow_key, dscp_counts, flow_class))
        
        flow_classes[flow_class].append(flow_key)
    
    # 统计
    print(f"\n流分类统计:")
    for cls, keys in sorted(flow_classes.items()):
        pkt_count = sum(len(flows[k]['packets']) for k in keys)
        print(f"  {cls}: {len(keys)} 流, {pkt_count} 包")
    
    if excluded_flows:
        print(f"\n排除的流: {len(excluded_flows)}")
        for fk, reason in excluded_flows[:5]:
            print(f"  {fk[0]} {fk[1]}:{fk[2]} <-> {fk[3]}:{fk[4]} ({reason})")
        if len(excluded_flows) > 5:
            print(f"  ... 还有 {len(excluded_flows) - 5} 个")
    
    if mixed_flows:
        print(f"\n混合 DSCP 的流 (同一流有多个 DSCP 值): {len(mixed_flows)}")
        for fk, dscp_counts, assigned_class in mixed_flows[:10]:
            dscp_str = ", ".join(f"0x{d:02x}:{c}" for d, c in dscp_counts.most_common())
            print(f"  {fk[0]} {fk[1]}:{fk[2]} <-> {fk[3]}:{fk[4]}")
            print(f"    DSCP分布: {dscp_str}")
            print(f"    分配类别: {assigned_class} (使用 {method} 方法)")
        if len(mixed_flows) > 10:
            print(f"  ... 还有 {len(mixed_flows) - 10} 个混合流")
    
    # 写入文件
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n写入文件...")
    for cls, keys in flow_classes.items():
        if cls in ['mixed', 'control']:
            continue
        
        output_path = os.path.join(output_dir, f'flow_{cls}.pcap')
        
        with PcapWriter(output_path, append=False, sync=True) as writer:
            pkt_count = 0
            for flow_key in keys:
                for pkt in flows[flow_key]['packets']:
                    writer.write(pkt)
                    pkt_count += 1
        
        print(f"  ✓ {output_path}: {len(keys)} 流, {pkt_count} 包")
    
    # 保存混合流（可选）
    if mixed_flows:
        mixed_path = os.path.join(output_dir, 'flow_mixed.pcap')
        with PcapWriter(mixed_path, append=False, sync=True) as writer:
            pkt_count = 0
            for flow_key, _, _ in mixed_flows:
                for pkt in flows[flow_key]['packets']:
                    writer.write(pkt)
                    pkt_count += 1
        print(f"  ✓ {mixed_path}: {len(mixed_flows)} 流, {pkt_count} 包 (混合DSCP)")
    
    # 保存详细报告
    report_path = os.path.join(output_dir, 'split_report.json')
    report = {
        'source_file': pcap_path,
        'method': method,
        'exclude_icmp': exclude_icmp,
        'exclude_control': exclude_control,
        'statistics': {
            cls: {
                'flows': len(keys),
                'packets': sum(len(flows[k]['packets']) for k in keys)
            }
            for cls, keys in flow_classes.items()
        },
        'mixed_flows_count': len(mixed_flows),
        'excluded_flows_count': len(excluded_flows),
        'mixed_flows_detail': [
            {
                'flow_key': {
                    'proto': fk[0],
                    'src_ip': fk[1],
                    'src_port': fk[2],
                    'dst_ip': fk[3],
                    'dst_port': fk[4],
                },
                'dscp_distribution': {f"0x{d:02x}": c for d, c in dscp_counts.items()},
                'assigned_class': assigned_class,
            }
            for fk, dscp_counts, assigned_class in mixed_flows
        ]
    }
    
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ {report_path}")
    
    return flow_classes, mixed_flows


def main():
    parser = argparse.ArgumentParser(description='按流（五元组）划分 PCAP')
    parser.add_argument('input', help='输入 PCAP 文件')
    parser.add_argument('-o', '--output', default='./flow_split', help='输出目录')
    parser.add_argument('-m', '--method', choices=['majority', 'first', 'strict'],
                        default='majority', help='分类方法: majority=多数DSCP, first=首包DSCP, strict=严格(混合标记)')
    parser.add_argument('--keep-icmp', action='store_true', help='保留 ICMP 流')
    parser.add_argument('--keep-control', action='store_true', help='保留控制流量 (CS6/CS7)')
    parser.add_argument('--analyze-only', action='store_true', help='只分析，不写入文件')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("按流划分 PCAP")
    print("=" * 70)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"分类方法: {args.method}")
    print(f"保留ICMP: {args.keep_icmp}")
    print(f"保留控制: {args.keep_control}")
    
    if args.analyze_only:
        flows = analyze_pcap(args.input)
        
        # 分析混合 DSCP 流
        mixed = []
        for flow_key, flow_data in flows.items():
            if len(flow_data['dscp_counts']) > 1:
                mixed.append((flow_key, flow_data['dscp_counts']))
        
        print(f"\n混合 DSCP 流: {len(mixed)}/{len(flows)}")
        for fk, dscp_counts in mixed[:20]:
            dscp_str = ", ".join(f"0x{d:02x}:{c}" for d, c in dscp_counts.most_common())
            print(f"  {fk[0]} {fk[1]}:{fk[2]} <-> {fk[3]}:{fk[4]}: {dscp_str}")
    else:
        split_by_flow(
            args.input, 
            args.output, 
            method=args.method,
            exclude_icmp=not args.keep_icmp,
            exclude_control=not args.keep_control
        )
    
    print("\n完成!")


if __name__ == '__main__':
    main()
