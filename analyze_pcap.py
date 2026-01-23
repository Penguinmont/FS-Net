#!/usr/bin/env python3
"""
分析 PCAP 文件的流和 DSCP 分布
在重划分数据前先了解数据情况

使用方法:
    python analyze_pcap.py origin.pcap
"""

import os
import sys
from collections import defaultdict, Counter
from scapy.all import PcapReader, IP, TCP, UDP
import numpy as np

# DSCP 到类别的映射
DSCP_TO_CLASS = {
    0x00: 'mid',    # CS0 / Best Effort
    0x20: 'low',    # CS1
    0x48: 'low',    # AF21
    0x90: 'high',   # EF 或 AF42
    0xb8: 'high',   # EF
}

DSCP_NAMES = {
    0x00: 'CS0 (Best Effort)',
    0x20: 'CS1',
    0x28: 'AF11',
    0x30: 'AF12',
    0x38: 'AF13',
    0x40: 'CS2',
    0x48: 'AF21',
    0x50: 'AF22',
    0x58: 'AF23',
    0x60: 'CS3',
    0x68: 'AF31',
    0x70: 'AF32',
    0x78: 'AF33',
    0x80: 'CS4',
    0x88: 'AF41',
    0x90: 'AF42',
    0x98: 'AF43',
    0xa0: 'CS5',
    0xb8: 'EF',
    0xc0: 'CS6',
    0xe0: 'CS7',
}


def get_dscp(tos_field):
    """从 TOS 字段提取 DSCP"""
    return tos_field & 0xfc


def get_flow_key(pkt):
    """获取双向流的五元组"""
    if not pkt.haslayer(IP):
        return None, None, None
    
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
        return None, None, None
    
    # 双向流 key
    if (ip.src, sport) < (ip.dst, dport):
        flow_key = (proto, ip.src, sport, ip.dst, dport)
    else:
        flow_key = (proto, ip.dst, dport, ip.src, sport)
    
    dscp = get_dscp(ip.tos)
    pkt_len = len(pkt)
    
    return flow_key, dscp, pkt_len


def analyze_pcap(pcap_path, min_packets=3):
    """分析 PCAP 文件"""
    print(f"\n读取: {pcap_path}")
    print(f"文件大小: {os.path.getsize(pcap_path) / 1024 / 1024:.1f} MB")
    
    flows = defaultdict(lambda: {
        'dscp_counts': Counter(),
        'packet_sizes': [],
        'first_dscp': None
    })
    
    # 全局统计
    total_packets = 0
    dscp_packet_counts = Counter()
    proto_counts = Counter()
    
    print("\n解析中...")
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            total_packets += 1
            if total_packets % 500000 == 0:
                print(f"  已处理 {total_packets:,} 包, 当前流数: {len(flows):,}")
            
            flow_key, dscp, pkt_len = get_flow_key(pkt)
            if flow_key is None:
                continue
            
            proto_counts[flow_key[0]] += 1
            dscp_packet_counts[dscp] += 1
            
            flows[flow_key]['dscp_counts'][dscp] += 1
            flows[flow_key]['packet_sizes'].append(pkt_len)
            if flows[flow_key]['first_dscp'] is None:
                flows[flow_key]['first_dscp'] = dscp
    
    print(f"\n{'='*70}")
    print("基本统计")
    print(f"{'='*70}")
    print(f"总包数: {total_packets:,}")
    print(f"总流数: {len(flows):,}")
    print(f"有效流 (>={min_packets}包): {sum(1 for f in flows.values() if len(f['packet_sizes']) >= min_packets):,}")
    
    # 协议分布
    print(f"\n协议分布:")
    for proto, count in proto_counts.most_common():
        print(f"  {proto}: {count:,} 包 ({count/total_packets*100:.1f}%)")
    
    # DSCP 分布（按包）
    print(f"\n{'='*70}")
    print("DSCP 分布 (按包)")
    print(f"{'='*70}")
    for dscp, count in sorted(dscp_packet_counts.items(), key=lambda x: -x[1]):
        dscp_name = DSCP_NAMES.get(dscp, f'Unknown(0x{dscp:02x})')
        mapped_class = DSCP_TO_CLASS.get(dscp, '?')
        print(f"  0x{dscp:02x} ({dscp_name:20s}) -> {mapped_class:6s}: {count:>10,} 包 ({count/total_packets*100:>5.1f}%)")
    
    # 按流分类
    print(f"\n{'='*70}")
    print("按流分类统计 (majority 方法)")
    print(f"{'='*70}")
    
    class_flows = defaultdict(list)
    mixed_flows = []
    
    for flow_key, flow_data in flows.items():
        if len(flow_data['packet_sizes']) < min_packets:
            continue
        
        # 检查混合 DSCP
        if len(flow_data['dscp_counts']) > 1:
            mixed_flows.append((flow_key, flow_data))
        
        # 使用 majority 方法分类
        majority_dscp = flow_data['dscp_counts'].most_common(1)[0][0]
        if majority_dscp in DSCP_TO_CLASS:
            flow_class = DSCP_TO_CLASS[majority_dscp]
        else:
            dscp_value = majority_dscp >> 2
            if dscp_value >= 40:
                flow_class = 'high'
            elif dscp_value >= 16:
                flow_class = 'low'
            else:
                flow_class = 'mid'
        
        class_flows[flow_class].append(flow_data)
    
    # 输出各类别统计
    print(f"\n{'类别':<10s} {'流数':>10s} {'包数':>12s} {'流占比':>10s} {'包占比':>10s}")
    print("-" * 55)
    
    total_valid_flows = sum(len(v) for v in class_flows.values())
    total_valid_packets = sum(sum(len(f['packet_sizes']) for f in v) for v in class_flows.values())
    
    for cls in ['mid', 'low', 'high']:
        flow_list = class_flows.get(cls, [])
        flow_count = len(flow_list)
        pkt_count = sum(len(f['packet_sizes']) for f in flow_list)
        
        flow_pct = flow_count / total_valid_flows * 100 if total_valid_flows else 0
        pkt_pct = pkt_count / total_valid_packets * 100 if total_valid_packets else 0
        
        print(f"{cls:<10s} {flow_count:>10,} {pkt_count:>12,} {flow_pct:>9.1f}% {pkt_pct:>9.1f}%")
    
    print("-" * 55)
    print(f"{'总计':<10s} {total_valid_flows:>10,} {total_valid_packets:>12,}")
    
    # 类别不平衡分析
    print(f"\n{'='*70}")
    print("数据平衡分析")
    print(f"{'='*70}")
    
    flow_counts = {cls: len(class_flows.get(cls, [])) for cls in ['mid', 'low', 'high']}
    max_count = max(flow_counts.values()) if flow_counts.values() else 1
    min_count = min(flow_counts.values()) if flow_counts.values() else 1
    
    print(f"\n各类别流数:")
    for cls, count in flow_counts.items():
        ratio = max_count / count if count > 0 else float('inf')
        bar = '█' * int(count / max_count * 30)
        print(f"  {cls:<6s}: {count:>6,} {bar}")
    
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n不平衡比: {imbalance_ratio:.1f}:1 (最大类/最小类)")
    
    if imbalance_ratio > 10:
        print("⚠️  数据严重不平衡，强烈建议使用 balance_data.py 进行过采样")
    elif imbalance_ratio > 3:
        print("⚠️  数据有一定不平衡，建议使用 balance_data.py 或类别权重")
    else:
        print("✓ 数据相对平衡")
    
    # 流长度分布
    print(f"\n{'='*70}")
    print("流长度分布 (各类别)")
    print(f"{'='*70}")
    
    for cls in ['mid', 'low', 'high']:
        flow_list = class_flows.get(cls, [])
        if not flow_list:
            continue
        
        lengths = [len(f['packet_sizes']) for f in flow_list]
        print(f"\n{cls}:")
        print(f"  流数: {len(lengths)}")
        print(f"  包数/流 - 最小: {min(lengths)}, 最大: {max(lengths)}, "
              f"平均: {np.mean(lengths):.1f}, 中位数: {np.median(lengths):.0f}")
        
        # 长度分布
        bins = [3, 10, 50, 100, 200, 500, float('inf')]
        bin_labels = ['3-9', '10-49', '50-99', '100-199', '200-499', '500+']
        
        for i, label in enumerate(bin_labels):
            count = sum(1 for l in lengths if bins[i] <= l < bins[i+1])
            pct = count / len(lengths) * 100
            if count > 0:
                print(f"    {label:>8s}: {count:>6,} ({pct:>5.1f}%)")
    
    # 混合 DSCP 流统计
    print(f"\n{'='*70}")
    print("混合 DSCP 流分析")
    print(f"{'='*70}")
    print(f"\n混合 DSCP 流数: {len(mixed_flows)} ({len(mixed_flows)/total_valid_flows*100:.1f}% of valid flows)")
    
    if mixed_flows:
        print(f"\n前 10 个混合流示例:")
        for i, (flow_key, flow_data) in enumerate(mixed_flows[:10]):
            dscp_str = ", ".join(
                f"0x{d:02x}:{c}" 
                for d, c in flow_data['dscp_counts'].most_common(3)
            )
            print(f"  {i+1}. {flow_key[0]} {flow_key[1]}:{flow_key[2]} <-> {flow_key[3]}:{flow_key[4]}")
            print(f"     DSCP: {dscp_str}, 总包数: {len(flow_data['packet_sizes'])}")
    
    # 输出建议
    print(f"\n{'='*70}")
    print("建议的下一步")
    print(f"{'='*70}")
    
    print(f"""
1. 运行数据重划分:
   python resplit_data.py --pcap {pcap_path} --output_dir ./

2. 预处理:
   python main_tf2.py --mode=prepro --class_num=3 --data_dir=./filter
""")
    
    if imbalance_ratio > 3:
        print(f"""3. 数据平衡 (推荐，因为不平衡比为 {imbalance_ratio:.1f}:1):
   python balance_data.py --input_dir ./filter --output_dir ./filter_balanced
   python main_tf2.py --mode=prepro --class_num=3 --data_dir=./filter_balanced
""")
    
    print(f"""4. 训练:
   python main_tf2.py --mode=train --class_num=3
""")

    return {
        'total_packets': total_packets,
        'total_flows': len(flows),
        'valid_flows': total_valid_flows,
        'class_distribution': flow_counts,
        'imbalance_ratio': imbalance_ratio,
        'mixed_flows': len(mixed_flows)
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_pcap.py <pcap_file> [min_packets]")
        print("示例: python analyze_pcap.py origin.pcap 3")
        sys.exit(1)
    
    pcap_path = sys.argv[1]
    min_packets = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    if not os.path.exists(pcap_path):
        print(f"错误: 文件不存在 - {pcap_path}")
        sys.exit(1)
    
    analyze_pcap(pcap_path, min_packets)


if __name__ == '__main__':
    main()
