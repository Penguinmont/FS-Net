#!/usr/bin/env python3
"""
完整数据重划分流程
===================
1. 备份旧的 filter/ 和 record/ 目录
2. 按流（五元组）重新划分原始 PCAP
3. 生成新的 .num 文件
4. 运行预处理生成 train/test JSON

使用方法:
    python resplit_data.py --pcap ./raw_capture.pcap --output_dir ./

    或者如果你已经有按 DSCP 分类的 PCAP:
    python resplit_data.py --pcap_dir ./dscp_split --output_dir ./
"""

import os
import sys
import shutil
import argparse
from datetime import datetime
from collections import defaultdict, Counter
from scapy.all import PcapReader, PcapWriter, IP, TCP, UDP
from tqdm import tqdm


# ========== DSCP 映射配置 ==========
DSCP_TO_CLASS = {
    0x00: 'mid',    # CS0 / Best Effort -> Chat
    0x20: 'low',    # CS1 -> Streaming
    0x48: 'low',    # AF21
    0x90: 'high',   # EF 或 AF42 -> VoIP
    0xb8: 'high',   # EF
}

CLASS_LABELS = {'mid': 0, 'low': 1, 'high': 2}


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


def backup_directories(base_dir, dirs_to_backup):
    """备份指定目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_base = os.path.join(base_dir, f'backup_{timestamp}')
    
    backed_up = []
    for dir_name in dirs_to_backup:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            backup_path = os.path.join(backup_base, dir_name)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copytree(dir_path, backup_path)
            backed_up.append(dir_name)
            print(f"  ✓ 备份: {dir_path} -> {backup_path}")
    
    if backed_up:
        print(f"\n备份目录: {backup_base}")
    else:
        print("  没有需要备份的目录")
    
    return backup_base if backed_up else None


def extract_flows_from_pcap(pcap_path, min_packets=3):
    """从 PCAP 提取所有流，记录每个包的信息"""
    print(f"\n读取: {pcap_path}")
    
    flows = defaultdict(lambda: {
        'dscp_counts': Counter(),
        'packets': [],  # [(pkt_len, timestamp), ...]
        'first_dscp': None
    })
    
    pkt_count = 0
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            pkt_count += 1
            if pkt_count % 500000 == 0:
                print(f"  已处理 {pkt_count:,} 包, 当前流数: {len(flows):,}")
            
            flow_key, dscp, pkt_len = get_flow_key(pkt)
            if flow_key is None:
                continue
            
            flows[flow_key]['dscp_counts'][dscp] += 1
            flows[flow_key]['packets'].append(pkt_len)
            if flows[flow_key]['first_dscp'] is None:
                flows[flow_key]['first_dscp'] = dscp
    
    print(f"  总包数: {pkt_count:,}")
    print(f"  总流数: {len(flows):,}")
    
    # 过滤短流
    valid_flows = {k: v for k, v in flows.items() if len(v['packets']) >= min_packets}
    print(f"  有效流 (>={min_packets}包): {len(valid_flows):,}")
    
    return valid_flows


def classify_flow_by_dscp(flow_data, method='majority'):
    """根据 DSCP 确定流的类别"""
    dscp_counts = flow_data['dscp_counts']
    
    if method == 'majority':
        dscp = dscp_counts.most_common(1)[0][0]
    else:  # first
        dscp = flow_data['first_dscp']
    
    if dscp in DSCP_TO_CLASS:
        return DSCP_TO_CLASS[dscp]
    else:
        # 未知 DSCP，根据值猜测
        dscp_value = dscp >> 2
        if dscp_value >= 40:
            return 'high'
        elif dscp_value >= 16:
            return 'low'
        else:
            return 'mid'


def encode_time_interval(interval_ms):
    """时间间隔编码（与原始 pcap_preprocess.py 一致）"""
    time_bins = [0, 1, 5, 10, 50, 100, 500, 1000]
    for i, threshold in enumerate(time_bins):
        if interval_ms < threshold:
            return i
    return len(time_bins)


def flow_to_num_record(packets, max_packet_length=5000):
    """
    将流转换为 .num 格式记录
    格式: 状态序列;包长度序列
    
    注意: 这里我们没有时间戳，所以状态序列全部设为0
    如果需要更准确的状态序列，需要保留原始包的时间戳
    """
    length_seq = []
    status_seq = []
    
    for pkt_len in packets:
        # 限制最大长度
        pkt_len = min(pkt_len, max_packet_length)
        length_seq.append(pkt_len)
        # 没有时间戳时，状态设为0
        status_seq.append(0)
    
    status_str = '\t'.join(map(str, status_seq))
    length_str = '\t'.join(map(str, length_seq))
    
    return f"{status_str};{length_str}"


def split_and_generate_num(pcap_path, output_dir, method='majority', min_packets=3, max_packet_length=5000):
    """
    从单个 PCAP 按流划分并直接生成 .num 文件
    """
    # 提取流
    flows = extract_flows_from_pcap(pcap_path, min_packets)
    
    # 按类别分组
    class_flows = defaultdict(list)
    mixed_count = 0
    
    for flow_key, flow_data in flows.items():
        # 检查是否混合 DSCP
        if len(flow_data['dscp_counts']) > 1:
            mixed_count += 1
        
        flow_class = classify_flow_by_dscp(flow_data, method)
        class_flows[flow_class].append(flow_data['packets'])
    
    print(f"\n流分类结果:")
    for cls, flow_list in sorted(class_flows.items()):
        total_pkts = sum(len(f) for f in flow_list)
        print(f"  {cls}: {len(flow_list)} 流, {total_pkts:,} 包")
    
    if mixed_count > 0:
        print(f"\n  注意: {mixed_count} 个流有混合 DSCP 值 (使用 {method} 方法分配)")
    
    # 生成 .num 文件
    filter_dir = os.path.join(output_dir, 'filter')
    os.makedirs(filter_dir, exist_ok=True)
    
    print(f"\n生成 .num 文件...")
    for cls, flow_list in class_flows.items():
        output_path = os.path.join(filter_dir, f'{cls}.num')
        
        with open(output_path, 'w') as f:
            for packets in flow_list:
                record = flow_to_num_record(packets, max_packet_length)
                f.write(record + '\n')
        
        print(f"  ✓ {output_path}: {len(flow_list)} 流")
    
    return class_flows


def process_existing_split(pcap_dir, output_dir, min_packets=3, max_packet_length=5000):
    """
    处理已经按 DSCP 划分好的 PCAP 文件
    
    预期文件:
      - cap_mid.pcap 或 flow_mid.pcap -> mid.num
      - cap_low.pcap 或 flow_low.pcap -> low.num  
      - cap_high.pcap 或 flow_high.pcap -> high.num
    """
    # 查找文件
    file_patterns = {
        'mid': ['cap_mid.pcap', 'flow_mid.pcap', 'mid.pcap'],
        'low': ['cap_low.pcap', 'flow_low.pcap', 'low.pcap'],
        'high': ['cap_high.pcap', 'flow_high.pcap', 'high.pcap'],
    }
    
    found_files = {}
    for cls, patterns in file_patterns.items():
        for pattern in patterns:
            fpath = os.path.join(pcap_dir, pattern)
            if os.path.exists(fpath):
                found_files[cls] = fpath
                break
    
    if not found_files:
        print(f"错误: 在 {pcap_dir} 中没有找到标签 PCAP 文件")
        print(f"预期文件名: {file_patterns}")
        return None
    
    print(f"\n找到文件:")
    for cls, fpath in found_files.items():
        print(f"  {cls}: {os.path.basename(fpath)}")
    
    filter_dir = os.path.join(output_dir, 'filter')
    os.makedirs(filter_dir, exist_ok=True)
    
    class_flows = {}
    
    for cls, pcap_path in found_files.items():
        print(f"\n处理 {cls}...")
        
        # 提取流
        flows = extract_flows_from_pcap(pcap_path, min_packets)
        
        # 直接使用文件标签，不再按 DSCP 分类
        flow_list = [fd['packets'] for fd in flows.values()]
        class_flows[cls] = flow_list
        
        # 写入 .num 文件
        output_path = os.path.join(filter_dir, f'{cls}.num')
        with open(output_path, 'w') as f:
            for packets in flow_list:
                record = flow_to_num_record(packets, max_packet_length)
                f.write(record + '\n')
        
        print(f"  ✓ {output_path}: {len(flow_list)} 流")
    
    return class_flows


def analyze_num_files(filter_dir):
    """分析生成的 .num 文件"""
    print("\n" + "=" * 60)
    print(".num 文件分析")
    print("=" * 60)
    
    for fname in sorted(os.listdir(filter_dir)):
        if not fname.endswith('.num'):
            continue
        
        fpath = os.path.join(filter_dir, fname)
        class_name = fname.replace('.num', '')
        
        flow_lengths = []
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) >= 2:
                    length_seq = parts[1].split('\t')
                    flow_lengths.append(len(length_seq))
        
        if flow_lengths:
            import numpy as np
            print(f"\n{class_name} (label={CLASS_LABELS.get(class_name, '?')}):")
            print(f"  流数量: {len(flow_lengths)}")
            print(f"  包数量 - 最小: {min(flow_lengths)}, 最大: {max(flow_lengths)}, "
                  f"平均: {np.mean(flow_lengths):.1f}, 中位数: {np.median(flow_lengths):.0f}")


def main():
    parser = argparse.ArgumentParser(
        description='完整数据重划分流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  # 从单个原始 PCAP 重新划分
  python resplit_data.py --pcap ./raw_capture.pcap --output_dir ./fs-net

  # 处理已有的按 DSCP 划分的 PCAP 文件
  python resplit_data.py --pcap_dir ./labeled_pcaps --output_dir ./fs-net

  # 不备份直接处理
  python resplit_data.py --pcap_dir ./labeled_pcaps --output_dir ./fs-net --no-backup

流程:
  1. 备份旧的 filter/ 和 record/ 目录
  2. 按流（五元组）提取并分类
  3. 生成新的 .num 文件
  4. 提示运行 preprocess 和 train
        """
    )
    
    parser.add_argument('--pcap', type=str, help='单个原始 PCAP 文件（会按 DSCP 自动划分）')
    parser.add_argument('--pcap_dir', type=str, help='包含已划分 PCAP 的目录')
    parser.add_argument('--output_dir', type=str, default='.', help='输出目录（fs-net 项目根目录）')
    parser.add_argument('--method', choices=['majority', 'first'], default='majority',
                        help='DSCP 分类方法: majority=多数, first=首包')
    parser.add_argument('--min_packets', type=int, default=3, help='流的最小包数')
    parser.add_argument('--max_packet_length', type=int, default=5000, help='包长度上限')
    parser.add_argument('--no-backup', action='store_true', help='不备份旧文件')
    
    args = parser.parse_args()
    
    if not args.pcap and not args.pcap_dir:
        parser.error("必须指定 --pcap 或 --pcap_dir")
    
    print("=" * 60)
    print("数据重划分流程")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    print(f"最小包数: {args.min_packets}")
    print(f"最大包长: {args.max_packet_length}")
    
    # 1. 备份
    if not args.no_backup:
        print("\n" + "-" * 60)
        print("[1] 备份旧数据")
        print("-" * 60)
        backup_directories(args.output_dir, ['filter', 'record', 'log'])
    else:
        print("\n跳过备份")
    
    # 2. 划分并生成 .num 文件
    print("\n" + "-" * 60)
    print("[2] 按流划分并生成 .num 文件")
    print("-" * 60)
    
    if args.pcap:
        # 从单个 PCAP 划分
        class_flows = split_and_generate_num(
            args.pcap, args.output_dir,
            method=args.method,
            min_packets=args.min_packets,
            max_packet_length=args.max_packet_length
        )
    else:
        # 处理已划分的 PCAP
        class_flows = process_existing_split(
            args.pcap_dir, args.output_dir,
            min_packets=args.min_packets,
            max_packet_length=args.max_packet_length
        )
    
    if class_flows is None:
        print("\n处理失败!")
        sys.exit(1)
    
    # 3. 分析结果
    filter_dir = os.path.join(args.output_dir, 'filter')
    analyze_num_files(filter_dir)
    
    # 4. 下一步指南
    print("\n" + "=" * 60)
    print("完成! 下一步操作:")
    print("=" * 60)
    
    print(f"""
1. 预处理数据:
   cd {args.output_dir}
   python main_tf2.py --mode=prepro --class_num=3 --data_dir=./filter

2. 训练模型:
   python main_tf2.py --mode=train --class_num=3

3. (可选) 如果数据不平衡，先运行平衡:
   python balance_data.py --input_dir ./filter --output_dir ./filter_balanced
   python main_tf2.py --mode=prepro --class_num=3 --data_dir=./filter_balanced
   python main_tf2.py --mode=train --class_num=3

4. 测试模型:
   python main_tf2.py --mode=test --class_num=3
""")


if __name__ == '__main__':
    main()
