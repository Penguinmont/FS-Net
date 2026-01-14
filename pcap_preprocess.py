#!/usr/bin/env python3
"""
PCAP预处理脚本 - 将pcap文件转换为FS-Net所需的.num格式

数据格式说明：
- 输入: pcap文件（每个文件包含同一类别的流量）
- 输出: .num文件，每行格式为: 状态序列;包长度序列
  - 状态序列: 包间时间间隔的编码值（用\t分隔）
  - 包长度序列: 每个包的长度（用\t分隔）

使用方法:
    python pcap_preprocess.py --input_dir ./pcap_data --output_dir ./filter

目录结构示例:
    pcap_data/
    ├── mid.pcap    -> 输出 filter/mid.num (label=0)
    ├── low.pcap    -> 输出 filter/low.num (label=1)
    └── high.pcap   -> 输出 filter/high.num (label=2)
"""

import os
import argparse
from collections import defaultdict
from scapy.all import rdpcap, IP, TCP, UDP
from tqdm import tqdm
import numpy as np


def get_flow_key(packet):
    """
    从数据包提取5元组作为流的标识
    5元组: (源IP, 目的IP, 源端口, 目的端口, 协议)
    """
    if IP not in packet:
        return None
    
    ip_layer = packet[IP]
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    protocol = ip_layer.proto
    
    # 提取端口信息
    if TCP in packet:
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    elif UDP in packet:
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
    else:
        src_port = 0
        dst_port = 0
    
    # 双向流合并：统一排序使得A->B和B->A属于同一流
    if (src_ip, src_port) > (dst_ip, dst_port):
        return (dst_ip, src_ip, dst_port, src_port, protocol)
    else:
        return (src_ip, dst_ip, src_port, dst_port, protocol)


def encode_time_interval(interval_ms, time_bins=None):
    """
    将时间间隔（毫秒）编码为状态值
    
    默认分箱策略（可根据实际数据调整）:
    - 0: [0, 1) ms
    - 1: [1, 5) ms
    - 2: [5, 10) ms
    - 3: [10, 50) ms
    - 4: [50, 100) ms
    - 5: [100, 500) ms
    - 6: [500, 1000) ms
    - 7: >= 1000 ms
    """
    if time_bins is None:
        time_bins = [0, 1, 5, 10, 50, 100, 500, 1000]
    
    for i, threshold in enumerate(time_bins):
        if interval_ms < threshold:
            return i
    return len(time_bins)


def extract_flows_from_pcap(pcap_file, min_packets=2):
    """
    从pcap文件提取所有流
    
    Args:
        pcap_file: pcap文件路径
        min_packets: 流的最小包数量（过滤太短的流）
    
    Returns:
        flows: dict, key为流标识，value为包列表
    """
    print(f"正在读取: {pcap_file}")
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"读取pcap文件失败: {e}")
        return {}
    
    # 按流分组
    flows = defaultdict(list)
    for pkt in tqdm(packets, desc="解析数据包", ascii=True):
        flow_key = get_flow_key(pkt)
        if flow_key is not None:
            flows[flow_key].append(pkt)
    
    # 过滤太短的流
    flows = {k: v for k, v in flows.items() if len(v) >= min_packets}
    print(f"提取到 {len(flows)} 个有效流")
    
    return flows


def process_flow(packets, max_packet_length=5000):
    """
    处理单个流，提取状态序列和包长度序列
    
    Args:
        packets: 数据包列表
        max_packet_length: 包长度上限
    
    Returns:
        status_seq: 状态序列（时间间隔编码）
        length_seq: 包长度序列
    """
    # 按时间排序
    packets = sorted(packets, key=lambda p: float(p.time))
    
    length_seq = []
    status_seq = []
    
    prev_time = None
    for pkt in packets:
        # 提取包长度
        pkt_len = len(pkt)
        pkt_len = min(pkt_len, max_packet_length)  # 限制最大长度
        length_seq.append(pkt_len)
        
        # 计算时间间隔
        curr_time = float(pkt.time)
        if prev_time is not None:
            interval_ms = (curr_time - prev_time) * 1000  # 转换为毫秒
            status = encode_time_interval(interval_ms)
            status_seq.append(status)
        else:
            status_seq.append(0)  # 第一个包的状态设为0
        
        prev_time = curr_time
    
    return status_seq, length_seq


def convert_pcap_to_num(pcap_file, output_file, min_packets=2, max_packet_length=5000):
    """
    将pcap文件转换为.num格式
    
    Args:
        pcap_file: 输入pcap文件路径
        output_file: 输出.num文件路径
        min_packets: 流的最小包数
        max_packet_length: 包长度上限
    """
    flows = extract_flows_from_pcap(pcap_file, min_packets)
    
    if not flows:
        print(f"警告: {pcap_file} 中没有有效流")
        return 0
    
    with open(output_file, 'w') as fp:
        for flow_key, packets in tqdm(flows.items(), desc="处理流", ascii=True):
            status_seq, length_seq = process_flow(packets, max_packet_length)
            
            # 格式: 状态序列;包长度序列
            status_str = '\t'.join(map(str, status_seq))
            length_str = '\t'.join(map(str, length_seq))
            fp.write(f"{status_str};{length_str}\n")
    
    print(f"已保存到: {output_file} ({len(flows)} 个流)")
    return len(flows)


def preprocess_all(input_dir, output_dir, label_mapping=None, min_packets=2, max_packet_length=5000):
    """
    批量处理所有pcap文件
    
    Args:
        input_dir: pcap文件目录
        output_dir: 输出目录
        label_mapping: 文件名到标签的映射，如 {'mid': 0, 'low': 1, 'high': 2}
        min_packets: 流的最小包数
        max_packet_length: 包长度上限
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认标签映射
    if label_mapping is None:
        label_mapping = {
            'mid': 0,
            'low': 1, 
            'high': 2
        }
    
    # 查找所有pcap文件
    pcap_files = []
    for fname in os.listdir(input_dir):
        if fname.endswith(('.pcap', '.pcapng', '.cap')):
            pcap_files.append(fname)
    
    if not pcap_files:
        print(f"错误: 在 {input_dir} 中没有找到pcap文件")
        return
    
    print(f"找到 {len(pcap_files)} 个pcap文件")
    print(f"标签映射: {label_mapping}")
    print("-" * 50)
    
    stats = {}
    for pcap_fname in pcap_files:
        pcap_path = os.path.join(input_dir, pcap_fname)
        
        # 获取输出文件名（去掉扩展名，加上.num）
        base_name = os.path.splitext(pcap_fname)[0]
        output_path = os.path.join(output_dir, f"{base_name}.num")
        
        print(f"\n处理: {pcap_fname}")
        flow_count = convert_pcap_to_num(
            pcap_path, 
            output_path,
            min_packets=min_packets,
            max_packet_length=max_packet_length
        )
        stats[base_name] = flow_count
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("预处理完成! 统计信息:")
    print("=" * 50)
    for name, count in stats.items():
        label = label_mapping.get(name, '未知')
        print(f"  {name}.num: {count} 个流 (label={label})")
    print(f"\n输出目录: {output_dir}")


def analyze_pcap(pcap_file):
    """
    分析pcap文件，打印统计信息
    """
    flows = extract_flows_from_pcap(pcap_file, min_packets=1)
    
    if not flows:
        print("没有找到有效流")
        return
    
    # 统计信息
    flow_lengths = [len(pkts) for pkts in flows.values()]
    
    print("\n数据分析:")
    print(f"  总流数: {len(flows)}")
    print(f"  流长度 - 最小: {min(flow_lengths)}, 最大: {max(flow_lengths)}, 平均: {np.mean(flow_lengths):.1f}")
    print(f"  流长度分布:")
    print(f"    < 5 包: {sum(1 for l in flow_lengths if l < 5)}")
    print(f"    5-20 包: {sum(1 for l in flow_lengths if 5 <= l < 20)}")
    print(f"    20-100 包: {sum(1 for l in flow_lengths if 20 <= l < 100)}")
    print(f"    >= 100 包: {sum(1 for l in flow_lengths if l >= 100)}")


def main():
    parser = argparse.ArgumentParser(
        description='将pcap文件转换为FS-Net所需的.num格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 批量处理所有pcap文件
  python pcap_preprocess.py --input_dir ./pcap_data --output_dir ./filter

  # 处理单个文件
  python pcap_preprocess.py --single ./data/mid.pcap --output ./filter/mid.num

  # 分析pcap文件
  python pcap_preprocess.py --analyze ./data/mid.pcap

目录结构:
  pcap_data/
  ├── mid.pcap    -> filter/mid.num (label=0)
  ├── low.pcap    -> filter/low.num (label=1)  
  └── high.pcap   -> filter/high.num (label=2)
        """
    )
    
    parser.add_argument('--input_dir', type=str, default='./pcap_data',
                        help='pcap文件所在目录')
    parser.add_argument('--output_dir', type=str, default='./filter',
                        help='输出.num文件的目录')
    parser.add_argument('--single', type=str, default=None,
                        help='处理单个pcap文件')
    parser.add_argument('--output', type=str, default=None,
                        help='单个文件的输出路径')
    parser.add_argument('--analyze', type=str, default=None,
                        help='分析pcap文件')
    parser.add_argument('--min_packets', type=int, default=2,
                        help='流的最小包数量 (默认: 2)')
    parser.add_argument('--max_packet_length', type=int, default=5000,
                        help='包长度上限 (默认: 5000)')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_pcap(args.analyze)
    elif args.single:
        output = args.output or args.single.replace('.pcap', '.num')
        convert_pcap_to_num(
            args.single, 
            output,
            min_packets=args.min_packets,
            max_packet_length=args.max_packet_length
        )
    else:
        preprocess_all(
            args.input_dir,
            args.output_dir,
            min_packets=args.min_packets,
            max_packet_length=args.max_packet_length
        )


if __name__ == '__main__':
    main()