#!/usr/bin/env python3
"""
从预处理后的 PCAP 文件生成 FS-Net 需要的 .num 文件

.num 文件格式: status_sequence;length_sequence
  - status_sequence: 包状态序列 (方向+TCP标志)
  - length_sequence: 包长度序列

用法:
    python pcap_to_num.py ./preprocessed -o ./filter
"""

import os
import sys
import argparse
from collections import defaultdict
from scapy.all import PcapReader, IP, IPv6, TCP, UDP


def get_flow_key(pkt):
    """提取流的五元组"""
    if pkt.haslayer(IP):
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
    elif pkt.haslayer(IPv6):
        src_ip = pkt[IPv6].src
        dst_ip = pkt[IPv6].dst
    else:
        return None, None
    
    if pkt.haslayer(TCP):
        proto = 'TCP'
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
    elif pkt.haslayer(UDP):
        proto = 'UDP'
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    else:
        return None, None
    
    # 规范化: 返回 (规范化key, 方向)
    if (src_ip, sport) < (dst_ip, dport):
        return (proto, src_ip, sport, dst_ip, dport), 0  # 正向
    else:
        return (proto, dst_ip, dport, src_ip, sport), 1  # 反向


def get_packet_features(pkt, direction):
    """
    提取包特征
    
    status: 方向(1bit) + TCP标志(如果是TCP)
    length: IP 包长度
    """
    # 获取包长度
    if pkt.haslayer(IP):
        length = pkt[IP].len
    elif pkt.haslayer(IPv6):
        length = pkt[IPv6].plen + 40  # IPv6 header = 40 bytes
    else:
        length = len(pkt)
    
    # 计算 status
    # 基础: 方向 (0=正向, 1=反向)
    status = direction
    
    # 如果是 TCP，加入标志位信息
    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        flags = int(tcp.flags)  # 转换为整数
        # 编码: direction * 128 + flags
        status = direction * 128 + (flags & 0x3F)  # 取低6位标志
    else:
        # UDP: 只用方向
        status = direction * 128
    
    return status, length


def pcap_to_flows(pcap_path):
    """从 PCAP 文件提取流"""
    flows = defaultdict(lambda: {'status': [], 'length': []})
    
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            flow_key, direction = get_flow_key(pkt)
            if flow_key is None:
                continue
            
            status, length = get_packet_features(pkt, direction)
            flows[flow_key]['status'].append(status)
            flows[flow_key]['length'].append(length)
    
    return flows


def write_num_file(flows, output_path):
    """写入 .num 文件"""
    with open(output_path, 'w') as f:
        for flow_key, data in flows.items():
            # 用 tab 分隔数字，用分号分隔 status 和 length
            status_seq = '\t'.join(map(str, data['status']))
            length_seq = '\t'.join(map(str, data['length']))
            f.write(f"{status_seq};{length_seq}\n")
    
    return len(flows)


def main():
    parser = argparse.ArgumentParser(description='从 PCAP 生成 .num 文件')
    parser.add_argument('input_dir', help='预处理后的目录 (包含 flow_*.pcap)')
    parser.add_argument('-o', '--output', default='./filter', help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--test_only', action='store_true', help='仅生成测试集（用于评估新数据）')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # test_only 模式
    if args.test_only:
        args.train_ratio = 0
    
    print(f"{'='*60}")
    print(f"PCAP → .num 转换")
    print(f"{'='*60}")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output}")
    if args.test_only:
        print(f"模式: 仅测试集")
    else:
        print(f"训练集比例: {args.train_ratio}")
    
    # 处理每个类别
    class_map = {'high': 2, 'mid': 0, 'low': 1}  # FS-Net 类别编号
    
    all_train = {0: [], 1: [], 2: []}
    all_test = {0: [], 1: [], 2: []}
    
    for class_name, class_id in class_map.items():
        pcap_path = os.path.join(args.input_dir, f'flow_{class_name}.pcap')
        
        if not os.path.exists(pcap_path):
            print(f"\n跳过 {class_name}: 文件不存在")
            continue
        
        print(f"\n处理 {class_name} ({pcap_path})...")
        
        # 提取流
        flows = pcap_to_flows(pcap_path)
        flow_list = list(flows.values())
        
        print(f"  提取到 {len(flow_list)} 个流")
        
        # 划分训练/测试集
        import random
        random.seed(42)
        random.shuffle(flow_list)
        
        split_idx = int(len(flow_list) * args.train_ratio)
        train_flows = flow_list[:split_idx]
        test_flows = flow_list[split_idx:]
        
        print(f"  训练集: {len(train_flows)}, 测试集: {len(test_flows)}")
        
        all_train[class_id].extend(train_flows)
        all_test[class_id].extend(test_flows)
    
    # 写入 .num 文件
    print(f"\n写入文件...")
    
    for class_id, class_name in [(0, 'mid'), (1, 'low'), (2, 'high')]:
        if args.test_only:
            # test_only 模式：所有数据写入不带 _test 后缀的文件
            output_path = os.path.join(args.output, f'{class_name}.num')
            all_flows = all_test[class_id]  # train_ratio=0 时数据都在 test
            with open(output_path, 'w') as f:
                for flow in all_flows:
                    status_seq = '\t'.join(map(str, flow['status']))
                    length_seq = '\t'.join(map(str, flow['length']))
                    f.write(f"{status_seq};{length_seq}\n")
            print(f"  ✓ {output_path}: {len(all_flows)} 流")
        else:
            # 正常模式
            # 训练集
            train_path = os.path.join(args.output, f'{class_name}.num')
            with open(train_path, 'w') as f:
                for flow in all_train[class_id]:
                    status_seq = '\t'.join(map(str, flow['status']))
                    length_seq = '\t'.join(map(str, flow['length']))
                    f.write(f"{status_seq};{length_seq}\n")
            print(f"  ✓ {train_path}: {len(all_train[class_id])} 流")
            
            # 测试集
            test_path = os.path.join(args.output, f'{class_name}_test.num')
            with open(test_path, 'w') as f:
                for flow in all_test[class_id]:
                    status_seq = '\t'.join(map(str, flow['status']))
                    length_seq = '\t'.join(map(str, flow['length']))
                    f.write(f"{status_seq};{length_seq}\n")
            print(f"  ✓ {test_path}: {len(all_test[class_id])} 流")
    
    # 统计
    print(f"\n{'='*60}")
    print(f"统计")
    print(f"{'='*60}")
    
    if args.test_only:
        total = sum(len(all_test[c]) for c in [0, 1, 2])
        print(f"\n{'类别':<8s} {'样本数':>8s} {'比例':>8s}")
        print(f"{'-'*30}")
        for class_id, class_name in [(0, 'mid'), (1, 'low'), (2, 'high')]:
            n = len(all_test[class_id])
            ratio = n / total * 100 if total > 0 else 0
            print(f"{class_name:<8s} {n:>8d} {ratio:>7.1f}%")
        print(f"{'-'*30}")
        print(f"{'总计':<8s} {total:>8d}")
    else:
        total_train = sum(len(v) for v in all_train.values())
        total_test = sum(len(v) for v in all_test.values())
        
        print(f"\n{'类别':<8s} {'训练':>8s} {'测试':>8s} {'总计':>8s} {'比例':>8s}")
        print(f"{'-'*40}")
        
        for class_id, class_name in [(0, 'mid'), (1, 'low'), (2, 'high')]:
            train_n = len(all_train[class_id])
            test_n = len(all_test[class_id])
            total = train_n + test_n
            ratio = total / (total_train + total_test) * 100 if (total_train + total_test) > 0 else 0
            print(f"{class_name:<8s} {train_n:>8d} {test_n:>8d} {total:>8d} {ratio:>7.1f}%")
        
        print(f"{'-'*40}")
        print(f"{'总计':<8s} {total_train:>8d} {total_test:>8d} {total_train+total_test:>8d}")
        
        # 计算推荐的类别权重
        if total_train > 0:
            print(f"\n推荐的类别权重 (基于训练集):")
            max_count = max(len(v) for v in all_train.values()) if any(all_train.values()) else 1
            weights = []
            for class_id in [0, 1, 2]:
                count = len(all_train[class_id])
                if count > 0:
                    weight = max_count / count
                else:
                    weight = 1.0
                weights.append(weight)
            
            # 归一化
            min_weight = min(w for w in weights if w > 0) if any(w > 0 for w in weights) else 1
            weights = [w / min_weight for w in weights]
            
            print(f"  --class_weights=\"{weights[0]:.1f},{weights[1]:.1f},{weights[2]:.1f}\"")
            print(f"  (mid, low, high)")
    
    print(f"\n完成!")


if __name__ == '__main__':
    main()