#!/usr/bin/env python3
"""
PCAP 数据预处理 - 智能流分类和截取

分类策略：
1. 纯净流：所有包DSCP一样 → 按DSCP严格分类
2. 混合短流（<=max_packets）：70%-95%区间majority，且last_quarter一致时才分类
3. 混合长流（2类DSCP，只有low或只有high）：滑动窗口找变化点，以变化点为中心截取
4. 混合长流（3类DSCP，同时有low和high）：滑动窗口找变化点（忽略mid），以变化点为中心截取

输出结构：
    output_dir/
    ├── flow_high.pcap      # 所有 high 流（合并）
    ├── flow_mid.pcap       # 所有 mid 流（合并）
    ├── flow_low.pcap       # 所有 low 流（合并）
    ├── pure/               # 纯净流（用于训练1）
    │   ├── flow_high.pcap
    │   ├── flow_mid.pcap
    │   └── flow_low.pcap
    ├── mixed/              # 混合流（用于训练2）
    │   ├── flow_high.pcap
    │   ├── flow_mid.pcap
    │   └── flow_low.pcap
    └── preprocess_report.json

用法:
    python preprocess_pcap_v3.py capture.pcap -o ./output
    python preprocess_pcap_v3.py capture.pcap -o ./output --max_packets 200 --min_packets 3
    python preprocess_pcap_v3.py capture.pcap -o ./output --no_separate  # 不分离纯净/混合
"""

import os
import sys
import json
import argparse
from collections import defaultdict, Counter
from scapy.all import PcapReader, PcapWriter, IP, IPv6, TCP, UDP

# DSCP 到类别的映射
DSCP_TO_CLASS = {
    0x00: 'mid',   # CS0 / Best Effort
    0x20: 'low',   # CS1
    0x48: 'low',   # AF21
    0x90: 'high',  # AF42
    0xb8: 'high',  # EF
}

def get_dscp(pkt):
    """从包中提取 DSCP 值"""
    if pkt.haslayer(IP):
        return pkt[IP].tos & 0xfc
    elif pkt.haslayer(IPv6):
        return (pkt[IPv6].tc >> 2) << 2
    return None


def get_flow_key(pkt):
    """提取流的五元组"""
    if pkt.haslayer(IP):
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
    elif pkt.haslayer(IPv6):
        src_ip = pkt[IPv6].src
        dst_ip = pkt[IPv6].dst
    else:
        return None
    
    if pkt.haslayer(TCP):
        proto = 'TCP'
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
    elif pkt.haslayer(UDP):
        proto = 'UDP'
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    else:
        return None
    
    # 规范化方向
    if (src_ip, sport) < (dst_ip, dport):
        return (proto, src_ip, sport, dst_ip, dport)
    else:
        return (proto, dst_ip, dport, src_ip, sport)


def dscp_to_class(dscp):
    """DSCP 值转类别"""
    return DSCP_TO_CLASS.get(dscp, 'unknown')


def get_class_set(dscp_counts):
    """获取流中包含的类别集合（忽略 unknown）"""
    classes = set()
    for dscp in dscp_counts.keys():
        cls = dscp_to_class(dscp)
        if cls != 'unknown':
            classes.add(cls)
    return classes


def find_changepoint_sliding_window(dscp_list, window_size=50, ignore_mid=False, target_transition=None, target_class=None):
    """
    使用滑动窗口找到 DSCP 变化点
    
    参数:
        dscp_list: DSCP 值列表
        window_size: 滑动窗口大小 (默认50)
        ignore_mid: 是否忽略 mid(0x00)，只看 low/high 变化
        target_transition: 目标转变类型，如 ('high', 'low') 表示找 high↔low 变化
        target_class: 目标类别，找涉及该类别的变化 (如 'low' 找 mid↔low)
    
    返回:
        changepoint: 最后一个符合条件的变化点位置（索引）
        from_class: 变化前的类别
        to_class: 变化后的类别
    """
    n = len(dscp_list)
    if n < window_size * 2:
        return None, None, None
    
    def get_window_majority(start, end, ignore_mid=False):
        """计算窗口内的 majority 类别"""
        window = dscp_list[start:end]
        counts = Counter()
        for dscp in window:
            cls = dscp_to_class(dscp)
            if ignore_mid and cls == 'mid':
                continue
            if cls != 'unknown':
                counts[cls] += 1
        
        if not counts:
            return None
        return counts.most_common(1)[0][0]
    
    # 滑动窗口，收集所有变化点
    all_changepoints = []
    prev_class = None
    for i in range(n - window_size + 1):
        current_class = get_window_majority(i, i + window_size, ignore_mid)
        
        if current_class is None:
            continue
        
        if prev_class is not None and current_class != prev_class:
            all_changepoints.append((i, prev_class, current_class))
        
        prev_class = current_class
    
    if not all_changepoints:
        return None, None, None
    
    # 如果指定了目标类别，找涉及该类别的变化
    if target_class:
        # 找涉及 target_class 的变化 (from 或 to 包含 target_class)
        matching = [(i, f, t) for i, f, t in all_changepoints 
                    if f == target_class or t == target_class]
        if matching:
            last_cp = matching[-1]
            return last_cp[0], last_cp[1], last_cp[2]
    
    # 如果指定了目标转变类型，找 A↔B 的变化
    if target_transition:
        from_target, to_target = target_transition
        # 找 from→to 或 to→from 的变化
        matching = [(i, f, t) for i, f, t in all_changepoints 
                    if (f == from_target and t == to_target) or (f == to_target and t == from_target)]
        if matching:
            last_cp = matching[-1]
            return last_cp[0], last_cp[1], last_cp[2]
    
    # 否则返回最后一个变化点
    last_cp = all_changepoints[-1]
    return last_cp[0], last_cp[1], last_cp[2]


def classify_and_extract_flow(dscp_list, packets, max_packets=200):
    """
    根据4种策略分类和截取流
    
    返回:
        class_label: 类别 ('high', 'mid', 'low', 'skip')
        extracted_packets: 截取后的包列表
        extracted_dscp: 截取后的 DSCP 列表
        method: 使用的方法
        info: 附加信息
    """
    n = len(dscp_list)
    dscp_counts = Counter(dscp_list)
    class_set = get_class_set(dscp_counts)
    
    info = {
        'original_length': n,
        'dscp_distribution': dict(dscp_counts),
        'class_set': list(class_set)
    }
    
    # ========== 情况1：纯净流 ==========
    if len(dscp_counts) == 1:
        dscp = list(dscp_counts.keys())[0]
        cls = dscp_to_class(dscp)
        
        if cls == 'unknown':
            return 'skip', [], [], 'pure_unknown', info
        
        # 截取（如果超过 max_packets，取中间部分）
        if n <= max_packets:
            extracted = list(range(n))
        else:
            mid = n // 2
            half = max_packets // 2
            start = mid - half
            end = start + max_packets
            extracted = list(range(start, end))
        
        info['method_detail'] = f'pure_{cls}'
        return cls, [packets[i] for i in extracted], [dscp_list[i] for i in extracted], 'pure', info
    
    # ========== 混合流 ==========
    
    # ========== 情况2：混合短流 (<=max_packets) ==========
    if n <= max_packets:
        # 计算 tail_segment (70%-95%)
        start_70 = int(n * 0.70)
        end_95 = int(n * 0.95)
        if end_95 <= start_70:
            end_95 = n
        
        segment_70_95 = dscp_list[start_70:end_95]
        if segment_70_95:
            counts_70_95 = Counter(segment_70_95)
            tail_segment_dscp = counts_70_95.most_common(1)[0][0]
            tail_segment_class = dscp_to_class(tail_segment_dscp)
        else:
            tail_segment_class = None
        
        # 计算 last_quarter (后25%)
        start_75 = int(n * 0.75)
        segment_75_100 = dscp_list[start_75:]
        if segment_75_100:
            counts_75_100 = Counter(segment_75_100)
            last_quarter_dscp = counts_75_100.most_common(1)[0][0]
            last_quarter_class = dscp_to_class(last_quarter_dscp)
        else:
            last_quarter_class = None
        
        # 检查一致性
        if tail_segment_class and last_quarter_class and tail_segment_class == last_quarter_class:
            cls = tail_segment_class
            if cls == 'unknown':
                return 'skip', [], [], 'mixed_short_unknown', info
            
            info['tail_segment_class'] = tail_segment_class
            info['last_quarter_class'] = last_quarter_class
            return cls, packets, dscp_list, 'mixed_short', info
        else:
            # 不一致，跳过
            info['tail_segment_class'] = tail_segment_class
            info['last_quarter_class'] = last_quarter_class
            info['skip_reason'] = 'tail_segment and last_quarter disagree'
            return 'skip', [], [], 'mixed_short_disagree', info
    
    # ========== 混合长流 (>max_packets) ==========
    
    has_low = 'low' in class_set
    has_high = 'high' in class_set
    has_mid = 'mid' in class_set
    
    # 判断是2类还是3类DSCP
    non_mid_classes = class_set - {'mid'}
    
    # ========== 情况3：混合长流，2类DSCP（只有low或只有high） ==========
    if len(non_mid_classes) == 1:
        # 只有 low 或只有 high（可能有 mid）
        target_class = list(non_mid_classes)[0]  # 'low' 或 'high'
        
        # 使用滑动窗口找变化点：找涉及 target_class 的变化
        # 可能是 mid → target 或 target → mid
        changepoint, from_cls, to_cls = find_changepoint_sliding_window(
            dscp_list, window_size=50, ignore_mid=False,
            target_class=target_class  # 找涉及 low 或 high 的变化
        )
        
        if changepoint is not None:
            # 以变化点为中心截取
            half = max_packets // 2
            start = max(0, changepoint - half)
            end = min(n, start + max_packets)
            if end - start < max_packets and start > 0:
                start = max(0, end - max_packets)
            
            extracted_dscp = dscp_list[start:end]
            extracted_packets = packets[start:end]
            
            # 用截取后的 last_quarter (后25%) 判断类别
            ext_len = len(extracted_dscp)
            last_quarter_start = int(ext_len * 0.75)
            last_quarter_dscp = extracted_dscp[last_quarter_start:]
            
            if last_quarter_dscp:
                lq_counts = Counter(last_quarter_dscp)
                final_dscp = lq_counts.most_common(1)[0][0]
                cls = dscp_to_class(final_dscp)
            else:
                ext_counts = Counter(extracted_dscp)
                final_dscp = ext_counts.most_common(1)[0][0]
                cls = dscp_to_class(final_dscp)
            
            if cls == 'unknown':
                return 'skip', [], [], 'mixed_long_2class_unknown', info
            
            info['changepoint'] = changepoint
            info['from_class'] = from_cls
            info['to_class'] = to_cls
            info['extract_range'] = [start, end]
            info['classification_method'] = 'last_quarter'
            return cls, extracted_packets, extracted_dscp, 'mixed_long_2class', info
        else:
            # 没找到变化点，使用后段
            start = int(n * 0.70)
            end = int(n * 0.95)
            if end - start > max_packets:
                mid_point = (start + end) // 2
                half = max_packets // 2
                start = mid_point - half
                end = start + max_packets
            
            extracted_dscp = dscp_list[start:end]
            extracted_packets = packets[start:end]
            
            # 用截取后的 last_quarter 判断类别
            ext_len = len(extracted_dscp)
            last_quarter_start = int(ext_len * 0.75)
            last_quarter_dscp = extracted_dscp[last_quarter_start:]
            
            if last_quarter_dscp:
                lq_counts = Counter(last_quarter_dscp)
                final_dscp = lq_counts.most_common(1)[0][0]
                cls = dscp_to_class(final_dscp)
            else:
                ext_counts = Counter(extracted_dscp)
                final_dscp = ext_counts.most_common(1)[0][0]
                cls = dscp_to_class(final_dscp)
            
            if cls == 'unknown':
                return 'skip', [], [], 'mixed_long_2class_no_cp_unknown', info
            
            info['extract_range'] = [start, end]
            info['no_changepoint'] = True
            info['classification_method'] = 'last_quarter'
            return cls, extracted_packets, extracted_dscp, 'mixed_long_2class_no_cp', info
    
    # ========== 情况4：混合长流，3类DSCP（同时有low和high） ==========
    elif has_low and has_high:
        # 同时有 low 和 high，使用滑动窗口找 high ↔ low 变化点
        changepoint, from_cls, to_cls = find_changepoint_sliding_window(
            dscp_list, window_size=50, ignore_mid=True,
            target_transition=('high', 'low')  # 找 high → low 或 low → high 变化
        )
        
        if changepoint is not None:
            # 以变化点为中心截取
            half = max_packets // 2
            start = max(0, changepoint - half)
            end = min(n, start + max_packets)
            if end - start < max_packets and start > 0:
                start = max(0, end - max_packets)
            
            extracted_dscp = dscp_list[start:end]
            extracted_packets = packets[start:end]
            
            # 用截取后的 last_quarter 判断类别（忽略 mid）
            ext_len = len(extracted_dscp)
            last_quarter_start = int(ext_len * 0.75)
            last_quarter_dscp = extracted_dscp[last_quarter_start:]
            
            # 忽略 mid，只看 low/high
            non_mid_lq = [d for d in last_quarter_dscp if dscp_to_class(d) != 'mid']
            if non_mid_lq:
                lq_counts = Counter(non_mid_lq)
                final_dscp = lq_counts.most_common(1)[0][0]
                cls = dscp_to_class(final_dscp)
            else:
                # 全是 mid，用全部的 majority
                ext_counts = Counter(extracted_dscp)
                final_dscp = ext_counts.most_common(1)[0][0]
                cls = dscp_to_class(final_dscp)
            
            if cls == 'unknown':
                return 'skip', [], [], 'mixed_long_3class_unknown', info
            
            info['changepoint'] = changepoint
            info['from_class'] = from_cls
            info['to_class'] = to_cls
            info['extract_range'] = [start, end]
            info['classification_method'] = 'last_quarter_ignore_mid'
            return cls, extracted_packets, extracted_dscp, 'mixed_long_3class', info
        else:
            # 没找到 low/high 变化点，使用后段并忽略 mid 判断
            start = int(n * 0.70)
            end = int(n * 0.95)
            if end - start > max_packets:
                mid_point = (start + end) // 2
                half = max_packets // 2
                start = mid_point - half
                end = start + max_packets
            
            extracted_dscp = dscp_list[start:end]
            extracted_packets = packets[start:end]
            
            # 用截取后的 last_quarter 判断类别（忽略 mid）
            ext_len = len(extracted_dscp)
            last_quarter_start = int(ext_len * 0.75)
            last_quarter_dscp = extracted_dscp[last_quarter_start:]
            
            non_mid_lq = [d for d in last_quarter_dscp if dscp_to_class(d) != 'mid']
            if non_mid_lq:
                lq_counts = Counter(non_mid_lq)
                final_dscp = lq_counts.most_common(1)[0][0]
                cls = dscp_to_class(final_dscp)
            else:
                ext_counts = Counter(extracted_dscp)
                final_dscp = ext_counts.most_common(1)[0][0]
                cls = dscp_to_class(final_dscp)
            
            if cls == 'unknown':
                return 'skip', [], [], 'mixed_long_3class_no_cp_unknown', info
            
            info['extract_range'] = [start, end]
            info['no_changepoint'] = True
            info['classification_method'] = 'last_quarter_ignore_mid'
            return cls, extracted_packets, extracted_dscp, 'mixed_long_3class_no_cp', info
    
    # 其他情况（理论上不应该到这里）
    return 'skip', [], [], 'unknown_case', info


def process_pcap(pcap_path, output_dir, max_packets=200, min_packets=5, separate_pure_mixed=True):
    """
    处理 PCAP 文件，按流分类和截取
    
    参数:
        separate_pure_mixed: 是否分开保存纯净流和混合流（用于分别训练）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if separate_pure_mixed:
        os.makedirs(os.path.join(output_dir, 'pure'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'mixed'), exist_ok=True)
    
    print(f"{'='*70}")
    print(f"PCAP 预处理")
    print(f"{'='*70}")
    print(f"输入: {pcap_path}")
    print(f"输出: {output_dir}")
    print(f"最大包数: {max_packets}")
    print(f"最小包数: {min_packets}")
    print(f"分离纯净/混合流: {separate_pure_mixed}")
    
    # 第一遍：收集流信息
    print(f"\n读取 PCAP...")
    flows = defaultdict(lambda: {'packets': [], 'dscp_list': []})
    
    pkt_count = 0
    with PcapReader(pcap_path) as reader:
        for pkt in reader:
            pkt_count += 1
            if pkt_count % 500000 == 0:
                print(f"  ...{pkt_count//1000}k 包")
            
            flow_key = get_flow_key(pkt)
            if flow_key is None:
                continue
            
            dscp = get_dscp(pkt)
            if dscp is None:
                continue
            
            flows[flow_key]['packets'].append(pkt)
            flows[flow_key]['dscp_list'].append(dscp)
    
    print(f"  总包数: {pkt_count:,}")
    print(f"  总流数: {len(flows):,}")
    
    # 处理每个流
    print(f"\n处理流...")
    
    # 分别存储纯净流和混合流
    results_pure = {'high': [], 'mid': [], 'low': []}
    results_mixed = {'high': [], 'mid': [], 'low': []}
    results_skip = []
    method_stats = Counter()
    
    for flow_key, data in flows.items():
        if len(data['packets']) < min_packets:
            continue
        
        cls, extracted_pkts, extracted_dscp, method, info = classify_and_extract_flow(
            data['dscp_list'], data['packets'], max_packets
        )
        
        method_stats[method] += 1
        
        if cls != 'skip' and extracted_pkts:
            flow_data = {
                'flow_key': flow_key,
                'packets': extracted_pkts,
                'dscp_list': extracted_dscp,
                'method': method,
                'info': info
            }
            
            if method == 'pure':
                results_pure[cls].append(flow_data)
            else:
                results_mixed[cls].append(flow_data)
        else:
            results_skip.append({
                'flow_key': flow_key,
                'method': method,
                'info': info
            })
    
    # 统计
    print(f"\n{'='*70}")
    print(f"分类结果")
    print(f"{'='*70}")
    
    print(f"\n纯净流 (所有包DSCP相同):")
    total_pure = 0
    for cls in ['high', 'mid', 'low']:
        count = len(results_pure[cls])
        total_pure += count
        if count > 0:
            pkt_count = sum(len(r['packets']) for r in results_pure[cls])
            print(f"  {cls}: {count:,} 流, {pkt_count:,} 包")
    print(f"  总计: {total_pure:,} 流")
    
    print(f"\n混合流 (包含多种DSCP):")
    total_mixed = 0
    for cls in ['high', 'mid', 'low']:
        count = len(results_mixed[cls])
        total_mixed += count
        if count > 0:
            pkt_count = sum(len(r['packets']) for r in results_mixed[cls])
            print(f"  {cls}: {count:,} 流, {pkt_count:,} 包")
    print(f"  总计: {total_mixed:,} 流")
    
    print(f"\n跳过: {len(results_skip):,} 流")
    
    print(f"\n方法统计:")
    for method, count in method_stats.most_common():
        print(f"  {method}: {count:,}")
    
    # 写入文件
    print(f"\n写入文件...")
    
    def write_flows(flows_dict, sub_dir=''):
        """写入流到 PCAP 文件"""
        dir_path = os.path.join(output_dir, sub_dir) if sub_dir else output_dir
        
        for cls in ['high', 'mid', 'low']:
            if not flows_dict[cls]:
                continue
            
            output_path = os.path.join(dir_path, f'flow_{cls}.pcap')
            with PcapWriter(output_path, append=False, sync=True) as writer:
                for flow_data in flows_dict[cls]:
                    for pkt in flow_data['packets']:
                        writer.write(pkt)
            
            pkt_count = sum(len(r['packets']) for r in flows_dict[cls])
            print(f"  ✓ {output_path}: {len(flows_dict[cls]):,} 流, {pkt_count:,} 包")
    
    if separate_pure_mixed:
        print(f"\n纯净流:")
        write_flows(results_pure, 'pure')
        
        print(f"\n混合流:")
        write_flows(results_mixed, 'mixed')
        
        # 同时写入合并版本
        print(f"\n合并版本:")
        results_all = {cls: results_pure[cls] + results_mixed[cls] for cls in ['high', 'mid', 'low']}
        write_flows(results_all, '')
    else:
        results_all = {cls: results_pure[cls] + results_mixed[cls] for cls in ['high', 'mid', 'low']}
        write_flows(results_all, '')
    
    # 写入报告
    report = {
        'input_file': pcap_path,
        'max_packets': max_packets,
        'min_packets': min_packets,
        'pure_flow_counts': {cls: len(results_pure[cls]) for cls in ['high', 'mid', 'low']},
        'mixed_flow_counts': {cls: len(results_mixed[cls]) for cls in ['high', 'mid', 'low']},
        'skip_count': len(results_skip),
        'method_stats': dict(method_stats),
    }
    
    report_path = os.path.join(output_dir, 'preprocess_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ {report_path}")
    
    # 写入详细信息
    def save_details(flows_dict, filename):
        details = []
        for cls in ['high', 'mid', 'low']:
            for flow_data in flows_dict[cls]:
                fk = flow_data['flow_key']
                details.append({
                    'flow': f"{fk[0]} {fk[1]}:{fk[2]} <-> {fk[3]}:{fk[4]}",
                    'class': cls,
                    'method': flow_data['method'],
                    'length': len(flow_data['packets']),
                    'original_length': flow_data['info']['original_length'],
                })
        
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(details, f, indent=2)
    
    if separate_pure_mixed:
        save_details(results_pure, 'pure/flow_details.json')
        save_details(results_mixed, 'mixed/flow_details.json')
    
    save_details({cls: results_pure[cls] + results_mixed[cls] for cls in ['high', 'mid', 'low']}, 
                 'flow_details.json')
    
    print(f"\n完成!")
    
    return {
        'pure': results_pure,
        'mixed': results_mixed,
        'skip': results_skip
    }


def main():
    parser = argparse.ArgumentParser(description='PCAP 数据预处理')
    parser.add_argument('pcap', help='输入 PCAP 文件')
    parser.add_argument('-o', '--output', default='./preprocessed', help='输出目录')
    parser.add_argument('--max_packets', type=int, default=200, help='最大包数 (默认 200)')
    parser.add_argument('--min_packets', type=int, default=5, help='最小包数 (默认 2)')
    parser.add_argument('--no_separate', action='store_true', help='不分离纯净流和混合流')
    args = parser.parse_args()
    
    process_pcap(args.pcap, args.output, args.max_packets, args.min_packets, 
                 separate_pure_mixed=not args.no_separate)


if __name__ == '__main__':
    main()