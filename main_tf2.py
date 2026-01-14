#!/usr/bin/env python3
"""
FS-Net 主入口 - TensorFlow 2.x 兼容版本

使用方法:
    # 预处理
    python main_tf2.py --mode=prepro --class_num=3 --data_dir=./filter_balanced
    
    # 训练
    python main_tf2.py --mode=train --class_num=3
    
    # 测试
    python main_tf2.py --mode=test --class_num=3
"""

import os
import argparse
import train_tf2
import preprocess


class Config:
    """配置类"""
    pass


def main():
    parser = argparse.ArgumentParser(description='FS-Net TensorFlow 2.x')
    
    # 路径参数
    parser.add_argument('--data_dir', type=str, default='./filter', help='数据目录')
    parser.add_argument('--log_dir', type=str, default='./log', help='日志目录')
    parser.add_argument('--model_dir', type=str, default='./log', help='模型保存目录')
    parser.add_argument('--pred_dir', type=str, default='./result', help='预测结果目录')
    parser.add_argument('--test_model_dir', type=str, default='./log', help='测试模型目录')
    
    # 数据参数
    parser.add_argument('--class_num', type=int, default=3, help='类别数量')
    parser.add_argument('--length_block', type=int, default=1, help='长度分块')
    parser.add_argument('--min_length', type=int, default=2, help='最小流长度')
    parser.add_argument('--max_packet_length', type=int, default=5000, help='最大包长度')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--keep_ratio', type=float, default=1.0, help='数据保留比例')
    parser.add_argument('--max_flow_length_train', type=int, default=200, help='训练时最大流长度')
    parser.add_argument('--max_flow_length_test', type=int, default=2000, help='测试时最大流长度')
    
    # 模型参数
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--hidden', type=int, default=128, help='GRU隐藏层维度')
    parser.add_argument('--layer', type=int, default=2, help='GRU层数')
    parser.add_argument('--length_dim', type=int, default=16, help='长度嵌入维度')
    
    # 训练参数
    parser.add_argument('--keep_prob', type=float, default=0.8, help='Dropout保留率')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--iter_num', type=int, default=15000, help='迭代次数 (约200 epochs)')
    parser.add_argument('--decay_step', type=str, default='auto', help='学习率衰减步数')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='学习率衰减率')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='梯度裁剪')
    parser.add_argument('--rec_loss', type=float, default=0.5, help='重构损失权重')
    parser.add_argument('--early_stop', type=int, default=20, help='早停轮数 (验证集无提升时停止)')
    
    # 其他参数
    parser.add_argument('--mode', type=str, default='train', choices=['prepro', 'train', 'test'],
                        help='运行模式: prepro/train/test')
    parser.add_argument('--capacity', type=int, default=1000, help='数据集shuffle缓冲区大小')
    parser.add_argument('--loss_save', type=int, default=100, help='损失保存间隔')
    parser.add_argument('--checkpoint', type=int, default=5000, help='检查点保存间隔')
    
    # 类别权重（处理数据不平衡）
    parser.add_argument('--class_weights', type=str, default=None,
                        help='类别权重，逗号分隔，如 "1.0,72.0,1.44"')
    
    args = parser.parse_args()
    
    # 创建配置对象
    config = Config()
    for key, value in vars(args).items():
        setattr(config, key, value)
    
    # 设置路径
    home = os.getcwd()
    record_dir = os.path.join(home, 'record')
    
    config.train_json = os.path.join(record_dir, 'train.json')
    config.test_json = os.path.join(record_dir, 'test.json')
    config.train_meta = os.path.join(record_dir, 'train.meta')
    config.test_meta = os.path.join(record_dir, 'test.meta')
    
    # 创建目录
    for d in [config.log_dir, config.model_dir, config.pred_dir, record_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 计算 length_num
    config.length_num = config.max_packet_length // config.length_block + 4
    
    # 处理 decay_step
    if config.decay_step != 'auto':
        config.decay_step = int(config.decay_step)
    
    # 处理类别权重
    if config.class_weights is not None:
        try:
            config.class_weights = [float(w.strip()) for w in config.class_weights.split(',')]
            print(f"类别权重: {config.class_weights}")
        except ValueError as e:
            print(f"警告: 无法解析类别权重 '{config.class_weights}', 将使用默认权重")
            config.class_weights = None
    
    # 运行
    if config.mode == 'prepro':
        print(f"预处理数据: {config.data_dir}")
        preprocess.preprocess(config)
    elif config.mode == 'train':
        train_tf2.train(config)
    elif config.mode == 'test':
        train_tf2.predict(config)
    else:
        print(f"未知模式: {config.mode}")


if __name__ == '__main__':
    main()
