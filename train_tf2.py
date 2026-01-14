"""
训练脚本 - TensorFlow 2.x 兼容版本
使用 pickle 保存权重，绕过 Keras 3 序列化问题
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import os
import pickle

from model_tf2 import create_model
from dataset_tf2 import create_dataset, compute_metrics


def get_all_weights(model):
    """递归获取模型所有权重（包括嵌套层）"""
    weights = []
    for layer in model.layers:
        weights.extend(layer.get_weights())
    # 还要获取模型顶层的变量
    for var in model.trainable_variables:
        # 避免重复
        pass
    return model.get_weights()


def save_model_weights(model, filepath):
    """保存模型所有权重"""
    weights = model.get_weights()
    with open(filepath, 'wb') as f:
        pickle.dump(weights, f)
    print(f"    保存了 {len(weights)} 个权重数组")


def load_model_weights(model, filepath):
    """加载模型权重"""
    with open(filepath, 'rb') as f:
        weights = pickle.load(f)
    model.set_weights(weights)
    print(f"    加载了 {len(weights)} 个权重数组")


def train(config):
    """训练模型"""
    print("=" * 60)
    print("FS-Net Training (TensorFlow 2.x)")
    print("=" * 60)
    
    max_len = config.max_flow_length_train
    
    # 加载数据集
    print("\n[1] 加载数据集...")
    train_dataset, train_num = create_dataset(config.train_json, config, max_len, shuffle=True)
    test_dataset, test_num = create_dataset(config.test_json, config, max_len, shuffle=False)
    
    print(f"    训练集: {train_num} 样本")
    print(f"    测试集: {test_num} 样本")
    
    # 计算训练步数
    steps_per_epoch = train_num // config.batch_size
    if config.decay_step == 'auto':
        config.decay_step = steps_per_epoch * 2
    
    print(f"    每轮步数: {steps_per_epoch}")
    print(f"    衰减步数: {config.decay_step}")
    
    # 创建模型
    print("\n[2] 创建模型...")
    model = create_model(config)
    
    # 学习率调度
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.decay_step,
        decay_rate=config.decay_rate,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)
    
    # TensorBoard
    log_dir = config.log_dir
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # 检查点目录
    checkpoint_dir = config.model_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存配置
    config_to_save = {
        'class_num': config.class_num,
        'length_num': config.length_num,
        'length_dim': config.length_dim,
        'hidden': config.hidden,
        'layer': config.layer,
        'keep_prob': config.keep_prob,
        'rec_loss': config.rec_loss,
        'max_len': max_len,
    }
    with open(os.path.join(checkpoint_dir, 'model_config.json'), 'w') as f:
        json.dump(config_to_save, f)
    
    # 训练循环
    print("\n[3] 开始训练...")
    global_step = 0
    best_accuracy = 0.0
    epochs_without_improvement = 0
    early_stop = getattr(config, 'early_stop', 20)
    
    num_epochs = config.iter_num // steps_per_epoch + 1
    
    for epoch in range(num_epochs):
        epoch_loss = []
        
        pbar = tqdm(train_dataset, desc=f'Epoch {epoch+1}/{num_epochs}', ascii=True)
        for batch in pbar:
            flow = batch['flow']
            labels = batch['label']
            mask = tf.cast(flow, tf.bool)
            
            with tf.GradientTape() as tape:
                logits, pred, rec_logits = model((flow, mask), training=True)
                loss, c_loss, rec_loss = model.compute_loss(flow, labels, mask, logits, rec_logits)
            
            # 梯度裁剪
            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss.append(float(loss))
            global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{float(loss):.4f}',
                'c_loss': f'{float(c_loss):.4f}',
                'rec_loss': f'{float(rec_loss):.4f}'
            })
            
            # 记录损失
            if global_step % config.loss_save == 0:
                with summary_writer.as_default():
                    tf.summary.scalar('loss/total', loss, step=global_step)
                    tf.summary.scalar('loss/classify', c_loss, step=global_step)
                    tf.summary.scalar('loss/reconstruct', rec_loss, step=global_step)
            
            if global_step >= config.iter_num:
                break
        
        if global_step >= config.iter_num:
            break
        
        # 每轮评估
        avg_loss = np.mean(epoch_loss)
        
        # 在测试集上评估
        test_acc, test_metrics = evaluate(model, test_dataset, config, verbose=False)
        print(f"\n  Epoch {epoch+1} | 损失: {avg_loss:.4f} | 准确率: {test_acc:.4f}")
        
        # 每10轮打印详细指标
        if (epoch + 1) % 10 == 0:
            for c in range(config.class_num):
                m = test_metrics.get(f'class_{c}', {})
                print(f"    Class {c}: TPR={m.get('TPR', 0):.3f}, F1={m.get('F1', 0):.3f}, Support={m.get('Support', 0)}")
        
        with summary_writer.as_default():
            tf.summary.scalar('accuracy/test', test_acc, step=global_step)
        
        # 保存最佳模型 & 早停检查
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            epochs_without_improvement = 0
            # 使用pickle保存权重
            save_model_weights(model, os.path.join(checkpoint_dir, 'best_model.pkl'))
            print(f"  ✓ 保存最佳模型 (accuracy={best_accuracy:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop:
                print(f"\n  ⚠ 早停: {early_stop} 轮无提升，停止训练")
                break
    
    # 保存最终模型
    save_model_weights(model, os.path.join(checkpoint_dir, 'final_model.pkl'))
    print(f"\n训练完成! 最佳准确率: {best_accuracy:.4f}")
    
    # ===== 立即验证 =====
    print("\n" + "=" * 60)
    print("验证模型保存/加载...")
    print("=" * 60)
    
    # 创建新模型
    model2 = create_model(config)
    
    # 先构建模型
    test_dataset_build, _ = create_dataset(config.test_json, config, max_len, shuffle=False)
    for batch in test_dataset_build:
        flow = batch['flow']
        mask = tf.cast(flow, tf.bool)
        _ = model2((flow, mask), training=False)
        break
    
    # 加载权重
    load_model_weights(model2, os.path.join(checkpoint_dir, 'best_model.pkl'))
    
    # 测试
    test_dataset_verify, _ = create_dataset(config.test_json, config, max_len, shuffle=False)
    reload_acc, reload_metrics = evaluate(model2, test_dataset_verify, config, verbose=True)
    print(f"加载后准确率: {reload_acc:.4f}")
    
    for c in range(config.class_num):
        m = reload_metrics.get(f'class_{c}', {})
        print(f"  Class {c}: TPR={m.get('TPR', 0):.3f}, Support={m.get('Support', 0)}")
    
    summary_writer.close()


def evaluate(model, dataset, config, verbose=False):
    """评估模型"""
    all_labels = []
    all_preds = []
    
    for batch in dataset:
        flow = batch['flow']
        labels = batch['label']
        mask = tf.cast(flow, tf.bool)
        
        logits, pred, _ = model((flow, mask), training=False)
        
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(pred.numpy().tolist())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # 打印预测分布
    if verbose:
        unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
        print(f"    预测分布: {dict(zip(unique_preds.tolist(), pred_counts.tolist()))}")
    
    metrics = compute_metrics(all_labels, all_preds, config.class_num)
    accuracy = metrics['accuracy']
    
    return accuracy, metrics


def predict(config):
    """预测并评估"""
    print("=" * 60)
    print("FS-Net Prediction (TensorFlow 2.x)")
    print("=" * 60)
    
    # 加载保存的配置
    config_path = os.path.join(config.test_model_dir, 'model_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        print(f"\n加载配置: {saved_config}")
        
        # 用保存的配置更新当前配置
        for k, v in saved_config.items():
            if k != 'max_len':
                setattr(config, k, v)
        max_len = saved_config.get('max_len', config.max_flow_length_test)
    else:
        print("\n警告: 未找到保存的配置，使用命令行参数")
        max_len = config.max_flow_length_test
    
    print(f"\n配置信息:")
    print(f"  max_len: {max_len}")
    print(f"  class_num: {config.class_num}")
    print(f"  length_num: {config.length_num}")
    
    # 加载测试数据
    print("\n[1] 加载测试数据...")
    test_dataset, test_num = create_dataset(config.test_json, config, max_len, shuffle=False)
    print(f"    测试集: {test_num} 样本")
    
    # 创建模型
    print("\n[2] 加载模型...")
    model = create_model(config)
    
    # 先构建模型
    print("    构建模型...")
    for batch in test_dataset:
        flow = batch['flow']
        mask = tf.cast(flow, tf.bool)
        _ = model((flow, mask), training=False)
        break
    
    # 重新创建数据集
    test_dataset, _ = create_dataset(config.test_json, config, max_len, shuffle=False)
    
    # 加载权重
    weights_path = os.path.join(config.test_model_dir, 'best_model.pkl')
    if not os.path.exists(weights_path):
        weights_path = os.path.join(config.test_model_dir, 'final_model.pkl')
    
    load_model_weights(model, weights_path)
    print(f"    已加载权重: {weights_path}")
    
    # 评估
    print("\n[3] 评估中...")
    accuracy, metrics = evaluate(model, test_dataset, config, verbose=True)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"\n总体准确率: {accuracy:.4f}")
    print("\n各类别指标:")
    
    class_names = {0: 'mid', 1: 'low', 2: 'high'}
    
    for c in range(config.class_num):
        m = metrics.get(f'class_{c}', {})
        name = class_names.get(c, f'class_{c}')
        print(f"  {name} (label={c}):")
        print(f"    TPR/Recall: {m.get('TPR', 0):.4f}")
        print(f"    Precision:  {m.get('Precision', 0):.4f}")
        print(f"    F1-Score:   {m.get('F1', 0):.4f}")
        print(f"    Support:    {m.get('Support', 0)}")
    
    # 保存结果
    result_path = os.path.join(config.pred_dir, 'FSNet_result.json')
    os.makedirs(config.pred_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n结果已保存到: {result_path}")
    
    return metrics