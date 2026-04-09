#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二分类任务评测脚本
计算准确率、精确率、召回率、F1、AUC等所有统计指标
"""

import json
import pandas as pd
import numpy as np
import argparse
import re
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

def extract_prediction_and_thinking(response_text):
    """
    从response中提取预测结果和思考过程
    
    Args:
        response_text: 模型回答文本
        
    Returns:
        tuple: (prediction, thinking)
    """
    if pd.isna(response_text) or response_text == "":
        return "", ""
    
    response_text = str(response_text)
    
    # 提取<think>标签内的内容
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, response_text, re.DOTALL)
    thinking = ' '.join(think_matches).strip() if think_matches else ""
    
    # 如果没有找到</think>标签，但有<think>标签，则提取<think>后的所有内容作为thinking
    if not thinking and response_text.startswith('<think>'):
        # 移除开头的<think>标签
        thinking = response_text[7:].strip()
    
    # 提取</think>之后的内容作为预测结果
    after_think_pattern = r'</think>\s*(.*?)$'
    after_think_matches = re.findall(after_think_pattern, response_text, re.DOTALL)
    prediction_text = after_think_matches[0].strip() if after_think_matches else ""
    
    # 如果没有找到</think>标签，尝试从整个response中提取预测结果
    if not prediction_text:
        # 如果response以<think>开头但没有</think>，则整个内容都是thinking，没有明确的预测结果
        if response_text.startswith('<think>'):
            prediction_text = ""
        else:
            prediction_text = response_text.strip()
    
    # 从预测文本中提取"是"或"否"
    if "是" in prediction_text and "否" not in prediction_text:
        prediction = "是"
    elif "否" in prediction_text and "是" not in prediction_text:
        prediction = "否"
    else:
        # 如果同时包含"是"和"否"，取最后一个
        if prediction_text.rfind("是") > prediction_text.rfind("否"):
            prediction = "是"
        elif prediction_text.rfind("否") > prediction_text.rfind("是"):
            prediction = "否"
        else:
            prediction = ""  # 无法确定
    
    return prediction, thinking

def extract_user_content(messages):
    """
    从messages中提取user的content字段
    
    Args:
        messages: 消息列表
        
    Returns:
        str: user的content内容
    """
    if not messages or not isinstance(messages, list):
        return ""
    
    for message in messages:
        if isinstance(message, dict) and message.get('role') == 'user':
            return str(message.get('content', ''))
    
    return ""

def count_tokens(text):
    """
    统计文本的token长度（简单估算）
    
    Args:
        text: 输入文本
        
    Returns:
        int: token数量
    """
    if pd.isna(text) or text == "":
        return 0
    
    # 简单的中文token估算：按字符数计算，中文1个字符约等于1个token
    text = str(text)
    # 去除空白字符
    text = text.strip()
    return len(text)

def calculate_binary_metrics(y_true, y_pred, y_prob=None):
    """
    计算二分类任务的所有指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选）
        
    Returns:
        dict: 包含所有指标的字典
    """
    # 转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 直接计算匹配情况，与correctness列保持一致
    # correctness列的计算逻辑：predictions[i] == labels[i]
    matches = (y_pred == y_true)
    
    # 转换为数值标签（是=1，否=0）
    y_true_binary = (y_true == "是").astype(int)
    y_pred_binary = (y_pred == "是").astype(int)
    
    metrics = {}
    
    # 基础指标
    # accuracy直接基于匹配情况计算，与correctness列保持一致
    metrics['accuracy'] = np.mean(matches)
    metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    metrics['f1_score'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # 混淆矩阵
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # 基于混淆矩阵的指标
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # 如果有概率预测，计算AUC相关指标
    if y_prob is not None:
        y_prob_valid = np.array(y_prob)[valid_mask]
        if len(y_prob_valid) > 0:
            try:
                # 将概率转换为数值
                y_prob_binary = np.array([1 if p == "是" else 0 for p in y_prob_valid])
                metrics['auc_roc'] = roc_auc_score(y_true_binary, y_prob_binary)
                metrics['auc_pr'] = average_precision_score(y_true_binary, y_prob_binary)
            except:
                metrics['auc_roc'] = None
                metrics['auc_pr'] = None
    
    # 样本统计
    metrics['total_samples'] = len(y_true)
    metrics['valid_samples'] = len(y_true)  # 现在所有样本都参与计算
    metrics['invalid_samples'] = 0  # 空值被处理为"否"，不再视为无效
    
    # 类别分布
    metrics['true_positive_rate'] = np.mean(y_true_binary)
    metrics['predicted_positive_rate'] = np.mean(y_pred_binary)
    
    return metrics

def process_jsonl_file(input_path, output_path=None):
    """
    处理JSONL文件，计算评测指标
    
    Args:
        input_path: 输入JSONL文件路径
        output_path: 输出文件路径（可选）
        
    Returns:
        dict: 评测结果
    """
    print(f"正在处理文件: {input_path}")
    
    # 读取数据
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析错误: {e}")
                continue
    
    print(f"成功读取 {len(data)} 条数据")
    
    # 提取预测结果和思考过程
    predictions = []
    thinkings = []
    labels = []
    prompts = []
    thinking_lengths = []
    
    for item in data:
        # 提取预测结果和思考过程
        prediction, thinking = extract_prediction_and_thinking(item.get('response', ''))
        predictions.append(prediction)
        thinkings.append(thinking)
        
        # 提取user content作为prompt
        prompt = extract_user_content(item.get('messages', []))
        prompts.append(prompt)
        
        # 计算thinking的token长度
        thinking_length = count_tokens(thinking)
        thinking_lengths.append(thinking_length)
        
        # 提取真实标签
        label = item.get('labels', '')
        labels.append(label)
    
    # 计算指标
    metrics = calculate_binary_metrics(labels, predictions)
    
    # 创建结果DataFrame
    result_data = []
    for i, item in enumerate(data):
        result_item = item.copy()  # 保留原始字段
        result_item['extracted_prediction'] = predictions[i]
        result_item['extracted_thinking'] = thinkings[i]
        result_item['len'] = thinking_lengths[i]  # 添加len列
        result_item['prompt'] = prompts[i]  # 添加prompt列
        result_item['extracted_label'] = labels[i]
        result_item['prediction_correct'] = predictions[i] == labels[i]
        result_item['correctness'] = 1 if predictions[i] == labels[i] else 0
        result_data.append(result_item)
    
    # 保存结果
    if output_path:
        # 创建DataFrame
        df = pd.DataFrame(result_data)
        
        # 将指标转换为百分比形式，保留2位小数
        percentage_columns = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 
                             'sensitivity', 'positive_predictive_value', 'negative_predictive_value',
                             'true_positive_rate', 'predicted_positive_rate']
        
        for col in percentage_columns:
            if col in metrics:
                metrics[col] = round(metrics[col] * 100, 2)
        
        # 创建指标数据DataFrame（14行×2列）
        metrics_data = []
        for key, value in metrics.items():
            if key not in ['total_samples', 'valid_samples', 'invalid_samples']:
                metrics_data.append({'metric_name': key, 'metric_value': str(value)})
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # 创建完整的指标列，前14行是指标数据，后面都是空值
        metric_name_col = [None] * len(df)
        metric_value_col = [None] * len(df)
        
        # 将前14行的指标数据填入
        for i, (_, row) in enumerate(metrics_df.iterrows()):
            if i < len(df):
                metric_name_col[i] = row['metric_name']
                metric_value_col[i] = row['metric_value']
        
        # 将指标列添加到原始DataFrame
        df['metric_name'] = metric_name_col
        df['metric_value'] = metric_value_col
        
        # 获取文件名作为sheet名称
        file_name = Path(input_path).stem
        
        # 保存到Excel文件
        df.to_excel(output_path, sheet_name=file_name, index=False)
        
        print(f"结果已保存到: {output_path}")
        print(f"  - Sheet '{file_name}': 包含所有原始数据、提取字段和评测指标（横向排列）")
    
    return metrics, result_data

def print_metrics_summary(metrics):
    """
    打印指标摘要
    """
    print("\n" + "="*50)
    print("评测指标摘要")
    print("="*50)
    
    if 'error' in metrics:
        print(f"错误: {metrics['error']}")
        return
    
    print(f"总样本数: {metrics['total_samples']}")
    print(f"有效样本数: {metrics['valid_samples']}")
    print(f"无效样本数: {metrics['invalid_samples']}")
    print()
    
    print("基础指标:")
    print(f"  准确率 (Accuracy): {metrics['accuracy']:.2f}")
    print(f"  精确率 (Precision): {metrics['precision']:.2f}")
    print(f"  召回率 (Recall): {metrics['recall']:.2f}")
    print(f"  F1分数: {metrics['f1_score']:.2f}")
    print()
    
    if 'true_positives' in metrics:
        print("混淆矩阵:")
        print(f"  真正例 (TP): {metrics['true_positives']}")
        print(f"  假正例 (FP): {metrics['false_positives']}")
        print(f"  假负例 (FN): {metrics['false_negatives']}")
        print(f"  真负例 (TN): {metrics['true_negatives']}")
        print()
        
        print("详细指标:")
        print(f"  特异性 (Specificity): {metrics['specificity']:.2f}")
        print(f"  敏感性 (Sensitivity): {metrics['sensitivity']:.2f}")
        print(f"  阳性预测值 (PPV): {metrics['positive_predictive_value']:.2f}")
        print(f"  阴性预测值 (NPV): {metrics['negative_predictive_value']:.2f}")
        print()
    
    if 'auc_roc' in metrics and metrics['auc_roc'] is not None:
        print("AUC指标:")
        print(f"  ROC-AUC: {metrics['auc_roc']:.2f}")
        print(f"  PR-AUC: {metrics['auc_pr']:.2f}")
        print()
    
    print("类别分布:")
    print(f"  真实阳性率: {metrics['true_positive_rate']:.2f}")
    print(f"  预测阳性率: {metrics['predicted_positive_rate']:.2f}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='二分类任务评测脚本')
    parser.add_argument('--pred_file', type=str, required=True, help='预测结果JSONL文件路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件路径')
    return parser.parse_args()

def main(input_path, output_path=None):
    """
    主函数
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    # 处理文件
    metrics, result_data = process_jsonl_file(input_path, output_path)
    
    # 打印指标摘要
    print_metrics_summary(metrics)
    
    return metrics, result_data

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    main(input_path=args.pred_file, output_path=args.output_file)

# python step2_evaluation.py --pred_file infer_res/v0_m0.jsonl --output_file infer_res/v0_m0_score.xlsx
