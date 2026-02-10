import os
import json
import glob
import csv
from collections import defaultdict
import statistics
import numpy as np

# 定义指标列表（按照要求的顺序：SSIM, LPIPS, DINO Similarity, CLIP, Text Similarity, CVQI）
METRICS = [
    'ssim',
    'lpips_distance',
    'dino_similarity',
    'clip_similarity',
    'text_similarity',
    'comprehensive_visual_score'
]

# 用于计算S_str的原始指标
S_STR_COMPONENTS = [
    'ssim',
    'lpips_distance',
    'dino_similarity'
]

# 用于计算S_sty的原始指标
S_STY_COMPONENTS = [
    'color_histogram_similarity',
    'dominant_color_similarity',
    'style_loss'
]

# 整体指标
GLOBAL_METRICS = [
    'visual_score',
    'success_ratio'
]

def is_valid_sequence(metadata):
    """检查sequence是否有效（非空且有成功步骤）"""
    sequence = metadata.get('sequence', [])
    successful_steps = metadata.get('successful_steps', 0)
    return len(sequence) > 0 and successful_steps > 0

def load_all_metadata(base_dir):
    """加载所有任务的metadata.json文件"""
    all_data = []
    
    # 遍历所有子目录
    for task_dir in os.listdir(base_dir):
        metadata_path = os.path.join(base_dir, task_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            continue
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
                # 提取模型名称（从id字段中提取，格式如 "gemini-3-flash_Stripe_1766583969"）
                model_name = metadata.get('id', task_dir).split('_')[0]
                
                all_data.append({
                    'model': model_name,
                    'task_id': metadata.get('id', task_dir),
                    'metadata': metadata
                })
        except Exception as e:
            print(f"Error loading {metadata_path}: {e}")
    
    return all_data

def calculate_metrics(all_data):
    """计算各模型的指标平均值"""
    # 按模型分组
    model_data = defaultdict(lambda: {'all': [], 'valid': []})
    
    for item in all_data:
        model = item['model']
        metadata = item['metadata']
        
        # 存储到all组（所有数据）
        model_data[model]['all'].append(metadata)
        
        # 存储到valid组（有效数据）
        if is_valid_sequence(metadata):
            model_data[model]['valid'].append(metadata)
    
    # 找到所有模型中的最大任务数量作为基准
    max_count = max(len(groups['all']) for groups in model_data.values())
    print(f"基准数量（最大任务数）: {max_count}")
    
    # 计算平均值
    results = []
    
    for model, groups in sorted(model_data.items()):
        # 计算所有数据的平均值（基于基准数量max_count）
        all_metrics = calculate_average_for_group(groups['all'], 'all', max_count)
        
        # 计算有效数据的平均值（只基于实际数量）
        valid_metrics = calculate_average_for_group(groups['valid'], 'valid')
        
        results.append({
            'model': model,
            'all_count': len(groups['all']),
            'valid_count': len(groups['valid']),
            'baseline_count': max_count,
            'all_metrics': all_metrics,
            'valid_metrics': valid_metrics
        })
    
    return results

def calculate_average_for_group(group, group_type, baseline_count=None):
    """计算一个组的指标平均值
    
    Args:
        group: 任务列表
        group_type: 'all' 或 'valid'
        baseline_count: 基准数量（仅在group_type='all'时使用）
    """
    if not group:
        # 返回所有指标和S_str、S_sty为0的字典
        result = {metric: 0 for metric in METRICS}
        result['s_str'] = 0
        result['s_sty'] = 0
        for metric in GLOBAL_METRICS:
            result[metric] = 0
        return result
    
    # 先计算每个任务的平均指标
    task_metrics = []
    
    for metadata in group:
        sequence = metadata.get('sequence', [])
        
        # 计算该任务所有步骤的平均指标
        task_avg = {}
        
        if sequence:
            # 计算步骤级指标的平均值
            for metric in METRICS:
                values = [step.get(metric) for step in sequence if metric in step and step.get(metric) is not None]
                if values:
                    task_avg[metric] = sum(values) / len(values)
            
            # 直接从sequence中读取s_str和s_sty
            values_s_str = [step.get('s_str') for step in sequence if 's_str' in step and step.get('s_str') is not None]
            values_s_sty = [step.get('s_sty') for step in sequence if 's_sty' in step and step.get('s_sty') is not None]
            
            task_avg['s_str'] = sum(values_s_str) / len(values_s_str) if values_s_str else 0.0
            task_avg['s_sty'] = sum(values_s_sty) / len(values_s_sty) if values_s_sty else 0.0
        else:
            # 空sequence，所有指标为0
            for metric in METRICS:
                task_avg[metric] = 0.0
            task_avg['s_str'] = 0.0
            task_avg['s_sty'] = 0.0
        
        # 添加全局指标
        for metric in GLOBAL_METRICS:
            task_avg[metric] = metadata.get(metric, 0.0)
        
        task_metrics.append(task_avg)
    
    # 对所有任务的指标求平均
    averages = {}
    
    # METRICS的平均值
    for metric in METRICS:
        if group_type == 'all' and baseline_count is not None:
            total_sum = sum(task_avg.get(metric, 0) for task_avg in task_metrics)
            averages[metric] = total_sum / baseline_count
        else:
            if task_metrics:
                total_sum = sum(task_avg.get(metric, 0) for task_avg in task_metrics)
                averages[metric] = total_sum / len(task_metrics)
            else:
                averages[metric] = 0.0
    
    # S_str的平均值
    if group_type == 'all' and baseline_count is not None:
        total_sum = sum(task_avg.get('s_str', 0) for task_avg in task_metrics)
        averages['s_str'] = total_sum / baseline_count
    else:
        if task_metrics:
            total_sum = sum(task_avg.get('s_str', 0) for task_avg in task_metrics)
            averages['s_str'] = total_sum / len(task_metrics)
        else:
            averages['s_str'] = 0.0
    
    # S_sty的平均值
    if group_type == 'all' and baseline_count is not None:
        total_sum = sum(task_avg.get('s_sty', 0) for task_avg in task_metrics)
        averages['s_sty'] = total_sum / baseline_count
    else:
        if task_metrics:
            total_sum = sum(task_avg.get('s_sty', 0) for task_avg in task_metrics)
            averages['s_sty'] = total_sum / len(task_metrics)
        else:
            averages['s_sty'] = 0.0
    
    # 全局指标的平均值
    for metric in GLOBAL_METRICS:
        if group_type == 'all' and baseline_count is not None:
            total_sum = sum(task_avg.get(metric, 0) for task_avg in task_metrics)
            averages[metric] = total_sum / baseline_count
        else:
            if task_metrics:
                total_sum = sum(task_avg.get(metric, 0) for task_avg in task_metrics)
                averages[metric] = total_sum / len(task_metrics)
            else:
                averages[metric] = 0.0
    
    return averages

def save_to_csv(results, output_path):
    """保存结果到CSV文件"""
    # 定义模型排序顺序
    MODEL_ORDER = [
        'gemini-3-pro',
        'gemini-3-flash',
        'Doubao-Seed-1.8',
        'doubao-seed-1-6',
        'gpt-4o',
        'Qwen3-VL-235B',
        'Qwen3-VL-30B',
        'Qwen3-VL-8B'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 获取基准数量（所有模型都有相同的baseline_count）
        baseline_count = results[0]['baseline_count'] if results else 0
        
        # 写入表头 - 计算所有
        # 顺序: SSIM, LPIPS, DINO Similarity, S_str, CLIP, Text Similarity, S_sty, CVQI, Success Ratio, Visual Score
        header_all = ['Model', f'Count (Baseline: {baseline_count})', 'SSIM', 'LPIPS', 'DINO Similarity', 
                      'S_str', 'CLIP', 'Text Similarity', 'S_sty', 'CVQI', 'Success Ratio', 'Visual Score']
        writer.writerow(header_all)
        
        # 按照指定顺序排序模型
        def get_model_order(model_name):
            try:
                return MODEL_ORDER.index(model_name)
            except ValueError:
                # 如果模型不在列表中，放在最后
                return len(MODEL_ORDER)
        
        # 写入所有模型的数据（计算所有）
        for result in sorted(results, key=lambda x: get_model_order(x['model'])):
            row = [result['model'], "100%"]
            metrics = result['all_metrics']
            row.extend([
                f"{metrics.get('ssim', 0):.4f}",
                f"{metrics.get('lpips_distance', 0):.4f}",
                f"{metrics.get('dino_similarity', 0):.4f}",
                f"{metrics.get('s_str', 0):.4f}",
                f"{metrics.get('clip_similarity', 0):.4f}",
                f"{metrics.get('text_similarity', 0):.4f}",
                f"{metrics.get('s_sty', 0):.4f}",
                f"{metrics.get('comprehensive_visual_score', 0):.4f}",
                f"{metrics.get('success_ratio', 0):.4f}",
                f"{metrics.get('visual_score', 0):.4f}"
            ])
            writer.writerow(row)
        
        # 写入空行分隔
        writer.writerow([])
        
        # 写入表头 - 只计算有效
        header_valid = ['Model', 'Count (Valid)', 'SSIM', 'LPIPS', 'DINO Similarity',
                        'S_str', 'CLIP', 'Text Similarity', 'S_sty', 'CVQI', 'Success Ratio', 'Visual Score']
        writer.writerow(header_valid)
        
        # 写入所有模型的数据（只计算有效）
        for result in sorted(results, key=lambda x: get_model_order(x['model'])):
            percentage = (result['valid_count'] / baseline_count * 100) if baseline_count > 0 else 0
            row = [result['model'], f"{percentage:.2f}%"]
            metrics = result['valid_metrics']
            row.extend([
                f"{metrics.get('ssim', 0):.4f}",
                f"{metrics.get('lpips_distance', 0):.4f}",
                f"{metrics.get('dino_similarity', 0):.4f}",
                f"{metrics.get('s_str', 0):.4f}",
                f"{metrics.get('clip_similarity', 0):.4f}",
                f"{metrics.get('text_similarity', 0):.4f}",
                f"{metrics.get('s_sty', 0):.4f}",
                f"{metrics.get('comprehensive_visual_score', 0):.4f}",
                f"{metrics.get('success_ratio', 0):.4f}",
                f"{metrics.get('visual_score', 0):.4f}"
            ])
            writer.writerow(row)
    
    print(f"Results saved to {output_path}")

def calculate_style_flow_metrics(all_data):
    """计算每个模型按 style、flow、context 分组的指标（只统计有效数据）"""
    # 按模型、style 分组
    model_style_data = defaultdict(lambda: defaultdict(list))
    
    # 按模型、flow 分组
    model_flow_data = defaultdict(lambda: defaultdict(list))
    
    # 按模型、context 分组
    model_context_data = defaultdict(lambda: defaultdict(list))
    
    # 整体 style 分组（不分模型）
    overall_style_data = defaultdict(list)
    
    # 整体 flow 分组（不分模型）
    overall_flow_data = defaultdict(list)
    
    # 整体 context 分组（不分模型）
    overall_context_data = defaultdict(list)
    
    for item in all_data:
        model = item['model']
        metadata = item['metadata']
        
        # 只处理有效数据
        if not is_valid_sequence(metadata):
            continue
        
        style = metadata.get('style', 'unknown')
        flow = metadata.get('flow', 'unknown')
        context = metadata.get('context', 'unknown')
        
        # 计算该任务的平均指标
        sequence = metadata.get('sequence', [])
        task_avg = {}
        
        if sequence:
            # 计算步骤级指标的平均值
            values_cvqi = [step.get('comprehensive_visual_score') for step in sequence if 'comprehensive_visual_score' in step and step.get('comprehensive_visual_score') is not None]
            if values_cvqi:
                task_avg['cvqi'] = sum(values_cvqi) / len(values_cvqi)
            else:
                task_avg['cvqi'] = 0.0
        else:
            task_avg['cvqi'] = 0.0
        
        # 添加全局指标
        task_avg['success_ratio'] = metadata.get('success_ratio', 0.0)
        task_avg['visual_score'] = metadata.get('visual_score', 0.0)
        
        # 存储到对应分组（按 style，分模型）
        model_style_data[model][style].append(task_avg)
        
        # 存储到对应分组（按 flow，分模型）
        model_flow_data[model][flow].append(task_avg)
        
        # 存储到对应分组（按 style，整体）
        overall_style_data[style].append(task_avg)
        
        # 存储到对应分组（按 flow，整体）
        overall_flow_data[flow].append(task_avg)
        
        # 存储到对应分组（按 context，分模型）
        model_context_data[model][context].append(task_avg)
        
        # 存储到对应分组（按 context，整体）
        overall_context_data[context].append(task_avg)
    
    # 计算按 style 分组的平均值（分模型）
    style_results = []
    for model in sorted(model_style_data.keys()):
        for style in sorted(model_style_data[model].keys()):
            task_metrics = model_style_data[model][style]
            
            # 计算平均值
            avg_cvqi = sum(t['cvqi'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
            avg_success_ratio = sum(t['success_ratio'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
            avg_visual_score = sum(t['visual_score'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
            
            style_results.append({
                'model': model,
                'group_type': 'style',
                'group_name': style,
                'count': len(task_metrics),
                'cvqi': avg_cvqi,
                'success_ratio': avg_success_ratio,
                'visual_score': avg_visual_score
            })
    
    # 计算按 flow 分组的平均值（分模型）
    flow_results = []
    for model in sorted(model_flow_data.keys()):
        for flow in sorted(model_flow_data[model].keys()):
            task_metrics = model_flow_data[model][flow]
            
            # 计算平均值
            avg_cvqi = sum(t['cvqi'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
            avg_success_ratio = sum(t['success_ratio'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
            avg_visual_score = sum(t['visual_score'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
            
            flow_results.append({
                'model': model,
                'group_type': 'flow',
                'group_name': flow,
                'count': len(task_metrics),
                'cvqi': avg_cvqi,
                'success_ratio': avg_success_ratio,
                'visual_score': avg_visual_score
            })
    
    # 计算整体 style 分组的平均值（不分模型）
    overall_style_results = []
    for style in sorted(overall_style_data.keys()):
        task_metrics = overall_style_data[style]
        
        # 计算平均值
        avg_cvqi = sum(t['cvqi'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
        avg_success_ratio = sum(t['success_ratio'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
        avg_visual_score = sum(t['visual_score'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
        
        overall_style_results.append({
            'group_type': 'style',
            'group_name': style,
            'count': len(task_metrics),
            'cvqi': avg_cvqi,
            'success_ratio': avg_success_ratio,
            'visual_score': avg_visual_score
        })
    
    # 计算整体 flow 分组的平均值（不分模型）
    overall_flow_results = []
    for flow in sorted(overall_flow_data.keys()):
        task_metrics = overall_flow_data[flow]
        
        # 计算平均值
        avg_cvqi = sum(t['cvqi'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
        avg_success_ratio = sum(t['success_ratio'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
        avg_visual_score = sum(t['visual_score'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
        
        overall_flow_results.append({
            'group_type': 'flow',
            'group_name': flow,
            'count': len(task_metrics),
            'cvqi': avg_cvqi,
            'success_ratio': avg_success_ratio,
            'visual_score': avg_visual_score
        })
    
    # 计算按 context 分组的平均值（分模型）
    context_results = []
    for model in sorted(model_context_data.keys()):
        for context in sorted(model_context_data[model].keys()):
            task_metrics = model_context_data[model][context]
            
            # 计算平均值
            avg_cvqi = sum(t['cvqi'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
            avg_success_ratio = sum(t['success_ratio'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
            avg_visual_score = sum(t['visual_score'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
            
            context_results.append({
                'model': model,
                'group_type': 'context',
                'group_name': context,
                'count': len(task_metrics),
                'cvqi': avg_cvqi,
                'success_ratio': avg_success_ratio,
                'visual_score': avg_visual_score
            })
    
    # 计算整体 context 分组的平均值（不分模型）
    overall_context_results = []
    for context in sorted(overall_context_data.keys()):
        task_metrics = overall_context_data[context]
        
        # 计算平均值
        avg_cvqi = sum(t['cvqi'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
        avg_success_ratio = sum(t['success_ratio'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
        avg_visual_score = sum(t['visual_score'] for t in task_metrics) / len(task_metrics) if task_metrics else 0.0
        
        overall_context_results.append({
            'group_type': 'context',
            'group_name': context,
            'count': len(task_metrics),
            'cvqi': avg_cvqi,
            'success_ratio': avg_success_ratio,
            'visual_score': avg_visual_score
        })
    
    return style_results, flow_results, context_results, overall_style_results, overall_flow_results, overall_context_results

def save_style_flow_metrics_to_csv(style_results, flow_results, context_results, overall_style_results, overall_flow_results, overall_context_results, output_path):
    """保存按 style、flow、context 分组的指标到CSV文件"""
    # 定义模型排序顺序
    MODEL_ORDER = [
        'gemini-3-pro',
        'gemini-3-flash',
        'Doubao-Seed-1.8',
        'doubao-seed-1-6',
        'gpt-4o',
        'Qwen3-VL-235B',
        'Qwen3-VL-30B',
        'Qwen3-VL-8B'
    ]
    
    def get_model_order(model_name):
        try:
            return MODEL_ORDER.index(model_name)
        except ValueError:
            return len(MODEL_ORDER)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入整体 Style 统计（不分模型）
        header_overall_style = ['=== OVERALL BY STYLE (ALL MODELS) ===', '', '', '', '', '']
        writer.writerow(header_overall_style)
        header_overall_style2 = ['Style', 'Count', 'CVQI', 'Success Ratio', 'Visual Score', '', '', '']
        writer.writerow(header_overall_style2)
        
        # 按 style 排序
        sorted_overall_style = sorted(overall_style_results, key=lambda x: x['group_name'])
        
        # 写入整体 style 统计
        for result in sorted_overall_style:
            row = [
                result['group_name'],
                result['count'],
                f"{result['cvqi']:.4f}",
                f"{result['success_ratio']:.4f}",
                f"{result['visual_score']:.4f}",
                '', '', ''
            ]
            writer.writerow(row)
        
        # 写入空行分隔
        writer.writerow([])
        
        # 写入整体 Flow 统计（不分模型）
        header_overall_flow = ['=== OVERALL BY FLOW (ALL MODELS) ===', '', '', '', '', '']
        writer.writerow(header_overall_flow)
        header_overall_flow2 = ['Flow', 'Count', 'CVQI', 'Success Ratio', 'Visual Score', '', '', '']
        writer.writerow(header_overall_flow2)
        
        # 按 flow 排序
        sorted_overall_flow = sorted(overall_flow_results, key=lambda x: x['group_name'])
        
        # 写入整体 flow 统计
        for result in sorted_overall_flow:
            row = [
                result['group_name'],
                result['count'],
                f"{result['cvqi']:.4f}",
                f"{result['success_ratio']:.4f}",
                f"{result['visual_score']:.4f}",
                '', '', ''
            ]
            writer.writerow(row)
        
        # 写入空行分隔
        writer.writerow([])
        
        # 写入整体 Context 统计（不分模型）
        header_overall_context = ['=== OVERALL BY CONTEXT (ALL MODELS) ===', '', '', '', '', '']
        writer.writerow(header_overall_context)
        header_overall_context2 = ['Context', 'Count', 'CVQI', 'Success Ratio', 'Visual Score', '', '', '']
        writer.writerow(header_overall_context2)
        
        # 按 context 排序
        sorted_overall_context = sorted(overall_context_results, key=lambda x: x['group_name'])
        
        # 写入整体 context 统计
        for result in sorted_overall_context:
            row = [
                result['group_name'],
                result['count'],
                f"{result['cvqi']:.4f}",
                f"{result['success_ratio']:.4f}",
                f"{result['visual_score']:.4f}",
                '', '', ''
            ]
            writer.writerow(row)
        
        # 写入空行分隔
        writer.writerow([])
        writer.writerow([])
        
        # 写入按 Style 分组的表头（分模型）
        header_style = ['=== BY STYLE (BY MODEL) ===', '', '', '', '', '']
        writer.writerow(header_style)
        header_style2 = ['Model', 'Style', 'Count', 'CVQI', 'Success Ratio', 'Visual Score']
        writer.writerow(header_style2)
        
        # 按 model、style 排序
        sorted_style_results = sorted(style_results, key=lambda x: (
            get_model_order(x['model']),
            x['group_name']
        ))
        
        # 写入按 style 分组的数据
        for result in sorted_style_results:
            row = [
                result['model'],
                result['group_name'],
                result['count'],
                f"{result['cvqi']:.4f}",
                f"{result['success_ratio']:.4f}",
                f"{result['visual_score']:.4f}"
            ]
            writer.writerow(row)
        
        # 写入空行分隔
        writer.writerow([])
        
        # 写入按 Flow 分组的表头（分模型）
        header_flow = ['=== BY FLOW (BY MODEL) ===', '', '', '', '', '']
        writer.writerow(header_flow)
        header_flow2 = ['Model', 'Flow', 'Count', 'CVQI', 'Success Ratio', 'Visual Score']
        writer.writerow(header_flow2)
        
        # 按 model、flow 排序
        sorted_flow_results = sorted(flow_results, key=lambda x: (
            get_model_order(x['model']),
            x['group_name']
        ))
        
        # 写入按 flow 分组的数据
        for result in sorted_flow_results:
            row = [
                result['model'],
                result['group_name'],
                result['count'],
                f"{result['cvqi']:.4f}",
                f"{result['success_ratio']:.4f}",
                f"{result['visual_score']:.4f}"
            ]
            writer.writerow(row)
        
        # 写入空行分隔
        writer.writerow([])
        
        # 写入按 Context 分组的表头（分模型）
        header_context = ['=== BY CONTEXT (BY MODEL) ===', '', '', '', '', '']
        writer.writerow(header_context)
        header_context2 = ['Model', 'Context', 'Count', 'CVQI', 'Success Ratio', 'Visual Score']
        writer.writerow(header_context2)
        
        # 按 model、context 排序
        sorted_context_results = sorted(context_results, key=lambda x: (
            get_model_order(x['model']),
            x['group_name']
        ))
        
        # 写入按 context 分组的数据
        for result in sorted_context_results:
            row = [
                result['model'],
                result['group_name'],
                result['count'],
                f"{result['cvqi']:.4f}",
                f"{result['success_ratio']:.4f}",
                f"{result['visual_score']:.4f}"
            ]
            writer.writerow(row)
    
    print(f"Style-Flow metrics saved to {output_path}")

def main():
    # 设置路径
    base_dir = "./build_dataset_multi_turn"
    output_csv = "./model_metrics_summary.csv"
    output_style_flow_csv = "./model_style_flow_metrics.csv"
    
    # 加载所有数据
    print("Loading all metadata files...")
    all_data = load_all_metadata(base_dir)
    print(f"Loaded {len(all_data)} tasks")
    
    # 按模型统计
    model_stats = defaultdict(int)
    for item in all_data:
        model_stats[item['model']] += 1
    
    print("\nModels found:")
    for model, count in sorted(model_stats.items()):
        print(f"  {model}: {count} tasks")
    
    # 计算指标平均值
    print("\nCalculating metrics...")
    results = calculate_metrics(all_data)
    
    # 保存到CSV
    print("\nSaving results to CSV...")
    save_to_csv(results, output_csv)
    
    # 计算按 style、flow、context 分组的指标
    print("\nCalculating style-flow-context metrics...")
    style_results, flow_results, context_results, overall_style_results, overall_flow_results, overall_context_results = calculate_style_flow_metrics(all_data)
    
    # 保存到新的CSV
    print("\nSaving style-flow-context metrics to CSV...")
    save_style_flow_metrics_to_csv(style_results, flow_results, context_results, overall_style_results, overall_flow_results, overall_context_results, output_style_flow_csv)
    
    # 打印摘要
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal models: {len(results)}")
    print(f"Total tasks: {len(all_data)}")
    
    print("\n" + "-"*80)
    print("ALL DATA (including empty sequences)")
    print("-"*80)
    for result in sorted(results, key=lambda x: x['model']):
        print(f"\n{result['model']} ({result['all_count']} tasks):")
        metrics = result['all_metrics']
        print(f"  SSIM: {metrics.get('ssim', 0):.4f}")
        print(f"  LPIPS: {metrics.get('lpips_distance', 0):.4f}")
        print(f"  DINO Similarity: {metrics.get('dino_similarity', 0):.4f}")
        print(f"  S_str: {metrics.get('s_str', 0):.4f}")
        print(f"  CLIP: {metrics.get('clip_similarity', 0):.4f}")
        print(f"  Text Similarity: {metrics.get('text_similarity', 0):.4f}")
        print(f"  S_sty: {metrics.get('s_sty', 0):.4f}")
        print(f"  CVQI: {metrics.get('comprehensive_visual_score', 0):.4f}")
        print(f"  Visual Score: {metrics.get('visual_score', 0):.4f}")
        print(f"  Success Ratio: {metrics.get('success_ratio', 0):.4f}")
    
    print("\n" + "-"*80)
    print("VALID DATA (only non-empty sequences)")
    print("-"*80)
    for result in sorted(results, key=lambda x: x['model']):
        print(f"\n{result['model']} ({result['valid_count']} valid tasks):")
        metrics = result['valid_metrics']
        print(f"  SSIM: {metrics.get('ssim', 0):.4f}")
        print(f"  LPIPS: {metrics.get('lpips_distance', 0):.4f}")
        print(f"  DINO Similarity: {metrics.get('dino_similarity', 0):.4f}")
        print(f"  S_str: {metrics.get('s_str', 0):.4f}")
        print(f"  CLIP: {metrics.get('clip_similarity', 0):.4f}")
        print(f"  Text Similarity: {metrics.get('text_similarity', 0):.4f}")
        print(f"  S_sty: {metrics.get('s_sty', 0):.4f}")
        print(f"  CVQI: {metrics.get('comprehensive_visual_score', 0):.4f}")
        print(f"  Visual Score: {metrics.get('visual_score', 0):.4f}")
        print(f"  Success Ratio: {metrics.get('success_ratio', 0):.4f}")

if __name__ == "__main__":
    main()
