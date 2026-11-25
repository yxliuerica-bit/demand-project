# -*- coding: utf-8 -*-
"""
为每个省份绘制"汽车产量"的历史与预测趋势图
每个省份单独一张图，包含真实数据和预测数据
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 12  # 全局字体大小

def load_railway_data(data_dir):
    """
    加载汽车产量数据
    data_dir: 数据文件所在目录
    return: 包含所有省份数据的DataFrame
    """
    # 文件名和列名配置
    filename = '分省汽车产量完整结果.csv'
    history_col = '真实汽车产量'
    predict_col = '预测汽车产量'
    file_path = os.path.join(data_dir, filename)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 读取数据
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"读取文件失败: {str(e)}")
    
    # 检查必要的列是否存在
    required_cols = ['地区', '年份', history_col, predict_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"文件 {filename} 中缺少必要的列: {col}，现有列: {df.columns.tolist()}")
    
    # 处理年份数据类型
    df['年份'] = pd.to_numeric(df['年份'], errors='coerce')
    df = df.dropna(subset=['年份', '地区'])
    df['年份'] = df['年份'].astype(int)
    
    return df, history_col, predict_col

def plot_province_railway_trend(data, history_col, predict_col, output_dir):
    """
    为每个省份绘制汽车产量的历史与预测趋势图
    """
    # 创建输出目录
    indicator_dir = os.path.join(output_dir, '汽车产量')
    os.makedirs(indicator_dir, exist_ok=True)
    
    # 获取所有省份
    provinces = data['地区'].unique()
    
    for province in provinces:
        # 筛选该省份的数据
        prov_data = data[data['地区'] == province].sort_values('年份')
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 绘制历史数据
        history_data = prov_data.dropna(subset=[history_col])
        if not history_data.empty:
            plt.plot(history_data['年份'], history_data[history_col], 
                     'bo-', markersize=6, linewidth=2, 
                     label='历史汽车产量')
        
        # 绘制预测数据
        predict_data = prov_data.dropna(subset=[predict_col])
        if not predict_data.empty:
            plt.plot(predict_data['年份'], predict_data[predict_col], 
                     'r--', markersize=6, linewidth=2, 
                     label='预测汽车产量')
        
        # 设置图表属性
        plt.title(f'{province}汽车产量历史与预测趋势', fontsize=16, pad=15)
        plt.xlabel('年份', fontsize=14)
        plt.ylabel('汽车产量', fontsize=14)
        
        # 设置坐标轴范围
        all_years = prov_data['年份'].unique()
        if len(all_years) > 0:
            plt.xlim(min(all_years) - 1, max(all_years) + 1)
        
        # 计算y轴上限
        y_max = 0
        y_max = max(y_max, prov_data[history_col].dropna().max())
        y_max = max(y_max, prov_data[predict_col].dropna().max())
        
        if y_max > 0:
            plt.ylim(0, y_max * 1.1)
        
        # 添加网格和图例
        plt.grid(True, linestyle='--', alpha=0.7)
        if len(plt.gca().get_legend_handles_labels()[0]) > 0:
            plt.legend(fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(indicator_dir, f'{province}_汽车产量趋势图.png')
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存: {save_path}")
        except Exception as e:
            print(f"保存{province}汽车产量图表失败: {str(e)}")
        
        plt.close()

def main():
    # 数据目录（CSV文件所在目录）
    data_dir = '/lustre/home/litong/intern/京津冀货运发展/output/predict_results/Sce_Pop_BAU_GDP_BAU'
    
    # 输出图表目录
    output_root = os.path.join(data_dir, '分省份指标趋势图')
    os.makedirs(output_root, exist_ok=True)
    
    try:
        print("开始处理汽车产量数据...")
        
        # 加载数据
        data, history_col, predict_col = load_railway_data(data_dir)
        
        # 绘制每个省份的趋势图
        plot_province_railway_trend(data, history_col, predict_col, output_root)
        
        print("汽车产量指标处理完成")
    
    except Exception as e:
        print(f"程序出错: {str(e)}")

if __name__ == "__main__":
    main()
    