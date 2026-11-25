# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# -------------------------- 1. 基础配置 --------------------------
PROVINCE_DATA_DIR = "/lustre/home/litong/intern/京津冀货运发展/output/predict_results/Sce_Pop_BAU_GDP_BAU/5/省份完整数据"
TARGET_PROVINCES = ["浙江省", "江苏省", "北京市"]
FACTOR_NAME = "新增建筑面积"

# 绘图配置
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# 创建输出目录
OUTPUT_DIR = os.path.join(PROVINCE_DATA_DIR, "对比分析图")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------- 2. 数据读取与预处理 --------------------------
def load_province_data(data_dir, target_provinces, factor_name):
    """读取并合并各省数据，确保关键列存在"""
    all_data = []
    
    for province in target_provinces:
        # 查找省份数据文件
        file_pattern = f"{province}{factor_name}完整数据.csv"
        file_path = os.path.join(data_dir, file_pattern)
        
        # 查找文件的容错机制
        if not os.path.exists(file_path):
            search_pattern = os.path.join(data_dir, f"{province}*{factor_name}*.csv")
            matching_files = glob.glob(search_pattern)
            if not matching_files:
                raise FileNotFoundError(f"未找到{province}的数据文件: {file_path}")
            file_path = matching_files[0]
        
        # 读取数据并标准化
        df = pd.read_csv(file_path, encoding='utf-8')
        df['地区'] = province  # 确保地区名称正确
        df['年份'] = pd.to_numeric(df['年份'], errors='coerce').fillna(0).astype(int)
        
        # 确保关键列存在
        required_cols = ['GDP', '人口', '第三产业GDP', '城镇人口']
        for col in required_cols:
            if col not in df.columns:
                # 尝试相似列名匹配
                similar_cols = [c for c in df.columns if col in c]
                if similar_cols:
                    df[col] = df[similar_cols[0]]
                else:
                    df[col] = 0  # 缺失值填充为0
        
        # 处理目标指标列
        if f'原始_{factor_name}' in df.columns:
            df['历史值'] = df[f'原始_{factor_name}']
        elif f'平滑_{factor_name}' in df.columns:
            df['历史值'] = df[f'平滑_{factor_name}']
        else:
            df['历史值'] = 0
            
        if f'预测_原始数据_{factor_name}' in df.columns:
            df['预测值'] = df[f'预测_原始数据_{factor_name}']
        elif f'预测_平滑数据_{factor_name}' in df.columns:
            df['预测值'] = df[f'预测_平滑数据_{factor_name}']
        else:
            df['预测值'] = 0
        
        all_data.append(df)
    
    # 合并数据并清理
    df_combined = pd.concat(all_data, ignore_index=True)
    return df_combined[['地区', '年份', 'GDP', '人口', '第三产业GDP', '城镇人口', '历史值', '预测值']]


# -------------------------- 3. 生成所有图表 --------------------------
def generate_all_charts(df):
    """生成所有要求的分析图表"""
    # 1. 三省趋势对比图（基础图表）
    plot_trend_comparison(df)
    
    # 2. 特征相关性热力图
    plot_correlation_heatmap(df)
    
    # 3. 关键特征对比图
    plot_feature_comparison(df)
    
    # 4. 特征与目标指标散点图
    plot_feature_scatter(df)


def plot_trend_comparison(df):
    """绘制三省趋势对比图"""
    plt.figure(figsize=(12, 6))
    styles = {
        "浙江省": ("#1f77b4", "o-"),
        "江苏省": ("#ff7f0e", "s-"),
        "北京市": ("#2ca02c", "^-")
    }
    
    for province, (color, style) in styles.items():
        prov_data = df[df['地区'] == province].sort_values('年份')
        
        # 绘制历史数据
        hist_data = prov_data[prov_data['年份'] < 2025]
        plt.plot(hist_data['年份'], hist_data['历史值'], style, color=color,
                 linewidth=2, markersize=5, label=f'{province}（历史）')
        
        # 绘制预测数据
        pred_data = prov_data[prov_data['年份'] >= 2025]
        plt.plot(pred_data['年份'], pred_data['预测值'], style, color=color,
                 linewidth=2, markersize=5, alpha=0.7, label=f'{province}（预测）')
    
    plt.axvline(x=2025, color='gray', linestyle='--', label='预测起点')
    plt.xlabel('年份')
    plt.ylabel(f'{FACTOR_NAME}（万平方米）')
    plt.title('三省新增建筑面积趋势对比')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '三省趋势对比图.png'), bbox_inches='tight')
    plt.close()
    print("已生成：三省趋势对比图")


def plot_correlation_heatmap(df):
    """绘制特征相关性热力图"""
    # 使用历史数据计算相关性
    hist_data = df[df['年份'] < 2025].copy()
    if len(hist_data) < 5:  # 确保有足够数据
        hist_data = df.copy()  # 数据不足时使用全部数据
    
    # 选择用于计算相关性的列
    corr_cols = ['GDP', '人口', '第三产业GDP', '城镇人口', '历史值']
    corr_data = hist_data[corr_cols].dropna()
    
    if len(corr_data) < 3:
        print("警告：数据不足，无法生成相关性热力图")
        return
    
    # 计算相关性
    corr_matrix = corr_data.corr()
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # 添加文本标签
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center', color='black')
    
    plt.colorbar(label='相关系数')
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45)
    plt.yticks(range(len(corr_cols)), corr_cols)
    plt.title('特征与新增建筑面积相关性热力图')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '相关性热力图.png'))
    plt.close()
    print("已生成：相关性热力图")


def plot_feature_comparison(df):
    """绘制关键特征对比图"""
    # 选择2025年或最近的预测年份
    target_year = 2025
    if target_year not in df['年份'].unique():
        target_year = df[df['年份'] >= 2025]['年份'].min() if len(df[df['年份'] >= 2025]) > 0 else df['年份'].max()
    
    year_data = df[df['年份'] == target_year]
    if len(year_data) < 3:  # 确保有三个省份的数据
        year_data = df.groupby('地区').last().reset_index()  # 使用最新数据
    
    # 绘制两个关键特征
    features = ['GDP', '城镇人口']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, feature in enumerate(features):
        axes[i].bar(year_data['地区'], year_data[feature], color=colors)
        axes[i].set_title(f'{target_year}年{feature}对比')
        axes[i].set_ylabel('数值' + ('（亿元）' if 'GDP' in feature else '（万人）'))
        # 添加数值标签
        for idx, val in enumerate(year_data[feature]):
            axes[i].text(idx, val*1.02, f'{int(val)}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '关键特征对比图.png'))
    plt.close()
    print("已生成：关键特征对比图")


def plot_feature_scatter(df):
    """绘制特征与目标指标的散点图（分析影响因素）"""
    features = ['GDP', '城镇人口']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    markers = {'浙江省': 'o', '江苏省': 's', '北京市': '^'}
    colors = {'浙江省': "#1f77b4", '江苏省': "#ff7f0e", '北京市': "#2ca02c"}
    
    for i, feature in enumerate(features):
        for province in TARGET_PROVINCES:
            prov_data = df[df['地区'] == province]
            axes[i].scatter(prov_data[feature], prov_data['历史值'], 
                           label=province, marker=markers[province], 
                           color=colors[province], alpha=0.7)
        
        axes[i].set_xlabel(feature + ('（亿元）' if 'GDP' in feature else '（万人）'))
        axes[i].set_ylabel(f'{FACTOR_NAME}（万平方米）')
        axes[i].set_title(f'{feature}与新增建筑面积关系')
        axes[i].legend()
        axes[i].grid(linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '特征影响散点图.png'))
    plt.close()
    print("已生成：特征影响散点图")


# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    try:
        # 加载数据
        print("正在读取数据...")
        df = load_province_data(PROVINCE_DATA_DIR, TARGET_PROVINCES, FACTOR_NAME)
        
        # 生成所有图表
        print("正在生成分析图表...")
        generate_all_charts(df)
        
        print(f"\n所有图表已保存至：{OUTPUT_DIR}")
        print("包含以下图表：")
        print("1. 三省趋势对比图")
        print("2. 相关性热力图")
        print("3. 关键特征对比图")
        print("4. 特征影响散点图")
        
    except Exception as e:
        print(f"执行出错：{str(e)}")
    