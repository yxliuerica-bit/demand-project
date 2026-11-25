# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================================================
# 1. 配置参数
# =========================================================
# 请确保这些基础路径与您之前保存 CSV 的路径完全一致
paths = {
    'GDP总量': '/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/1.1GDP总量预测/v2_师姐约束/gdp_predictions_基准_完整年份.csv',
    '第三产业GDP': '/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/1.3第三产业GDP预测/third_gdp_predictions_基准_完整年份.csv',
    '总人口': '/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/1.2总人口预测/v2_师姐约束/population_predictions_基准_完整年份.csv',
    '城镇人口': '/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/1.4城镇人口预测/urban_population_predictions_基准_完整年份.csv'
}

# 输出目录
save_dir = '/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/饼图'
os.makedirs(save_dir, exist_ok=True)

# 指定 8 省市名单（绘图顺序）
provinces = ['北京市', '天津市', '河北省', '河南省', '山东省', '山西省', '内蒙古自治区', '陕西省']

# 设置配色（匹配示例图的柔和色调）
colors = ['#e1f0f9', '#b3cde3', '#e4f4d1', '#b2df8a', '#fbb4ae', '#e78ac3', '#fee0b6', '#fdbf6f']

# 设置字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 2. 核心绘图函数
# =========================================================
def draw_dashboard(indicator_name, file_path):
    if not os.path.exists(file_path):
        print(f"跳过 {indicator_name}: 找不到文件 {file_path}")
        return

    # 读取数据
    df = pd.read_csv(file_path, index_col=0)
    df.index = df.index.astype(int)
    
    # 提取 8 省市 2024-2035 数据
    plot_years = np.arange(2024, 2036)
    data_plot = df.loc[plot_years, provinces].astype(float)
    
    # 根据指标调整单位（GDP 类除以 10000 变万亿，人口类除以 10000 变亿人）
    is_gdp = 'GDP' in indicator_name
    unit = '万亿元' if is_gdp else '亿人'
    data_plot = data_plot / 10000

    # 创建画布 (1行2列)
    fig = plt.figure(figsize=(12, 5), dpi=150)
    
    # --- A. 左侧饼图 (2035年占比) ---
    ax1 = fig.add_subplot(1, 2, 1)
    data_2035 = data_plot.loc[2035]
    wedges, texts, autotexts = ax1.pie(
        data_2035, labels=provinces, autopct='%1.1f%%',
        startangle=90, colors=colors, pctdistance=0.75,
        textprops={'fontsize': 14}
    )
    ax1.set_title(f"8省市{indicator_name}预测构成 (2035年)", fontsize=14, pad=20, weight='bold')

    # --- B. 右侧堆叠柱状图 (2024-2035趋势) ---
    ax2 = fig.add_subplot(1, 2, 2)
    bottom = np.zeros(len(plot_years))
    
    for i, prov in enumerate(provinces):
        values = data_plot[prov].values
        ax2.bar(plot_years, values, bottom=bottom, color=colors[i], label=prov, width=0.7)
        bottom += values
    
    # 在柱状图顶部标注总数
    totals = data_plot.sum(axis=1)
    for year, total in zip(plot_years, totals):
        ax2.text(year, total + (totals.max()*0.02), f'{total:.1f}', 
                 ha='center', va='bottom', fontsize=14, rotation=90)

    ax2.set_ylabel(f"{indicator_name} ({unit})", fontsize=14)
    ax2.set_xticks(plot_years)
    ax2.set_xticklabels(plot_years, rotation=90)
    ax2.set_ylim(0, totals.max() * 1.2)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 图例
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=14)

    # 整体修饰
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_dir, f'{indicator_name}_看板.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"已生成看板: {save_path}")
    plt.show()

# =========================================================
# 3. 循环生成
# =========================================================
if __name__ == "__main__":
    for name, path in paths.items():
        draw_dashboard(name, path)