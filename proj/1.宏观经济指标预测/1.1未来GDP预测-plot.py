# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------
# 1. 配置参数与路径
# ---------------------------------------------------------
base_dir = '/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/1.1GDP总量预测/v2_师姐约束'
output_fig = os.path.join(base_dir, 'gdp_final_comparison_trillion.png')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 2. 读取数据
# ---------------------------------------------------------
def load_data():
    scenarios = {
        '基准': 'gdp_predictions_基准_完整年份.csv',
        '低增长': 'gdp_predictions_低增长_完整年份.csv',
        '中增长': 'gdp_predictions_中增长_完整年份.csv',
        '高增长': 'gdp_predictions_高增长_完整年份.csv'
    }
    
    data_dict = {}
    for name, file in scenarios.items():
        path = os.path.join(base_dir, file)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            df.index = df.index.astype(int)
            data_dict[name] = df
        else:
            print(f"警告：未找到文件 {path}")
    return data_dict

# ---------------------------------------------------------
# 3. 核心绘图函数
# ---------------------------------------------------------
def plot_dual_gdp_trillion(data_dict):
    if not data_dict:
        return

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9), dpi=200)
    
    # 颜色与标记配置 (匹配参考图风格)
    style_cfg = {
        '低增长': {'color': '#1f77b4', 'label': '低增速情景', 'marker': 'o'},
        '中增长': {'color': '#008b45', 'label': '中增速情景', 'marker': 'o'},
        '高增长': {'color': '#ffa500', 'label': '高增速情景', 'marker': 'o'},
        '基准':   {'color': '#d62728', 'label': '历史趋势外推', 'marker': '.'}
    }

    plot_targets = [
        ('全国', ax1, '全国GDP (万亿元)'),
        ('京津冀及周边8省市', ax2, '京津冀及周边8省市GDP (万亿元)')
    ]

    for column, ax, ylabel in plot_targets:
        # --- 1. 绘制历史段 (2015-2024) ---
        hist_years = np.arange(2015, 2025)
        # 换算单位：除以 10000
        hist_values = data_dict['基准'].loc[hist_years, column] / 10000
        
        ax.plot(hist_years, hist_values, 
                color=style_cfg['基准']['color'], marker='.', markersize=7, 
                linewidth=1.8, zorder=5)

        # --- 2. 绘制各情景预测段 (2024-2035) ---
        pred_years = np.arange(2024, 2036)
        for scn_name, df in data_dict.items():
            cfg = style_cfg[scn_name]
            # 换算单位：除以 10000
            y_values = df.loc[pred_years, column] / 10000
            
            ax.plot(pred_years, y_values, 
                    color=cfg['color'], marker=cfg['marker'], markersize=4,
                    linewidth=1.3, label=cfg['label'], alpha=0.9)

        # --- 3. 细节修饰 ---
        # 垂直分界虚线 (2024年)
        ax.axvline(x=2024, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 坐标轴标签与网格
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_xlim(2015, 2035)
        ax.set_xticks(np.arange(2015, 2036, 5))
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        
        # 自动调整纵轴范围，留出一点顶部空间
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min * 0.98, y_max * 1.05)

        # 图例设置
        ax.legend(loc='best', fontsize=15, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    
    # 保存结果
    plt.savefig(output_fig, bbox_inches='tight')
    print(f"图像已保存，单位已切换为万亿元：{output_fig}")
    plt.show()

# ---------------------------------------------------------
# 4. 运行
# ---------------------------------------------------------
if __name__ == "__main__":
    gdp_data = load_data()
    plot_dual_gdp_trillion(gdp_data)