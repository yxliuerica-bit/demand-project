# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================================================
# 1. 配置参数 (请根据实际情况修改路径)
# =========================================================
# 数据读取路径
base_dir = '/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/1.3第三产业GDP预测'
# 图片保存路径
output_fig = os.path.join(base_dir, 'third_gdp_comparison_final_plot.png')

# 字体大小统一调整
FONT_SIZE_LABEL = 12    # 坐标轴标签大小
FONT_SIZE_TICK = 10     # 刻度数字大小
FONT_SIZE_LEGEND = 18   # 图例文字大小

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 2. 加载数据函数
# =========================================================
def load_scenario_data():
    scenarios = {
        '基准': 'third_gdp_predictions_基准_完整年份.csv',
        '低增长': 'third_gdp_predictions_低增长.csv',   # 预测期数据
        '中增长': 'third_gdp_predictions_中增长.csv',
        '高增长': 'third_gdp_predictions_高增长.csv'
    }
    
    data_dict = {}
    # 首先读取基准（含历史）作为底表
    base_full = pd.read_csv(os.path.join(base_dir, scenarios['基准']), index_col=0)
    base_full.index = base_full.index.astype(int)
    data_dict['基准'] = base_full
    
    # 历史数据部分 (2015-2024)
    history_data = base_full.loc[2015:2024]

    # 读取其他情景并合并历史，确保预测线从2024年连续分叉
    for name in ['低增长', '中增长', '高增长']:
        path = os.path.join(base_dir, scenarios[name])
        if os.path.exists(path):
            df_pred = pd.read_csv(path, index_col=0)
            df_pred.index = df_pred.index.astype(int)
            # 合并历史(2015-2023) + 预测(2024-2035) 确保连接点连续
            combined = pd.concat([history_data.loc[2015:2023], df_pred.loc[2024:2035]])
            data_dict[name] = combined
            
    return data_dict

# =========================================================
# 3. 核心绘图程序
# =========================================================
def draw_comparison_plot(data_dict):
    # 创建 2行1列 的画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), dpi=200)
    
    # 配色与样式说明
    # 历史趋势/基准: 红色; 低增长: 蓝色; 中增长: 绿色; 高增长: 橙色
    style_cfg = {
        '低增长': {'color': '#1f77b4', 'label': '低增速情景', 'marker': 'o'},
        '中增长': {'color': '#008b45', 'label': '中增速情景', 'marker': 'o'},
        '高增长': {'color': '#ffa500', 'label': '高增速情景', 'marker': 'o'},
        '基准':   {'color': '#d62728', 'label': '历史趋势外推', 'marker': '.'}
    }

    targets = [
        ('全国', ax1, '全国第三产业GDP (万亿元)'),
        ('京津冀及周边8省市', ax2, '京津冀及周边8省市第三产业GDP (万亿元)')
    ]

    for col_name, ax, ylabel in targets:
        # --- A. 绘制历史实测线 (2015-2024) ---
        # 使用基准情景的数据作为历史展示，除以10000换算为万亿
        hist_years = np.arange(2015, 2025)
        hist_vals = data_dict['基准'].loc[hist_years, col_name] / 10000
        ax.plot(hist_years, hist_vals, color=style_cfg['基准']['color'], 
                marker='.', markersize=8, linewidth=2, zorder=5)

        # --- B. 绘制各情景预测分支 (2024-2035) ---
        pred_years = np.arange(2024, 2036)
        for scn in ['低增长', '中增长', '高增长', '基准']:
            cfg = style_cfg[scn]
            # 换算为万亿
            y_vals = data_dict[scn].loc[pred_years, col_name] / 10000
            
            ax.plot(pred_years, y_vals, color=cfg['color'], 
                    marker=cfg['marker'], markersize=4, 
                    linewidth=1.2, label=cfg['label'], alpha=0.8)

        # --- C. 细节修饰 ---
        # 2024垂直分界线
        ax.axvline(x=2024, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 坐标轴设置
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_xlim(2015, 2035)
        ax.set_xticks(range(2015, 2036, 5))
        ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
        
        # 网格与背景
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # 图例：第一个子图放在左上，第二个放在左下（避开数据点）
        loc = 'upper left' if col_name == '全国' else 'lower left'
        ax.legend(loc=loc, fontsize=FONT_SIZE_LEGEND, frameon=True, shadow=False)

    # 自动调整布局
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.22) # 调整上下子图间距
    
    # 保存图片
    plt.savefig(output_fig, bbox_inches='tight')
    print(f">>> 成功绘制对比折线图！保存路径: {output_fig}")
    plt.show()

# =========================================================
# 4. 执行
# =========================================================
if __name__ == "__main__":
    try:
        data = load_scenario_data()
        draw_comparison_plot(data)
    except Exception as e:
        print(f"运行绘图脚本时出错: {e}")