# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================================================
# 1. 配置参数 (请根据实际情况确认路径)
# =========================================================
# 数据读取路径 (与您预测脚本中的 out_dir 一致)
base_dir = '/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/1.4城镇人口预测'
# 图片保存路径
output_fig = os.path.join(base_dir, 'urban_population_scenarios_final_plot.png')

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
def load_urban_population_data():
    scenarios = {
        '基准': 'urban_population_predictions_基准_完整年份.csv',
        '低增长': 'urban_population_predictions_低增长.csv',   # 预测期模型文件
        '中增长': 'urban_population_predictions_中增长.csv',
        '高增长': 'urban_population_predictions_高增长.csv'
    }
    
    data_dict = {}
    
    # 1. 加载基准情景（含2019-2024历史段）
    base_full_path = os.path.join(base_dir, scenarios['基准'])
    if not os.path.exists(base_full_path):
        raise FileNotFoundError(f"未找到基准文件: {base_full_path}")
        
    base_full = pd.read_csv(base_full_path, index_col=0)
    base_full.index = base_full.index.astype(int)
    data_dict['基准'] = base_full
    
    # 2. 提取历史段 (2019-2024) 用于其他情景的拼接
    history_part = base_full.loc[2019:2024]

    # 3. 读取约束情景并拼接历史，确保从2024年开始平滑分叉
    for name in ['低增长', '中增长', '高增长']:
        path = os.path.join(base_dir, scenarios[name])
        if os.path.exists(path):
            df_pred = pd.read_csv(path, index_col=0)
            df_pred.index = df_pred.index.astype(int)
            # 拼接：历史(2019-2023) + 预测(2024-2035)
            # 2024年作为公共衔接点
            combined = pd.concat([history_part.loc[2019:2023], df_pred.loc[2024:2035]])
            data_dict[name] = combined
            
    return data_dict

# =========================================================
# 3. 核心绘图程序
# =========================================================
def draw_urban_population_plot(data_dict):
    # 创建 2行1列 的画布
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), dpi=200)
    
    # 配色样式说明 (参考图风格)
    # 历史趋势/基准: 红色; 低增长: 蓝色; 中增长: 绿色; 高增长: 橙色
    style_cfg = {
        '低增长': {'color': '#1f77b4', 'label': '低增长情景', 'marker': 'o'},
        '中增长': {'color': '#008b45', 'label': '中增长情景', 'marker': 'o'},
        '高增长': {'color': '#ffa500', 'label': '高增长情景', 'marker': 'o'},
        '基准':   {'color': '#d62728', 'label': '历史趋势外推', 'marker': '.'}
    }

    targets = [
        ('全国', ax1, '全国城镇人口 (亿人)'),
        ('京津冀及周边8省市', ax2, '京津冀及周边8省市城镇人口 (亿人)')
    ]

    for col_name, ax, ylabel in targets:
        # --- A. 绘制历史段 (2019-2024) ---
        # 单位：万人 -> 亿人 (除以 10000)
        hist_years = np.arange(2019, 2025)
        hist_vals = data_dict['基准'].loc[hist_years, col_name] / 10000
        # 绘制主干基准线（红色实线加点）
        ax.plot(hist_years, hist_vals, color=style_cfg['基准']['color'], 
                marker='.', markersize=8, linewidth=2, zorder=5)

        # --- B. 绘制预测分支 (2024-2035) ---
        pred_years = np.arange(2024, 2036)
        for scn in ['低增长', '中增长', '高增长', '基准']:
            cfg = style_cfg[scn]
            y_vals = data_dict[scn].loc[pred_years, col_name] / 10000
            
            ax.plot(pred_years, y_vals, color=cfg['color'], 
                    marker=cfg['marker'], markersize=4, 
                    linewidth=1.2, label=cfg['label'], alpha=0.8)

        # --- C. 细节装饰 ---
        # 1. 2024垂直分界线
        ax.axvline(x=2024, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 2. 坐标轴与标签
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL, fontweight='bold')
        ax.set_xlim(2019, 2035)
        ax.set_xticks(range(2019, 2036, 4))
        ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
        
        # 3. 背景网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # 4. 图例
        ax.legend(loc='lower left', fontsize=FONT_SIZE_LEGEND, frameon=True)

    # 布局优化
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.22)
    
    # 保存并显示
    plt.savefig(output_fig, bbox_inches='tight')
    print(f">>> 城镇人口情景对比图已保存至: {output_fig}")
    plt.show()

# =========================================================
# 4. 执行
# =========================================================
if __name__ == "__main__":
    try:
        urban_data = load_urban_population_data()
        draw_urban_population_plot(urban_data)
    except Exception as e:
        print(f"运行绘图出错: {e}")