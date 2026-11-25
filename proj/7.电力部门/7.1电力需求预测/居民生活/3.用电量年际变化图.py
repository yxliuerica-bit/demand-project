import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 初始化设置 ---
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
input_file = "total_residential_electricity_calibrated.csv"
output_image = "京津冀及周边8省份居民生活用电需求.png"

# 定义省份列表（按照你参考图中的顺序或默认顺序）
provlist_bthsa = ['北京市', '天津市', '河北省', '河南省', '山东省', '山西省', '内蒙古自治区', '陕西省']
colors = ['#DEECF6','#AFC8E2','#E2F2CD','#B6DAA7','#F9D5D5','#EF98A1','#FBE3C0','#FBC99A']

# --- 2. 读取与预处理数据 ---
try:
    df = pd.read_csv(input_file)
    # 筛选年份范围（参考图中为2010-2035）
    years = np.arange(2010, 2036, 1)
    df_subset = df[df['Year'].isin(years)].set_index('Year')
    
    # 将单位从 亿kWh 转换为 万亿kWh (根据你附图量级调整，
    # 如果原始单位就是亿kWh且数值在0-5之间，则无需除以1000。
    # 这里假设附图纵轴单位是“千亿千瓦时”，请根据需要调整缩放)
    scale = 1000  # 1万亿kWh = 1000千亿kWh
    df_plot = df_subset / scale 
    unit_label = "千亿千瓦时" 

    # --- 3. 绘图初始化 ---
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4)) # 调整比例以适应双轴

    # --- 4. 绘制左轴：分省堆叠柱状图 ---
    bottom = np.zeros(len(years))
    for i, province in enumerate(provlist_bthsa):
        province_data = df_plot[province].fillna(0).clip(lower=0)
        ax1.bar(years, province_data, bottom=bottom, color=colors[i], 
                width=0.7, alpha=0.9, label=province)
        bottom += province_data

    # 设置左轴属性
    ax1.set_ylabel(f'分省居民生活用电需求 ({unit_label})', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, bottom.max() * 1.2) # 自动留白

    # --- 5. 绘制右轴：全国趋势折线图 ---
    ax2 = ax1.twinx()
    national_data = df_plot['全国'].fillna(0).clip(lower=0)
    line_plot, = ax2.plot(years, national_data, color='#d62728', marker='o', 
                          markersize=4, linewidth=1.5, label='全国')
    
    ax2.set_ylabel(f'全国居民生活用电需求 ({unit_label})', fontsize=10)
    ax2.set_ylim(0, national_data.max() * 1.2)
    ax2.grid(visible=False) # 关闭右轴网格线

    # --- 6. 添加全国线的数值标签 ---
    # 每隔 5 年标注一次，或标注关键年份（2010, 2015, 2020, 2025, 2030, 2035）
    label_years = [2010, 2015, 2020, 2025, 2030, 2035]
    for y in label_years:
        if y in national_data.index:
            val = national_data.loc[y]
            ax2.text(y, val + (national_data.max() * 0.03), f'{val:.1f}', 
                     ha='center', va='bottom', fontsize=8, color='#d62728', fontweight='bold')

    # --- 7. 轴刻度与图例合并 ---
    ax1.set_xticks(np.arange(2010, 2036, 5))
    
    # 获取柱状图和折线图的句柄和标签
    handles_bar, labels_bar = ax1.get_legend_handles_labels()
    # 将全国折线加入图例
    ax1.legend(handles_bar + [line_plot], labels_bar + ['全国'], 
               loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=9, frameon=True)

    # 移除顶部边框
    plt.gca().spines['top'].set_visible(False)

    # --- 8. 保存输出 ---
    plt.tight_layout()
    plt.savefig(output_image, dpi=500, bbox_inches='tight')
    plt.show()
    print(f"居民生活用电需求图表生成完成: {output_image}")

except Exception as e:
    print(f"绘图过程中出错: {e}")