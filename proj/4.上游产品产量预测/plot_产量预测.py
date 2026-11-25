# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============ 1. 路径配置 ============
scenario_name = '基准'
# 输入文件路径
steel_path  = f"/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_{scenario_name}/分省粗钢产量_历史加预测.xlsx"
cement_path = f"/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_{scenario_name}/分省水泥产量_历史加预测.xlsx"
power_path  = f"/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_{scenario_name}/分省煤炭产量_历史加预测.xlsx" # 新增火电路径

out_fig_dir = f"/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_{scenario_name}/预测图_最终优化版"
Path(out_fig_dir).mkdir(parents=True, exist_ok=True)

# ============ 2. 绘图环境配置 ============
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False

# 绘图指定的 8 省市顺序
PROV_8_LIST = ['北京市', '天津市', '河北省', '河南省', '山东省', '山西省', '内蒙古自治区', '陕西省']
# 堆叠图颜色
STACK_COLORS = ['#DEECF6','#AFC8E2','#E2F2CD','#B6DAA7','#F9D5D5','#EF98A1','#FBE3C0','#FBC99A']

# ============ 3. 绘图核心函数 ============

def plot_production_forecast_final(xlsx_path, prod_name, out_png):
    """
    绘制产量预测图：8 省堆叠 + 全国折线 (风格对齐需求量图)
    """
    if not os.path.exists(xlsx_path):
        print(f"⚠️ 跳过：找不到文件 {xlsx_path}")
        return

    # 1. 行业适配逻辑 (与代码二保持一致)
    # 根据产品名称匹配精度和特殊年份
    # 映射关系：粗钢->钢铁, 水泥->水泥, 煤炭->火电
    mapping = {"粗钢": "钢铁", "水泥": "水泥", "煤炭": "火电"}
    internal_key = mapping.get(prod_name, "钢铁")
    
    precision_map = {'钢铁': '.1f', '水泥': '.1f', '火电': '.3f'}
    fmt = precision_map.get(internal_key, '.1f')
    special_year = 2024 if internal_key == '火电' else 2025

    # 2. 读取与预处理
    df = pd.read_excel(xlsx_path, index_col=0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 单位转换：吨 -> 亿吨
    scale = 1e8
    df_unit = df / scale
    years = df_unit.index.values
    
    # 3. 创建画布 (风格对齐代码二的 7x3.5)
    fig, ax1 = plt.subplots(figsize=(7, 3.5), dpi=300)
    
    # A. 绘制 8 省堆叠柱状图 (调整 width 为 0.7，对齐代码二)
    available_8 = [c for c in PROV_8_LIST if c in df_unit.columns]
    bottom = np.zeros(len(years))
    for i, prov in enumerate(available_8):
        vals = df_unit[prov].clip(lower=0).values
        ax1.bar(years, vals, bottom=bottom, color=STACK_COLORS[i % len(STACK_COLORS)], 
                width=0.7, alpha=0.9, label=prov) # 去掉 edgecolor 增加干净度
        bottom += vals

    ax1.set_ylabel(f'分省{prod_name}产量 (亿吨)', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # B. 绘制全国总量折线图 (右轴, 颜色与 marker 对齐代码二)
    ax2 = ax1.twinx()
    nat_series = df_unit['全国'].clip(lower=0) if '全国' in df_unit.columns else df_unit[available_8].sum(axis=1)
    
    line_nat, = ax2.plot(years, nat_series, color='#d62728', marker='o', 
                         markersize=4, linewidth=1.2, label='全国总量', zorder=10)
    
    ax2.set_ylabel(f'全国{prod_name}总量 (亿吨)', fontsize=9)
    # 裕量从 1.4 缩减为 1.25，防止图像显得太“矮”
    ax2.set_ylim(0, nat_series.max() * 1.25)
    
    # C. 数值标注逻辑 (核心修改：5年一标 + 特殊年份红粗标注)
    for i, yr in enumerate(years):
        val = nat_series.iloc[i]
        if val > 0:
            # 判断逻辑：5年倍数 或者是 行业的特殊年份
            if (yr % 5 == 0) or (yr == special_year):
                is_special = (yr == special_year)
                # 垂直偏移量系数对齐代码二的 0.03
                ax2.text(yr, val + (nat_series.max() * 0.03), 
                         f'{val:{fmt}}', 
                         ha='center', va='bottom', 
                         fontsize=7, 
                         fontweight='bold' if is_special else 'normal',
                         color='#d62728' if is_special else 'black')

    # D. 细节微调
    ax1.set_xticks(np.arange(2010, 2036, 5))
    ax2.grid(visible=False) # 确保右轴不重复画网格
    
    # 合并图例 (对齐代码二：上左放置，bbox_to_anchor 调整)
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1 + [line_nat], l1 + ['全国总量'], 
               loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=8)

    plt.title(f"2010-2035年 {prod_name}产量预测 (单位: 亿吨)", fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

# ============ 4. 执行流程 ============

def main():
    # 1. 钢铁预测图
    plot_production_forecast_final(steel_path, "粗钢", os.path.join(out_fig_dir, "京津冀及周边8省_粗钢产量预测_最终优化版.png"))
    
    # 2. 水泥预测图
    plot_production_forecast_final(cement_path, "水泥", os.path.join(out_fig_dir, "京津冀及周边8省_水泥产量预测_最终优化版.png"))
    
    # 3. 火电（煤炭）预测图 - 新增执行逻辑
    plot_production_forecast_final(power_path, "煤炭", os.path.join(out_fig_dir, "京津冀及周边8省_煤炭产量预测_最终优化版.png"))

if __name__ == "__main__":
    main()