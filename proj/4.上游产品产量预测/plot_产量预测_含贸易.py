# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============ 1. 路径配置 ============
scenario_name = '基准'
BASE_DIR = "/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据"
steel_path  = f"/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_{scenario_name}/分省粗钢产量_历史加预测.xlsx"
cement_path = f"/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_{scenario_name}/分省水泥产量_历史加预测.xlsx"
power_path  = f"/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_{scenario_name}/分省煤炭产量_历史加预测.xlsx"

EXP_FILE = os.path.join(BASE_DIR, "上游产品出口年度数据.xlsx")
IMP_FILE = os.path.join(BASE_DIR, "上游产品进口年度数据.xlsx")

out_fig_dir = f"/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_{scenario_name}/预测图_最终优化版"
Path(out_fig_dir).mkdir(parents=True, exist_ok=True)

# ============ 2. 绘图环境配置 ============
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False


# ============ 2. 数据读取与外推函数 ============

def load_net_export_and_extrapolate(target_years):
    """
    读取贸易数据，计算净出口，并外推至 2035 年
    target_years: 绘图需要的完整年份序列 (如 2010-2035)
    """
    mapping = {
        "钢铁": {"exp": "钢材出口数量 (万吨)", "imp": "钢材进口数量 (万吨)"},
        "水泥": {"exp": "水泥及水泥熟料出口数量 (万吨)", "imp": None},
        "煤炭": {"exp": "煤出口数量 (万吨)", "imp": "煤进口数量 (万吨)"}
    }
    
    df_exp = pd.read_excel(EXP_FILE)
    df_imp = pd.read_excel(IMP_FILE)
    
    # 清洗列名
    df_exp.columns = [str(c).strip() for c in df_exp.columns]
    df_imp.columns = [str(c).strip() for c in df_imp.columns]
    
    # 提取年份
    for df in [df_exp, df_imp]:
        time_col = next((c for c in df.columns if '时间' in c), None)
        df['year'] = df[time_col].astype(str).str.extract(r'(\d{4})').astype(int)

    res = {}
    for category, cols in mapping.items():
        # 1. 匹配出口列 (正则表达式兼容括号)
        pat_exp = cols['exp'].replace('(', r'[(（]').replace(')', r'[)）]')
        col_exp = next((c for c in df_exp.columns if re.search(pat_exp, c)), None)
        ser_exp = df_exp.set_index('year')[col_exp] / 10000.0 if col_exp else pd.Series(0.0, index=df_exp['year'])

        # 2. 匹配进口列
        ser_imp = pd.Series(0.0, index=ser_exp.index)
        if cols['imp']:
            pat_imp = cols['imp'].replace('(', r'[(（]').replace(')', r'[)）]')
            col_imp = next((c for c in df_imp.columns if re.search(pat_imp, c)), None)
            if col_imp:
                ser_imp = df_imp.set_index('year')[col_imp] / 10000.0

        # 3. 计算历史净出口 (亿吨)
        hist_net_exp = (ser_exp.fillna(0) - ser_imp.fillna(0))
        
        # 4. 【核心逻辑】历史外推至 2035
        # 创建完整年份序列，并将历史值填入
        full_ser = hist_net_exp.reindex(target_years)
        
        # 使用 Forward Fill (ffill) 将最后一份观测数据延伸至 2035
        # 这种外推方式假设未来的进出口量级与历史末期(2024/2025)持平
        full_ser = full_ser.ffill().fillna(0.0)
        
        res[category] = full_ser
        
    return res

# ============ 3. 绘图核心函数 ============

# 先定义要绘图的年份范围
PLOT_YEARS = np.arange(2010, 2036)
NET_EXPORT_DICT = load_net_export_and_extrapolate(PLOT_YEARS)

def plot_production_forecast_final(xlsx_path, prod_name, out_png):
    if not os.path.exists(xlsx_path): return

    mapping = {"粗钢": "钢铁", "水泥": "水泥", "煤炭": "煤炭"}
    cat_key = mapping.get(prod_name, "钢铁")
    
    precision_map = {'钢铁': '.1f', '水泥': '.1f', '煤炭': '.3f'}
    fmt = precision_map.get(cat_key, '.1f')
    special_year = 2024 if cat_key == '煤炭' else 2025

    df = pd.read_excel(xlsx_path, index_col=0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_unit = df / 1e8
    years = df_unit.index.values
    net_exp_series = NET_EXPORT_DICT.get(cat_key)

    fig, ax1 = plt.subplots(figsize=(7, 3.5), dpi=300)
    PROV_8_LIST = ['北京市', '天津市', '河北省', '河南省', '山东省', '山西省', '内蒙古自治区', '陕西省']
    STACK_COLORS = ['#DEECF6','#AFC8E2','#E2F2CD','#B6DAA7','#F9D5D5','#EF98A1','#FBE3C0','#FBC99A']
    
    available_8 = [c for c in PROV_8_LIST if c in df_unit.columns]
    bottom = np.zeros(len(years))
    for i, prov in enumerate(available_8):
        vals = df_unit[prov].clip(lower=0).values
        ax1.bar(years, vals, bottom=bottom, color=STACK_COLORS[i % len(STACK_COLORS)], width=0.7, alpha=0.9, label=prov)
        bottom += vals

    ax1.set_ylabel(f'分省{prod_name}产量 (亿吨)', fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    nat_series = df_unit['全国'].clip(lower=0) if '全国' in df_unit.columns else df_unit[available_8].sum(axis=1)
    line_nat, = ax2.plot(years, nat_series, color='#d62728', marker='o', markersize=4, linewidth=1.2, label='全国总产量', zorder=10)
    
    ax2.set_ylabel(f'全国{prod_name}总量 (亿吨)', fontsize=9)
    ax2.set_ylim(0, nat_series.max() * 1.4) 
    
    # 标注逻辑
    for i, yr in enumerate(years):
        val = nat_series.iloc[i]
        if val > 0:
            if (yr % 5 == 0) or (yr == special_year):
                is_special = (yr == special_year)
                # 产量标注 (上方)
                ax2.text(yr, val + (nat_series.max() * 0.03), f'{val:{fmt}}', ha='center', va='bottom', 
                         fontsize=7, fontweight='bold' if is_special else 'normal', color='#d62728' if is_special else 'black')
                
                # 净出口外推标注 (下方)
                net_val = net_exp_series.get(yr, 0)
                # 2025年后的标注加上“(est.)”表示是预测值
                # tag = "(est.)" if yr > 2024 else ""
                ax2.text(yr, val - (nat_series.max() * 0.07), f"净出口:{net_val:.2f}", 
                         ha='center', va='top', fontsize=10, color='blue', fontstyle='italic')

    ax1.set_xticks(np.arange(2010, 2036, 5))
    ax2.grid(visible=False)
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1 + [line_nat], l1 + ['全国总产量'], loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=8)

    # plt.title(f"2010-2035年 {prod_name}产量预测 (含贸易外推) (单位: 亿吨)", fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

# ============ 4. 执行流程 ============
if __name__ == "__main__":
    plot_production_forecast_final(steel_path, "粗钢", os.path.join(out_fig_dir, "京津冀及周边8省_粗钢产量预测_贸易外推版.png"))
    plot_production_forecast_final(cement_path, "水泥", os.path.join(out_fig_dir, "京津冀及周边8省_水泥产量预测_贸易外推版.png"))
    plot_production_forecast_final(power_path, "煤炭", os.path.join(out_fig_dir, "京津冀及周边8省_煤炭产量预测_贸易外推版.png"))