# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# =================== 1. 环境与路径配置 ===================
PROD_PATH = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/预测表/分省粗钢产量_历史+预测_2010-2035_逐年衰减版.xlsx"
CONS_PATH = "/lustre/home/litong/intern/京津冀货运发展/output/predict_results/Sce_Pop_BAU_GDP_BAU/12_跨省应用模型预测/整合分省完整数据/分省钢铁消费量预测.xlsx"
OUT_DIR = "/lustre/home/litong/intern/京津冀货运发展/output/清洁运输最终看板_亿吨版/"
os.makedirs(OUT_DIR, exist_ok=True)

# 目标省份
TARGET_PROVS = ['北京', '天津', '河北', '河南', '山东', '山西', '内蒙古', '陕西']
YEARS = list(range(2010, 2036))

# 原料运量系数
UF_DICT = {
    "燃料煤": 0.22, "焦炭": 0.46, "铁矿石": 1.60, "石灰石": 0.20, "固废": 0.68, "废钢": 1.20, "其他辅料": 0.53
}

# 绘图配色
PROV_COLORS = ['#DEECF6','#AFC8E2','#E2F2CD','#B6DAA7','#F9D5D5','#EF98A1','#FBE3C0','#FBC99A']
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =================== 2. 数据清洗函数 ===================

def std_name(s):
    if pd.isna(s): return s
    s = str(s).strip()
    if s == "全国": return s
    return re.sub(r"(省|市|自治区|壮族|维吾尔|回族)$", "", s)

def load_data_safe(path, sheet_name=0):
    df = pd.read_excel(path, sheet_name=sheet_name)
    y_col = next((c for c in df.columns if str(c).lower() in ['year', '年份']), None)
    if y_col: df = df.set_index(y_col)
    
    df = df.groupby(level=0).sum()
    df.columns = [std_name(c) for c in df.columns]
    
    return df.reindex(index=YEARS).fillna(0)

def get_ratio_curve(points, years):
    return pd.Series(points).reindex(range(min(years), max(years)+1)).interpolate().reindex(years).ffill().bfill()

# =================== 3. 绘图标准函数 (增强版) ===================

def plot_master_chart(df, title, subtype, show_national_line=True):
    """
    show_national_line: 是否绘制全国总量红色折线
    """
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=130)
    
    # 堆叠柱状图 (始终绘制)
    bottom = np.zeros(len(YEARS))
    cols = [c for c in TARGET_PROVS if c in df.columns]
    for i, prov in enumerate(cols):
        ax1.bar(YEARS, df[prov], bottom=bottom, color=PROV_COLORS[i % len(PROV_COLORS)], 
                label=prov, width=0.7, edgecolor='white', linewidth=0.2)
        bottom += df[prov].values
    
    # --- 新增：在每个柱子顶部标注八省市总和 (保留一位小数) ---
    for yr_idx, total_val in enumerate(bottom):
        if total_val > 0:
            ax1.text(YEARS[yr_idx], total_val + (bottom.max() * 0.01), f'{total_val:.2f}', 
                     ha='center', va='bottom', fontsize=8, color='#555555')
        
    ax1.set_title(f"2010-2035年 {title} 清洁运输规模", fontsize=14, pad=20)
    ax1.set_ylabel("八省市规模 (亿吨)", fontsize=11)
    ax1.set_xticks(YEARS[::2])
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    # 根据参数决定是否绘制右侧轴和全国折线
    if show_national_line and '全国' in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(YEARS, df['全国'], color='#D7263D', marker='o', markersize=3, 
                 linewidth=2, label='全国总清洁运输规模')
        
        # --- 修改：标注全国数值 (保留一位小数) ---
        for yr in [2010, 2024, 2030, 2035]:
            val = df.loc[yr, '全国']
            ax2.text(yr, val + (df['全国'].max()*0.01), f'{val:.1f}', 
                     color='#D7263D', ha='center', fontweight='bold', fontsize=9)

        ax2.set_ylabel("全国总规模 (亿吨)", fontsize=11, color='#D7263D')
        
        # 合并图例
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left', bbox_to_anchor=(1.12, 1), title="省份及汇总")
    else:
        # 不绘制折线时，仅显示省份图例
        ax1.legend(loc='upper left', bbox_to_anchor=(1.12, 1), title="省份")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{title}_最终版_亿吨.png"), bbox_inches='tight')
    plt.close()

# =================== 4. 核心计算与生成逻辑 ===================

def generate_final_charts():
    # A. 加载数据
    df_prod = load_data_safe(PROD_PATH, sheet_name="wide_tons")
    df_cons = load_data_safe(CONS_PATH)
    
    # B. 比例曲线准备
    clean_ratio = get_ratio_curve({2010: 0.35, 2024: 0.35, 2030: 0.80, 2035: 1.00}, YEARS)
    eaf_share = get_ratio_curve({2010: 0.08, 2020: 0.10, 2025: 0.12, 2030: 0.17, 2035: 0.22}, YEARS)

    # C. 计算原材料 (基于生产驱动)
    for mat, uf in UF_DICT.items():
        res_df = pd.DataFrame(index=YEARS)
        for prov in TARGET_PROVS:
            if prov in df_prod.columns:
                share_factor = (1 - eaf_share) if mat in ["燃料煤", "焦炭", "铁矿石", "石灰石", "固废"] else (eaf_share if mat == "废钢" else 1.0)
                res_df[prov] = (df_prod[prov] * share_factor * uf * clean_ratio) / 1e8 
        
        res_df['全国'] = (df_prod['全国'] * 1.0 * uf * clean_ratio) / 1e8 
        plot_master_chart(res_df, mat, "生产驱动型", show_national_line=True)

    # D. 计算钢铁成品 (基于供需驱动/净流量)
    net_diff_raw = (df_cons - df_prod.reindex(columns=df_cons.columns).fillna(0)).abs()
    
    steel_df = pd.DataFrame(index=YEARS)
    for prov in TARGET_PROVS:
        if prov in net_diff_raw.columns:
            steel_df[prov] = (net_diff_raw[prov] * 1.0 * clean_ratio) / 1e8
            
    national_net = (df_cons['全国'] - df_prod['全国']).abs()
    steel_df['全国'] = (national_net * 1.0 * clean_ratio) / 1e8
    
    # 钢铁成品图：不显示全国折线
    plot_master_chart(steel_df, "钢铁成品(跨省流量)", "供需驱动型", show_national_line=False)

if __name__ == "__main__":
    generate_final_charts()
    print(f"✅ 处理完成！结果见: {OUT_DIR}")