# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# =================== 1. 路径与配置 ===================
PROD_PATH = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/预测表/分省粗钢产量_历史+预测_2010-2035_逐年衰减版.xlsx"
CONS_PATH = "/lustre/home/litong/intern/京津冀货运发展/output/predict_results/Sce_Pop_BAU_GDP_BAU/12_跨省应用模型预测/整合分省完整数据/分省钢铁消费量预测.xlsx"
OUT_BASE_DIR = "/lustre/home/litong/intern/京津冀货运发展/output/分省详细货物清洁运输图/"
os.makedirs(OUT_BASE_DIR, exist_ok=True)

# 目标省份
TARGET_PROVS = ['北京', '天津', '河北', '河南', '山东', '山西', '内蒙古', '陕西']
YEARS = list(range(2010, 2036))

# 运量系数与物料列表
UF_DICT = {
    "燃料煤": 0.22, "焦炭": 0.46, "铁矿石": 1.60, "石灰石": 0.20, 
    "废钢": 1.20, "固废": 0.68, "其他辅料": 0.53
}

# 绘图颜色（参考附图配色方案）
COLORS = ["#45A6C6", "#DAEDF7", "#F5E57B", "#FFE7A6", "#EE9B5A", "#D7263D", "#B56576", "#9A7D9A"]
BAR_WIDTH = 0.7

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# =================== 2. 数据处理工具 ===================

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

# =================== 3. 核心计算与分省绘图 ===================

def generate_provincial_stacked_charts():
    # A. 加载数据
    df_prod = load_data_safe(PROD_PATH, sheet_name="wide_tons")
    df_cons = load_data_safe(CONS_PATH)
    
    # B. 比例准备
    clean_ratio = get_ratio_curve({2010: 0.35, 2024: 0.35, 2030: 0.80, 2035: 1.00}, YEARS)
    eaf_share = get_ratio_curve({2010: 0.08, 2020: 0.10, 2025: 0.12, 2030: 0.17, 2035: 0.22}, YEARS)

    # C. 逐省循环
    for prov in TARGET_PROVS:
        if prov not in df_prod.columns or prov not in df_cons.columns:
            continue
        
        print(f"正在生成 {prov} 的详细货物清洁运输图...")
        
        # 1. 计算各项货物的清洁运输量 (单位：亿吨)
        prov_data = pd.DataFrame(index=YEARS)
        
        # —— 计算生产物料 ——
        prod_vol = df_prod[prov]
        for mat, uf in UF_DICT.items():
            if mat in ["燃料煤", "焦炭", "铁矿石", "石灰石", "固废"]:
                vol = prod_vol * (1 - eaf_share) * uf
            elif mat == "废钢":
                vol = prod_vol * eaf_share * uf
            else: # 其他辅料
                vol = prod_vol * uf
            prov_data[mat] = (vol * clean_ratio) / 1e8
            
        # —— 计算钢铁产品 (跨省净流量) ——
        net_flow = (df_cons[prov] - df_prod[prov]).abs()
        prov_data["钢铁产品(净流)"] = (net_flow * 1.0 * clean_ratio) / 1e8
        
        # 2. 绘图 (堆叠柱状图)
        fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
        bottom = np.zeros(len(YEARS))
        
        for i, col in enumerate(prov_data.columns):
            vals = prov_data[col].values
            ax.bar(YEARS, vals, bottom=bottom, width=BAR_WIDTH,
                   color=COLORS[i % len(COLORS)], 
                   edgecolor='black', linewidth=0.2, label=col)
            bottom += vals
            
        # 设置标题与轴标签
        ax.set_title(f"{prov} 钢铁相关大宗货物清洁运输预测 (2010-2035)", fontsize=15, pad=15)
        ax.set_ylabel("清洁运输量（亿吨）", fontsize=12)
        ax.set_xticks(YEARS)
        ax.set_xticklabels(YEARS, rotation=45)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
        # 图例位置参考附图置于右侧
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="货物类别", frameon=True)
        
        plt.tight_layout()
        save_path = os.path.join(OUT_BASE_DIR, f"{prov}_详细货物清洁运输.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generate_provincial_stacked_charts()
    print(f"\n[任务完成] 各省详细图表已导出至: {OUT_BASE_DIR}")