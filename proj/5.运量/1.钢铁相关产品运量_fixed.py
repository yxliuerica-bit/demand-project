# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================== 1. 参数与路径配置 ===================
XLSX_PATH = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/预测表/分省粗钢产量_历史+预测_2010-2035_逐年衰减版.xlsx"
SHEET_NAME = "wide_tons"
YEARS = list(range(2020, 2036))

# 输出目录
OUT_BASE_DIR = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/相关产品运量/"
IMG_DIR = os.path.join(OUT_BASE_DIR, "images")
CSV_DIR = os.path.join(OUT_BASE_DIR, "data")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# —— 单位产品运量系数（吨/吨粗钢） ——
UF_COAL         = 0.22  # 燃料煤（仅长流程）
UF_COKE         = 0.46  # 焦炭（仅长流程）
UF_ORE          = 1.60  # 铁矿石（仅长流程）
UF_LIMESTONE    = 0.20  # 石灰石（仅长流程）
UF_SOLID_WASTE  = 0.68  # 固废（仅长流程）
UF_SCRAP        = 1.20  # 废钢（仅短流程）
UF_OTHER        = 0.53  # 其他辅料（总粗钢）
UF_STEEL_OUT    = 1.0   # 钢材成品（产出端）

# —— 短流程占比（EAF share）线性插值 ——
EAF_POINTS = {2020: 0.10, 2025: 0.12, 2030: 0.17, 2035: 0.22}

# 绘图颜色（剔除成品后的原材料配色）
RAW_MAT_COLORS = ["#45A6C6", "#DAEDF7", "#F5E57B", "#FFE7A6", "#EE9B5A", "#D7263D", "#B56576"]
BAR_WIDTH = 0.6

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# =================== 2. 数据预处理 ===================
df_wide = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)
if "year" in df_wide.columns:
    df_wide = df_wide.set_index("year")

all_provinces = df_wide.columns.tolist()

# 生成逐年短流程占比
eaf_series = pd.Series(EAF_POINTS).sort_index()
eaf_series = eaf_series.reindex(range(min(YEARS), max(YEARS)+1)).interpolate().reindex(YEARS).ffill().bfill()
eaf_values = eaf_series.clip(0.0, 1.0).values

master_data_list = []

# =================== 3. 循环计算与绘图 ===================
for province in all_provinces:
    print(f"正在处理: {province} ...")
    
    # 基础产量数据
    steel_tons = df_wide[province].reindex(YEARS).astype(float).fillna(0.0)
    steel_short = steel_tons * eaf_values
    steel_long  = steel_tons - steel_short
    
    # --- 分类计算运量 ---
    # 1. 生产物料（堆叠图展示对象）
    material_data = {
        "燃料煤": steel_long * UF_COAL,
        "焦炭": steel_long * UF_COKE,
        "铁矿石": steel_long * UF_ORE,
        "石灰石": steel_long * UF_LIMESTONE,
        "废钢": steel_short * UF_SCRAP,
        "固废": steel_long * UF_SOLID_WASTE,
        "其他辅料": steel_tons * UF_OTHER
    }
    
    # 2. 产成品（仅保存到CSV，不进堆叠图）
    product_data = {
        "钢材成品外运": steel_tons * UF_STEEL_OUT
    }
    
    # 合并所有数据用于保存
    full_prov_df = pd.DataFrame({**material_data, **product_data}, index=YEARS)
    full_prov_df.index.name = "年份"
    
    # 保存该省份 CSV (单位：吨)
    csv_path = os.path.join(CSV_DIR, f"{province}_钢铁全口径运量.csv")
    full_prov_df.to_csv(csv_path, encoding='utf-8-sig')
    
    # 汇总至总表
    temp_df = full_prov_df.reset_index()
    temp_df['省份'] = province
    master_data_list.append(temp_df)
    
    # --- 绘图：生产物料堆叠图 (单位：万吨) ---
    plot_df = pd.DataFrame(material_data, index=YEARS) / 1e4
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(YEARS))
    
    for i, col in enumerate(plot_df.columns):
        vals = plot_df[col].values
        ax.bar(YEARS, vals, bottom=bottom, width=BAR_WIDTH,
               color=RAW_MAT_COLORS[i % len(RAW_MAT_COLORS)], 
               edgecolor='white', linewidth=0.3, label=col)
        bottom += vals
        
    ax.set_title(f"{province}钢铁生产大宗物料需求预测 (不含成品, 2020-2035)")
    ax.set_ylabel("运量（万吨）")
    ax.set_xticks(YEARS)
    ax.set_xticklabels(YEARS, rotation=45)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="物料类别")
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"{province}_生产物料运量.png"), dpi=300)
    plt.close()

# 保存全省汇总 CSV
master_df = pd.concat(master_data_list, ignore_index=True)
master_df = master_df[['省份', '年份'] + [c for c in master_df.columns if c not in ['省份', '年份']]]
master_df.to_csv(os.path.join(OUT_BASE_DIR, "全国分省钢铁运量汇总_2020-2035.csv"), index=False, encoding='utf-8-sig')

print(f"\n[任务完成] 数据与图表已导出至 {OUT_BASE_DIR}")