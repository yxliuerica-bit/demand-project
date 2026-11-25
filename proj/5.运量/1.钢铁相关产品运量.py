# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================== 参数区（按需修改） ===================
XLSX_PATH = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/预测表/分省粗钢产量_历史+预测_2010-2035_逐年衰减版.xlsx"
SHEET_NAME = "wide_tons"      # 原始产量数据所在 Sheet
YEARS = list(range(2020, 2036))

# 输出路径设定
OUT_BASE_DIR = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/相关产品运量/"
IMG_DIR = os.path.join(OUT_BASE_DIR, "images")  # 存放图片
CSV_DIR = os.path.join(OUT_BASE_DIR, "data")    # 存放CSV数据
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# —— 单位产品运量系数（保持不变） ——
UF_COAL         = 0.22
UF_COKE         = 0.46
UF_ORE          = 1.60
UF_LIMESTONE    = 0.20
UF_SOLID_WASTE  = 0.68
UF_SCRAP        = 1.20
UF_LONG_STEEL   = 1.0 
UF_SHORT_STEEL  = 1.0 
UF_OTHER        = 0.53

# —— 短流程占比（EAF share）线性插值 ——
EAF_POINTS = {2020: 0.10, 2025: 0.12, 2030: 0.17, 2035: 0.22}

# 绘图外观
COLORS = ["#2F5597", "#6FA8DC", "#45A6C6", "#DAEDF7", "#F5E57B", "#FFE7A6", "#EE9B5A", "#D7263D", "#B56576"]
BAR_WIDTH = 0.6

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# =================== 1. 数据准备 ===================
df_wide = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME)
if "year" in df_wide.columns:
    df_wide = df_wide.set_index("year")

# 自动获取所有省份列表（除去可能存在的非省份列）
all_provinces = df_wide.columns.tolist()

# 生成 EAF 占比序列
eaf_series = pd.Series(EAF_POINTS).sort_index()
eaf_series = eaf_series.reindex(range(min(YEARS), max(YEARS)+1)).interpolate().reindex(YEARS).ffill().bfill()
eaf_values = eaf_series.clip(0.0, 1.0).values

# 用于存储所有省份汇总数据的长表
master_data_list = []

# =================== 2. 循环计算每个省份 ===================
for province in all_provinces:
    print(f"正在处理: {province} ...")
    
    # 获取该省粗钢产量
    steel_tons = df_wide[province].reindex(YEARS).astype(float).fillna(0.0)
    
    # 计算长短流程产量
    steel_short = steel_tons * eaf_values
    steel_long  = steel_tons - steel_short
    
    # 计算各项运量（吨）
    data = {
        "长流程钢材成品": steel_long * UF_LONG_STEEL,
        "短流程钢材成品": steel_short * UF_SHORT_STEEL,
        "燃料煤": steel_long * UF_COAL,
        "焦炭": steel_long * UF_COKE,
        "铁矿石": steel_long * UF_ORE,
        "石灰石": steel_long * UF_LIMESTONE,
        "固废": steel_long * UF_SOLID_WASTE,
        "废钢": steel_short * UF_SCRAP,
        "其他辅料": steel_tons * UF_OTHER
    }
    
    # 构建当前省份的 DataFrame
    prov_df = pd.DataFrame(data, index=YEARS)
    prov_df.index.name = "年份"
    
    # --- 保存该省的原始数据 CSV (单位：吨) ---
    prov_csv_path = os.path.join(CSV_DIR, f"{province}_钢铁相关运量_2020-2035.csv")
    prov_df.to_csv(prov_csv_path, encoding='utf-8-sig')
    
    # 存入汇总列表（增加省份标识）
    temp_df = prov_df.reset_index()
    temp_df['省份'] = province
    master_data_list.append(temp_df)
    
    # --- 绘图 (单位：万吨) ---
    prov_df_10k = prov_df / 1e4
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(YEARS))
    
    for i, col in enumerate(prov_df_10k.columns):
        vals = prov_df_10k[col].values
        ax.bar(YEARS, vals, bottom=bottom, width=BAR_WIDTH,
               color=COLORS[i % len(COLORS)], edgecolor='white', linewidth=0.3, label=col)
        bottom += vals
        
    ax.set_title(f"{province}钢铁行业相关运量预测 (2020-2035)")
    ax.set_ylabel("运量（万吨）")
    ax.set_xticks(YEARS)
    ax.set_xticklabels(YEARS, rotation=45)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    
    plt.tight_layout()
    img_path = os.path.join(IMG_DIR, f"{province}_钢铁运量.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

# =================== 3. 保存全省份汇总 CSV ===================
master_df = pd.concat(master_data_list, ignore_index=True)
# 调整列顺序，把省份和年份放在最前面
cols = ['省份', '年份'] + [c for c in master_df.columns if c not in ['省份', '年份']]
master_df = master_df[cols]

master_csv_path = os.path.join(OUT_BASE_DIR, "所有省份钢铁相关运量汇总_2020-2035.csv")
master_df.to_csv(master_csv_path, index=False, encoding='utf-8-sig')

print(f"\n[完成] 所有省份计算结束！")
print(f"统计图表已保存至: {IMG_DIR}")
print(f"各省数据已保存至: {CSV_DIR}")
print(f"全省汇总表已保存至: {master_csv_path}")