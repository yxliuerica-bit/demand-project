# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =================== 1. 路径与参数配置 ===================
PROD_PATH = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/预测表/分省粗钢产量_历史+预测_2010-2035_逐年衰减版.xlsx"
CONS_PATH = "/lustre/home/litong/intern/京津冀货运发展/output/predict_results/Sce_Pop_BAU_GDP_BAU/12_跨省应用模型预测/整合分省完整数据/分省钢铁消费量预测.xlsx"
OUT_DIR = "/lustre/home/litong/intern/京津冀货运发展/output/OD矩阵可视化/"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_YEAR = 2020
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# =================== 2. 数据处理与分配逻辑 ===================

def std_name(s):
    if pd.isna(s): return s
    return re.sub(r"(省|市|自治区|壮族|维吾尔|回族)$", "", str(s).strip())

def generate_visual_od():
    # A. 加载数据
    df_prod = pd.read_excel(PROD_PATH, sheet_name="wide_tons").set_index('year')
    df_cons = pd.read_excel(CONS_PATH).set_index('year')
    
    # 清洗列名
    df_prod.columns = [std_name(c) for c in df_prod.columns]
    df_cons.columns = [std_name(c) for c in df_cons.columns]
    
    # 提取2035截面
    p2035 = df_prod.loc[TARGET_YEAR]
    c2035 = df_cons.loc[TARGET_YEAR]
    
    # 剔除汇总项并对齐省份
    exclude = ["全国", "京津冀及周边8省市", "京津冀及周边8省"]
    provs = [p for p in set(p2035.index) | set(c2035.index) if p not in exclude]
    
    balance = (p2035.reindex(provs).fillna(0) - c2035.reindex(provs).fillna(0)) / 1e8 # 亿吨
    
    # B. 绘制产销平衡柱状图
    plt.figure(figsize=(14, 6))
    balance_sorted = balance.sort_values(ascending=False)
    colors = ['#E06C75' if x < 0 else '#61AFEF' for x in balance_sorted]
    plt.bar(balance_sorted.index, balance_sorted.values, color=colors, edgecolor='black', alpha=0.8)
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"{TARGET_YEAR}年各省钢铁产销平衡状态（正值盈余/流出，负值缺口/流入）", fontsize=14)
    plt.ylabel("规模（亿吨）")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_产销平衡对比图.png"), dpi=300)
    
    # C. 计算 O-D 矩阵
    origins = balance[balance > 0]
    dests = balance[balance < 0].abs()
    
    # 比例分配
    od_matrix = pd.DataFrame(index=origins.index, columns=dests.index)
    total_d = dests.sum()
    for o in origins.index:
        for d in dests.index:
            od_matrix.loc[o, d] = origins[o] * (dests[d] / total_d)
            
    # D. 绘制流量热力图
    od_matrix = od_matrix.astype(float)
    plt.figure(figsize=(12, 8))
    sns.heatmap(od_matrix, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': '运输流量 (亿吨)'})
    plt.title(f"{TARGET_YEAR}年省际钢铁产品清洁运输流量热力图 (O-D Matrix)", fontsize=14)
    plt.xlabel("流入省份 (Destinations)")
    plt.ylabel("流出省份 (Origins)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_OD流量热力图.png"), dpi=300)
    
    # 保存 CSV
    od_matrix.to_csv(os.path.join(OUT_DIR, "2035_OD矩阵_最终版.csv"), encoding='utf-8-sig')

if __name__ == "__main__":
    generate_visual_od()
    print(f"可视化完成！请查看目录：{OUT_DIR}")