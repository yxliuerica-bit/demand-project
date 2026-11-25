# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 参数 ======
XLSX = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/预测表/分省粗钢产量_历史+预测_2010-2035_逐年衰减版.xlsx"
SHEET = "wide_tons"   # 宽表：行=year，列=省份（单位：吨）
YEARS = [2025, 2030, 2035]
HEBEI = "河北省"
NEIGHBORS = ["北京市","天津市","河南省","山东省","山西省","内蒙古自治区","陕西省"]

SAVE_PNG = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/预测图/河北省_与周边省份钢铁产量对比_2025_2030_2035.png"

# ====== 中文与风格 ======
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# ====== 读取数据 ======
df = pd.read_excel(XLSX, sheet_name=SHEET)
if "year" in df.columns:
    df = df.set_index("year")

# 选年、取数（吨→万吨）
hebei = (df[HEBEI].reindex(YEARS).astype(float) / 1e4).fillna(0.0)
neis = (df[NEIGHBORS].reindex(YEARS).astype(float) / 1e4).fillna(0.0)

# ====== 作图 ======
fig, ax1 = plt.subplots(figsize=(8.5, 5))

x = np.arange(len(YEARS), dtype=float)
w = 0.30  # 每组中单柱宽度

# — 左轴：河北（单色柱，无边框）
bar_hebei = ax1.bar(x - w/2, hebei.values, width=w,
                    color="#CFE7C9", label=HEBEI)
ax1.set_ylabel("河北省钢铁产量（万吨）")
ax1.set_xticks(x)
ax1.set_xticklabels(YEARS)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# 在河北柱上标注数值
for xi, v in zip(x - w/2, hebei.values):
    ax1.text(xi, v*1.01, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

# — 右轴：周边省份（堆叠柱）
ax2 = ax1.twinx()
bottom = np.zeros(len(YEARS))
colors = ["#E8F0FE","#C9DAF8","#9FC5E8","#C5E0B4","#F4CCCC","#F9CB9C","#EAD1DC"]
handles = []
for i, prov in enumerate(NEIGHBORS):
    vals = neis[prov].values
    h = ax2.bar(x + w/2, vals, width=w, bottom=bottom,
                color=colors[i % len(colors)], edgecolor='white', linewidth=0.6, label=prov)
    bottom += vals
    handles.append(h)

ax2.set_ylabel("周边省份钢铁产量（万吨）")
max_nei = (neis.sum(axis=1)).max()
ax2.set_ylim(0, max_nei * 1.15)

# ==== 图例合并：河北 + 周边 ====
handles_all = [bar_hebei] + [h[0] for h in handles]
labels_all = [HEBEI] + NEIGHBORS
ax2.legend(handles_all, labels_all, loc='center left', bbox_to_anchor=(1.15, 0.5), frameon=True)

# 标题
ax1.set_title("河北省与周边省份钢铁产量对比")



plt.tight_layout()
os.makedirs(os.path.dirname(SAVE_PNG), exist_ok=True)
plt.savefig(SAVE_PNG, dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] 已保存：{SAVE_PNG}")
