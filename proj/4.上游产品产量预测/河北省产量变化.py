# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 参数 =========
xlsx_path = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/预测表/分省粗钢产量_历史+预测_2010-2035_逐年衰减版.xlsx"
sheet     = "wide_tons"     # 宽表（行=year，列=各省，单位：吨）
province  = "河北省"
years     = list(range(2020, 2036))
save_png  = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/预测图/河北省_粗钢产量变化_2020-2035.png"
to_yidun  = True            # True=转换为亿吨；False=保持吨

# ========= 中文显示 =========
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# ========= 读取数据 =========
df = pd.read_excel(xlsx_path, sheet_name=sheet)
if "year" in df.columns:
    df = df.set_index("year")

# 取河北省
s = df[province].reindex(years).astype(float)

# 单位转换
if to_yidun:
    y = s / 1e8
    y_label = "粗钢产量（亿吨）"
    fmt = "{:.1f}"
else:
    y = s
    y_label = "粗钢产量（吨）"
    fmt = "{:,.0f}"

# ========= 绘图 =========
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(years, y.values, marker="o", linewidth=1.8)
ax.set_title("河北省2020–2035年粗钢产量变化")
ax.set_xlabel("年份")
ax.set_ylabel(y_label)
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=35, ha="right")
ax.grid(axis="y", linestyle="--", alpha=0.5)

# 数据标签
for X, V in zip(years, y.values):
    ax.text(X, V*(1.01 if V>=0 else 0.99), fmt.format(V), ha="center", va="bottom", fontsize=8)

plt.tight_layout()
os.makedirs(os.path.dirname(save_png), exist_ok=True)
plt.savefig(save_png, dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] 已保存：{save_png}")
