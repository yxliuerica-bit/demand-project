# -*- coding: utf-8 -*-
"""
只绘制京津冀周边（河北/天津/北京/山东/山西/辽宁/内蒙古）
2020年钢铁 消费量/产量 上正下负 柱状图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import FuncFormatter

# ========= 中文显示 =========
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False

SPECIES = "钢铁"
# ========= 文件路径 =========
CONS_PATH = f"/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_基准/全国完整数据/分省{SPECIES}消费量预测.xlsx"
PROD_PATH = f"/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_基准/分省{SPECIES}产量_历史加预测.xlsx"
TARGET_YEAR = 2035
OUT_PATH = f"/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_基准/省际流入流出图/{SPECIES}消费产量{TARGET_YEAR}_京津冀周边.png"

# ========= 省份规范 & 读取函数 =========
def norm_prov(name: str) -> str:
    if pd.isna(name): return name
    s = str(name).strip()
    s = re.sub(r'(省|市|特别行政区)$', '', s).replace('自治区', '')
    s = (s.replace('广西壮族', '广西')
           .replace('宁夏回族', '宁夏')
           .replace('新疆维吾尔', '新疆'))
    return s

def load_consumption(path, year=f"{TARGET_YEAR}", sheet_name=0):
    """读取消费量；兼容 长表(province/year/value) 和 两种宽表"""
    df = pd.read_excel(path, sheet_name=sheet_name)
    cols_lower = {c: str(c).strip().lower() for c in df.columns}

    # --- 长表 ---
    if {"province","year"}.issubset(set(cols_lower.values())):
        ren = {}
        for c in df.columns:
            lc = cols_lower[c]
            if lc == "province": ren[c] = "province"
            elif lc == "year":   ren[c] = "year"
            elif lc in ["value","consumption","消费量","数值"]: ren[c] = "value"
        if "value" not in ren.values():
            # 合并其它数值列兜底
            num_cols = [c for c in df.columns if c not in ren]
            for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
            df["value"] = df[num_cols].sum(axis=1, skipna=True)
            ren["value"] = "value"
        out = df.rename(columns=ren)
        out = out[out["year"] == year][["province","value"]]
        out["province"] = out["province"].map(norm_prov)
        out["consumption"] = pd.to_numeric(out["value"], errors="coerce")
        return out[["province","consumption"]].dropna(subset=["consumption"])

    # --- 宽表（行=year, 列=省份）---
    if any(v == "year" for v in cols_lower.values()):
        wide = df.copy()
        # year 作为索引，从该年取一行
        ycol = [c for c in df.columns if str(c).strip().lower() == "year"][0]
        wide = wide.set_index(ycol)
        if year not in wide.index:
            # 若没有精确年份，用最接近的一年
            idx_years = pd.to_numeric(wide.index, errors="coerce")
            near = idx_years.sub(year).abs().idxmin()
            year_key = wide.index[near]
        else:
            year_key = year
        row = wide.loc[year_key]
        ser = row.dropna()
        cons = ser.reset_index()
        cons.columns = ["province","consumption"]
        cons["province"] = cons["province"].map(norm_prov)
        cons["consumption"] = pd.to_numeric(cons["consumption"], errors="coerce")
        return cons.dropna(subset=["consumption"])

    # --- 宽表（行=省, 列=年份）---
    # 寻找“省份”列
    prov_col = None
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["province","省份","地区","省"]:
            prov_col = c; break
    if prov_col is not None:
        # 选择最接近的年份列
        year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
        if not year_cols:
            raise ValueError("未找到年份列")
        ycol = min(year_cols, key=lambda c: abs(int(str(c)) - int(year)))
        out = df[[prov_col, ycol]].rename(columns={prov_col:"province", ycol:"consumption"})
        out["province"] = out["province"].map(norm_prov)
        out["consumption"] = pd.to_numeric(out["consumption"], errors="coerce")
        return out.dropna(subset=["consumption"])

    raise RuntimeError("无法识别消费量表结构")


def load_production(path, year=f"{TARGET_YEAR}", sheet_name=0):
    """读取产量；兼容 长表(province/year/value) 和 两种宽表"""
    # 尝试“long”“wide_tons”等常见 sheet
    try:
        xls = pd.ExcelFile(path)
        cand = [s for s in ["long","wide_tons","wide","Sheet1",sheet_name] if s in xls.sheet_names]
        for sh in cand or [sheet_name]:
            df = pd.read_excel(path, sheet_name=sh)
            if df.dropna(how="all").empty:
                continue
            cols_lower = {c: str(c).strip().lower() for c in df.columns}

            # --- 长表 ---
            if {"province","year"}.issubset(set(cols_lower.values())):
                ren = {}
                for c in df.columns:
                    lc = cols_lower[c]
                    if lc == "province": ren[c] = "province"
                    elif lc == "year":   ren[c] = "year"
                    elif lc in ["value","production","产量","数值"]: ren[c] = "value"
                if "value" not in ren.values():
                    num_cols = [c for c in df.columns if c not in ren]
                    for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
                    df["value"] = df[num_cols].sum(axis=1, skipna=True)
                    ren["value"] = "value"
                out = df.rename(columns=ren)
                out = out[out["year"] == year][["province","value"]]
                out["province"] = out["province"].map(norm_prov)
                out["production"] = pd.to_numeric(out["value"], errors="coerce")
                return out[["province","production"]].dropna(subset=["production"])

            # --- 宽表（行=year, 列=省份）---  ← 新文件多半是这种
            if any(v == "year" for v in cols_lower.values()):
                wide = df.copy()
                ycol = [c for c in df.columns if str(c).strip().lower() == "year"][0]
                wide = wide.set_index(ycol)
                if year not in wide.index:
                    idx_years = pd.to_numeric(wide.index, errors="coerce")
                    near = idx_years.sub(year).abs().idxmin()
                    year_key = wide.index[near]
                else:
                    year_key = year
                row = wide.loc[year_key]
                ser = row.dropna()
                prod = ser.reset_index()
                prod.columns = ["province","production"]
                prod["province"] = prod["province"].map(norm_prov)
                prod["production"] = pd.to_numeric(prod["production"], errors="coerce")
                return prod.dropna(subset=["production"])

            # --- 宽表（行=省, 列=年份）---
            prov_col = None
            for c in df.columns:
                lc = str(c).strip().lower()
                if lc in ["province","省份","地区","省"]:
                    prov_col = c; break
            if prov_col is not None:
                year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
                if year_cols:
                    ycol = min(year_cols, key=lambda c: abs(int(str(c)) - int(year)))
                    out = df[[prov_col, ycol]].rename(columns={prov_col:"province", ycol:"production"})
                    out["province"] = out["province"].map(norm_prov)
                    out["production"] = pd.to_numeric(out["production"], errors="coerce")
                    return out.dropna(subset=["production"])
    except Exception:
        pass

    # 兜底
    df = pd.read_excel(path, sheet_name=sheet_name)
    raise RuntimeError("无法识别产量表结构，请确认是长表或宽表（行year/列省份）。")


# ========= 读取、筛选、排序 =========
cons = load_consumption(CONS_PATH, TARGET_YEAR)
prod = load_production(PROD_PATH, TARGET_YEAR)
df = pd.merge(cons, prod, on='province', how='inner')

# 仅保留“京津冀周边”并按例图顺序
keep_order = ['河北', '天津', '北京', '山东', '山西', '辽宁', '内蒙古']
df = df[df['province'].isin(keep_order)].copy()
df['province'] = pd.Categorical(df['province'], categories=keep_order, ordered=True)
df = df.sort_values('province')

# 单位：转“万吨”（若原始已是万吨，把 UNIT_DIV 改为 1）
UNIT_DIV = 1e4
df['cons_wt'] = df['consumption']/UNIT_DIV
df['prod_wt'] = df['production']/UNIT_DIV

# ========= 绘图 (镜像对比 - Y轴全正数标注版) =========
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

x = np.arange(len(df))
bw = 0.6  # 镜像图通常单柱居中更美观

# 1. 绘制消费量 (向上，正值)
ax.bar(x, df['cons_wt'], width=bw, label='生产需求',
       color='#4B8BBE', edgecolor='black', linewidth=0.8)

# 2. 绘制产量 (向下，负值)
ax.bar(x, -df['prod_wt'], width=bw, label='产量',
       color='#E06C75', edgecolor='black', linewidth=0.8)

# 3. 核心：设置 Y 轴刻度格式，将负值显示为正值
def abs_formatter(x, pos):
    return f"{int(abs(x))}"

ax.yaxis.set_major_formatter(FuncFormatter(abs_formatter))

# 4. 辅助线与范围设置
# 根据数据动态设置 Y 轴对称范围，确保上下空间充足
ymax = 10000 # 消费量上限
ymin = -25000 # 产量下限（注意是负值）
ax.set_ylim(ymin - 2000, ymax + 2000)

# 0 轴基准线
ax.axhline(0, color='black', linewidth=1.2)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

# 5. 数字标注
for i, (c, p) in enumerate(zip(df['cons_wt'], df['prod_wt'])):
    # 消费量标注 (上方)
    ax.text(i, c + 500, f"{int(round(c))}",
            ha='center', va='bottom', fontsize=9, color='black')
    # 产量标注 (下方)
    ax.text(i, -p - 500, f"{int(round(p))}",
            ha='center', va='top', fontsize=9, color='black')

# 6. 坐标轴名称与刻度
ax.set_xticks(x)
ax.set_xticklabels(df['province'], rotation=0, fontsize=17)
ax.set_ylabel("万吨", color='black', fontsize=17, fontweight='bold')
# ax.set_xlabel("省份", color='black', fontsize=15)

# 7. 图例
ax.legend(loc='upper right', frameon=True, fontsize=17)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
print(f"镜像图已保存：{OUT_PATH} (Y轴标签已取绝对值)")