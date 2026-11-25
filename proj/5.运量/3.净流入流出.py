# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ======================== 1. 路径配置 ========================
# 产量文件路径
prod_steel_path  = "/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_基准/分省粗钢产量_历史加预测.xlsx"
prod_cement_path = "/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_基准/分省水泥产量_历史加预测.xlsx"

# 消费量文件路径
cons_steel_path  = "/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_基准/全国完整数据/分省钢铁消费量预测.xlsx"
cons_cement_path = "/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_基准/全国完整数据/分省水泥消费量预测.xlsx"

OUT_DIR = "/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_基准/省际流入流出图"
os.makedirs(OUT_DIR, exist_ok=True)

# ======================== 2. 绘图环境配置 ========================
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# ======================== 3. 数据处理工具 ========================

VALID_PROV = ['北京','天津','河北','山西','内蒙古','辽宁','吉林','黑龙江',
              '上海','江苏','浙江','安徽','福建','江西','山东','河南',
              '湖北','湖南','广东','广西','海南','重庆','四川','贵州',
              '云南','西藏','陕西','甘肃','青海','宁夏','新疆']

def std_province_name(s):
    if pd.isna(s): return s
    s = str(s).strip()
    s = re.sub(r"(省|市|自治区|壮族自治区|维吾尔自治区|回族自治区)$","", s)
    return s

def clean_province_df(df):
    df = df.copy()
    df["province"] = df["province"].astype(str).str.strip()
    drop_keys = {"China","china","CHINA","全国","合计","总计","小计","中国"}
    df = df[~df["province"].isin(drop_keys)]
    df["province"] = df["province"].map(std_province_name)
    df = df[df["province"].isin(VALID_PROV)]
    return df

def read_any_prov_year_to_long(xlsx_path):
    """通用读取函数：将各种Excel结构转为[province, year, value]长表"""
    xls = pd.ExcelFile(xlsx_path)
    # 尝试寻找包含数据的 sheet
    sheets = [s for s in ["wide_tons","wide","long","Sheet1"] if s in xls.sheet_names]
    if not sheets: sheets = [xls.sheet_names[0]]
    
    df = pd.read_excel(xlsx_path, sheet_name=sheets[0])
    lower_cols = {c: str(c).strip().lower() for c in df.columns}
    
    # 逻辑 A: 宽表 (行是 year, 列是省份)
    if any(v == "year" for v in lower_cols.values()):
        ycol = [c for c in df.columns if str(c).strip().lower() == "year"][0]
        prov_cols = [c for c in df.columns if c != ycol]
        long_df = df.melt(id_vars=[ycol], value_vars=prov_cols, var_name="province", value_name="value")
        long_df = long_df.rename(columns={ycol: "year"})
        return clean_province_df(long_df)
    
    # 逻辑 B: 长表
    if {"province","year"}.issubset(set(lower_cols.values())):
        ren = {c: lower_cols[c] for c in df.columns if lower_cols[c] in ["province","year"]}
        val_col = next((c for c in df.columns if lower_cols[c] in ["value","产量","消费量","数值"]), None)
        if val_col: ren[val_col] = "value"
        return clean_province_df(df.rename(columns=ren)[["province","year","value"]])

    return None

# ======================== 4. 核心绘图函数 ========================

def plot_net_inflow_mirrored_no_negative(cons_long, prod_long, title, out_png, year=2025):
    """
    绘制净流入镜像图：
    - 消费 > 产量: 向上生长 (蓝色)
    - 产量 > 消费: 向下生长 (红色)
    - Y轴标签取绝对值，不显示负数
    """
    # 1. 数据合并与计算
    c = cons_long[cons_long["year"] == year][["province","value"]].rename(columns={"value":"cons"})
    p = prod_long[prod_long["year"] == year][["province","value"]].rename(columns={"value":"prod"})
    df = pd.merge(c, p, on="province", how="inner")
    
    # 计算净流入量 (单位：万吨)
    df["net_val"] = (df["cons"] - df["prod"]) / 10000.0
    
    # 排序：按净流入量从大到小排序
    df = df.sort_values("net_val", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    x = np.arange(len(df))
    bw = 0.7

    # 2. 绘制柱状图
    # 消费缺口（向上）
    mask_in = df["net_val"] >= 0
    ax.bar(x[mask_in], df.loc[mask_in, "net_val"], bw, 
           color="#4B8BBE", edgecolor="black", linewidth=0.8, label="净流入")
    
    # 产量富余（向下）
    mask_out = df["net_val"] < 0
    ax.bar(x[mask_out], df.loc[mask_out, "net_val"], bw, 
           color="#E06C75", edgecolor="black", linewidth=0.8, label="净流出")

    # 3. 【关键】格式化Y轴，将负数显示为正数
    def abs_formatter(x, pos):
        return f"{int(abs(x))}"
    ax.yaxis.set_major_formatter(FuncFormatter(abs_formatter))

    # 4. 美化坐标轴
    ax.axhline(0, color='black', linewidth=1.2) # 0 轴加粗
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_ylabel("万吨", fontsize=15, fontweight='bold')
    # ax.set_title(f"{title} - 各省供需平衡趋势 ({year}年)", fontsize=16, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df["province"], rotation=45, fontsize=12, ha="right")

    # 5. 数值标注
    max_abs = df["net_val"].abs().max()
    offset = max_abs * 0.02
    
    for i, val in enumerate(df["net_val"]):
        if val >= 0:
            # 向上标注
            ax.text(i, val + offset, f"{int(round(val))}", 
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            # 向下标注，显示绝对值
            ax.text(i, val - offset, f"{int(round(abs(val)))}", 
                    ha='center', va='top', fontsize=9, fontweight='bold')

    # 6. 设置范围对称
    # limit = max_abs * 1.25
    # 直接设置数值
    ax.set_ylim(-23000, 11000)
    # ax.set_ylim(-limit, limit)

    # 7. 图例
    ax.legend(loc="upper right", frameon=True, fontsize=18, shadow=True)

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"✅ 已生成镜像图: {out_png}")

# ======================== 5. 执行流程 ========================

def main():
    # 读取原始数据
    print("正在读取并清洗数据...")
    cons_steel = read_any_prov_year_to_long(cons_steel_path)
    prod_steel = read_any_prov_year_to_long(prod_steel_path)
    cons_cement = read_any_prov_year_to_long(cons_cement_path)
    prod_cement = read_any_prov_year_to_long(prod_cement_path)

    # 绘制 2025 年镜像图
    plot_net_inflow_mirrored_no_negative(cons_steel, prod_steel, "钢铁行业", 
                                         os.path.join(OUT_DIR, "钢铁_供需平衡镜像图_2025.png"), year=2025)
    
    plot_net_inflow_mirrored_no_negative(cons_cement, prod_cement, "水泥行业", 
                                         os.path.join(OUT_DIR, "水泥_供需平衡镜像图_2025.png"), year=2025)

if __name__ == "__main__":
    main()