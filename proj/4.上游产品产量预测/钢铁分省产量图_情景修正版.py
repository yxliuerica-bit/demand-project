# -*- coding: utf-8 -*-
"""根据最新钢铁需求情景更新全国及分省粗钢产量并绘图。"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


PROJECT = Path(r"C:\Users\admin\Desktop\货运预测李童+高洁\京津冀货运发展")
DEMAND_FILE = (PROJECT / "output" / "2.消费需求预测" / "钢铁需求情景修正版_20261020" /
               "Sce_基准" / "全国完整数据" / "分省钢铁消费量预测_情景修正版.xlsx")
PROD_TEMPLATE = (PROJECT / "output" / "3.产量分配" / "Sce_基准" /
                 "分省钢铁产量_历史加预测.xlsx")
EXPORT_FILE = PROJECT / "data" / "预测原始数据" / "上游产品出口年度数据.xlsx"
IMPORT_FILE = PROJECT / "data" / "预测原始数据" / "上游产品进口年度数据.xlsx"
OUT_DIR = PROJECT / "output" / "3.产量分配" / "钢铁需求情景修正版_20261020"

YEARS = np.arange(2010, 2036)
FORECAST_START = 2025
CONVERSION_FACTOR = 0.95
PROV_8 = ["北京市", "天津市", "河北省", "河南省", "山东省", "山西省", "内蒙古自治区", "陕西省"]
COLORS = ["#DCEAF5", "#AFC8E2", "#DDEEC8", "#B5D79D", "#F7D4D7", "#EB9BA3", "#F8E2BE", "#F5C794"]

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 17,
    "axes.titlesize": 22,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


def trade_series(path: Path, keyword: str) -> pd.Series:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    time_col = next(c for c in df.columns if "时间" in c)
    value_col = next(c for c in df.columns if keyword in c and "万吨" in c)
    years = df[time_col].astype(str).str.extract(r"(\d{4})")[0].astype(int)
    values = pd.to_numeric(df[value_col], errors="coerce") * 1e4 / CONVERSION_FACTOR
    series = pd.Series(values.values, index=years).sort_index()
    return series.reindex(YEARS).ffill().bfill()


def build_production() -> tuple[pd.DataFrame, pd.DataFrame]:
    demand = pd.read_excel(DEMAND_FILE, index_col=0)
    demand.index = pd.to_numeric(demand.index, errors="coerce")
    template = pd.read_excel(PROD_TEMPLATE, index_col=0)
    template.index = pd.to_numeric(template.index, errors="coerce")
    demand = demand.apply(pd.to_numeric, errors="coerce").fillna(0)
    template = template.apply(pd.to_numeric, errors="coerce").fillna(0)

    exports = trade_series(EXPORT_FILE, "钢材出口")
    imports = trade_series(IMPORT_FILE, "钢材进口")
    national = template["全国"].reindex(YEARS).copy()
    future = YEARS >= FORECAST_START
    national.loc[future] = demand.loc[YEARS[future], "全国"] + exports.loc[YEARS[future]] - imports.loc[YEARS[future]]

    excluded = {"全国", "京津冀及周边8省市"}
    provinces = [c for c in template.columns if c not in excluded]
    shares = template[provinces].reindex(YEARS).clip(lower=0)
    shares = shares.div(shares.sum(axis=1).replace(0, np.nan), axis=0).ffill().fillna(0)
    output = shares.mul(national, axis=0)
    output["全国"] = output[provinces].sum(axis=1)
    available_8 = [p for p in PROV_8 if p in output.columns]
    output["京津冀及周边8省市"] = output[available_8].sum(axis=1)
    output.index.name = "year"

    check = pd.DataFrame({
        "年份": YEARS,
        "国内表观需求_亿吨": demand["全国"].reindex(YEARS).values / 1e8,
        "钢材出口粗钢当量_亿吨": exports.values / 1e8,
        "钢材进口粗钢当量_亿吨": imports.values / 1e8,
        "全国粗钢产量_亿吨": output["全国"].values / 1e8,
        "省级合计差额_吨": output[provinces].sum(axis=1).values - output["全国"].values,
    })
    return output, check


def plot(output: pd.DataFrame) -> Path:
    data = output.loc[2020:2035] / 1e8
    years = data.index.to_numpy()
    fig, ax1 = plt.subplots(figsize=(13.33, 5.2), dpi=180)
    bottom = np.zeros(len(data))
    for color, province in zip(COLORS, PROV_8):
        if province not in data.columns:
            continue
        vals = data[province].clip(lower=0).to_numpy()
        ax1.bar(years, vals, bottom=bottom, width=0.72, color=color,
                edgecolor="white", linewidth=0.25, label=province)
        bottom += vals

    ax1.set_ylabel("重点省份粗钢产量（亿吨）")
    ax1.set_xlabel("年份")
    ax1.set_xlim(2019.4, 2035.6)
    ax1.set_ylim(0, max(bottom) * 1.16)
    ax1.set_xticks(np.arange(2020, 2036, 5))
    ax1.yaxis.set_major_locator(MultipleLocator(1))
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.spines[["top"]].set_visible(False)

    ax2 = ax1.twinx()
    national = data["全国"]
    line, = ax2.plot(years, national.values, color="#D62728", marker="o",
                     markersize=5.5, linewidth=2.6, label="全国粗钢产量", zorder=8)
    ax2.set_ylabel("全国粗钢产量（亿吨）")
    ax2.set_ylim(0, national.max() * 1.18)
    ax2.spines[["top"]].set_visible(False)
    for year in [2020, 2024, 2025, 2030, 2035]:
        value = float(national.loc[year])
        ax2.text(year, value + national.max() * 0.025, f"{value:.2f}",
                 ha="center", va="bottom", fontsize=15, color="#B22222", fontweight="bold")

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles + [line], labels + ["全国粗钢产量"], loc="center left",
               bbox_to_anchor=(1.08, 0.5), frameon=False, labelspacing=0.75)
    ax1.set_title("全国及重点省份粗钢产量（最新基准情景）", fontweight="bold", pad=18)
    fig.subplots_adjust(left=0.10, right=0.75, top=0.96, bottom=0.18)
    out = OUT_DIR / "全国及重点省份粗钢产量_最新基准情景.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    production, audit = build_production()
    production.to_excel(OUT_DIR / "分省钢铁产量_情景修正版.xlsx")
    audit.to_csv(OUT_DIR / "钢铁产量更新_核查表.csv", index=False, encoding="utf-8-sig")
    print(plot(production))
    print(f"省级合计最大差额：{audit['省级合计差额_吨'].abs().max():.6f} 吨")

