# -*- coding: utf-8 -*-
"""钢铁下游需求结构情景修正版。

使用基准模型计算的逐年下游行业份额，将修正后的钢铁总需求分解至建筑、
汽车、铁路和家电行业。这样四个情景与钢铁总需求情景图完全一致，且行业
合计逐年等于对应情景的全国钢铁需求。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


PROJECT = Path(r"C:\Users\admin\Desktop\货运预测李童+高洁\京津冀货运发展")
RAW_BASE = PROJECT / "output" / "2.消费需求预测" / "Sce_基准" / "全国完整数据"
CORRECTED_ROOT = PROJECT / "output" / "2.消费需求预测" / "钢铁需求情景修正版_20261020"
OUT_DIR = CORRECTED_ROOT / "下游产品需求结构"

YEARS = np.arange(2010, 2036)
SCENARIOS = ["低增长", "基准", "中增长", "高增长"]
SECTORS = ["新增建筑面积", "汽车产量", "新增铁路里程", "洗衣机产量", "电冰箱产量", "空调产量"]
DISPLAY_NAMES = {
    "新增建筑面积": "建筑业",
    "汽车产量": "汽车制造",
    "新增铁路里程": "铁路建设",
    "洗衣机产量": "洗衣机",
    "电冰箱产量": "电冰箱",
    "空调产量": "空调",
}
COLORS = {
    "新增建筑面积": "#4E79A7",
    "汽车产量": "#F28E2B",
    "新增铁路里程": "#A0A0A0",
    "空调产量": "#59A14F",
    "电冰箱产量": "#B6992D",
    "洗衣机产量": "#E15759",
}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 40,
    "axes.titlesize": 52,
    "axes.labelsize": 44,
    "xtick.labelsize": 36,
    "ytick.labelsize": 36,
    "legend.fontsize": 36,
})


def national_sector_shares() -> pd.DataFrame:
    source = RAW_BASE / "下游行业用钢需求预测.xlsx"
    parts = {}
    for sector in SECTORS:
        df = pd.read_excel(source, sheet_name=f"{sector}_中间结果", index_col=0)
        df.index = pd.to_numeric(df.index, errors="coerce")
        numeric = df.select_dtypes(include="number")
        parts[sector] = numeric.sum(axis=1).reindex(YEARS).clip(lower=0).fillna(0)
    values = pd.DataFrame(parts, index=YEARS)
    row_sum = values.sum(axis=1).replace(0, np.nan)
    shares = values.div(row_sum, axis=0).fillna(0)
    return shares


def corrected_total(scenario: str) -> pd.Series:
    path = (CORRECTED_ROOT / f"Sce_{scenario}" / "全国完整数据" /
            "分省钢铁消费量预测_情景修正版.xlsx")
    df = pd.read_excel(path, index_col=0)
    df.index = pd.to_numeric(df.index, errors="coerce")
    return pd.to_numeric(df["全国"], errors="coerce").reindex(YEARS)


def build_structures():
    shares = national_sector_shares()
    structures = {}
    audit = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        total = corrected_total(scenario)
        structure = shares.mul(total, axis=0)
        structures[scenario] = structure
        structure.to_excel(OUT_DIR / f"{scenario}_钢铁下游行业需求分解.xlsx")
        gap = structure.sum(axis=1) - total
        for year in YEARS:
            audit.append({
                "情景": scenario,
                "年份": int(year),
                "情景总需求_亿吨": float(total.loc[year] / 1e8),
                "行业分解合计_亿吨": float(structure.loc[year].sum() / 1e8),
                "差额_吨": float(gap.loc[year]),
            })
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(OUT_DIR / "下游行业分解_质量检查.csv", index=False, encoding="utf-8-sig")
    return structures, float(audit_df["差额_吨"].abs().max())


def draw_stack(ax: plt.Axes, df: pd.DataFrame, scenario: str, show_legend=True, show_title=True):
    order = df.mean().sort_values(ascending=False).index.tolist()
    values = df[order] / 1e8
    ax.stackplot(values.index, values.T.values,
                 labels=[DISPLAY_NAMES[x] for x in order],
                 colors=[COLORS[x] for x in order], alpha=0.92, linewidth=0)
    if show_title:
        ax.set_title(f"{scenario}情景", fontweight="bold", pad=12)
    ax.set_xlim(2009, 2037)
    ax.set_ylim(0, values.sum(axis=1).max() * 1.35)
    ax.set_xticks(np.arange(2010, 2036, 5))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(axis="y", ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.99),
            frameon=True,
            facecolor="white",
            framealpha=0.85,
            edgecolor="none",
        )


def plot_single(structure: pd.DataFrame, scenario: str):
    fig, ax = plt.subplots(figsize=(15, 13), dpi=180)
    draw_stack(ax, structure, scenario, show_legend=True, show_title=False)
    ax.set_xlabel("年份")
    ax.set_ylabel("钢铁需求量（亿吨）")
    fig.subplots_adjust(left=0.13, right=0.97, top=0.94, bottom=0.18)
    out = OUT_DIR / f"全国钢铁下游需求结构_{scenario}情景.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


def plot_panel(structures):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=180, sharex=True, sharey=True)
    for ax, scenario in zip(axes.flat, SCENARIOS):
        draw_stack(ax, structures[scenario], scenario, show_legend=False)
    axes[1, 0].set_xlabel("年份")
    axes[1, 1].set_xlabel("年份")
    axes[0, 0].set_ylabel("钢铁需求量（亿吨）")
    axes[1, 0].set_ylabel("钢铁需求量（亿吨）")
    order = structures["基准"].mean().sort_values(ascending=False).index.tolist()
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[x]) for x in order]
    labels = [DISPLAY_NAMES[x] for x in order]
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("全国钢铁下游行业需求结构：四情景对比", fontsize=48, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.13, hspace=0.28, wspace=0.16)
    out = OUT_DIR / "全国钢铁下游需求结构_四情景对比.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    data, max_gap = build_structures()
    plot_single(data["基准"], "基准")
    print(f"输出目录：{OUT_DIR}")
    print(f"行业合计与情景总需求最大差额：{max_gap:.6f} 吨")
