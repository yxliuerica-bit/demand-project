# -*- coding: utf-8 -*-
"""钢铁需求情景一致性修正（MVP）。

以现有基准情景的省级需求为中心，用有界情景偏差生成低/中/高增长需求。
历史至 2024 年保持一致，预测从 2025 年开始分叉，并继承钢铁需求结构性回落趋势。
原始结果只读，新结果写入本脚本所在目录。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


PROJECT = Path(r"C:\Users\admin\Desktop\货运预测李童+高洁\京津冀货运发展")
RAW_DEMAND_ROOT = PROJECT / "output" / "2.消费需求预测"
GDP_ROOT = PROJECT / "output" / "scenarios"
OUT_ROOT = PROJECT / "output" / "2.消费需求预测" / "钢铁需求情景修正版_20261020"
SCENARIOS = ["低增长", "基准", "中增长", "高增长"]
HISTORY_END = 2024
MIN_ANNUAL_DECLINE = 0.002
# 2035 年相对基准需求的终值偏差。钢铁需求已进入平台回落期，宏观增长
# 只改变下降幅度，不能抵消材料强度下降并造成远期需求反弹。
TERMINAL_DEVIATION = {
    "低增长": -0.08,
    "基准": 0.00,
    "中增长": 0.01,
    "高增长": 0.025,
}

STYLES = {
    "低增长": {"color": "#4E79A7", "ls": "--"},
    "基准": {"color": "#D62728", "ls": "-"},
    "中增长": {"color": "#F28E2B", "ls": "-."},
    "高增长": {"color": "#59A14F", "ls": ":"},
}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 18,
    "axes.titlesize": 26,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 17,
})


def read_gdp(name: str) -> pd.Series:
    p = GDP_ROOT / f"gdp_predictions_{name}_完整年份.csv"
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_numeric(df.index, errors="coerce")
    return pd.to_numeric(df["全国"], errors="coerce")


def read_baseline() -> pd.DataFrame:
    p = RAW_DEMAND_ROOT / "Sce_基准" / "全国完整数据" / "分省钢铁消费量预测.xlsx"
    df = pd.read_excel(p, index_col=0)
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    # 现有文件的“全国”列与省级列合计不一致。以项目使用的全国列为总量锚点，
    # 保留各省相对份额并归一化，防止情景修正额外抬高全国总量。
    province_cols = [c for c in df.columns if c != "全国"]
    province_sum = df[province_cols].sum(axis=1).replace(0, np.nan)
    df[province_cols] = df[province_cols].mul(df["全国"] / province_sum, axis=0).fillna(0)
    return df


def correct_scenarios():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    baseline = read_baseline()
    gdp_base = read_gdp("基准")
    corrected = {}
    audit_rows = []

    for name in SCENARIOS:
        gdp = read_gdp(name)
        factor = pd.Series(1.0, index=baseline.index, dtype=float)
        future = factor.index > HISTORY_END
        # 从 2024 年的共同起点线性过渡到 2035 年的有界偏差。
        progress = ((factor.index.to_series() - HISTORY_END) / (2035 - HISTORY_END)).clip(0, 1)
        factor.loc[future] = 1.0 + TERMINAL_DEVIATION[name] * progress.loc[future]

        result = baseline.mul(factor, axis=0)
        # 消除基准序列在 2032 年后的小幅反弹。预测期需求逐年不增加，并至少
        # 保持 0.2% 的年降幅；按同一年度系数缩放各省，保留省际结构。
        for year in range(HISTORY_END + 1, 2036):
            previous = float(result.loc[year - 1, "全国"])
            current = float(result.loc[year, "全国"])
            target = min(current, previous * (1 - MIN_ANNUAL_DECLINE))
            if current > 0:
                result.loc[year] = result.loc[year] * (target / current)
        # 保证“全国”与省级列之和一致。
        province_cols = [c for c in result.columns if c != "全国"]
        result["全国"] = result[province_cols].sum(axis=1)
        corrected[name] = result

        scenario_dir = OUT_ROOT / f"Sce_{name}" / "全国完整数据"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        result.to_excel(scenario_dir / "分省钢铁消费量预测_情景修正版.xlsx")

        for year in range(2024, 2036):
            audit_rows.append({
                "情景": name,
                "年份": year,
                "GDP相对基准系数": float((gdp.loc[year] / gdp_base.loc[year]) if year in gdp.index else np.nan),
                "需求调整系数": float(factor.loc[year]),
                "全国钢铁需求_亿吨": float(result.loc[year, "全国"] / 1e8),
            })

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_ROOT / "钢铁需求情景修正_核查表.csv", index=False, encoding="utf-8-sig")

    # 逐年检查：低增长 <= 基准 <= 中增长 <= 高增长。
    totals = pd.DataFrame({k: v["全国"] for k, v in corrected.items()})
    future_totals = totals.loc[2025:2035, SCENARIOS]
    monotonic_ok = bool((future_totals.diff(axis=1).iloc[:, 1:] >= -1e-6).all().all())
    decline_2030_2035 = bool((totals.loc[2035, SCENARIOS] < totals.loc[2030, SCENARIOS]).all())
    yearly_nonincrease = bool((totals.loc[2025:2035, SCENARIOS].diff().iloc[1:] <= 1e-6).all().all())
    conservation_gap = max(
        float((df["全国"] - df[[c for c in df.columns if c != "全国"]].sum(axis=1)).abs().max())
        for df in corrected.values()
    )
    qa = pd.DataFrame({
        "检查项": ["预测期情景顺序", "2025—2035逐年不回升", "各情景2035年低于2030年", "省级合计与全国差额最大值（吨）", "负值数量", "2035年相对基准偏差"],
        "结果": ["通过" if monotonic_ok else "未通过", "通过" if yearly_nonincrease else "未通过", "通过" if decline_2030_2035 else "未通过",
                 conservation_gap, sum(int((df < 0).sum().sum()) for df in corrected.values()),
                 str(TERMINAL_DEVIATION)],
    })
    qa.to_csv(OUT_ROOT / "质量检查.csv", index=False, encoding="utf-8-sig")
    return corrected


def plot(corrected):
    years = np.arange(2010, 2036)
    data = pd.DataFrame({k: v["全国"].reindex(years) / 1e8 for k, v in corrected.items()}, index=years)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    ax.axvspan(2024, 2035, color="#F2F2F2", alpha=0.8, zorder=0)
    ax.fill_between(data.loc[2024:].index, data.loc[2024:].min(axis=1), data.loc[2024:].max(axis=1),
                    color="#BDBDBD", alpha=0.18, label="情景范围")
    ax.plot(data.loc[:2024].index, data.loc[:2024, "基准"], color="#7F7F7F", lw=3, label="历史/校准值")

    label_y = {"低增长": 6.84, "基准": 7.22, "中增长": 7.52, "高增长": 7.82}
    for name in SCENARIOS:
        s = data.loc[2024:, name]
        st = STYLES[name]
        ax.plot(s.index, s.values, color=st["color"], ls=st["ls"], lw=3, label=f"{name}情景")
        ax.annotate(
            f"{name} {s.loc[2035]:.2f}",
            xy=(2035, s.loc[2035]),
            xytext=(2035.15, label_y[name]),
            color=st["color"],
            fontsize=16,
            fontweight="bold",
            va="center",
            arrowprops=dict(arrowstyle="-", color=st["color"], lw=1.2),
        )

    anchor = data.loc[2024, "基准"]
    ax.scatter([2024], [anchor], color="#D62728", s=80, zorder=8)
    ax.annotate(
        f"2024年\n{anchor:.2f}亿吨",
        xy=(2024, anchor),
        xytext=(16, 5),
        textcoords="offset points",
        fontsize=18,
        fontweight="bold",
        color="#8B1A1A",
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color="#8B1A1A", lw=1.8),
    )
    ax.axvline(2024, color="#777777", ls="--", lw=1.2)
    ax.text(2017, 6.15, "历史阶段", color="#666666", fontsize=19, ha="center")
    ax.text(2029.5, 6.15, "预测阶段", color="#666666", fontsize=19, ha="center")
    ax.set_xlabel("年份")
    ax.set_ylabel("钢铁消费需求（亿吨）")
    ax.set_xlim(2010, 2036.3)
    ax.set_ylim(6.0, 11.1)
    ax.set_xticks(np.arange(2010, 2036, 5))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(ls="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.subplots_adjust(left=0.09, right=0.84, top=0.88, bottom=0.13)
    out = OUT_ROOT / "全国钢铁需求多情景_一致性修正版.png"
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    results = correct_scenarios()
    print(plot(results))
    print(OUT_ROOT)
