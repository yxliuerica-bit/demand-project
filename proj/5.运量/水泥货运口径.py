"""水泥综合货运统一口径；可用已有结果更新，不运行需求、产量分配或OD模型。"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

SOURCE = "《京津冀及周边地区大宗货类全链条调查及货运发展趋势研判研究报告-v4(tp)》正文第94—95页、图5-6"
REGION = ["北京市", "天津市", "河北省", "河南省", "山东省", "山西省", "内蒙古自治区", "陕西省"]
TOTAL = "水泥行业综合货运需求_吨"
RATE = "水泥行业货运综合系数_吨每吨水泥"
INPUT = "生产端原辅料投入参考量_吨"
NOTE = f"""系数来源：{SOURCE}。
全国、各省及八省市统一采用：水泥产量×年度综合货运系数。
2025/2030/2035年系数为2.80/2.79/2.78吨/吨水泥，中间年份线性插值。
2024年暂按2.80回算，为历史产量对应的模型估算，不是实测运输量。
综合量不再加上原辅料分项、熟料消耗、炉渣产生或省际水泥流入流出。
原辅料投入参考量与成品水泥OD单独保留；旧字段“生产端原辅料运输需求”仅为兼容别名，不证明全部投入均发生外部运输。
报告2022年八省市行业运输量为14.2亿吨；图5-6约9.5亿吨对应2035年，不作为2024年现状基准。
"""


def freight_rate(year):
    """2024历史回算使用2.80；预测沿用报告系数，禁止静默外推。"""
    if not 2024 <= int(year) <= 2035:
        raise ValueError(f"无已确认的水泥货运系数：{year}")
    return float(np.interp(year, [2025, 2030, 2035], [2.80, 2.79, 2.78]))


def consistent_tables(bal, ann):
    """保留原分项、OD和兼容字段，仅增加同口径综合量及投入参考量。"""
    bal, ann = bal.copy(), ann.copy()
    bal[RATE] = bal["年份"].map(freight_rate)
    bal[TOTAL] = bal["产量_吨"] * bal[RATE]
    bal[INPUT] = bal["生产端原辅料运输需求_吨"]
    ann[RATE] = ann["年份"].map(freight_rate)
    ann["水泥行业综合货运需求_亿吨"] = ann["全国水泥产量_亿吨"] * ann[RATE]
    # 保留全国原字段兼容下游，但确保它与新字段只有一个计算来源。
    ann["报告口径水泥货运总需求_亿吨"] = ann["水泥行业综合货运需求_亿吨"]
    ann["生产端原辅料投入参考量_亿吨"] = ann["生产端原辅料运输需求_亿吨"]
    regional = bal[bal["省份"].isin(REGION)].groupby("年份")[TOTAL].sum() / 1e8
    ann["八省市水泥行业综合货运需求_亿吨"] = ann["年份"].map(regional)
    national = bal.groupby("年份")[TOTAL].sum() / 1e8
    if not np.allclose(ann["年份"].map(national), ann["水泥行业综合货运需求_亿吨"], rtol=1e-10, atol=1e-8):
        raise ValueError("分省综合货运需求与全国控制量不一致")
    return bal, ann


def historical_freight(production, provinces):
    """同一产量表的2024历史回算和2025—2035预测；不拼接旧报告产量。"""
    rows = []
    for year, values in production.loc[2024:2035, provinces].iterrows():
        for province in provinces:
            amount = float(values[province])
            rows.append({"年份": int(year), "省份": province, "产量_吨": amount,
                         RATE: freight_rate(year), TOTAL: amount * freight_rate(year),
                         "数据属性": "历史产量回算（非实测运量）" if year == 2024 else "产量情景对应货运估算"})
    result = pd.DataFrame(rows)
    if result[["产量_吨", TOTAL]].isna().any().any() or (result["产量_吨"] < 0).any():
        raise ValueError("综合货运计算所用产量缺失或为负")
    return result


def export_history_chart(history, data_dir, figure_dir):
    import matplotlib.pyplot as plt
    data_dir, figure_dir = Path(data_dir), Path(figure_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(data_dir / "2024-2035分省水泥综合货运需求.csv", index=False, encoding="utf-8-sig")
    regional = history[history["省份"].isin(REGION)].pivot(index="年份", columns="省份", values=TOTAL).reindex(columns=REGION) / 1e8
    national = history.groupby("年份")[TOTAL].sum() / 1e8
    summary = pd.DataFrame({"全国综合货运需求_亿吨": national, "八省市综合货运需求_亿吨": regional.sum(axis=1)})
    summary[RATE] = summary.index.map(freight_rate)
    summary["数据属性"] = ["历史产量回算（非实测运量）" if y == 2024 else "产量情景对应货运估算" for y in summary.index]
    summary.to_csv(data_dir / "2024-2035全国及八省市水泥综合货运需求.csv", encoding="utf-8-sig")
    short = {p: p.replace("省", "").replace("市", "").replace("自治区", "") for p in REGION}
    regional.rename(columns=short).to_csv(figure_dir / "图1_八省市水泥货运量_数据.csv", encoding="utf-8-sig")
    colors = ["#DCEAF5", "#AFC9E2", "#DDECCB", "#A8CF95", "#F3D1D3", "#E89CA5", "#F8E1BE", "#F3BB82"]
    # 与原整体预测脚本的图1使用相同画布、字号和布局。
    plt.rcParams.update({"font.size": 20, "axes.labelsize": 20,
                         "xtick.labelsize": 18, "ytick.labelsize": 18,
                         "legend.fontsize": 20})
    fig, ax = plt.subplots(figsize=(15, 5))
    bottom = np.zeros(len(regional))
    for province, color in zip(REGION, colors):
        ax.bar(regional.index, regional[province], bottom=bottom, color=color, edgecolor="white", lw=.3, label=short[province])
        bottom += regional[province].to_numpy()
    ax2 = ax.twinx()
    line, = ax2.plot(national.index, national, color="#D62728", marker="o", lw=2.4, label="全国水泥货运总需求")
    for year, value in national.items():
        ax2.text(year, value + .20, f"{value:.1f}", ha="center", va="bottom", fontsize=15, color="#D62728")
    ax2.set_ylim(top=national.max() + 1.5)
    ax.axvline(2024.5, color="#666", ls="--", lw=1)
    ax.set_xlabel("年份")
    ax.set_ylabel("八省市水泥综合运量（亿吨）")
    ax2.set_ylabel("全国水泥综合货运需求（亿吨）")
    ax.set_xticks(regional.index)
    ax.grid(axis="y", ls="--", alpha=.3)
    ax.set_axisbelow(True)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles + [line], [h.get_label() for h in handles] + [line.get_label()], loc="center left", bbox_to_anchor=(1.2, .5), frameon=True)
    # 口径和2024回算说明保留在配套数据说明中，不增加图外脚注高度。
    fig.tight_layout()
    fig.savefig(figure_dir / "01_京津冀8省市水泥货运量.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    (data_dir / "水泥综合货运统计口径.txt").write_text(NOTE, encoding="utf-8")
    return summary


def refresh_existing(root, date="20261022"):
    """只读原模型结果，在独立目录输出修正数据和主图；原工作簿及OD不覆盖。"""
    root = Path(root)
    source = root / "output" / "4.运量" / f"水泥货运区域匹配_情景修正版_{date}"
    dest = source / "口径一致性修正_20261026"
    bal = pd.read_csv(source / "数据" / "2025-2035省级水泥供需与货运汇总.csv")
    ann = pd.read_csv(source / "数据" / "2025-2035全国水泥货运量汇总.csv")
    production = pd.read_excel(root / "output" / "3.产量分配" / f"水泥需求情景修正版_{date}" / "分省水泥产量_基准情景_修正版.xlsx").set_index("year")
    provinces = bal["省份"].drop_duplicates().tolist()
    revised_bal, revised_ann = consistent_tables(bal, ann)
    # 原列值必须全部保留；全国兼容字段只容许浮点舍入差。
    pd.testing.assert_frame_equal(bal, revised_bal[bal.columns])
    for column in ann.columns:
        if pd.api.types.is_numeric_dtype(ann[column]):
            np.testing.assert_allclose(ann[column], revised_ann[column], rtol=1e-12, atol=1e-8)
        else:
            pd.testing.assert_series_equal(ann[column], revised_ann[column])
    history = historical_freight(production, provinces)
    h = history[history["年份"] >= 2025].set_index(["年份", "省份"])[TOTAL].sort_index()
    np.testing.assert_allclose(h, revised_bal.set_index(["年份", "省份"])[TOTAL].sort_index(), rtol=1e-12)
    summary = export_history_chart(history, dest / "数据", dest / "京津冀8省市专题图")
    revised_bal.to_csv(dest / "数据" / "2025-2035省级水泥供需与货运汇总.csv", index=False, encoding="utf-8-sig")
    revised_ann.to_csv(dest / "数据" / "2025-2035全国水泥货运量汇总.csv", index=False, encoding="utf-8-sig")
    checks = {"未运行完整模型": True, "原始列与分项保持一致": True, "分省全国加总一致": True,
              "2024八省市综合货运需求_亿吨": float(summary.loc[2024, "八省市综合货运需求_亿吨"]),
              "2025八省市综合货运需求_亿吨": float(summary.loc[2025, "八省市综合货运需求_亿吨"]),
              "2025全国综合货运需求_亿吨": float(summary.loc[2025, "全国综合货运需求_亿吨"])}
    (dest / "核验结果.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "说明.md").write_text("# 水泥货运口径一致性修正\n\n" + NOTE + "\n本目录由已有结果静态更新，供主图替换使用；上一级原结果及Excel工作簿保留为修正前版本。未重新运行产量、需求或OD模型。\n", encoding="utf-8")
    return dest, checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--date", default="20261022")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    dest, checks = refresh_existing(args.root, args.date)
    print(json.dumps({"output": str(dest), **checks}, ensure_ascii=False, indent=2))
