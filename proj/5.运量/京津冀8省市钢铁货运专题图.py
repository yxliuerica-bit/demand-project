from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath

ROOT = Path(r"C:\Users\admin\Desktop\货运预测李童+高洁\京津冀货运发展")
BASE = ROOT / "output" / "4.运量" / "钢铁货运区域匹配_情景修正版_20261020"
DATA = BASE / "数据"
OUT = BASE / "京津冀8省市专题图"
OUT.mkdir(parents=True, exist_ok=True)

for font in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    if any(x.name == font for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [font]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size":20,"axes.labelsize":20,"xtick.labelsize":18,"ytick.labelsize":18,"legend.fontsize":20})

REGION = ["北京市","天津市","河北省","河南省","山东省","山西省","内蒙古自治区","陕西省"]
SHORT = {"北京市":"北京","天津市":"天津","河北省":"河北","山西省":"山西","内蒙古自治区":"内蒙古","山东省":"山东","河南省":"河南","陕西省":"陕西"}
COLORS = {p:c for p,c in zip(REGION,["#DCEAF5","#AFC9E2","#DDECCB","#A8CF95","#F3D1D3","#E89CA5","#F8E1BE","#F3BB82"])}

bal = pd.read_csv(DATA / "2025-2035省级钢铁供需与货运汇总.csv")
od = pd.read_csv(DATA / "2025-2035省际钢铁OD预测.csv")
national = pd.read_csv(DATA / "2025-2035全国钢铁货运量汇总.csv").set_index("年份")

# 省级货运需求采用装卸/运输触发口径：生产端原料运入 + 成品钢省际流入 + 省际流出。
r = bal[bal["省份"].isin(REGION)].copy()
r["钢铁相关货运量_亿吨"] = (r["生产端原料运输量_吨"] + r["省际流入量_吨"] + r["省际流出量_吨"]) / 1e8
pivot = r.pivot(index="年份", columns="省份", values="钢铁相关货运量_亿吨").reindex(columns=REGION)
pivot.rename(columns=SHORT).to_csv(OUT / "图1_京津冀8省市钢铁货运量_数据.csv", encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(15,5))
bottom = np.zeros(len(pivot))
for p in REGION:
    vals = pivot[p].to_numpy()
    ax.bar(pivot.index, vals, bottom=bottom, width=.72, color=COLORS[p], label=SHORT[p], edgecolor="white", linewidth=.25)
    bottom += vals
ax2=ax.twinx(); national_line=national.loc[pivot.index,"钢铁物流总量_亿吨"]
line,=ax2.plot(pivot.index,national_line,color="#D62728",marker="o",lw=2.4,ms=5,label="全国钢铁物流总量",zorder=5)
for x,y in zip(pivot.index,national_line): ax2.text(x,y+.20,f"{y:.1f}",ha="center",va="bottom",fontsize=15,color="#D62728")
ax2.set_ylim(top=national_line.max() + 1.5)
ax.set_xlabel("年份"); ax.set_ylabel("八省市钢铁相关货运量（亿吨）"); ax2.set_ylabel("全国钢铁物流总量（亿吨）")
ax.set_xticks(pivot.index); ax.grid(axis="y",ls="--",alpha=.30); ax.set_axisbelow(True)
handles,labels=ax.get_legend_handles_labels(); ax.legend(handles+[line],labels+[line.get_label()],ncol=1,frameon=True,loc="center left",bbox_to_anchor=(1.2,.5))
fig.tight_layout(); fig.savefig(OUT / "01_京津冀8省市钢铁货运量.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# 2035弦图：八省市内部OD + 八省市与区外的双向联系；区外流向按每个省聚合。
yod = od[od["年份"]==2035].copy()
inside = yod[yod["起点省份"].isin(REGION) & yod["终点省份"].isin(REGION)].copy()
out_ext = yod[yod["起点省份"].isin(REGION) & ~yod["终点省份"].isin(REGION)].groupby("起点省份",as_index=False)["省际成品钢预测量_吨"].sum()
out_ext["终点省份"]="区外需求"
in_ext = yod[~yod["起点省份"].isin(REGION) & yod["终点省份"].isin(REGION)].groupby("终点省份",as_index=False)["省际成品钢预测量_吨"].sum()
in_ext["起点省份"]="区外供给"
flows = pd.concat([inside[["起点省份","终点省份","省际成品钢预测量_吨"]],out_ext[["起点省份","终点省份","省际成品钢预测量_吨"]],in_ext[["起点省份","终点省份","省际成品钢预测量_吨"]]],ignore_index=True)
flows = flows[flows["省际成品钢预测量_吨"] > 0].sort_values("省际成品钢预测量_吨",ascending=False)
flows.assign(预测量_万吨=flows["省际成品钢预测量_吨"]/1e4).to_csv(OUT / "图2_2035京津冀8省市货运流向_数据.csv",index=False,encoding="utf-8-sig")

nodes = REGION + ["区外需求","区外供给"]
labels = {**SHORT,"区外需求":"区外需求","区外供给":"区外供给"}
node_colors = {**COLORS,"区外需求":"#A8A8A8","区外供给":"#777777"}
# 圆环弦图：每个节点扇区按流入+流出总量分配，内部带宽与OD量成比例。
incident={n:0.0 for n in nodes}
for _,q in flows.iterrows():
    incident[q["起点省份"]]+=q["省际成品钢预测量_吨"]; incident[q["终点省份"]]+=q["省际成品钢预测量_吨"]
gap=np.deg2rad(3.0); available=2*np.pi-gap*len(nodes); total=sum(incident.values())
spans={}; cur=np.pi/2
for n in nodes:
    span=available*incident[n]/total
    spans[n]=(cur-span,cur); cur-=span+gap
cursor={n:spans[n][0] for n in nodes}
segments=[]
for _,q in flows.sort_values("省际成品钢预测量_吨",ascending=False).iterrows():
    o,t,w=q["起点省份"],q["终点省份"],q["省际成品钢预测量_吨"]
    so=available*w/total; st=available*w/total
    oa=(cursor[o],cursor[o]+so); ta=(cursor[t],cursor[t]+st); cursor[o]+=so; cursor[t]+=st
    segments.append((o,t,w,oa,ta))
fig,ax=plt.subplots(figsize=(13.8,12.9)); ax.set_aspect("equal"); ax.axis("off")
for n,(a0,a1) in spans.items():
    ax.add_patch(Wedge((0,0),1.0,np.degrees(a0),np.degrees(a1),width=.105,facecolor=node_colors[n],edgecolor="white",lw=1.2,zorder=3))
    am=(a0+a1)/2; x,y=1.13*np.cos(am),1.13*np.sin(am)
    ha="left" if x>.08 else ("right" if x<-.08 else "center")
    ax.text(x,y,labels[n],ha=ha,va="center",fontsize=19.5,fontweight="bold",rotation=0)
def point(theta,r=.885): return np.array([r*np.cos(theta),r*np.sin(theta)])
for o,t,w,oa,ta in sorted(segments,key=lambda z:z[2]):
    s0,s1=point(oa[0]),point(oa[1]); t0,t1=point(ta[0]),point(ta[1])
    verts=[s0,(0,0),(0,0),t1,t0,(0,0),(0,0),s1,s0]
    codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts,codes),facecolor=node_colors[o],edgecolor="none",alpha=.38,zorder=1))
ax.text(0,-1.22,"带宽表示2035年省际成品钢预测运输量；颜色表示起点",ha="center",fontsize=16.5,color="#555")
ax.set_xlim(-1.34,1.34); ax.set_ylim(-1.31,1.28)
fig.tight_layout(); fig.savefig(OUT / "02_2035京津冀8省市钢铁货运流向弦图.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)

# 2035桑基图：起点省份 -> 运输方式 -> 终点省份。
mode_columns = {"公路": "公路_吨", "水运": "水运_吨", "铁路": "铁路_吨"}
mode_colors = {"公路": "#ED7D31", "水运": "#5B9BD5", "铁路": "#4472C4"}
mode_od = pd.read_csv(DATA / "2025-2035省际钢铁OD运输方式匹配.csv")
mode_od = mode_od[mode_od["年份"] == 2035].copy()
mode_od = mode_od[
    mode_od["起点省份"].isin(REGION) | mode_od["终点省份"].isin(REGION)
].copy()
mode_od["图起点"] = np.where(mode_od["起点省份"].isin(REGION), mode_od["起点省份"], "区外供给")
mode_od["图终点"] = np.where(mode_od["终点省份"].isin(REGION), mode_od["终点省份"], "区外需求")
mode_flows = mode_od.groupby(["图起点", "图终点"], as_index=False)[list(mode_columns.values())].sum()
selected_pairs = flows[["起点省份", "终点省份"]].rename(
    columns={"起点省份": "图起点", "终点省份": "图终点"}
)
mode_flows = mode_flows.merge(
    selected_pairs, on=["图起点", "图终点"], how="inner", validate="one_to_one"
)
if len(mode_flows) != len(selected_pairs):
    raise ValueError("部分已筛选OD在运输方式匹配数据中缺失，无法绘制三列桑基图。")
mode_flows["方式合计_吨"] = mode_flows[list(mode_columns.values())].sum(axis=1)
expected_flows = flows[["起点省份", "终点省份", "省际成品钢预测量_吨"]].rename(
    columns={
        "起点省份": "图起点",
        "终点省份": "图终点",
        "省际成品钢预测量_吨": "OD预测量_吨",
    }
)
mode_check = mode_flows.merge(expected_flows, on=["图起点", "图终点"], validate="one_to_one")
if not np.allclose(mode_check["方式合计_吨"], mode_check["OD预测量_吨"], rtol=1e-9, atol=1e-3):
    raise ValueError("运输方式分项合计与OD预测量不一致，无法绘制三列桑基图。")

local = bal[(bal["年份"] == 2035) & bal["省份"].isin(REGION)][
    ["省份", "本地匹配量_吨"]
].copy()
mode_share = pd.read_csv(DATA / "钢铁调查校准省级方式份额.csv").set_index("province")
local_water_provinces = {"天津市", "河北省", "山东省"}
local_mode_rows = []
for _, row in local.iterrows():
    province = row["省份"]
    base = mode_share.loc[province, ["公路", "铁路", "水运"]].to_numpy(dtype=float)
    score = base * np.array([0.65, 2.00, 1.30]) * np.array([1.75, 0.75, 0.35])
    if province not in local_water_provinces:
        score[2] = 0
    if score.sum() <= 0:
        raise ValueError(f"{province}缺少有效的本地运输方式份额，无法分配本地匹配量。")
    share = score / score.sum()
    local_mode_rows.append({
        "图起点": province,
        "图终点": province,
        "公路_吨": row["本地匹配量_吨"] * share[0],
        "水运_吨": row["本地匹配量_吨"] * share[2],
        "铁路_吨": row["本地匹配量_吨"] * share[1],
        "本地匹配量_吨": row["本地匹配量_吨"],
    })
local_flows = pd.DataFrame(local_mode_rows)
mode_flows["流向类别"] = "省际OD"
local_flows["流向类别"] = "省内本地匹配"
plot_flows = pd.concat([mode_flows, local_flows], ignore_index=True)

mode_names = {column: name for name, column in mode_columns.items()}
plot_flows["方式合计_吨"] = plot_flows[list(mode_columns.values())].sum(axis=1)
origin_mode = plot_flows.melt(
    id_vars=["图起点"],
    value_vars=list(mode_columns.values()),
    var_name="方式字段",
    value_name="运输量_吨",
).rename(columns={"图起点": "起点"})
origin_mode["运输方式"] = origin_mode["方式字段"].map(mode_names)
origin_mode = origin_mode[origin_mode["运输量_吨"] > 0].reset_index(drop=True)
mode_destination = plot_flows.melt(
    id_vars=["图终点"],
    value_vars=list(mode_columns.values()),
    var_name="方式字段",
    value_name="运输量_吨",
).rename(columns={"图终点": "终点"})
mode_destination["运输方式"] = mode_destination["方式字段"].map(mode_names)
mode_destination = mode_destination[mode_destination["运输量_吨"] > 0].reset_index(drop=True)

origin_totals = origin_mode.groupby("起点")["运输量_吨"].sum().to_dict()
mode_totals = origin_mode.groupby("运输方式")["运输量_吨"].sum().to_dict()
destination_totals = mode_destination.groupby("终点")["运输量_吨"].sum().to_dict()
total_flow = origin_mode["运输量_吨"].sum()
sankey_top, sankey_bottom, sankey_gap = .91, .09, .04

def sankey_positions(totals, order):
    names = [name for name in order if totals.get(name, 0) > 0]
    scale = (sankey_top - sankey_bottom - sankey_gap * (len(names) - 1)) / total_flow
    positions = {}
    top_edge = sankey_top
    for name in names:
        height = totals[name] * scale
        positions[name] = (top_edge - height, top_edge)
        top_edge -= height + sankey_gap
    return positions, scale

source_order = sorted(origin_totals, key=origin_totals.get, reverse=True)
destination_order = sorted(destination_totals, key=destination_totals.get, reverse=True)
mode_order = list(mode_columns)
source_pos, sankey_scale = sankey_positions(origin_totals, source_order)
mode_pos, _ = sankey_positions(mode_totals, mode_order)
destination_pos, _ = sankey_positions(destination_totals, destination_order)

def add_sankey_links(ax, edges, source_key, target_key, source_positions, target_positions, x0, x1, color_for):
    source_cursors = {name: bounds[0] for name, bounds in source_positions.items()}
    target_cursors = {name: bounds[0] for name, bounds in target_positions.items()}
    source_segments, target_segments = {}, {}
    for source, group in edges.groupby(source_key):
        ordered = group.sort_values(
            target_key,
            key=lambda col: col.map(lambda name: sum(target_positions[name]) / 2),
        )
        for index, row in ordered.iterrows():
            height = row["运输量_吨"] * sankey_scale
            lower = source_cursors[source]
            source_segments[index] = (lower, lower + height)
            source_cursors[source] += height
    for target, group in edges.groupby(target_key):
        ordered = group.sort_values(
            source_key,
            key=lambda col: col.map(lambda name: sum(source_positions[name]) / 2),
        )
        for index, row in ordered.iterrows():
            height = row["运输量_吨"] * sankey_scale
            lower = target_cursors[target]
            target_segments[index] = (lower, lower + height)
            target_cursors[target] += height

    control = (x1 - x0) * .42
    for index, row in edges.iterrows():
        source_low, source_high = source_segments[index]
        target_low, target_high = target_segments[index]
        vertices = [
            (x0, source_low), (x0 + control, source_low),
            (x1 - control, target_low), (x1, target_low),
            (x1, target_high), (x1 - control, target_high),
            (x0 + control, source_high), (x0, source_high), (x0, source_low),
        ]
        codes = [
            MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        ax.add_patch(PathPatch(
            MplPath(vertices, codes),
            facecolor=color_for(row),
            edgecolor="none",
            alpha=.34,
            zorder=1,
        ))

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
source_x, mode_x, destination_x, node_width = .09, .49, .87, .018
origin_edges = origin_mode.rename(columns={"起点": "源节点"})
destination_edges = mode_destination.rename(columns={"终点": "目标节点"})
source_label_y = dict(zip(
    source_pos,
    np.linspace(sankey_top - .015, sankey_bottom + .015, len(source_pos)),
))
destination_label_y = dict(zip(
    destination_pos,
    np.linspace(sankey_top - .015, sankey_bottom + .015, len(destination_pos)),
))
add_sankey_links(
    ax, origin_edges, "源节点", "运输方式", source_pos, mode_pos,
    source_x + node_width, mode_x, lambda row: node_colors[row["源节点"]],
)
add_sankey_links(
    ax, destination_edges, "运输方式", "目标节点", mode_pos, destination_pos,
    mode_x + node_width, destination_x, lambda row: mode_colors[row["运输方式"]],
)
for name, (lower, upper) in source_pos.items():
    ax.add_patch(plt.Rectangle(
        (source_x, lower), node_width, upper-lower,
        facecolor=node_colors[name], edgecolor="white", lw=.7, zorder=2,
    ))
    center = (lower + upper) / 2
    label_y = source_label_y[name]
    ax.plot(
        [source_x-.006, source_x-.032], [center, label_y],
        color="#777777", lw=.6, alpha=.55, zorder=3,
    )
    ax.text(source_x-.038, label_y, labels[name], ha="right", va="center", fontsize=17)
for mode, (lower, upper) in mode_pos.items():
    ax.add_patch(plt.Rectangle(
        (mode_x, lower), node_width, upper-lower,
        facecolor=mode_colors[mode], edgecolor="white", lw=.7, zorder=2,
    ))
    ax.text(
        mode_x+node_width+.012, (lower+upper)/2,
        f"{mode} {mode_totals[mode]/total_flow:.1%}",
        ha="left", va="center", fontsize=18,
    )
for name, (lower, upper) in destination_pos.items():
    ax.add_patch(plt.Rectangle(
        (destination_x, lower), node_width, upper-lower,
        facecolor=node_colors[name], edgecolor="white", lw=.7, zorder=2,
    ))
    center = (lower + upper) / 2
    label_y = destination_label_y[name]
    ax.plot(
        [destination_x+node_width+.006, destination_x+node_width+.032], [center, label_y],
        color="#777777", lw=.6, alpha=.55, zorder=3,
    )
    ax.text(destination_x+node_width+.038, label_y, labels[name], ha="left", va="center", fontsize=17)
ax.text(source_x+node_width/2, .95, "起点省份", ha="center", va="bottom", fontsize=21, fontweight="bold")
ax.text(mode_x+node_width/2, .95, "运输方式及占比", ha="center", va="bottom", fontsize=21, fontweight="bold")
ax.text(destination_x+node_width/2, .95, "终点省份", ha="center", va="bottom", fontsize=21, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "03_2035京津冀8省市钢铁货运流向桑基图.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
plot_flows.assign(预测量_万吨=plot_flows["方式合计_吨"] / 1e4).to_csv(
    OUT / "图3_2035京津冀8省市桑基图流向数据.csv", index=False, encoding="utf-8-sig"
)

summary = pd.DataFrame({
    "指标":["2035八省市钢铁相关货运总需求（装卸口径，亿吨）","2035八省市内部OD（亿吨）","2035八省市流向区外（亿吨）","2035区外流入八省市（亿吨）"],
    "数值":[pivot.loc[2035].sum(),inside["省际成品钢预测量_吨"].sum()/1e8,out_ext["省际成品钢预测量_吨"].sum()/1e8,in_ext["省际成品钢预测量_吨"].sum()/1e8]
})
mode_value_columns = ["公路_吨", "水运_吨", "铁路_吨"]
mode_totals = plot_flows[mode_value_columns].sum()
mode_total = mode_totals.sum()
for mode, column in zip(["公路", "水运", "铁路"], mode_value_columns):
    summary.loc[len(summary)] = [f"2035桑基图{mode}占比", mode_totals[column] / mode_total]
summary.to_csv(OUT/"专题图结果摘要.csv",index=False,encoding="utf-8-sig")
mode_summary_rows = [{
    "流向类别": "合计",
    "运输量_吨": mode_total,
    **{f"{mode}_吨": mode_totals[column] for mode, column in zip(["公路", "水运", "铁路"], mode_value_columns)},
    **{f"{mode}_占比": mode_totals[column] / mode_total for mode, column in zip(["公路", "水运", "铁路"], mode_value_columns)},
}]
for flow_type, grouped in plot_flows.groupby("流向类别"):
    totals = grouped[mode_value_columns].sum()
    flow_total = totals.sum()
    mode_summary_rows.append({
        "流向类别": flow_type,
        "运输量_吨": flow_total,
        **{f"{mode}_吨": totals[column] for mode, column in zip(["公路", "水运", "铁路"], mode_value_columns)},
        **{f"{mode}_占比": totals[column] / flow_total for mode, column in zip(["公路", "水运", "铁路"], mode_value_columns)},
    })
pd.DataFrame(mode_summary_rows).to_csv(
    OUT / "2035公水铁运输方式占比.csv", index=False, encoding="utf-8-sig"
)
print(summary.to_json(orient="records",force_ascii=True,indent=2))
print(str(OUT).encode("unicode_escape").decode())
