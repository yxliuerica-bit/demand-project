from pathlib import Path
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

ROOT = Path(r"C:\Users\admin\Desktop\货运预测李童+高洁\京津冀货运发展")
DEMAND_FILE = ROOT / "output" / "2.消费需求预测" / "钢铁需求情景修正版_20261020" / "Sce_基准" / "全国完整数据" / "分省钢铁消费量预测_情景修正版.xlsx"
PROD_FILE = ROOT / "output" / "3.产量分配" / "钢铁需求情景修正版_20261020" / "分省钢铁产量_情景修正版.xlsx"
MODE_DIR = ROOT / "data" / "预测原始数据"
OUT = ROOT / "output" / "4.运量" / "钢铁货运区域匹配_情景修正版_20261020"
DATA_OUT, FIG_OUT = OUT / "数据", OUT / "图表"
DATA_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

for candidate in ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    if any(f.name == candidate for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [candidate]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})

COORD = {
"北京市":(116.4,39.9),"天津市":(117.2,39.1),"河北省":(114.5,38.0),"山西省":(112.5,37.9),"内蒙古自治区":(111.7,40.8),
"辽宁省":(123.4,41.8),"吉林省":(125.3,43.9),"黑龙江省":(126.6,45.8),"上海市":(121.5,31.2),"江苏省":(118.8,32.1),
"浙江省":(120.2,30.3),"安徽省":(117.3,31.9),"福建省":(119.3,26.1),"江西省":(115.9,28.7),"山东省":(117.0,36.7),
"河南省":(113.6,34.8),"湖北省":(114.3,30.6),"湖南省":(112.9,28.2),"广东省":(113.3,23.1),"广西壮族自治区":(108.3,22.8),
"海南省":(110.3,20.0),"重庆市":(106.6,29.6),"四川省":(104.1,30.7),"贵州省":(106.7,26.6),"云南省":(102.7,25.0),
"西藏自治区":(91.1,29.7),"陕西省":(108.9,34.3),"甘肃省":(103.8,36.1),"青海省":(101.8,36.6),"宁夏回族自治区":(106.2,38.5),
"新疆维吾尔自治区":(87.6,43.8)}
SHORT = {k:k.replace("省","").replace("市","").replace("自治区","").replace("壮族","").replace("回族","").replace("维吾尔","") for k in COORD}
WATER = {"天津市","河北省","辽宁省","上海市","江苏省","浙江省","福建省","山东省","湖北省","湖南省","广东省","广西壮族自治区","海南省","重庆市","四川省","江西省","安徽省"}

def norm(s):
    return str(s).strip().replace("黑_龙江省","黑龙江省").replace(" ","")

def haversine(a, b):
    lon1, lat1 = map(math.radians, COORD[a]); lon2, lat2 = map(math.radians, COORD[b])
    x = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371 * 2 * math.asin(math.sqrt(x))

def read_wide(path):
    d = pd.read_excel(path)
    d.columns = [norm(c) for c in d.columns]
    d = d.rename(columns={d.columns[0]: "year"})
    keep = ["year"] + [p for p in COORD if p in d.columns]
    return d[keep].set_index("year").apply(pd.to_numeric, errors="coerce")

def read_mode(name):
    d = pd.read_excel(MODE_DIR / name, header=3)
    d = d.iloc[:, :2].copy(); d.columns = ["province", "value"]
    d["province"] = d["province"].map(norm)
    d["value"] = pd.to_numeric(d["value"], errors="coerce").fillna(0)
    return d[d["province"].isin(COORD)].set_index("province")["value"]

def ipf(seed, rows, cols, n=1000, tol=1e-7):
    x = seed.copy().astype(float)
    for _ in range(n):
        rs = x.sum(axis=1); x *= np.divide(rows, rs, out=np.zeros_like(rows), where=rs>0)[:,None]
        cs = x.sum(axis=0); x *= np.divide(cols, cs, out=np.zeros_like(cols), where=cs>0)[None,:]
        if max(np.max(np.abs(x.sum(1)-rows)), np.max(np.abs(x.sum(0)-cols))) < tol: break
    return x

dem, prod = read_wide(DEMAND_FILE), read_wide(PROD_FILE)
# 上游需求文件的全国控制量与31省加总不一致；OD计算前按各省原有份额逐年归一到全国控制量。
dem_source = pd.read_excel(DEMAND_FILE)
dem_source.columns = [norm(c) for c in dem_source.columns]
national_demand = dem_source.set_index("year")["全国"].astype(float)
demand_scale = national_demand / dem.sum(axis=1)
dem = dem.mul(demand_scale, axis=0)
years = list(range(2025, 2036))
provinces = list(COORD)

road = read_mode("公路货运量分省年度数据.xls")
rail = read_mode("铁路货运量分省年度数据.xls")
water = read_mode("水运货运量分省年度数据.xls")
hist = pd.concat([road.rename("公路"), rail.rename("铁路"), water.rename("水运")], axis=1).reindex(provinces).fillna(0)
fallback = hist.sum() / hist.sum().sum()
hist_share = hist.div(hist.sum(axis=1).replace(0,np.nan), axis=0).fillna(fallback)
hist_share.to_csv(DATA_OUT / "2024省级货运方式基础份额.csv", encoding="utf-8-sig")
# Steel-enterprise survey benchmarks from the v4 regional freight report.
# Conveyor/belt volumes are outside the road/rail/water chart; preserve the
# generic water share where the survey does not report a steel-specific value.
steel_share = hist_share.copy()
survey_rows = []
survey_inputs = {
    "河北省": {
        "road": 27000.0, "rail": 5420.68, "water": None,
        "water_source": "2024综合货运方式份额代理；报告未单列钢铁水运量",
        "source": "趋势研究报告v4，第63-65页；公路约2.7亿吨、铁路5420.68万吨",
    },
    "山西省": {
        "road": 2895.8, "rail": 755.1, "water": None,
        "water_source": "省级综合货运水运份额；省内无水运",
        "source": "趋势研究报告v4，第48-50页；调查公路2895.8万吨、铁路755.1万吨",
    },
    "河南省": {
        "road": 4813.0, "rail": 4454.0, "water": 10.0,
        "water_source": "报告调查水运10万吨",
        "source": "趋势研究报告v4，第59页；公路4813万吨、铁路4454万吨、水运10万吨",
    },
}
for province, values in survey_inputs.items():
    base_water = (
        values["water"] / (values["road"] + values["rail"] + values["water"])
        if values["water"] is not None
        else float(hist_share.loc[province, "水运"])
    )
    land_total = values["road"] + values["rail"]
    steel_share.loc[province, "公路"] = (1 - base_water) * values["road"] / land_total
    steel_share.loc[province, "铁路"] = (1 - base_water) * values["rail"] / land_total
    steel_share.loc[province, "水运"] = base_water
    survey_rows.append({
        "province": province,
        "公路": steel_share.loc[province, "公路"],
        "铁路": steel_share.loc[province, "铁路"],
        "水运": steel_share.loc[province, "水运"],
        "调查样本公路量_万吨": values["road"],
        "调查样本铁路量_万吨": values["rail"],
        "调查样本水运量_万吨": values["water"],
        "水运份额处理": values["water_source"],
        "source_note": values["source"],
        "notes": "按调查公路/铁路运量校准陆路比例；皮带等未纳入公水铁分母；水运缺报时沿用省级综合货运水运份额。",
    })
steel_share.to_csv(DATA_OUT / "钢铁调查校准省级方式份额.csv", encoding="utf-8-sig")
pd.DataFrame(survey_rows).to_csv(DATA_OUT / "钢铁调查方式校准依据.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"全国控制量_吨":national_demand,"归一化前31省加总_吨":national_demand/demand_scale,"省级需求归一化系数":demand_scale}).to_csv(DATA_OUT / "省级需求归一化核查.csv", encoding="utf-8-sig")

eaf_points = {2025:.12, 2030:.17, 2035:.22}
def eaf(y): return np.interp(y, list(eaf_points), list(eaf_points.values()))
# 单耗：吨运输物料/吨粗钢；长流程项乘(1-电炉比)，废钢乘电炉比，其他物料覆盖全部流程。
long_coeff, scrap_coeff, other_coeff = 0.22+0.46+1.60+0.20+0.68, 1.20, 0.53

all_bal, all_od, all_modes, annual = [], [], [], []
for y in years:
    p = prod.loc[y, provinces].astype(float).clip(lower=0)
    d = dem.loc[y, provinces].astype(float).clip(lower=0)
    surplus, deficit = (p-d).clip(lower=0), (d-p).clip(lower=0)
    local = np.minimum(p,d)
    domestic_flow = min(surplus.sum(), deficit.sum())
    row_targets = surplus / surplus.sum() * domestic_flow if surplus.sum() else surplus
    dests = deficit[deficit>0].index.tolist(); origins = surplus[surplus>0].index.tolist()
    od_rows=[]
    if origins and dests and domestic_flow>0:
        dist = np.array([[haversine(o,t) for t in dests] for o in origins])
        seed = np.power(np.maximum(dist,100), -1.15)
        mat = ipf(seed, row_targets.loc[origins].to_numpy(), deficit.loc[dests].to_numpy())
        for i,o in enumerate(origins):
            for j,t in enumerate(dests):
                if mat[i,j] > 1:
                    od_rows.append([y,o,t,dist[i,j],mat[i,j]])
    od = pd.DataFrame(od_rows, columns=["年份","起点省份","终点省份","质心距离_km","省际成品钢预测量_吨"])
    # Use steel-survey land-mode benchmarks where available; otherwise use
    # provincial 2024 freight shares. Adjust for distance and water access.
    for _,r in od.iterrows():
        base = (steel_share.loc[r["起点省份"]] + steel_share.loc[r["终点省份"]]) / 2
        score = base.to_numpy() * np.array([0.65,2.00,1.30])
        km = r["质心距离_km"]
        score *= np.array([1.75,0.75,0.35]) if km<500 else (np.array([1.05,1.30,0.80]) if km<1000 else np.array([0.55,1.65,1.20]))
        if not (r["起点省份"] in WATER and r["终点省份"] in WATER): score[2]=0
        share = score/score.sum()
        all_modes.append({"年份":y,"起点省份":r["起点省份"],"终点省份":r["终点省份"],"质心距离_km":km,
                          "省际成品钢预测量_吨":r["省际成品钢预测量_吨"],"公路_吨":r["省际成品钢预测量_吨"]*share[0],
                          "铁路_吨":r["省际成品钢预测量_吨"]*share[1],"水运_吨":r["省际成品钢预测量_吨"]*share[2]})
    inbound = od.groupby("终点省份")["省际成品钢预测量_吨"].sum() if len(od) else pd.Series(dtype=float)
    outbound = od.groupby("起点省份")["省际成品钢预测量_吨"].sum() if len(od) else pd.Series(dtype=float)
    raw_coeff = long_coeff*(1-eaf(y)) + scrap_coeff*eaf(y) + other_coeff
    bal = pd.DataFrame({"年份":y,"省份":provinces,"产量_吨":p.values,"需求量_吨":d.values,"本地匹配量_吨":local.values,
                        "理论盈余_吨":surplus.values,"理论缺口_吨":deficit.values,"省际流出量_吨":[outbound.get(x,0) for x in provinces],
                        "省际流入量_吨":[inbound.get(x,0) for x in provinces],"系统外流出_吨":(surplus-row_targets).values,
                        "生产端原料运输量_吨":(p*raw_coeff).values})
    all_bal.append(bal); all_od.append(od)
    annual.append({"年份":y,"全国粗钢产量_亿吨":p.sum()/1e8,"全国钢铁需求_亿吨":d.sum()/1e8,"生产端原料运输量_亿吨":p.sum()*raw_coeff/1e8,
                   "成品钢本地匹配量_亿吨":local.sum()/1e8,"成品钢省际运输量_亿吨":domestic_flow/1e8,"系统外流出_亿吨":(surplus.sum()-domestic_flow)/1e8,
                   "钢铁物流总量_亿吨":(p.sum()*raw_coeff + local.sum()+domestic_flow+(surplus.sum()-domestic_flow))/1e8})

bal_all = pd.concat(all_bal, ignore_index=True); od_all = pd.concat(all_od, ignore_index=True); mode_all = pd.DataFrame(all_modes); annual_df = pd.DataFrame(annual)
bal_all.to_csv(DATA_OUT / "2025-2035省级钢铁供需与货运汇总.csv", index=False, encoding="utf-8-sig")
od_all.to_csv(DATA_OUT / "2025-2035省际钢铁OD预测.csv", index=False, encoding="utf-8-sig")
mode_all.to_csv(DATA_OUT / "2025-2035省际钢铁OD运输方式匹配.csv", index=False, encoding="utf-8-sig")
annual_df.to_csv(DATA_OUT / "2025-2035全国钢铁货运量汇总.csv", index=False, encoding="utf-8-sig")
with pd.ExcelWriter(DATA_OUT / "钢铁货运预测_区域供需_OD及方式匹配.xlsx", engine="openpyxl") as w:
    annual_df.to_excel(w, "全国汇总", index=False); bal_all.to_excel(w,"省级供需货运",index=False)
    od_all.to_excel(w, "省际OD", index=False); mode_all.to_excel(w,"OD运输方式",index=False)
    hist_share.to_excel(w,"2024方式基础份额"); steel_share.to_excel(w,"钢铁调查校准方式份额")

y=2035; b=bal_all[bal_all["年份"]==y].copy(); od=od_all[od_all["年份"]==y].copy(); md=mode_all[mode_all["年份"]==y].copy()
# 图1：供需平衡
b["净盈余_万吨"]=(b["产量_吨"]-b["需求量_吨"])/1e4; b=b.sort_values("净盈余_万吨")
fig,ax=plt.subplots(figsize=(13.3,5.2)); colors=np.where(b["净盈余_万吨"]>=0,"#2f5597","#e15759")
ax.bar(b["省份"].map(SHORT),b["净盈余_万吨"],color=colors); ax.axhline(0,color="#333",lw=.8); ax.set_ylabel("产量－需求量（万吨）"); ax.tick_params(axis="x",rotation=60); ax.grid(axis="y",ls="--",alpha=.35); fig.tight_layout(); fig.savefig(FIG_OUT/"01_2035省级钢铁供需平衡.png",dpi=300,bbox_inches="tight"); plt.close(fig)
# 图2：主要OD流向示意图
top=od.nlargest(25,"省际成品钢预测量_吨")
fig,ax=plt.subplots(figsize=(13.3,6.2)); ax.scatter([v[0] for v in COORD.values()],[v[1] for v in COORD.values()],s=16,c="#777",zorder=3)
mx=top["省际成品钢预测量_吨"].max()
for _,r in top.iterrows():
    a,bp=COORD[r["起点省份"]],COORD[r["终点省份"]]; width=.6+4*r["省际成品钢预测量_吨"]/mx
    ax.add_patch(FancyArrowPatch(a,bp,arrowstyle="-|>",mutation_scale=7,lw=width,color="#d62728",alpha=.38,connectionstyle="arc3,rad=.08"))
for p,(xv,yv) in COORD.items(): ax.text(xv,yv+.35,SHORT[p],fontsize=8,ha="center")
ax.set_xlim(78,133); ax.set_ylim(17,51); ax.set_xlabel("经度"); ax.set_ylabel("纬度"); ax.grid(ls=":",alpha=.25); fig.tight_layout(); fig.savefig(FIG_OUT/"02_2035省际钢铁主要流向图.png",dpi=300,bbox_inches="tight"); plt.close(fig)
# 图3：方式结构
s=md[["公路_吨","铁路_吨","水运_吨"]].sum()/1e8
fig,ax=plt.subplots(figsize=(9.5,5.2)); bars=ax.bar(["公路","铁路","水运"],s,color=["#ed7d31","#4472c4","#5b9bd5"],width=.58)
ax.set_ylabel("省际成品钢运输量（亿吨）"); ax.grid(axis="y",ls="--",alpha=.35)
for q,v in zip(bars,s): ax.text(q.get_x()+q.get_width()/2,v+.01,f"{v:.2f}\n({v/s.sum():.1%})",ha="center",fontsize=14)
fig.tight_layout(); fig.savefig(FIG_OUT/"03_2035省际钢铁运输方式结构.png",dpi=300,bbox_inches="tight"); plt.close(fig)
# 图4：货运趋势
fig,ax=plt.subplots(figsize=(13.3,5.2)); ax.plot(annual_df["年份"],annual_df["生产端原料运输量_亿吨"],marker="o",lw=2.4,label="生产端原料运输")
ax.plot(annual_df["年份"],annual_df["成品钢省际运输量_亿吨"],marker="o",lw=2.4,label="成品钢省际运输")
ax.plot(annual_df["年份"],annual_df["系统外流出_亿吨"],marker="o",lw=2.4,label="系统外流出")
ax.set_xlabel("年份"); ax.set_ylabel("货运量（亿吨）"); ax.set_xticks(years); ax.grid(ls="--",alpha=.35); ax.legend(ncol=3,frameon=False); fig.tight_layout(); fig.savefig(FIG_OUT/"04_2025-2035钢铁货运量趋势.png",dpi=300,bbox_inches="tight"); plt.close(fig)
# 图5：主要OD
t=top.head(15).sort_values("省际成品钢预测量_吨"); labels=[SHORT[o]+"→"+SHORT[d] for o,d in zip(t["起点省份"],t["终点省份"])]
fig,ax=plt.subplots(figsize=(11.5,6.2)); ax.barh(labels,t["省际成品钢预测量_吨"]/1e4,color="#4472c4"); ax.set_xlabel("预测运输量（万吨）"); ax.grid(axis="x",ls="--",alpha=.35); fig.tight_layout(); fig.savefig(FIG_OUT/"05_2035主要省际OD.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# 图6：2035年主要省际OD运输方式桑基图（展示运输量前10的OD）
top_sankey = md.groupby(["起点省份", "终点省份"], as_index=False)[
    ["公路_吨", "铁路_吨", "水运_吨"]
].sum()
top_sankey["总运输量_吨"] = top_sankey[["公路_吨", "铁路_吨", "水运_吨"]].sum(axis=1)
top_sankey = top_sankey.nlargest(10, "总运输量_吨")
mode_columns = {"公路": "公路_吨", "铁路": "铁路_吨", "水运": "水运_吨"}
mode_colors = {"公路": "#ed7d31", "铁路": "#4472c4", "水运": "#5b9bd5"}
province_colors = {
    province: plt.get_cmap("tab20")(index)
    for index, province in enumerate(provinces)
}

origin_mode = top_sankey.melt(
    id_vars=["起点省份"],
    value_vars=list(mode_columns.values()),
    var_name="运输方式",
    value_name="运输量_吨",
).groupby(["起点省份", "运输方式"], as_index=False)["运输量_吨"].sum()
column_modes = {column: mode for mode, column in mode_columns.items()}
origin_mode["运输方式"] = origin_mode["运输方式"].map(column_modes)
origin_mode = origin_mode[origin_mode["运输量_吨"] > 0]

mode_destination = top_sankey.melt(
    id_vars=["终点省份"],
    value_vars=list(mode_columns.values()),
    var_name="运输方式",
    value_name="运输量_吨",
).groupby(["运输方式", "终点省份"], as_index=False)["运输量_吨"].sum()
mode_destination["运输方式"] = mode_destination["运输方式"].map(column_modes)
mode_destination = mode_destination[mode_destination["运输量_吨"] > 0]

origin_totals = origin_mode.groupby("起点省份")["运输量_吨"].sum().to_dict()
mode_totals = origin_mode.groupby("运输方式")["运输量_吨"].sum().to_dict()
destination_totals = mode_destination.groupby("终点省份")["运输量_吨"].sum().to_dict()
total_flow = sum(origin_totals.values())
column_top, column_bottom, node_gap = 0.91, 0.09, 0.018

def sankey_positions(totals):
    names = sorted(totals, key=totals.get, reverse=True)
    scale = (column_top - column_bottom - node_gap * (len(names) - 1)) / total_flow
    positions = {}
    top_edge = column_top
    for name in names:
        height = totals[name] * scale
        positions[name] = (top_edge - height, top_edge)
        top_edge -= height + node_gap
    return positions, scale

origin_pos, sankey_scale = sankey_positions(origin_totals)
mode_pos, _ = sankey_positions(mode_totals)
destination_pos, _ = sankey_positions(destination_totals)

def sankey_segments(edges, source_key, target_key, source_positions, target_positions):
    source_cursors = {name: bounds[0] for name, bounds in source_positions.items()}
    target_cursors = {name: bounds[0] for name, bounds in target_positions.items()}
    source_segments, target_segments = {}, {}
    for source, group in edges.groupby(source_key):
        ordered = group.sort_values(
            by=target_key,
            key=lambda series: series.map(lambda name: sum(target_positions[name]) / 2),
        )
        for index, row in ordered.iterrows():
            height = row["运输量_吨"] * sankey_scale
            lower = source_cursors[source]
            source_segments[index] = (lower, lower + height)
            source_cursors[source] += height
    for target, group in edges.groupby(target_key):
        ordered = group.sort_values(
            by=source_key,
            key=lambda series: series.map(lambda name: sum(source_positions[name]) / 2),
        )
        for index, row in ordered.iterrows():
            height = row["运输量_吨"] * sankey_scale
            lower = target_cursors[target]
            target_segments[index] = (lower, lower + height)
            target_cursors[target] += height
    return source_segments, target_segments

def add_sankey_links(ax, edges, source_key, target_key, source_positions, target_positions, x0, x1, color_for):
    source_segments, target_segments = sankey_segments(
        edges, source_key, target_key, source_positions, target_positions
    )
    control = (x1 - x0) * 0.42
    for index, row in edges.iterrows():
        source_low, source_high = source_segments[index]
        target_low, target_high = target_segments[index]
        vertices = [
            (x0, source_low),
            (x0 + control, source_low),
            (x1 - control, target_low),
            (x1, target_low),
            (x1, target_high),
            (x1 - control, target_high),
            (x0 + control, source_high),
            (x0, source_high),
            (x0, source_low),
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
            alpha=0.34,
            zorder=1,
        ))

fig, ax = plt.subplots(figsize=(16, 5.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
origin_x, mode_x, destination_x, node_width = 0.11, 0.49, 0.87, 0.018
add_sankey_links(
    ax, origin_mode, "起点省份", "运输方式", origin_pos, mode_pos,
    origin_x + node_width, mode_x,
    lambda row: province_colors[row["起点省份"]],
)
add_sankey_links(
    ax, mode_destination, "运输方式", "终点省份", mode_pos, destination_pos,
    mode_x + node_width, destination_x,
    lambda row: mode_colors[row["运输方式"]],
)
for province, (lower, upper) in origin_pos.items():
    ax.add_patch(plt.Rectangle((origin_x, lower), node_width, upper-lower,
                               facecolor=province_colors[province], edgecolor="white", lw=0.5, zorder=2))
    ax.text(origin_x - 0.012, (lower+upper)/2, SHORT[province], ha="right", va="center", fontsize=9)
for mode, (lower, upper) in mode_pos.items():
    ax.add_patch(plt.Rectangle((mode_x, lower), node_width, upper-lower,
                               facecolor=mode_colors[mode], edgecolor="white", lw=0.5, zorder=2))
    ax.text(mode_x + node_width + 0.012, (lower+upper)/2, mode, ha="left", va="center", fontsize=9)
for province, (lower, upper) in destination_pos.items():
    ax.add_patch(plt.Rectangle((destination_x, lower), node_width, upper-lower,
                               facecolor=province_colors[province], edgecolor="white", lw=0.5, zorder=2))
    ax.text(destination_x + node_width + 0.012, (lower+upper)/2, SHORT[province], ha="left", va="center", fontsize=9)
ax.text(origin_x + node_width/2, 0.97, "起点省份", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.text(mode_x + node_width/2, 0.97, "运输方式", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.text(destination_x + node_width/2, 0.97, "终点省份", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.text(
    0.5, 0.015,
    "2035年省际成品钢运输量前10条OD；带宽表示运输量，单位：吨",
    ha="center", va="bottom", fontsize=9, color="#555555",
)
fig.savefig(FIG_OUT / "06_2035主要省际OD运输方式桑基图.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

checks = {
"target_year":2035,"province_count":len(provinces),"od_nonnegative":bool((od_all["省际成品钢预测量_吨"]>=0).all()),
"mode_nonnegative":bool((mode_all[["公路_吨","铁路_吨","水运_吨"]]>=0).all().all()),
"2035_od_total_ton":float(od["省际成品钢预测量_吨"].sum()),"2035_deficit_total_ton":float(bal_all.query("年份==2035")["理论缺口_吨"].sum()),
"2035_mode_sum_error_ton":float(abs(md[["公路_吨","铁路_吨","水运_吨"]].sum().sum()-od["省际成品钢预测量_吨"].sum())),
"2035_max_destination_error_ton":float((od.groupby("终点省份")["省际成品钢预测量_吨"].sum().sub(bal_all.query("年份==2035").set_index("省份")["理论缺口_吨"],fill_value=0)).abs().max())}
(DATA_OUT/"质量检查.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
latest=annual_df.iloc[-1]; modal=md[["公路_吨","铁路_吨","水运_吨"]].sum(); modal=modal/modal.sum()
report=f"""# 钢铁货运预测、区域供需匹配与运输方式匹配运行说明\n\n## 数据与年份\n- 采用钢铁需求情景修正版和分省产量情景修正版，计算2025—2035年，重点展示2035年。\n- 运输方式基础份额来自项目内2024年分省公路、铁路和水运货运量统计表。\n\n## 方法与假设\n- 原需求文件的全国控制量高于31省加总；为保证口径一致，逐年保留各省原有占比并缩放至全国控制量。2035年缩放系数为{demand_scale.loc[2035]:.6f}。\n- 省内匹配量取产量和需求量的较小值；产量高于需求形成供应盈余，反之形成需求缺口。\n- 省际OD是估算量，不是真实贸易观测。以省级盈余和缺口为双边约束，以省会/自治区首府质心距离的-1.15次方作为阻抗，通过IPF得到完整OD。\n- 全国供应盈余高于国内缺口的部分列为“系统外流出”，可理解为出口、库存变化及口径差，不分配给国内省份。\n- 方式匹配以2024年省级综合货运方式份额为基础，考虑钢铁重货属性（提高铁路权重）和运距；只有双方具备沿海或主要内河运输条件时保留水运。\n- 原料运输量按粗钢产量和物料单耗估算；长流程物料单耗3.16吨/吨钢、短流程废钢1.20吨/吨钢、其他物料0.53吨/吨钢，电炉钢比从2025年12%线性升至2035年22%。\n\n## 2035年结果\n- 粗钢产量：{latest['全国粗钢产量_亿吨']:.3f}亿吨；国内需求：{latest['全国钢铁需求_亿吨']:.3f}亿吨。\n- 生产端原料运输：{latest['生产端原料运输量_亿吨']:.3f}亿吨；成品钢省际运输：{latest['成品钢省际运输量_亿吨']:.3f}亿吨；系统外流出：{latest['系统外流出_亿吨']:.3f}亿吨。\n- 省际成品钢运输方式：公路{modal['公路_吨']:.1%}、铁路{modal['铁路_吨']:.1%}、水运{modal['水运_吨']:.1%}。\n\n## 质量检查与局限\n- 31省均纳入，OD和运输方式结果均为非负；目的地流入与需求缺口最大误差{checks['2035_max_destination_error_ton']:.2f}吨；方式分项加总误差{checks['2035_mode_sum_error_ton']:.2f}吨。\n- 本结果适合项目汇报MVP。由于缺少真实省际钢材贸易流、铁路运价/能力、港口吞吐和线路可达性数据，具体省际流向及方式份额应表述为情景估算。\n"""
report = report.replace(
    "## 方法与假设",
    "## 输出图表\n- 新增 `图表/06_2035主要省际OD运输方式桑基图.png`，横向展示2035年运输量前10条省际OD及其运输方式分配。\n\n## 方法与假设",
)
report = report.replace(
    "- 运输方式基础份额来自项目内2024年分省公路、铁路和水运货运量统计表。",
    "- 方式基准采用2024年省级综合货运结构，并以趋势研究报告v4中河北、山西、河南的钢铁企业调查公路/铁路运量比校准；水运调查缺报时保留省级综合货运水运份额代理。",
)
report = report.replace(
    "- 方式匹配以2024年省级综合货运方式份额为基础，考虑钢铁重货属性（提高铁路权重）和运距；只有双方具备沿海或主要内河运输条件时保留水运。",
    "- 方式匹配对调查覆盖省份采用钢铁企业调查的公路/铁路量比，对其他省份采用2024年综合货运方式份额；再考虑钢铁重货属性和运距，并按水运可达条件筛除水运。调查中的皮带等其他方式不在公、水、铁三类中，水运未明确覆盖的样本省份暂以综合货运水运份额作代理。\n- 生态环境部等五部门《关于推进实施钢铁行业超低排放的意见》提出，钢铁企业进出厂相关大宗物料和产品采用铁路、水路、管道或管状带式输送机等清洁方式运输比例不低于80%；未达到时，汽车运输部分另须采用新能源汽车或达到国六排放标准的汽车。该政策分母包括生产端原料和产品。本模型当前的公水铁结果只覆盖成品钢货流，未建模企业原料来源OD和合规车辆类型，因此不把80%误作铁路+水运份额硬约束；该政策作为运输结构优化的情景方向与边界条件说明。\n- 国务院办公厅《推进多式联运发展优化调整运输结构工作方案（2021—2025年）》关于铁路、水路增长及重点港口大宗货物绿色集疏运的目标用于趋势和政策一致性核验，不直接外推为2035年钢铁方式占比。",
)
(OUT/"运行说明.md").write_text(report,encoding="utf-8")
print(json.dumps({"output":str(OUT),"checks":checks,"2035":latest.to_dict(),"modal_share":modal.to_dict(),"top_od":top.head(5).to_dict("records")},ensure_ascii=True,indent=2))
