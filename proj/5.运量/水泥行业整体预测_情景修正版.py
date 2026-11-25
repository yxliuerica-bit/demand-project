from pathlib import Path
from 水泥货运口径 import freight_rate, consistent_tables, historical_freight, export_history_chart, NOTE
import math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Wedge, PathPatch
from matplotlib.path import Path as MplPath

ROOT=Path(r"C:\Users\admin\Desktop\货运预测李童+高洁\京津冀货运发展")
DATE="20261022"
DEM_OUT=ROOT/"output"/"2.消费需求预测"/f"水泥需求情景修正版_{DATE}"
PROD_OUT=ROOT/"output"/"3.产量分配"/f"水泥需求情景修正版_{DATE}"
OUT=ROOT/"output"/"4.运量"/f"水泥货运区域匹配_情景修正版_{DATE}"
DO,FO,RO=OUT/"数据",OUT/"图表",OUT/"京津冀8省市专题图"
for p in [DEM_OUT,PROD_OUT,DO,FO,RO]: p.mkdir(parents=True,exist_ok=True)
for f in ["Microsoft YaHei","SimHei"]:
 if any(x.name==f for x in font_manager.fontManager.ttflist): plt.rcParams["font.sans-serif"]=[f]; break
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams.update({"font.size":20,"axes.labelsize":20,"xtick.labelsize":18,"ytick.labelsize":18,"legend.fontsize":20})

COORD={"北京市":(116.4,39.9),"天津市":(117.2,39.1),"河北省":(114.5,38.0),"山西省":(112.5,37.9),"内蒙古自治区":(111.7,40.8),"辽宁省":(123.4,41.8),"吉林省":(125.3,43.9),"黑龙江省":(126.6,45.8),"上海市":(121.5,31.2),"江苏省":(118.8,32.1),"浙江省":(120.2,30.3),"安徽省":(117.3,31.9),"福建省":(119.3,26.1),"江西省":(115.9,28.7),"山东省":(117.0,36.7),"河南省":(113.6,34.8),"湖北省":(114.3,30.6),"湖南省":(112.9,28.2),"广东省":(113.3,23.1),"广西壮族自治区":(108.3,22.8),"海南省":(110.3,20.0),"重庆市":(106.6,29.6),"四川省":(104.1,30.7),"贵州省":(106.7,26.6),"云南省":(102.7,25.0),"西藏自治区":(91.1,29.7),"陕西省":(108.9,34.3),"甘肃省":(103.8,36.1),"青海省":(101.8,36.6),"宁夏回族自治区":(106.2,38.5),"新疆维吾尔自治区":(87.6,43.8)}
P=list(COORD); SHORT={x:x.replace("省","").replace("市","").replace("自治区","").replace("壮族","").replace("回族","").replace("维吾尔","") for x in P}
REGION=["北京市","天津市","河北省","河南省","山东省","山西省","内蒙古自治区","陕西省"]
RC={x:c for x,c in zip(REGION,["#DCEAF5","#AFC9E2","#DDECCB","#A8CF95","#F3D1D3","#E89CA5","#F8E1BE","#F3BB82"])}
WATER={"天津市","河北省","辽宁省","上海市","江苏省","浙江省","福建省","山东省","湖北省","湖南省","广东省","广西壮族自治区","海南省","重庆市","四川省","江西省","安徽省"}
def norm(x): return str(x).strip().replace("黑_龙江省","黑龙江省").replace(" ","")
def wide(path):
 d=pd.read_excel(path); d.columns=[norm(c) for c in d.columns]; d=d.rename(columns={d.columns[0]:"year"}); return d.set_index("year")
def hav(a,b):
 x1,y1=map(math.radians,COORD[a]); x2,y2=map(math.radians,COORD[b]); q=math.sin((y2-y1)/2)**2+math.cos(y1)*math.cos(y2)*math.sin((x2-x1)/2)**2; return 12742*math.asin(math.sqrt(q))
def ipf(seed,rows,cols):
 x=seed.astype(float)
 for _ in range(1000):
  rs=x.sum(1); x*=np.divide(rows,rs,out=np.zeros_like(rows),where=rs>0)[:,None]
  cs=x.sum(0); x*=np.divide(cols,cs,out=np.zeros_like(cols),where=cs>0)[None,:]
  if max(abs(x.sum(1)-rows).max(),abs(x.sum(0)-cols).max())<1e-6: break
 return x
def read_mode(name):
 d=pd.read_excel(ROOT/"data"/"预测原始数据"/name,header=3).iloc[:,:2]; d.columns=["省份","值"]; d["省份"]=d["省份"].map(norm); d["值"]=pd.to_numeric(d["值"],errors="coerce").fillna(0); return d[d["省份"].isin(P)].set_index("省份")["值"]
def read_official_cement_history():
 d=pd.read_excel(ROOT/"data"/"预测原始数据"/"分省历史水泥产量数据.xls",header=3)
 d["地区"]=d["地区"].map(norm)
 year_cols={int(str(c)[:4]):c for c in d.columns if str(c)[:4].isdigit()}
 history=d.set_index("地区").reindex(P)[list(year_cols.values())].T
 history.index=[int(str(c)[:4]) for c in year_cols.values()]
 history.index.name="年份"
 history.columns.name="省份"
 history=history.apply(pd.to_numeric,errors="coerce")
 if history.isna().any().any():
  missing=history.columns[history.isna().any()].tolist()
  raise ValueError(f"国家统计局水泥产量历史表存在缺失省份或数值：{missing}")
 return history*1e4
def project_provincial_shares(history,years):
 shares=history.div(history.sum(axis=1),axis=0)
 base_year=int(shares.index.max())
 start_year=max(int(shares.index.min()),base_year-5)
 start=shares.loc[start_year].clip(lower=1e-8)
 base=shares.loc[base_year].clip(lower=1e-8)
 annual_change=(np.log(base/start)/(base_year-start_year)).clip(-0.03,0.03)
 result={}
 for year in years:
  raw=base*np.exp(annual_change*(year-base_year))
  result[year]=raw/raw.sum()
 return pd.DataFrame(result).T.reindex(columns=history.columns)

# 1. 全国需求情景：保留旧模型曲线范围，按需求水平重新排列标签。
src={
 "高增长":wide(ROOT/"output"/"2.消费需求预测"/"Sce_基准"/"全国完整数据"/"分省水泥消费量预测.xlsx"),
 "基准":wide(ROOT/"output"/"2.消费需求预测"/"Sce_中增长"/"全国完整数据"/"分省水泥消费量预测.xlsx"),
 "低增长":wide(ROOT/"output"/"2.消费需求预测"/"Sce_低增长"/"全国完整数据"/"分省水泥消费量预测.xlsx")}
years=list(range(2025,2036)); hist=src["高增长"]
scenario=pd.DataFrame(index=years)
for s,d in src.items(): scenario[s]=d.loc[years,"全国"].astype(float)
scenario.index.name="年份"; scenario.to_csv(DEM_OUT/"全国水泥需求情景_修正版.csv",encoding="utf-8-sig")
fig,ax=plt.subplots(figsize=(13.3,5.2))
for s,c in [("高增长","#D62728"),("基准","#ED7D31"),("低增长","#4472C4")]: ax.plot(years,scenario[s]/1e8,lw=2.5,marker="o",label=s,color=c)
ax.set_xlabel("年份"); ax.set_ylabel("全国水泥需求（亿吨）"); ax.set_xticks(years); ax.grid(ls="--",alpha=.3); ax.legend(frameon=False,ncol=3); fig.tight_layout(); fig.savefig(DEM_OUT/"全国水泥需求情景_修正版.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# 2. 基准情景分省需求与产量：水泥以本地匹配为主，旧需求信号向省级产能份额收缩。
old_dem=src["基准"]
official_prod=read_official_cement_history()
new_dem=pd.DataFrame(index=years,columns=P,dtype=float); new_prod=pd.DataFrame(index=years,columns=P,dtype=float)
ratio_2024=float(official_prod.loc[2024].sum())/float(hist.loc[2024,"全国"])
future_shares=project_provincial_shares(official_prod,years)
for y in years:
 raw=old_dem.loc[y,P].clip(lower=0); raw=(raw+1)/(raw.sum()+len(P))
 ps=future_shares.loc[y]
 rel=(raw/ps).clip(.4,2.5); shrunk=ps*(.80+.20*rel); ds=shrunk/shrunk.sum()
 new_dem.loc[y]=ds*scenario.loc[y,"基准"]
 national_prod=16.9e8 if y==2025 else scenario.loc[y,"基准"]*ratio_2024
 new_prod.loc[y]=ps*national_prod
for d,name,out in [(new_dem,"分省水泥需求_基准情景_修正版.xlsx",DEM_OUT)]:
 z=d.copy(); z.insert(0,"year",z.index); z["全国"]=d.sum(axis=1); z["京津冀及周边8省市"]=d[REGION].sum(axis=1); z.to_excel(out/name,index=False)
prod_history_forecast=pd.concat([official_prod.loc[:,P],new_prod],axis=0).sort_index()
prod_history_forecast["全国"]=prod_history_forecast[P].sum(axis=1)
prod_history_forecast["京津冀及周边8省市"]=prod_history_forecast[REGION].sum(axis=1)
prod_history_forecast.insert(0,"year",prod_history_forecast.index)
prod_history_forecast.to_excel(PROD_OUT/"分省水泥产量_基准情景_修正版.xlsx",index=False)
status=pd.DataFrame({"年份":prod_history_forecast.index})
status["数据属性"]=status["年份"].map(lambda y:"国家统计局分省历史实绩" if y<=2024 else ("国家统计局公报全国值+省级结构估算" if y==2025 else "模型情景预测"))
status["产量全国合计_亿吨"]=prod_history_forecast["全国"].to_numpy()/1e8
status["省级结构说明"]=status["年份"].map(lambda y:"分省统计局历史值" if y<=2024 else ("以2024分省份额及2019-2024变化趋势分配" if y>=2025 else ""))
status.to_csv(DO/"水泥产量历史与预测口径.csv",index=False,encoding="utf-8-sig")

# 3. OD、货运量和运输方式。分项消耗系数来自前期报告，货运总量采用报告综合系数。
hist_mode=pd.concat([read_mode("公路货运量分省年度数据.xls").rename("公路"),read_mode("铁路货运量分省年度数据.xls").rename("铁路"),read_mode("水运货运量分省年度数据.xls").rename("水运")],axis=1).reindex(P).fillna(0)
fallback=hist_mode.sum()/hist_mode.sum().sum(); shares=hist_mode.div(hist_mode.sum(1).replace(0,np.nan),axis=0).fillna(fallback)
balances=[]; ods=[]; modes=[]; annual=[]
for y in years:
 p=new_prod.loc[y].astype(float); d=new_dem.loc[y].astype(float); surplus=(p-d).clip(lower=0); deficit=(d-p).clip(lower=0); local=np.minimum(p,d)
 # Report coefficients: 1.54 t limestone and 0.05 t slag per t clinker;
 # 0.8 t clinker and 0.05 t gypsum/mixed materials per t cement.
 coal_rate=float(np.interp(y,[2022,2030,2035],[.157,.147,.143]))
 total_freight_rate=freight_rate(y)
 regional_coal_target=float(np.interp(y,[2025,2030,2035],[.46,.42,.35]))*1e8
 limestone=p*.8*1.54
 kiln_coal=p*.8*coal_rate
 regional_base_coal=kiln_coal.reindex(REGION).sum()
 if regional_base_coal<=0:
  raise ValueError(f"{y}年京津冀及周边八省市缺少水泥产量，无法匹配报告煤炭运输需求。")
 kiln_coal.loc[REGION]=kiln_coal.loc[REGION]*(regional_coal_target/regional_base_coal)
 clinker_use=p*.8
 gypsum_mix=p*.05
 slag=p*.8*.05
 upstream_inputs=limestone+kiln_coal+gypsum_mix  # 投入参考量；未扣厂内自供，不等同实测外部运输
 total=min(surplus.sum(),deficit.sum()); rt=surplus/surplus.sum()*total if surplus.sum() else surplus
 ori=surplus[surplus>0].index.tolist(); dst=deficit[deficit>0].index.tolist(); rows=[]
 if ori and dst:
  dist=np.array([[hav(o,t) for t in dst] for o in ori]); seed=np.power(np.maximum(dist,100),-2.2)
  mat=ipf(seed,rt.loc[ori].to_numpy(),deficit.loc[dst].to_numpy())
  for i,o in enumerate(ori):
   for j,t in enumerate(dst):
    if mat[i,j]>1: rows.append([y,o,t,dist[i,j],mat[i,j]])
 od=pd.DataFrame(rows,columns=["年份","起点省份","终点省份","质心距离_km","省际水泥预测量_吨"]); ods.append(od)
 for _,q in od.iterrows():
  base=(shares.loc[q["起点省份"]]+shares.loc[q["终点省份"]])/2; score=base.to_numpy()*np.array([1.45,.75,.55]); km=q["质心距离_km"]
  score*=np.array([1.8,.65,.35]) if km<400 else (np.array([1.15,1.15,.7]) if km<800 else np.array([.65,1.6,1.1]))
  if not(q["起点省份"] in WATER and q["终点省份"] in WATER): score[2]=0
  sh=score/score.sum(); v=q["省际水泥预测量_吨"]
  modes.append({**q.to_dict(),"公路_吨":v*sh[0],"铁路_吨":v*sh[1],"水运_吨":v*sh[2]})
 inbound=od.groupby("终点省份")["省际水泥预测量_吨"].sum() if len(od) else pd.Series(dtype=float); outbound=od.groupby("起点省份")["省际水泥预测量_吨"].sum() if len(od) else pd.Series(dtype=float)
 balances.append(pd.DataFrame({
  "年份":y,"省份":P,"产量_吨":p.values,"需求量_吨":d.values,
  "本地匹配量_吨":local.values,"理论盈余_吨":surplus.values,"理论缺口_吨":deficit.values,
  "省际流出量_吨":[outbound.get(x,0) for x in P],
  "省际流入量_吨":[inbound.get(x,0) for x in P],
  "系统外流出_吨":(surplus-rt).values,
  "熟料消耗参考量_吨":clinker_use.values,
  "石灰质原料参考需求_吨":limestone.values,
  "水泥窑煤炭参考需求_吨":kiln_coal.values,
  "石膏及混合材参考需求_吨":gypsum_mix.values,
  "炉渣产生参考量_吨":slag.values,
  "生产端原辅料运输需求_吨":upstream_inputs.values,
 }))
 annual.append({
  "年份":y,"全国水泥产量_亿吨":p.sum()/1e8,"全国水泥需求_亿吨":d.sum()/1e8,
  "熟料消耗参考量_亿吨":clinker_use.sum()/1e8,
  "石灰质原料参考需求_亿吨":limestone.sum()/1e8,
  "水泥窑煤炭参考需求_亿吨":kiln_coal.sum()/1e8,
  "八省市报告煤炭需求校准目标_亿吨":regional_coal_target/1e8,
  "八省市报告煤炭需求校准值_亿吨":kiln_coal.reindex(REGION).sum()/1e8,
  "石膏及混合材参考需求_亿吨":gypsum_mix.sum()/1e8,
  "炉渣产生参考量_亿吨":slag.sum()/1e8,
  "生产端原辅料运输需求_亿吨":upstream_inputs.sum()/1e8,
  "水泥行业货运综合系数_吨每吨水泥":total_freight_rate,
  "报告口径水泥货运总需求_亿吨":p.sum()*total_freight_rate/1e8,
  "水泥省际运输量_亿吨":total/1e8,
  "系统外流出_亿吨":(surplus.sum()-total)/1e8,
 })
bal=pd.concat(balances,ignore_index=True); odall=pd.concat(ods,ignore_index=True); mode=pd.DataFrame(modes); ann=pd.DataFrame(annual)
# 全国、各省及八省市共用综合系数；原辅料与OD单列，不叠加进综合量。
bal,ann=consistent_tables(bal,ann)
freight_history=historical_freight(prod_history_forecast,P)
bal.to_csv(DO/"2025-2035省级水泥供需与货运汇总.csv",index=False,encoding="utf-8-sig"); odall.to_csv(DO/"2025-2035省际水泥OD预测.csv",index=False,encoding="utf-8-sig"); mode.to_csv(DO/"2025-2035省际水泥OD运输方式匹配.csv",index=False,encoding="utf-8-sig"); ann.to_csv(DO/"2025-2035全国水泥货运量汇总.csv",index=False,encoding="utf-8-sig")
with pd.ExcelWriter(DO/"水泥整体预测_供需_OD及方式匹配.xlsx",engine="openpyxl") as w:
 scenario.to_excel(w,"需求情景"); new_dem.to_excel(w,"分省需求"); new_prod.to_excel(w,"分省产量"); ann.to_excel(w,"全国货运",index=False); bal.to_excel(w,"省级供需货运",index=False); odall.to_excel(w,"省际OD",index=False); mode.to_excel(w,"OD运输方式",index=False)

# 4. 全国及OD汇报图。
y=2035; b=bal[bal.年份==y].copy(); od=odall[odall.年份==y]; md=mode[mode.年份==y]
b["净盈余_万吨"]=(b["产量_吨"]-b["需求量_吨"])/1e4; b=b.sort_values("净盈余_万吨")
fig,ax=plt.subplots(figsize=(13.3,5.2)); ax.bar(b["省份"].map(SHORT),b["净盈余_万吨"],color=np.where(b["净盈余_万吨"]>=0,"#4472C4","#E15759")); ax.axhline(0,color="#333",lw=.8); ax.set_ylabel("产量－需求量（万吨）"); ax.tick_params(axis="x",rotation=60); ax.grid(axis="y",ls="--",alpha=.35); fig.tight_layout(); fig.savefig(FO/"01_2035省级水泥供需平衡.png",dpi=300,bbox_inches="tight"); plt.close(fig)
top=od.nlargest(25,"省际水泥预测量_吨"); fig,ax=plt.subplots(figsize=(13.3,6.2)); ax.scatter([v[0] for v in COORD.values()],[v[1] for v in COORD.values()],s=24,c="#777",zorder=3); mx=top["省际水泥预测量_吨"].max()
for _,q in top.iterrows(): ax.add_patch(FancyArrowPatch(COORD[q["起点省份"]],COORD[q["终点省份"]],arrowstyle="-|>",mutation_scale=10,lw=.6+4*q["省际水泥预测量_吨"]/mx,color="#D62728",alpha=.38,connectionstyle="arc3,rad=.08"))
for p,(xv,yv) in COORD.items(): ax.text(xv,yv+.35,SHORT[p],fontsize=10,ha="center")
ax.set_xlim(78,133); ax.set_ylim(17,51); ax.set_xlabel("经度"); ax.set_ylabel("纬度"); ax.grid(ls=":",alpha=.25); fig.tight_layout(); fig.savefig(FO/"02_2035省际水泥主要流向图.png",dpi=300,bbox_inches="tight"); plt.close(fig)
s=md[["公路_吨","铁路_吨","水运_吨"]].sum()/1e8; fig,ax=plt.subplots(figsize=(9.5,5.2)); bars=ax.bar(["公路","铁路","水运"],s,color=["#ED7D31","#4472C4","#5B9BD5"],width=.58); ax.set_ylabel("省际水泥运输量（亿吨）"); ax.grid(axis="y",ls="--",alpha=.35); ax.set_ylim(0,s.max()*1.22)
for q,v in zip(bars,s): ax.text(q.get_x()+q.get_width()/2,v+max(s.max()*.015,.01),f"{v:.2f}\n({v/s.sum():.1%})",ha="center",fontsize=16)
fig.tight_layout(); fig.savefig(FO/"03_2035水泥运输方式结构.png",dpi=300,bbox_inches="tight"); plt.close(fig)
fig,ax=plt.subplots(figsize=(13.3,5.2)); ax.plot(ann.年份,ann["生产端原辅料运输需求_亿吨"],marker="o",lw=2.4,label="原辅料投入参考量（非实测运量）"); ax.plot(ann.年份,ann["水泥省际运输量_亿吨"],marker="o",lw=2.4,label="水泥省际运输"); ax.set_xlabel("年份"); ax.set_ylabel("货运量（亿吨）"); ax.set_xticks(years); ax.grid(ls="--",alpha=.35); ax.legend(frameon=False,ncol=2); fig.tight_layout(); fig.savefig(FO/"04_2025-2035水泥货运量趋势.png",dpi=300,bbox_inches="tight"); plt.close(fig)

# 5. 京津冀8省市专题图：同钢铁配色和图式。
# 主图加入2024历史产量回算；全国和八省市均为产量×同一系数。
export_history_chart(freight_history,DO,RO)

# 圆环弦图，含区域内部与区外联系。
inside=od[(od.起点省份.isin(REGION))&(od.终点省份.isin(REGION))]; oe=od[(od.起点省份.isin(REGION))&(~od.终点省份.isin(REGION))].groupby("起点省份",as_index=False)["省际水泥预测量_吨"].sum(); oe["终点省份"]="区外需求"; ie=od[(~od.起点省份.isin(REGION))&(od.终点省份.isin(REGION))].groupby("终点省份",as_index=False)["省际水泥预测量_吨"].sum(); ie["起点省份"]="区外供给"
fl=pd.concat([inside[["起点省份","终点省份","省际水泥预测量_吨"]],oe[["起点省份","终点省份","省际水泥预测量_吨"]],ie[["起点省份","终点省份","省际水泥预测量_吨"]]],ignore_index=True); fl=fl[fl["省际水泥预测量_吨"]>0]; fl.assign(预测量_万吨=fl["省际水泥预测量_吨"]/1e4).to_csv(RO/"图2_2035八省市水泥流向_数据.csv",index=False,encoding="utf-8-sig")
nodes=REGION+["区外需求","区外供给"]; nc={**RC,"区外需求":"#A8A8A8","区外供给":"#777777"}; lab={**SHORT,"区外需求":"区外需求","区外供给":"区外供给"}; inc={n:0. for n in nodes}
for _,q in fl.iterrows(): inc[q.起点省份]+=q.省际水泥预测量_吨; inc[q.终点省份]+=q.省际水泥预测量_吨
gap=np.deg2rad(3); avail=2*np.pi-gap*len(nodes); tot=sum(inc.values()); spans={}; cur=np.pi/2
for n in nodes: sp=avail*inc[n]/tot; spans[n]=(cur-sp,cur); cur-=sp+gap
cursor={n:spans[n][0] for n in nodes}; seg=[]
for _,q in fl.sort_values("省际水泥预测量_吨",ascending=False).iterrows():
 w=q.省际水泥预测量_吨; da=avail*w/tot; oa=(cursor[q.起点省份],cursor[q.起点省份]+da); ta=(cursor[q.终点省份],cursor[q.终点省份]+da); cursor[q.起点省份]+=da; cursor[q.终点省份]+=da; seg.append((q.起点省份,q.终点省份,w,oa,ta))
fig,ax=plt.subplots(figsize=(13.8,12.9)); ax.set_aspect("equal"); ax.axis("off")
for n,(a0,a1) in spans.items():
 ax.add_patch(Wedge((0,0),1,np.degrees(a0),np.degrees(a1),width=.105,facecolor=nc[n],edgecolor="white",lw=1.2,zorder=3)); am=(a0+a1)/2; x,yv=1.13*np.cos(am),1.13*np.sin(am); ax.text(x,yv,lab[n],ha="left" if x>.08 else("right" if x<-.08 else"center"),va="center",fontsize=19.5,fontweight="bold")
pt=lambda a:.885*np.array([np.cos(a),np.sin(a)])
for o,t,w,oa,ta in sorted(seg,key=lambda z:z[2]):
 s0,s1,t0,t1=pt(oa[0]),pt(oa[1]),pt(ta[0]),pt(ta[1]); vv=[s0,(0,0),(0,0),t1,t0,(0,0),(0,0),s1,s0]; cc=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]; ax.add_patch(PathPatch(MplPath(vv,cc),facecolor=nc[o],edgecolor="none",alpha=.38,zorder=1))
ax.text(0,-1.22,"带宽表示2035年省际水泥预测运输量；颜色表示起点",ha="center",fontsize=16.5,color="#555"); ax.set_xlim(-1.34,1.34); ax.set_ylim(-1.31,1.28); fig.tight_layout(); fig.savefig(RO/"02_2035京津冀8省市水泥货运流向弦图.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)

# 三列桑基图：起点省份 -> 运输方式 -> 终点省份，包含省内本地匹配量。
mode_columns={"公路":"公路_吨","水运":"水运_吨","铁路":"铁路_吨"}
mode_colors={"公路":"#ED7D31","水运":"#5B9BD5","铁路":"#4472C4"}
mode_od=pd.DataFrame(modes)
mode_od=mode_od[mode_od["年份"]==2035].copy()
mode_od=mode_od[mode_od["起点省份"].isin(REGION)|mode_od["终点省份"].isin(REGION)].copy()
mode_od["图起点"]=np.where(mode_od["起点省份"].isin(REGION),mode_od["起点省份"],"区外供给")
mode_od["图终点"]=np.where(mode_od["终点省份"].isin(REGION),mode_od["终点省份"],"区外需求")
mode_flows=mode_od.groupby(["图起点","图终点"],as_index=False)[list(mode_columns.values())].sum()
selected_pairs=fl[["起点省份","终点省份"]].rename(columns={"起点省份":"图起点","终点省份":"图终点"})
mode_flows=mode_flows.merge(selected_pairs,on=["图起点","图终点"],how="inner",validate="one_to_one")
if len(mode_flows)!=len(selected_pairs):
 raise ValueError("部分水泥OD在运输方式匹配数据中缺失，无法绘制三列桑基图。")
mode_flows["方式合计_吨"]=mode_flows[list(mode_columns.values())].sum(axis=1)
expected_flows=fl[["起点省份","终点省份","省际水泥预测量_吨"]].rename(columns={"起点省份":"图起点","终点省份":"图终点","省际水泥预测量_吨":"OD预测量_吨"})
mode_check=mode_flows.merge(expected_flows,on=["图起点","图终点"],validate="one_to_one")
if not np.allclose(mode_check["方式合计_吨"],mode_check["OD预测量_吨"],rtol=1e-9,atol=1e-3):
 raise ValueError("水泥运输方式分项合计与OD预测量不一致，无法绘制三列桑基图。")

local=bal[(bal["年份"]==2035)&bal["省份"].isin(REGION)][["省份","本地匹配量_吨"]].copy()
local_mode_rows=[]
for _,row in local.iterrows():
 province=row["省份"]
 base=shares.loc[province,["公路","铁路","水运"]].to_numpy(dtype=float)
 score=base*np.array([1.45,.75,.55])*np.array([1.8,.65,.35])
 if province not in WATER: score[2]=0
 if score.sum()<=0: raise ValueError(f"{province}缺少有效的本地运输方式份额，无法分配本地匹配量。")
 share=score/score.sum()
 local_mode_rows.append({"图起点":province,"图终点":province,"公路_吨":row["本地匹配量_吨"]*share[0],"水运_吨":row["本地匹配量_吨"]*share[2],"铁路_吨":row["本地匹配量_吨"]*share[1],"本地匹配量_吨":row["本地匹配量_吨"],"流向类别":"省内本地匹配"})
local_flows=pd.DataFrame(local_mode_rows)
mode_flows["流向类别"]="省际OD"
plot_flows=pd.concat([mode_flows,local_flows],ignore_index=True)
plot_flows["方式合计_吨"]=plot_flows[list(mode_columns.values())].sum(axis=1)
mode_names={column:name for name,column in mode_columns.items()}
origin_mode=plot_flows.melt(id_vars=["图起点"],value_vars=list(mode_columns.values()),var_name="方式字段",value_name="运输量_吨").rename(columns={"图起点":"起点"})
origin_mode["运输方式"]=origin_mode["方式字段"].map(mode_names)
origin_mode=origin_mode[origin_mode["运输量_吨"]>0].reset_index(drop=True)
mode_destination=plot_flows.melt(id_vars=["图终点"],value_vars=list(mode_columns.values()),var_name="方式字段",value_name="运输量_吨").rename(columns={"图终点":"终点"})
mode_destination["运输方式"]=mode_destination["方式字段"].map(mode_names)
mode_destination=mode_destination[mode_destination["运输量_吨"]>0].reset_index(drop=True)
origin_totals=origin_mode.groupby("起点")["运输量_吨"].sum().to_dict()
mode_totals=origin_mode.groupby("运输方式")["运输量_吨"].sum().to_dict()
destination_totals=mode_destination.groupby("终点")["运输量_吨"].sum().to_dict()
total_flow=origin_mode["运输量_吨"].sum()
sankey_top,sankey_bottom,sankey_gap=.86,.09,.04
def sankey_positions(totals,order):
 names=[name for name in order if totals.get(name,0)>0]
 scale=(sankey_top-sankey_bottom-sankey_gap*(len(names)-1))/total_flow
 positions={}; top_edge=sankey_top
 for name in names:
  height=totals[name]*scale; positions[name]=(top_edge-height,top_edge); top_edge-=height+sankey_gap
 return positions,scale
source_order=sorted(origin_totals,key=origin_totals.get,reverse=True)
destination_order=sorted(destination_totals,key=destination_totals.get,reverse=True)
mode_order=list(mode_columns)
source_pos,sankey_scale=sankey_positions(origin_totals,source_order)
mode_pos,_=sankey_positions(mode_totals,mode_order)
destination_pos,_=sankey_positions(destination_totals,destination_order)
def add_sankey_links(ax,edges,source_key,target_key,source_positions,target_positions,x0,x1,color_for):
 source_cursors={name:bounds[0] for name,bounds in source_positions.items()}
 target_cursors={name:bounds[0] for name,bounds in target_positions.items()}
 source_segments={}; target_segments={}
 for source,group in edges.groupby(source_key):
  ordered=group.sort_values(target_key,key=lambda col:col.map(lambda name:sum(target_positions[name])/2))
  for index,row in ordered.iterrows():
   height=row["运输量_吨"]*sankey_scale; lower=source_cursors[source]
   source_segments[index]=(lower,lower+height); source_cursors[source]+=height
 for target,group in edges.groupby(target_key):
  ordered=group.sort_values(source_key,key=lambda col:col.map(lambda name:sum(source_positions[name])/2))
  for index,row in ordered.iterrows():
   height=row["运输量_吨"]*sankey_scale; lower=target_cursors[target]
   target_segments[index]=(lower,lower+height); target_cursors[target]+=height
 control=(x1-x0)*.42
 for index,row in edges.iterrows():
  source_low,source_high=source_segments[index]; target_low,target_high=target_segments[index]
  vertices=[(x0,source_low),(x0+control,source_low),(x1-control,target_low),(x1,target_low),(x1,target_high),(x1-control,target_high),(x0+control,source_high),(x0,source_high),(x0,source_low)]
  codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
  ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor=color_for(row),edgecolor="none",alpha=.34,zorder=1))
fig,ax=plt.subplots(figsize=(14,4.5)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
source_x,mode_x,destination_x,node_width=.09,.49,.87,.018
origin_edges=origin_mode.rename(columns={"起点":"源节点"})
destination_edges=mode_destination.rename(columns={"终点":"目标节点"})
node_colors={**nc,"区外供给":"#777777","区外需求":"#A8A8A8"}
source_label_y=dict(zip(source_pos,np.linspace(sankey_top-.015,sankey_bottom+.015,len(source_pos))))
destination_label_y=dict(zip(destination_pos,np.linspace(sankey_top-.015,sankey_bottom+.015,len(destination_pos))))
add_sankey_links(ax,origin_edges,"源节点","运输方式",source_pos,mode_pos,source_x+node_width,mode_x,lambda row:node_colors[row["源节点"]])
add_sankey_links(ax,destination_edges,"运输方式","目标节点",mode_pos,destination_pos,mode_x+node_width,destination_x,lambda row:mode_colors[row["运输方式"]])
for name,(lower,upper) in source_pos.items():
 ax.add_patch(plt.Rectangle((source_x,lower),node_width,upper-lower,facecolor=node_colors[name],edgecolor="white",lw=.7,zorder=2))
 center=(lower+upper)/2; label_y=source_label_y[name]
 ax.plot([source_x-.006,source_x-.032],[center,label_y],color="#777777",lw=.6,alpha=.55,zorder=3)
 ax.text(source_x-.038,label_y,lab[name],ha="right",va="center",fontsize=17)
for name,(lower,upper) in mode_pos.items():
 ax.add_patch(plt.Rectangle((mode_x,lower),node_width,upper-lower,facecolor=mode_colors[name],edgecolor="white",lw=.7,zorder=2))
mode_label_y={name:(lower+upper)/2 for name,(lower,upper) in mode_pos.items()}
ordered_mode_labels=sorted(mode_label_y,key=mode_label_y.get,reverse=True)
for index in range(1,len(ordered_mode_labels)):
 previous,current=ordered_mode_labels[index-1:index+1]
 if mode_label_y[previous]-mode_label_y[current]<.11:
  mode_label_y[current]=mode_label_y[previous]-.11
if ordered_mode_labels and mode_label_y[ordered_mode_labels[-1]]<sankey_bottom+.015:
 shift=sankey_bottom+.015-mode_label_y[ordered_mode_labels[-1]]
 mode_label_y={name:y+shift for name,y in mode_label_y.items()}
for name,(lower,upper) in mode_pos.items():
 center=(lower+upper)/2; label_y=mode_label_y[name]
 ax.plot([mode_x+node_width+.003,mode_x+node_width+.009],[center,label_y],color="#777777",lw=.6,alpha=.6,zorder=3)
 ax.text(mode_x+node_width+.012,label_y,f"{name} {mode_totals[name]/total_flow:.1%}",ha="left",va="center",fontsize=18)
for name,(lower,upper) in destination_pos.items():
 ax.add_patch(plt.Rectangle((destination_x,lower),node_width,upper-lower,facecolor=node_colors[name],edgecolor="white",lw=.7,zorder=2))
 center=(lower+upper)/2; label_y=destination_label_y[name]
 ax.plot([destination_x+node_width+.006,destination_x+node_width+.032],[center,label_y],color="#777777",lw=.6,alpha=.55,zorder=3)
 ax.text(destination_x+node_width+.038,label_y,lab[name],ha="left",va="center",fontsize=17)
ax.text(source_x+node_width/2,.90,"起点省份",ha="center",va="bottom",fontsize=21,fontweight="bold")
ax.text(mode_x+node_width/2,.90,"运输方式及占比",ha="center",va="bottom",fontsize=21,fontweight="bold")
ax.text(destination_x+node_width/2,.90,"终点省份",ha="center",va="bottom",fontsize=21,fontweight="bold")
fig.tight_layout(); fig.savefig(RO/"03_2035京津冀8省市水泥货运流向桑基图.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
plot_flows.assign(预测量_万吨=plot_flows["方式合计_吨"]/1e4).to_csv(RO/"图3_2035京津冀8省市水泥桑基图流向数据.csv",index=False,encoding="utf-8-sig")

checks={"省份数":31,"OD非负":bool((odall["省际水泥预测量_吨"]>=0).all()),"方式非负":bool((mode[["公路_吨","铁路_吨","水运_吨"]]>=0).all().all()),"上游分项非负":bool((bal[["石灰质原料参考需求_吨","水泥窑煤炭参考需求_吨","石膏及混合材参考需求_吨","炉渣产生参考量_吨"]]>=0).all().all()),"煤炭报告校准最大误差_吨":float((ann["八省市报告煤炭需求校准值_亿吨"]-ann["八省市报告煤炭需求校准目标_亿吨"]).abs().max()*1e8),"2035需求_亿吨":float(new_dem.loc[2035].sum()/1e8),"2035产量_亿吨":float(new_prod.loc[2035].sum()/1e8),"2035省际OD_亿吨":float(od["省际水泥预测量_吨"].sum()/1e8),"2035原辅料参考需求_亿吨":float(ann.iloc[-1]["生产端原辅料运输需求_亿吨"]),"2035报告口径货运总需求_亿吨":float(ann.iloc[-1]["报告口径水泥货运总需求_亿吨"]),"2035报告综合系数_吨每吨水泥":float(ann.iloc[-1]["水泥行业货运综合系数_吨每吨水泥"]),"方式加总误差_吨":float(abs(md[["公路_吨","铁路_吨","水运_吨"]].sum().sum()-od["省际水泥预测量_吨"].sum()))}
(DO/"质量检查.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
modal=md[["公路_吨","铁路_吨","水运_吨"]].sum(); modal=modal/modal.sum()
report=f"""# 水泥行业整体预测运行说明\n\n- 全国需求情景沿用现有模型曲线范围，并按2035年需求水平重新排列为高、基准、低三种单调情景。\n- 基准情景2035年需求{checks['2035需求_亿吨']:.3f}亿吨，产量{checks['2035产量_亿吨']:.3f}亿吨。\n- 分省需求按80%本地产能份额与20%原需求模型信号收缩分配；产量按现有省级产能格局归一到全国控制量。\n- 省际水泥OD以盈余和缺口为约束，采用距离-2.2次方阻抗和IPF估算，不是真实贸易观测。2035年省际OD{checks['2035省际OD_亿吨']:.3f}亿吨。\n- 上游分项按前期报告单位消耗估算：每吨熟料约1.54吨石灰质原料、0.157吨煤炭（2022年基准，线性降至2030年0.147吨、2035年0.143吨）；每吨水泥按0.8吨熟料和0.05吨石膏/混合材估算，并列示约0.05吨/吨熟料炉渣产生量。八省市水泥窑煤炭需求按报告2025、2030、2035年0.46、0.42、0.35亿吨校准，并按各省水泥产量份额分配；其余省份按单位煤耗系数估算。熟料消耗量为工艺投入参考值，不表示熟料全部跨省运输。分项按水泥生产省归属，不代表原料来源OD。\n- 全国货运总需求采用报告综合系数插值：2025年2.80、2030年2.79、2035年2.78吨/吨水泥。该综合量包含多类生产和产品运输口径；与上游分项不可直接相加，避免将熟料/成品重复计量。2035年估算总量{checks['2035报告口径货运总需求_亿吨']:.3f}亿吨。\n- 分项参考需求与报告综合货运量并列输出；物料投入不自动等于外部运输。OD仅为成品水泥的供需匹配估算，未独立建立熟料OD。\n- 方式结构基于2024年分省综合货运份额，并按水泥短距离、公路优先特征修正：公路{modal['公路_吨']:.1%}、铁路{modal['铁路_吨']:.1%}、水运{modal['水运_吨']:.1%}。该方式结构只对应现有省际水泥OD。\n- 图表采用钢铁专题图的画布和字号规范；区域专题图包括八省市货运量图、全量省际OD弦图和含省内本地匹配量的三列桑基图。\n- 桑基图运输方式占比分配沿用水泥OD模式匹配结果，省内本地匹配按短距离方式权重估算；其展示口径与全省际OD弦图不同。\n- 31省完整，OD及方式均非负，上游分项均非负；八省市煤炭校准最大误差{checks['煤炭报告校准最大误差_吨']:.2f}吨，方式加总误差{checks['方式加总误差_吨']:.2f}吨。具体流向和方式份额适合作为项目MVP情景估算。\n"""
report += "\n统一口径说明：\n" + NOTE
(OUT/"运行说明.md").write_text(report,encoding="utf-8")
print(json.dumps({"output":str(OUT),"checks":checks,"mode_share":modal.to_dict(),"top_od":top.head(5).to_dict("records")},ensure_ascii=True,indent=2))
