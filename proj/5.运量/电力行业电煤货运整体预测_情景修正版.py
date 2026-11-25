from pathlib import Path
import math,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch,Wedge,PathPatch
from matplotlib.path import Path as MplPath

ROOT=Path(r"C:\Users\admin\Desktop\货运预测李童+高洁\京津冀货运发展"); DATE="20261022"
DEM=ROOT/"output"/"2.消费需求预测"/f"电力与电煤需求情景修正版_{DATE}"; PROD=ROOT/"output"/"3.产量分配"/f"电煤供给情景修正版_{DATE}"; OUT=ROOT/"output"/"4.运量"/f"电力行业电煤货运区域匹配_情景修正版_{DATE}"; DO,FO,RO=OUT/"数据",OUT/"图表",OUT/"京津冀8省市专题图"
for p in [DEM,PROD,DO,FO,RO]: p.mkdir(parents=True,exist_ok=True)
for f in ["Microsoft YaHei","SimHei"]:
 if any(x.name==f for x in font_manager.fontManager.ttflist): plt.rcParams["font.sans-serif"]=[f]; break
plt.rcParams["axes.unicode_minus"]=False; plt.rcParams.update({"font.size":20,"axes.labelsize":20,"xtick.labelsize":18,"ytick.labelsize":18,"legend.fontsize":20})
COORD={"北京市":(116.4,39.9),"天津市":(117.2,39.1),"河北省":(114.5,38.0),"山西省":(112.5,37.9),"内蒙古自治区":(111.7,40.8),"辽宁省":(123.4,41.8),"吉林省":(125.3,43.9),"黑龙江省":(126.6,45.8),"上海市":(121.5,31.2),"江苏省":(118.8,32.1),"浙江省":(120.2,30.3),"安徽省":(117.3,31.9),"福建省":(119.3,26.1),"江西省":(115.9,28.7),"山东省":(117,36.7),"河南省":(113.6,34.8),"湖北省":(114.3,30.6),"湖南省":(112.9,28.2),"广东省":(113.3,23.1),"广西壮族自治区":(108.3,22.8),"海南省":(110.3,20),"重庆市":(106.6,29.6),"四川省":(104.1,30.7),"贵州省":(106.7,26.6),"云南省":(102.7,25),"西藏自治区":(91.1,29.7),"陕西省":(108.9,34.3),"甘肃省":(103.8,36.1),"青海省":(101.8,36.6),"宁夏回族自治区":(106.2,38.5),"新疆维吾尔自治区":(87.6,43.8)}
P=list(COORD); SHORT={x:x.replace("省","").replace("市","").replace("自治区","").replace("壮族","").replace("回族","").replace("维吾尔","") for x in P}; REGION=["北京市","天津市","河北省","河南省","山东省","山西省","内蒙古自治区","陕西省"]; RC={x:c for x,c in zip(REGION,["#DCEAF5","#AFC9E2","#DDECCB","#A8CF95","#F3D1D3","#E89CA5","#F8E1BE","#F3BB82"])}; WATER={"天津市","河北省","辽宁省","上海市","江苏省","浙江省","福建省","山东省","湖北省","湖南省","广东省","广西壮族自治区","海南省","重庆市","四川省","江西省","安徽省"}
def norm(x): return str(x).strip().replace("黑_龙江省","黑龙江省")
def wide(path):
 d=pd.read_excel(path); d.columns=["year" if str(c) in ["年份","year"] else norm(c) for c in d.columns]; return d.set_index("year")
def hav(a,b):
 x1,y1=map(math.radians,COORD[a]);x2,y2=map(math.radians,COORD[b]);q=math.sin((y2-y1)/2)**2+math.cos(y1)*math.cos(y2)*math.sin((x2-x1)/2)**2;return 12742*math.asin(math.sqrt(q))
def ipf(seed,rows,cols):
 x=seed.astype(float)
 for _ in range(1000):
  rs=x.sum(1);x*=np.divide(rows,rs,out=np.zeros_like(rows),where=rs>0)[:,None];cs=x.sum(0);x*=np.divide(cols,cs,out=np.zeros_like(cols),where=cs>0)[None,:]
  if max(abs(x.sum(1)-rows).max(),abs(x.sum(0)-cols).max())<1e-6:break
 return x
def modefile(n):
 d=pd.read_excel(ROOT/"data"/"预测原始数据"/n,header=3).iloc[:,:2];d.columns=["省份","值"];d["省份"]=d["省份"].map(norm);d["值"]=pd.to_numeric(d["值"],errors="coerce").fillna(0);return d[d.省份.isin(P)].set_index("省份")["值"]
def official_fire_history():
 d=pd.read_excel(ROOT/"data"/"预测原始数据"/"火力发电量分省年度数据.xls",header=3)
 d["地区"]=d["地区"].map(norm)
 year_cols={int(str(c)[:4]):c for c in d.columns if str(c)[:4].isdigit()}
 history=d.set_index("地区").reindex(P)[list(year_cols.values())].T
 history.index=[int(str(c)[:4]) for c in year_cols.values()]
 history.index.name="年份";history.columns.name="省份"
 history=history.apply(pd.to_numeric,errors="coerce")
 if history.isna().any().any():
  missing=history.columns[history.isna().any()].tolist()
  raise ValueError(f"国家统计局火力发电量历史表存在缺失省份或数值：{missing}")
 return history
def forecast_fire_shares(history,years):
 shares=history.div(history.sum(axis=1),axis=0)
 base_year=int(shares.index.max());trend_start=max(int(shares.index.min()),base_year-5)
 start=shares.loc[trend_start].clip(lower=1e-8);base=shares.loc[base_year].clip(lower=1e-8)
 change=(np.log(base/start)/(base_year-trend_start)).clip(-0.03,0.03)
 projected={}
 for year in years:
  raw=base*np.exp(change*(year-base_year));projected[year]=raw/raw.sum()
 return pd.DataFrame(projected).T.reindex(columns=history.columns)

# 高、基准、低总用电情景分别采用原高增长、中增长、低增长。
mapping={"高增长":"高增长","基准":"中增长","低增长":"低增长"}; years=list(range(2025,2036)); elec={}; fire={}
for label,old in mapping.items():
 elec[label]=wide(ROOT/"output"/"2.消费需求预测"/f"Sce_{old}"/"全国完整数据"/"分省电力消费量数据(2005-2035).xlsx")
 fire[label]=wide(ROOT/"output"/"2.消费需求预测"/f"Sce_{old}"/"全国完整数据"/"分省火力发电量数据(2005-2035).xlsx")
elec_s=pd.DataFrame({s:elec[s].loc[years,"全国"] for s in mapping},index=years); elec_s.index.name="年份"
official_fire=official_fire_history()
fire_shares=forecast_fire_shares(official_fire,years)
generation={s:wide(ROOT/"output"/"2.消费需求预测"/f"Sce_{old}"/"全国完整数据"/"分省发电量数据(2005-2035).xlsx").loc[list(range(2024,2036)),"全国"].astype(float) for s,old in mapping.items()}
from 电力非火电供给情景 import predict
generation_frame,residual_fire=predict(generation,DEM)
generation={c:generation_frame[c] for c in generation_frame}
fire_s=pd.DataFrame(index=years,columns=mapping,dtype=float)
for label in mapping:
 national_fire=residual_fire.loc[years,label].copy()
 # NBS 2025 statistical communique, Table 4, reports 63,271.5 hundred-million kWh.
 # 2025为预测值，公报值仅事后核验。
 fire_s[label]=national_fire
 for year in years:
  fire[label].loc[year,P]=fire_s.loc[year,label]*fire_shares.loc[year,P]
  fire[label].loc[year,"全国"]=fire_s.loc[year,label]
assert (fire_s["高增长"]+1e-6>=fire_s["基准"]).all() and (fire_s["基准"]+1e-6>=fire_s["低增长"]).all()
from 分省电力预测衔接 import export as export_provincial
province_results=export_provincial(ROOT,DEM,P,REGION,mapping,generation_frame,residual_fire,fire_shares,official_fire,wide,norm)
fire={label:province_results[label][1] for label in mapping}
comparison=pd.concat([pd.DataFrame({"年份":years,"情景":label,"原模型火电量_亿千瓦时":wide(ROOT/"output"/"2.消费需求预测"/f"Sce_{old}"/"全国完整数据"/"分省火力发电量数据(2005-2035).xlsx").loc[years,"全国"].to_numpy(),"修正火电量_亿千瓦时":fire_s[label].to_numpy(),"总发电量_亿千瓦时":generation[label].loc[years].to_numpy()}) for label,old in mapping.items()],ignore_index=True)
comparison.to_csv(DEM/"火电情景修正前后对照.csv",index=False,encoding="utf-8-sig")
# The 2024 heat-rate source is 302.5 gce/kWh; improvement to 285 by 2035 is a scenario.
heat_rate_table=pd.read_excel(ROOT/"data"/"单位用料强度"/"单位发电煤耗.xlsx")
heat_rate_2024=float(heat_rate_table.loc[heat_rate_table["标识"]=="火力发电量","发电煤耗"].iloc[0])/1000
coal_rate=pd.Series(np.linspace(heat_rate_2024,.285,len(years)+1)[1:],index=years,name="吨标煤每MWh")
# Fire generation is in 100-million kWh; one unit equals 100,000 MWh.
fire_mwh=fire_s*1e5
coal_s=fire_mwh.mul(coal_rate,axis=0)
for label in mapping:
 provincial_coal=fire[label].loc[years,P].mul(coal_rate*1e5,axis=0)
 provincial_coal["全国"]=provincial_coal[P].sum(axis=1)
 provincial_coal["京津冀及周边8省市"]=provincial_coal[REGION].sum(axis=1)
 provincial_coal.to_csv(DEM/f"分省电煤折算需求_{label}_吨标准煤.csv",encoding="utf-8-sig")
assert (coal_s["高增长"]+1e-6>=coal_s["基准"]).all() and (coal_s["基准"]+1e-6>=coal_s["低增长"]).all()

elec_s.to_csv(DEM/"全国电力需求情景_亿千瓦时.csv",encoding="utf-8-sig"); fire_mwh.to_csv(DEM/"全国火电发电量情景_MWh.csv",encoding="utf-8-sig"); coal_s.to_csv(DEM/"全国电煤需求情景_吨标煤.csv",encoding="utf-8-sig")
fire_actual=official_fire.copy(); fire_actual.insert(0,"年份",fire_actual.index)
fire_actual.to_csv(DEM/"分省火力发电量_国家统计局历史实绩.csv",index=False,encoding="utf-8-sig")
pd.DataFrame([{"年份":year,"省份":province,"火力发电量_亿千瓦时":fire["基准"].loc[year,province],"数据属性":"2024基年模型预测，2025公报仅用于事后核验" if year==2025 else "模型情景预测"} for year in years for province in P]).to_csv(DEM/"分省火力发电量_预测口径.csv",index=False,encoding="utf-8-sig")
pd.DataFrame([
 {"项目":"全国火电预测方法","口径":"总发电量按2024基年实绩校准后，减水核风光供给；详见火电情景修正说明.md及非火电装机与效率假设.csv"},
 {"项目":"电煤折算限制","口径":"仍按全部火电乘煤耗，未剥离燃气、燃油、生物质等非煤火电，因此为标准煤代理，不是实际煤电耗煤"},
 {"项目":"电煤需求与省际运输量单位","口径":"吨标准煤；按发电量乘单位发电煤耗估算，不等同实际原煤吨数"},
 {"项目":"2025年火电事后核验值","口径":"2024基年预测；2025公报63271.5亿千瓦时仅事后核验，不参与校准；来源（https://www.stats.gov.cn/sj/zxfb/202602/t20260228_1962662.html）"},
 {"项目":"2024年单位发电煤耗","口径":f"{heat_rate_2024:.4f}吨标准煤/MWh，读取单位发电煤耗源表"},
 {"项目":"2025—2035年单位发电煤耗","口径":"由2024年基线线性下降至2035年0.285吨标准煤/MWh，属于情景假设"},
 {"项目":"省级供给份额","口径":"COAL_SOURCE情景代理，不是官方分省原煤产量或省际可外运量"},
 {"项目":"原煤实际货运吨数","口径":"尚未换算；需补充各省煤质/热值和官方分省原煤产量后重算"}
]).to_csv(OUT/"电力电煤预测数据口径.csv",index=False,encoding="utf-8-sig")
fig,ax=plt.subplots(figsize=(13.3,5.2))
for s,c in [("高增长","#D62728"),("基准","#ED7D31"),("低增长","#4472C4")]:ax.plot(years,elec_s[s]/1e4,lw=2.5,marker="o",label=s,color=c)
ax.set_xlabel("年份");ax.set_ylabel("全国用电量（万亿千瓦时）");ax.set_xticks(years);ax.grid(ls="--",alpha=.3);ax.legend(frameon=False,ncol=3);fig.tight_layout();fig.savefig(DEM/"全国电力需求情景.png",dpi=300,bbox_inches="tight");plt.close(fig)
fig,ax=plt.subplots(figsize=(13.3,5.2))
for s,c in [("高增长","#D62728"),("基准","#ED7D31"),("低增长","#4472C4")]:ax.plot(years,coal_s[s]/1e8,lw=2.5,marker="o",label=s,color=c)
ax.set_xlabel("年份");ax.set_ylabel("全国电煤需求（亿吨标煤）");ax.set_xticks(years);ax.grid(ls="--",alpha=.3);ax.legend(frameon=False,ncol=3);fig.tight_layout();fig.savefig(DEM/"全国电煤需求情景.png",dpi=300,bbox_inches="tight");plt.close(fig)

# 需求侧使用国家统计局火电实绩校准后的分省发电结构。下列供给权重为情景代理，
# 不是国家统计局分省原煤产量；需在获得可核验的分省原煤产量资料后替换。
COAL_SOURCE={"山西省":1.36,"内蒙古自治区":1.21,"陕西省":0.76,"新疆维吾尔自治区":0.46,
 "贵州省":0.13,"安徽省":0.11,"宁夏回族自治区":0.10,"河南省":0.10,"山东省":0.087,
 "云南省":0.07,"黑龙江省":0.067,"甘肃省":0.06,"河北省":0.046,"辽宁省":0.03,
 "四川省":0.02,"吉林省":0.012,"青海省":0.011,"湖南省":0.01,"重庆市":0.01,
 "广西壮族自治区":0.007,"湖北省":0.005,"江西省":0.004,"福建省":0.003,"江苏省":0.002}
fb=fire["基准"]; demand=pd.DataFrame(index=years,columns=P,dtype=float); supply=pd.DataFrame(index=years,columns=P,dtype=float)
for y in years:
 fs=fb.loc[y,P].clip(lower=0);fs=fs/fs.sum();ps=pd.Series(COAL_SOURCE,index=P,dtype=float).fillna(0)
 # 2025—2035年略提高新疆、内蒙古份额，降低东中部小型产区份额。
 trend=(y-2025)/10;ps["新疆维吾尔自治区"]*=1+.25*trend;ps["内蒙古自治区"]*=1+.06*trend
 ps[["安徽省","河南省","山东省","河北省","辽宁省"]]*=1-.12*trend;ps=ps/ps.sum()
 demand.loc[y]=fs*coal_s.loc[y,"基准"];supply.loc[y]=ps*coal_s.loc[y,"基准"]
for d,name,out in [(demand,"分省电煤需求_基准情景.xlsx",DEM),(supply,"分省电煤供给_基准情景.xlsx",PROD)]:
 z=d.copy();z.insert(0,"year",z.index);z["全国"]=d.sum(1);z["京津冀及周边8省市"]=d[REGION].sum(1);z.to_excel(out/name,index=False)
pd.DataFrame({"省份":list(COAL_SOURCE),"煤炭供给权重_相对值":list(COAL_SOURCE.values()),"数据属性":"项目情景代理，非官方分省原煤产量"}).to_csv(PROD/"电煤供给权重_情景假设.csv",index=False,encoding="utf-8-sig")

hm=pd.concat([modefile("公路货运量分省年度数据.xls").rename("公路"),modefile("铁路货运量分省年度数据.xls").rename("铁路"),modefile("水运货运量分省年度数据.xls").rename("水运")],axis=1).reindex(P).fillna(0);fallback=hm.sum()/hm.sum().sum();share=hm.div(hm.sum(1).replace(0,np.nan),axis=0).fillna(fallback)
balances=[];ods=[];modes=[];annual=[]
for y in years:
 p=supply.loc[y].astype(float);d=demand.loc[y].astype(float);sur=(p-d).clip(lower=0);de=(d-p).clip(lower=0);local=np.minimum(p,d);total=min(sur.sum(),de.sum());rt=sur/sur.sum()*total if sur.sum() else sur;ori=sur[sur>0].index.tolist();dst=de[de>0].index.tolist();rows=[]
 if ori and dst:
  dist=np.array([[hav(o,t) for t in dst] for o in ori]);mat=ipf(np.power(np.maximum(dist,100),-1.25),rt.loc[ori].to_numpy(),de.loc[dst].to_numpy())
  for i,o in enumerate(ori):
   for j,t in enumerate(dst):
    if mat[i,j]>1:rows.append([y,o,t,dist[i,j],mat[i,j]])
 od=pd.DataFrame(rows,columns=["年份","起点省份","终点省份","质心距离_km","省际电煤预测量_吨"]);ods.append(od)
 for _,q in od.iterrows():
  base=(share.loc[q.起点省份]+share.loc[q.终点省份])/2;score=base.to_numpy()*np.array([.35,2.3,1.15]);km=q.质心距离_km;score*=np.array([1.4,.9,.45]) if km<400 else(np.array([.8,1.55,.8]) if km<1000 else np.array([.4,1.85,1.3]));
  if not(q.起点省份 in WATER and q.终点省份 in WATER):score[2]=0
  sh=score/score.sum();v=q.省际电煤预测量_吨;modes.append({**q.to_dict(),"公路_吨":v*sh[0],"铁路_吨":v*sh[1],"水运_吨":v*sh[2]})
 inbound=od.groupby("终点省份")["省际电煤预测量_吨"].sum() if len(od) else pd.Series(dtype=float);outbound=od.groupby("起点省份")["省际电煤预测量_吨"].sum() if len(od) else pd.Series(dtype=float)
 balances.append(pd.DataFrame({"年份":y,"省份":P,"电煤供给_吨":p.values,"电煤需求_吨":d.values,"本地匹配量_吨":local.values,"理论盈余_吨":sur.values,"理论缺口_吨":de.values,"省际流出量_吨":[outbound.get(x,0) for x in P],"省际流入量_吨":[inbound.get(x,0) for x in P],"系统余量_吨":(sur-rt).values}))
 annual.append({"年份":y,"全国用电量_万亿千瓦时":elec_s.loc[y,"基准"]/1e4,"全国火电量_万亿千瓦时":fire_s.loc[y,"基准"]/1e4,"全国电煤需求_亿吨标煤":d.sum()/1e8,"省际电煤运输量_亿吨标煤":total/1e8,"系统余量_亿吨标煤":(sur.sum()-total)/1e8})
bal=pd.concat(balances,ignore_index=True);odall=pd.concat(ods,ignore_index=True);mode=pd.DataFrame(modes);ann=pd.DataFrame(annual)
bal.to_csv(DO/"2025-2035省级电煤供需与货运汇总.csv",index=False,encoding="utf-8-sig");odall.to_csv(DO/"2025-2035省际电煤OD预测.csv",index=False,encoding="utf-8-sig");mode.to_csv(DO/"2025-2035省际电煤OD运输方式匹配.csv",index=False,encoding="utf-8-sig");ann.to_csv(DO/"2025-2035全国电力与电煤汇总.csv",index=False,encoding="utf-8-sig")
with pd.ExcelWriter(DO/"电力行业预测_电煤供需_OD及方式匹配.xlsx",engine="openpyxl") as w:
 elec_s.to_excel(w,sheet_name="用电情景");fire_mwh.to_excel(w,sheet_name="火电情景");coal_s.to_excel(w,sheet_name="电煤情景");demand.to_excel(w,sheet_name="分省电煤需求");supply.to_excel(w,sheet_name="分省电煤供给");ann.to_excel(w,sheet_name="全国汇总",index=False);bal.to_excel(w,sheet_name="省级供需",index=False);odall.to_excel(w,sheet_name="省际OD",index=False);mode.to_excel(w,sheet_name="运输方式",index=False)

y=2035;b=bal[bal.年份==y].copy();od=odall[odall.年份==y];md=mode[mode.年份==y];b["净盈余_万吨"]=(b.电煤供给_吨-b.电煤需求_吨)/1e4;b=b.sort_values("净盈余_万吨")
fig,ax=plt.subplots(figsize=(13.3,5.2));ax.bar(b.省份.map(SHORT),b["净盈余_万吨"],color=np.where(b["净盈余_万吨"]>=0,"#4472C4","#E15759"));ax.axhline(0,color="#333",lw=.8);ax.set_ylabel("电煤供给－需求（万吨标准煤）");ax.tick_params(axis="x",rotation=60);ax.grid(axis="y",ls="--",alpha=.35);fig.tight_layout();fig.savefig(FO/"01_2035省级电煤供需平衡.png",dpi=300,bbox_inches="tight");plt.close(fig)
top=od.nlargest(25,"省际电煤预测量_吨");fig,ax=plt.subplots(figsize=(13.3,6.2));ax.scatter([v[0] for v in COORD.values()],[v[1] for v in COORD.values()],s=24,c="#777",zorder=3);mx=top["省际电煤预测量_吨"].max()
for _,q in top.iterrows():ax.add_patch(FancyArrowPatch(COORD[q.起点省份],COORD[q.终点省份],arrowstyle="-|>",mutation_scale=10,lw=.6+4*q.省际电煤预测量_吨/mx,color="#D62728",alpha=.38,connectionstyle="arc3,rad=.08"))
for p,(xv,yv) in COORD.items():ax.text(xv,yv+.35,SHORT[p],fontsize=10,ha="center")
ax.set_xlim(78,133);ax.set_ylim(17,51);ax.set_xlabel("经度");ax.set_ylabel("纬度");ax.grid(ls=":",alpha=.25);fig.tight_layout();fig.savefig(FO/"02_2035省际电煤主要流向图.png",dpi=300,bbox_inches="tight");plt.close(fig)
s=md[["公路_吨","铁路_吨","水运_吨"]].sum()/1e8;fig,ax=plt.subplots(figsize=(9.5,5.2));bars=ax.bar(["公路","铁路","水运"],s,color=["#ED7D31","#4472C4","#5B9BD5"],width=.58);ax.set_ylabel("省际电煤运输量（亿吨标准煤）");ax.grid(axis="y",ls="--",alpha=.35)
ax.set_ylim(0,s.max()*1.22)
for q,v in zip(bars,s):ax.text(q.get_x()+q.get_width()/2,v+max(s.max()*.015,.01),f"{v:.2f}\n({v/s.sum():.1%})",ha="center",fontsize=16)
fig.tight_layout();fig.savefig(FO/"03_2035电煤运输方式结构.png",dpi=300,bbox_inches="tight");plt.close(fig)
fig,ax=plt.subplots(figsize=(13.3,5.2));ax.plot(ann.年份,ann.全国电煤需求_亿吨标煤,marker="o",lw=2.4,label="全国电煤需求");ax.plot(ann.年份,ann.省际电煤运输量_亿吨标煤,marker="o",lw=2.4,label="省际电煤运输");ax.set_xlabel("年份");ax.set_ylabel("货运需求（亿吨标准煤）");ax.set_xticks(years);ax.grid(ls="--",alpha=.35);ax.legend(frameon=False,ncol=2);fig.tight_layout();fig.savefig(FO/"04_2025-2035电煤货运量趋势.png",dpi=300,bbox_inches="tight");plt.close(fig)

# 八省市货运量采用流入+流出装卸口径；全国折线为电煤需求。
rr=bal[bal.省份.isin(REGION)].copy();rr["货运量_亿吨"]=(rr.省际流入量_吨+rr.省际流出量_吨)/1e8;pv=rr.pivot(index="年份",columns="省份",values="货运量_亿吨").reindex(columns=REGION);pv.rename(columns=SHORT).to_csv(RO/"图1_八省市电煤货运量_数据.csv",encoding="utf-8-sig")
fig,ax=plt.subplots(figsize=(15,5));bottom=np.zeros(len(pv))
for p in REGION:ax.bar(pv.index,pv[p],bottom=bottom,color=RC[p],edgecolor="white",lw=.3,label=SHORT[p]);bottom+=pv[p].to_numpy()
ax2=ax.twinx();line,=ax2.plot(ann.年份,ann.全国电煤需求_亿吨标煤,color="#D62728",marker="o",lw=2.4,label="全国电煤需求")
for x,v in zip(ann.年份,ann.全国电煤需求_亿吨标煤):ax2.text(x,v+.20,f"{v:.1f}",ha="center",va="bottom",fontsize=15,color="#D62728")
ax2.set_ylim(top=ann.全国电煤需求_亿吨标煤.max()+1.5)
ax.set_xlabel("年份");ax.set_ylabel("八省市省际电煤货运量（亿吨标准煤）");ax2.set_ylabel("全国电煤需求（亿吨标准煤）");ax.set_xticks(years);ax.grid(axis="y",ls="--",alpha=.3);ax.set_axisbelow(True);h,l=ax.get_legend_handles_labels();ax.legend(h+[line],l+[line.get_label()],frameon=True,loc="center left",bbox_to_anchor=(1.2,.5));fig.tight_layout();fig.savefig(RO/"01_京津冀8省市电煤货运量.png",dpi=300,bbox_inches="tight");plt.close(fig)

inside=od[(od.起点省份.isin(REGION))&(od.终点省份.isin(REGION))];oe=od[(od.起点省份.isin(REGION))&(~od.终点省份.isin(REGION))].groupby("起点省份",as_index=False).省际电煤预测量_吨.sum();oe["终点省份"]="区外需求";ie=od[(~od.起点省份.isin(REGION))&(od.终点省份.isin(REGION))].groupby("终点省份",as_index=False).省际电煤预测量_吨.sum();ie["起点省份"]="区外供给";fl=pd.concat([inside[["起点省份","终点省份","省际电煤预测量_吨"]],oe[["起点省份","终点省份","省际电煤预测量_吨"]],ie[["起点省份","终点省份","省际电煤预测量_吨"]]],ignore_index=True);fl=fl[fl.省际电煤预测量_吨>0];fl.assign(预测量_万吨=fl.省际电煤预测量_吨/1e4).to_csv(RO/"图2_2035八省市电煤流向_数据.csv",index=False,encoding="utf-8-sig")
nodes=REGION+["区外需求","区外供给"];nc={**RC,"区外需求":"#A8A8A8","区外供给":"#777777"};lab={**SHORT,"区外需求":"区外需求","区外供给":"区外供给"};inc={n:0. for n in nodes}
for _,q in fl.iterrows():inc[q.起点省份]+=q.省际电煤预测量_吨;inc[q.终点省份]+=q.省际电煤预测量_吨
gap=np.deg2rad(3);avail=2*np.pi-gap*len(nodes);tot=sum(inc.values());spans={};cur=np.pi/2
for n in nodes:sp=avail*inc[n]/tot;spans[n]=(cur-sp,cur);cur-=sp+gap
cursor={n:spans[n][0] for n in nodes};seg=[]
for _,q in fl.sort_values("省际电煤预测量_吨",ascending=False).iterrows():w=q.省际电煤预测量_吨;da=avail*w/tot;oa=(cursor[q.起点省份],cursor[q.起点省份]+da);ta=(cursor[q.终点省份],cursor[q.终点省份]+da);cursor[q.起点省份]+=da;cursor[q.终点省份]+=da;seg.append((q.起点省份,q.终点省份,w,oa,ta))
fig,ax=plt.subplots(figsize=(13.8,12.9));ax.set_aspect("equal");ax.axis("off")
label_points={}
for n,(a0,a1) in spans.items():
 ax.add_patch(Wedge((0,0),1,np.degrees(a0),np.degrees(a1),width=.105,facecolor=nc[n],edgecolor="white",lw=1.2,zorder=3))
 am=(a0+a1)/2;x,yv=1.13*np.cos(am),1.13*np.sin(am);side=1 if x>=0 else -1
 label_points[n]=(side,yv,am)
for side in (-1,1):
 side_names=sorted([n for n,v in label_points.items() if v[0]==side],key=lambda n:label_points[n][1])
 label_y=[label_points[n][1] for n in side_names]
 for index in range(1,len(label_y)):
  label_y[index]=max(label_y[index],label_y[index-1]+.13)
 if label_y and label_y[-1]>1.2:
  shift=label_y[-1]-1.2;label_y=[y-shift for y in label_y]
 if label_y and label_y[0]<-1.2:
  shift=-1.2-label_y[0];label_y=[y+shift for y in label_y]
 for n,y in zip(side_names,label_y):
  _,_,am=label_points[n];x0,y0=1.02*np.cos(am),1.02*np.sin(am);x1=side*1.18
  ax.plot([x0,x1],[y0,y],color="#777777",lw=.6,alpha=.6,zorder=4)
  ax.text(side*1.21,y,lab[n],ha="left" if side>0 else "right",va="center",fontsize=19.5,fontweight="bold")
pt=lambda a:.885*np.array([np.cos(a),np.sin(a)])
for o,t,w,oa,ta in sorted(seg,key=lambda z:z[2]):s0,s1,t0,t1=pt(oa[0]),pt(oa[1]),pt(ta[0]),pt(ta[1]);vv=[s0,(0,0),(0,0),t1,t0,(0,0),(0,0),s1,s0];cc=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY];ax.add_patch(PathPatch(MplPath(vv,cc),facecolor=nc[o],edgecolor="none",alpha=.38,zorder=1))
ax.text(0,-1.22,"带宽表示2035年省际电煤预测运输量；颜色表示起点；数量为标准煤口径",ha="center",fontsize=16.5,color="#555");ax.set_xlim(-1.34,1.34);ax.set_ylim(-1.31,1.28);fig.tight_layout();fig.savefig(RO/"02_2035京津冀8省市电煤货运流向弦图.png",dpi=300,bbox_inches="tight",facecolor="white");plt.close(fig)

# 三列桑基图：起点省份 -> 运输方式 -> 终点省份，包含省内本地匹配量。
mode_columns={"公路":"公路_吨","水运":"水运_吨","铁路":"铁路_吨"}
mode_colors={"公路":"#ED7D31","水运":"#5B9BD5","铁路":"#4472C4"}
mode_od=mode[mode["年份"]==2035].copy()
mode_od=mode_od[mode_od["起点省份"].isin(REGION)|mode_od["终点省份"].isin(REGION)].copy()
mode_od["图起点"]=np.where(mode_od["起点省份"].isin(REGION),mode_od["起点省份"],"区外供给")
mode_od["图终点"]=np.where(mode_od["终点省份"].isin(REGION),mode_od["终点省份"],"区外需求")
mode_flows=mode_od.groupby(["图起点","图终点"],as_index=False)[list(mode_columns.values())].sum()
selected_pairs=fl[["起点省份","终点省份"]].rename(columns={"起点省份":"图起点","终点省份":"图终点"})
mode_flows=mode_flows.merge(selected_pairs,on=["图起点","图终点"],how="inner",validate="one_to_one")
if len(mode_flows)!=len(selected_pairs):
 raise ValueError("部分电煤OD在运输方式匹配数据中缺失，无法绘制三列桑基图。")
mode_flows["方式合计_吨"]=mode_flows[list(mode_columns.values())].sum(axis=1)
expected_flows=fl[["起点省份","终点省份","省际电煤预测量_吨"]].rename(columns={"起点省份":"图起点","终点省份":"图终点","省际电煤预测量_吨":"OD预测量_吨"})
mode_check=mode_flows.merge(expected_flows,on=["图起点","图终点"],validate="one_to_one")
if not np.allclose(mode_check["方式合计_吨"],mode_check["OD预测量_吨"],rtol=1e-9,atol=1e-3):
 raise ValueError("电煤运输方式分项合计与OD预测量不一致，无法绘制三列桑基图。")
local=bal[(bal["年份"]==2035)&bal["省份"].isin(REGION)][["省份","本地匹配量_吨"]].copy()
local_rows=[]
for _,row in local.iterrows():
 province=row["省份"]
 score=share.loc[province,["公路","铁路","水运"]].to_numpy(dtype=float)*np.array([.35,2.3,1.15])*np.array([1.4,.9,.45])
 if province not in WATER:score[2]=0
 if score.sum()<=0:raise ValueError(f"{province}缺少有效的本地运输方式份额，无法分配本地匹配量。")
 shares_local=score/score.sum()
 local_rows.append({"图起点":province,"图终点":province,"公路_吨":row["本地匹配量_吨"]*shares_local[0],"水运_吨":row["本地匹配量_吨"]*shares_local[2],"铁路_吨":row["本地匹配量_吨"]*shares_local[1],"本地匹配量_吨":row["本地匹配量_吨"],"流向类别":"省内本地匹配"})
local_flows=pd.DataFrame(local_rows)
mode_flows["流向类别"]="省际OD"
plot_flows=pd.concat([mode_flows,local_flows],ignore_index=True)
plot_flows["方式合计_吨"]=plot_flows[list(mode_columns.values())].sum(axis=1)
mode_names={column:name for name,column in mode_columns.items()}
origin_mode=plot_flows.melt(id_vars=["图起点"],value_vars=list(mode_columns.values()),var_name="方式字段",value_name="运输量_吨").rename(columns={"图起点":"起点"})
origin_mode["运输方式"]=origin_mode["方式字段"].map(mode_names)
origin_mode=origin_mode[origin_mode["运输量_吨"]>0].reset_index(drop=True)
destination_mode=plot_flows.melt(id_vars=["图终点"],value_vars=list(mode_columns.values()),var_name="方式字段",value_name="运输量_吨").rename(columns={"图终点":"终点"})
destination_mode["运输方式"]=destination_mode["方式字段"].map(mode_names)
destination_mode=destination_mode[destination_mode["运输量_吨"]>0].reset_index(drop=True)
origin_totals=origin_mode.groupby("起点")["运输量_吨"].sum().to_dict()
mode_totals=origin_mode.groupby("运输方式")["运输量_吨"].sum().to_dict()
destination_totals=destination_mode.groupby("终点")["运输量_吨"].sum().to_dict()
total_flow=origin_mode["运输量_吨"].sum()
top_edge,bottom_edge,gap=.91,.09,.04
def sankey_positions(totals,order):
 names=[name for name in order if totals.get(name,0)>0]
 scale=(top_edge-bottom_edge-gap*(len(names)-1))/total_flow
 positions={};edge=top_edge
 for name in names:
  height=totals[name]*scale;positions[name]=(edge-height,edge);edge-=height+gap
 return positions,scale
source_order=sorted(origin_totals,key=origin_totals.get,reverse=True)
destination_order=sorted(destination_totals,key=destination_totals.get,reverse=True)
source_pos,link_scale=sankey_positions(origin_totals,source_order)
mode_pos,_=sankey_positions(mode_totals,list(mode_columns))
destination_pos,_=sankey_positions(destination_totals,destination_order)
def draw_sankey_links(ax,edges,source_key,target_key,source_pos,target_pos,x0,x1):
 source_cursor={name:bounds[0] for name,bounds in source_pos.items()}
 target_cursor={name:bounds[0] for name,bounds in target_pos.items()}
 source_segments={};target_segments={}
 for source,group in edges.groupby(source_key):
  group=group.sort_values(target_key,key=lambda col:col.map(lambda name:sum(target_pos[name])/2))
  for index,row in group.iterrows():
   h=row["运输量_吨"]*link_scale;lo=source_cursor[source]
   source_segments[index]=(lo,lo+h);source_cursor[source]+=h
 for target,group in edges.groupby(target_key):
  group=group.sort_values(source_key,key=lambda col:col.map(lambda name:sum(source_pos[name])/2))
  for index,row in group.iterrows():
   h=row["运输量_吨"]*link_scale;lo=target_cursor[target]
   target_segments[index]=(lo,lo+h);target_cursor[target]+=h
 control=(x1-x0)*.42
 for index,row in edges.iterrows():
  sl,sh=source_segments[index];tl,th=target_segments[index]
  vertices=[(x0,sl),(x0+control,sl),(x1-control,tl),(x1,tl),(x1,th),(x1-control,th),(x0+control,sh),(x0,sh),(x0,sl)]
  codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
  ax.add_patch(PathPatch(MplPath(vertices,codes),facecolor=mode_colors[row["运输方式"]],edgecolor="none",alpha=.34,zorder=1))
fig,ax=plt.subplots(figsize=(14,4.5));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
source_x,mode_x,destination_x,node_width=.09,.49,.87,.018
origin_edges=origin_mode.rename(columns={"起点":"源节点"})
destination_edges=destination_mode.rename(columns={"终点":"目标节点"})
source_label_y=dict(zip(source_pos,np.linspace(top_edge-.015,bottom_edge+.015,len(source_pos))))
destination_label_y=dict(zip(destination_pos,np.linspace(top_edge-.015,bottom_edge+.015,len(destination_pos))))
draw_sankey_links(ax,origin_edges,"源节点","运输方式",source_pos,mode_pos,source_x+node_width,mode_x)
draw_sankey_links(ax,destination_edges,"运输方式","目标节点",mode_pos,destination_pos,mode_x+node_width,destination_x)
node_colors={**nc,"区外供给":"#777777","区外需求":"#A8A8A8"}
for name,(lo,hi) in source_pos.items():
 ax.add_patch(plt.Rectangle((source_x,lo),node_width,hi-lo,facecolor=node_colors[name],edgecolor="white",lw=.7,zorder=2))
 center=(lo+hi)/2;label_y=source_label_y[name]
 ax.plot([source_x-.006,source_x-.032],[center,label_y],color="#777777",lw=.6,alpha=.55,zorder=3)
 ax.text(source_x-.038,label_y,lab[name],ha="right",va="center",fontsize=17)
for name,(lo,hi) in mode_pos.items():
 ax.add_patch(plt.Rectangle((mode_x,lo),node_width,hi-lo,facecolor=mode_colors[name],edgecolor="white",lw=.7,zorder=2))
mode_label_y={name:(lo+hi)/2 for name,(lo,hi) in mode_pos.items()}
ordered_modes=sorted(mode_label_y,key=mode_label_y.get,reverse=True)
for index in range(1,len(ordered_modes)):
 previous,current=ordered_modes[index-1:index+1]
 if mode_label_y[previous]-mode_label_y[current]<.11:mode_label_y[current]=mode_label_y[previous]-.11
if ordered_modes and mode_label_y[ordered_modes[-1]]<bottom_edge+.015:
 shift=bottom_edge+.015-mode_label_y[ordered_modes[-1]]
 mode_label_y={name:y+shift for name,y in mode_label_y.items()}
for name,(lo,hi) in mode_pos.items():
 center=(lo+hi)/2;label_y=mode_label_y[name]
 ax.plot([mode_x+node_width+.003,mode_x+node_width+.009],[center,label_y],color="#777777",lw=.6,alpha=.6,zorder=3)
 ax.text(mode_x+node_width+.012,label_y,f"{name} {mode_totals[name]/total_flow:.1%}",ha="left",va="center",fontsize=18)
for name,(lo,hi) in destination_pos.items():
 ax.add_patch(plt.Rectangle((destination_x,lo),node_width,hi-lo,facecolor=node_colors[name],edgecolor="white",lw=.7,zorder=2))
 center=(lo+hi)/2;label_y=destination_label_y[name]
 ax.plot([destination_x+node_width+.006,destination_x+node_width+.032],[center,label_y],color="#777777",lw=.6,alpha=.55,zorder=3)
 ax.text(destination_x+node_width+.038,label_y,lab[name],ha="left",va="center",fontsize=17)
ax.text(source_x+node_width/2,.95,"起点省份",ha="center",va="bottom",fontsize=21,fontweight="bold")
ax.text(mode_x+node_width/2,.95,"运输方式及占比",ha="center",va="bottom",fontsize=21,fontweight="bold")
ax.text(destination_x+node_width/2,.95,"终点省份",ha="center",va="bottom",fontsize=21,fontweight="bold")
fig.text(.5,.015,"标准煤口径；供给为情景代理；含省内匹配量，与省际OD弦图口径不同",ha="center",fontsize=13,color="#555")
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(RO/"03_2035京津冀8省市电煤货运流向桑基图.png",dpi=300,bbox_inches="tight",facecolor="white");plt.close(fig)
plot_flows.assign(预测量_万吨=plot_flows["方式合计_吨"]/1e4).to_csv(RO/"图3_2035京津冀8省市电煤桑基图流向数据.csv",index=False,encoding="utf-8-sig")

checks={"省份数":31,"OD非负":bool((odall.省际电煤预测量_吨>=0).all()),"方式非负":bool((mode[["公路_吨","铁路_吨","水运_吨"]]>=0).all().all()),"2035用电量_万亿千瓦时":float(ann.iloc[-1].全国用电量_万亿千瓦时),"2035电煤需求_亿吨标煤":float(ann.iloc[-1].全国电煤需求_亿吨标煤),"2035省际电煤运输_亿吨标煤":float(ann.iloc[-1].省际电煤运输量_亿吨标煤),"方式加总误差_吨标煤":float(abs(md[["公路_吨","铁路_吨","水运_吨"]].sum().sum()-od.省际电煤预测量_吨.sum()))}
(DO/"质量检查.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
modal=md[["公路_吨","铁路_吨","水运_吨"]].sum()
modal=modal/modal.sum()
report=f"""# 电力行业预测与电煤货运说明

- 电力通过电网传输，不直接形成货运；本流程把电力行业货运定义为火电需求驱动的电煤运输。
- 总用电高、基准、低情景分别采用原高增长、中增长、低增长结果；基准2035年用电量{checks['2035用电量_万亿千瓦时']:.3f}万亿千瓦时。
- 火电量以2024年实绩为基年，2025年公报仅用于事后核验，2025年省级结构及此后省级火电份额按国家统计局历史份额趋势外推。单位发电煤耗由源表2024年302.5克标准煤/千瓦时线性降至2035年285克标准煤/千瓦时。
- 2025年火电事后核验值来源：[国家统计局2025年国民经济和社会发展统计公报](https://www.stats.gov.cn/sj/zxfb/202602/t20260228_1962662.html)。
- 电煤需求及省际运输结果均为吨标准煤，不是实际原煤吨数；实际货运吨数还需各省煤质/热值资料换算。基准2035年电煤需求{checks['2035电煤需求_亿吨标煤']:.3f}亿吨标准煤。
- 分省电煤需求按火电量分配；供给按资源型省份格局权重估算，不再添加无依据的系统余量。供给权重不是国家统计局分省原煤产量，也不等同于真实可外运量。
- OD采用供需双约束、距离-1.25次方阻抗和IPF估算，不是真实贸易观测。2035年省际电煤运输{checks['2035省际电煤运输_亿吨标煤']:.3f}亿吨标准煤。
- 方式结构：公路{modal['公路_吨']:.1%}、铁路{modal['铁路_吨']:.1%}、水运{modal['水运_吨']:.1%}。
- 图表沿用钢铁专题图的字号和画布设置；八省市弦图保留全部正值OD，三列桑基图另含省内本地匹配量，省内方式按省级综合货运份额及短距离铁路优先权重估算。
"""
(OUT/"运行说明.md").write_text(report,encoding="utf-8")
print(json.dumps({"output":str(OUT),"checks":checks,"mode_share":modal.to_dict(),"top_od":top.head(5).to_dict("records")},ensure_ascii=True,indent=2))
