# -*- coding: utf-8 -*-
"""按钢铁最新版式生成水泥及电力行业汇报图。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT=Path(r"C:\Users\admin\Desktop\货运预测李童+高洁\京津冀货运发展")
CEMENT_DATE="20261022"
POWER_DATE="20261022"
REGION=["北京市","天津市","河北省","河南省","山东省","山西省","内蒙古自治区","陕西省"]
COLORS=["#DCEAF5","#AFC8E2","#DDEEC8","#B5D79D","#F7D4D7","#EB9BA3","#F8E2BE","#F5C794"]
YEARS=np.arange(2020,2036)
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","Noto Sans CJK SC","DejaVu Sans"],"axes.unicode_minus":False,"font.size":17,"axes.titlesize":22,"axes.labelsize":19,"xtick.labelsize":16,"ytick.labelsize":16,"legend.fontsize":16})

def wide(path):
    d=pd.read_excel(path,index_col=0);d.index=pd.to_numeric(d.index,errors="coerce");return d.apply(pd.to_numeric,errors="coerce").fillna(0)

def steel_bar_line(data, bar_cols, national_col, left_label, right_label, line_label, out, title, years=YEARS):
    years=np.asarray(years);z=data.reindex(years).copy(); fig,ax1=plt.subplots(figsize=(13.33,5.2),dpi=180);bottom=np.zeros(len(z))
    for color,p in zip(COLORS,bar_cols):
        vals=z[p].clip(lower=0).to_numpy();ax1.bar(years,vals,bottom=bottom,width=.72,color=color,edgecolor="white",linewidth=.25,label=p);bottom+=vals
    ax1.set_ylabel(left_label);ax1.set_xlabel("年份");ax1.set_xlim(years.min()-.6,years.max()+.6);ax1.set_ylim(0,max(bottom)*1.16 if max(bottom)>0 else 1);ax1.set_xticks([x for x in [2020,2025,2030,2035] if x in years]);ax1.grid(axis="y",ls="--",alpha=.3);ax1.spines[["top"]].set_visible(False)
    ax2=ax1.twinx();nat=z[national_col];line,=ax2.plot(years,nat,color="#D62728",marker="o",markersize=5.5,lw=2.6,label=line_label,zorder=8);ax2.set_ylabel(right_label);ax2.set_ylim(0,nat.max()*1.18);ax2.spines[["top"]].set_visible(False)
    for y in [2020,2024,2025,2030,2035]:
        if y not in nat.index or pd.isna(nat.loc[y]):continue
        v=float(nat.loc[y]);ax2.text(y,v+nat.max()*.025,f"{v:.2f}",ha="center",va="bottom",fontsize=15,color="#B22222",fontweight="bold")
    h,l=ax1.get_legend_handles_labels();ax1.legend(h+[line],l+[line_label],loc="center left",bbox_to_anchor=(1.08,.5),frameon=False,labelspacing=.75)
    ax1.set_title(title,fontweight="bold",pad=18);fig.subplots_adjust(left=.10,right=.75,top=.90,bottom=.18);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=300,facecolor="white");plt.close(fig)

def cement():
    dem_root=ROOT/"output"/"2.消费需求预测"/f"水泥需求情景修正版_{CEMENT_DATE}"
    src=ROOT/"output"/"2.消费需求预测"/"Sce_中增长"/"全国完整数据"/"下游行业水泥需求预测.xlsx"
    prod_root=ROOT/"output"/"3.产量分配"/f"水泥需求情景修正版_{CEMENT_DATE}"
    out=dem_root/"钢铁格式图表";out.mkdir(parents=True,exist_ok=True)
    sectors=["新增建筑面积","新增高速公路里程","新增其他等级公路里程","新增铁路里程","农村水电基础设施投资"]
    names={"新增建筑面积":"房屋建筑","新增高速公路里程":"高速公路","新增其他等级公路里程":"其他等级公路","新增铁路里程":"铁路建设","农村水电基础设施投资":"农村水电投资项目"}
    cols={"新增建筑面积":"#26A69A","新增高速公路里程":"#7E57C2","新增其他等级公路里程":"#90CAF9","新增铁路里程":"#1E88E5","农村水电基础设施投资":"#BDBDBD"}
    vals={}
    for s in sectors:
        d=pd.read_excel(src,sheet_name=f"{s}_中间结果",index_col=0);d.index=pd.to_numeric(d.index,errors="coerce");vals[s]=d.select_dtypes(include="number").sum(axis=1).reindex(np.arange(2010,2036)).clip(lower=0).fillna(0)
    vals=pd.DataFrame(vals)
    total=wide(dem_root/"分省水泥需求_基准情景_修正版.xlsx")["全国"].reindex(np.arange(2010,2036))
    # 修正版文件从2025年开始；历史段采用原模型全国总量。
    raw=wide(ROOT/"output"/"2.消费需求预测"/"Sce_基准"/"全国完整数据"/"分省水泥消费量预测.xlsx")["全国"]
    total=total.combine_first(raw).reindex(np.arange(2010,2036))
    # 下游结构由各指标的“预测工程量×单位水泥强度”中间结果决定。
    # 旧版把农村水电人工固定为45%，无数据依据，且与原模型结果相差约10万倍。
    # 2024年部分基建指标为0是原数据缺口，将夹在正值之间的0视为缺失并按年份插值。
    for c in vals.columns:
        s=vals[c].astype(float)
        internal_zero=(s.eq(0) & s.shift(1).gt(0) & s.shift(-1).gt(0))
        vals[c]=s.mask(internal_zero).interpolate(method="index",limit_direction="both").clip(lower=0)
    raw_sum=vals.sum(axis=1).replace(0,np.nan)
    shares=vals.div(raw_sum,axis=0).fillna(0)
    structure=shares.mul(total,axis=0)
    shares.rename(columns=names).to_csv(out/"基准情景_水泥下游结构占比_模型计算.csv",encoding="utf-8-sig")
    structure.to_csv(out/"基准情景_水泥下游产品需求分解.csv",encoding="utf-8-sig")
    order=["农村水电基础设施投资","新增铁路里程","新增其他等级公路里程","新增高速公路里程","新增建筑面积"];v=structure[order]/1e8
    plt.rcParams.update({"font.size":40,"axes.labelsize":44,"xtick.labelsize":36,"ytick.labelsize":36,"legend.fontsize":34})
    fig,ax=plt.subplots(figsize=(15,13),dpi=180);ax.stackplot(v.index,v.T.values,labels=[names[x] for x in order],colors=[cols[x] for x in order],alpha=.92,linewidth=0);ax.set_xlim(2009,2037);ax.set_ylim(0,v.sum(axis=1).max()*1.35);ax.set_xticks(np.arange(2010,2036,5));ax.yaxis.set_major_locator(MultipleLocator(5));ax.grid(axis="y",ls="--",alpha=.25);ax.spines[["top","right"]].set_visible(False);ax.legend(loc="upper center",ncol=2,bbox_to_anchor=(.5,.99),frameon=True,facecolor="white",framealpha=.85,edgecolor="none");ax.set_xlabel("年份");ax.set_ylabel("水泥需求量（亿吨）");fig.subplots_adjust(left=.14,right=.97,top=.94,bottom=.18);fig.savefig(out/"全国水泥下游产品需求结构_基准情景.png",dpi=300,facecolor="white");plt.close(fig)
    plt.rcParams.update({"font.size":17,"axes.titlesize":22,"axes.labelsize":19,"xtick.labelsize":16,"ytick.labelsize":16,"legend.fontsize":16})
    pnew=wide(prod_root/"分省水泥产量_基准情景_修正版.xlsx")
    p=pnew.reindex(YEARS)/1e8
    assert not p[REGION+["全国"]].isna().any().any()
    steel_bar_line(p,REGION,"全国","重点省份水泥产量（亿吨）","全国水泥产量（亿吨）","全国水泥产量",out/"全国及重点省份水泥产量_最新基准情景.png","全国及重点省份水泥产量（最新基准情景）")
    steel_bar_line(p,REGION,"全国","重点省份水泥产量（亿吨）","全国水泥产量（亿吨）","全国水泥产量",prod_root/"钢铁格式图表"/"全国及重点省份水泥产量_最新基准情景.png","水泥产量（2025年省级分配为估算）")
    gap=float((structure.sum(axis=1)-total).abs().max());return {"水泥下游分解最大差额_吨":gap,"2035水泥需求_亿吨":float(total.loc[2035]/1e8),"2035水泥产量_亿吨":float(p.loc[2035,"全国"])}

def power():
    out=ROOT/"output"/"4.运量"/"电力行业电煤货运区域匹配_情景修正版_20261022"/"钢铁格式图表";out.mkdir(parents=True,exist_ok=True)
    base=ROOT/"output"/"2.消费需求预测"/"Sce_中增长"/"全国完整数据"
    gen=pd.read_csv(ROOT/"output"/"2.消费需求预测"/f"电力与电煤需求情景修正版_{POWER_DATE}"/"分省总发电量_历史及预测_基准_亿千瓦时.csv",index_col=0).reindex(YEARS)/1e4
    if "全国" not in gen: gen["全国"]=gen[[c for c in gen if c in REGION or c.endswith(("省","市","自治区"))]].sum(axis=1)
    steel_bar_line(gen,REGION,"全国","重点省份发电量（万亿千瓦时）","全国发电量（万亿千瓦时）","全国发电量",out/"全国及重点省份发电量_最新基准情景.png","全国及重点省份发电量（最新基准情景）")
    coal_new=wide(ROOT/"output"/"2.消费需求预测"/"电力与电煤需求情景修正版_20261022"/"分省电煤需求_基准情景.xlsx")
    dem=ROOT/"output"/"2.消费需求预测"/f"电力与电煤需求情景修正版_{POWER_DATE}"
    fire=pd.read_csv(dem/"分省火力发电量_国家统计局历史实绩.csv",index_col=0)
    fire.index=pd.to_numeric(fire.index)
    fire["全国"]=fire.sum(axis=1)
    rate_table=pd.read_excel(ROOT/"data"/"单位用料强度"/"单位发电煤耗.xlsx")
    rate=float(rate_table.loc[rate_table["标识"]=="火力发电量","发电煤耗"].iloc[0])/1000
    coal_hist=fire*1e5*rate  # 历史按2024煤耗基线折算，非逐年实测电煤
    coal=coal_new.combine_first(coal_hist).reindex(YEARS)/1e8
    forecast=pd.read_csv(dem/"分省火力发电量_预测口径.csv")
    fire_future=forecast.pivot(index="年份",columns="省份",values="火力发电量_亿千瓦时")
    fire_future["全国"]=fire_future.sum(axis=1)
    fire_all=fire_future.combine_first(fire).reindex(YEARS)/1e4
    steel_bar_line(fire_all,REGION,"全国","重点省份火电量（万亿千瓦时）","全国火电量（万亿千瓦时）","全国火电量",out/"全国及重点省份火力发电量_最新基准情景.png","火力发电量（2025年省级分配为估算）")
    power_prod=ROOT/"output"/"3.产量分配"/f"电煤供给情景修正版_{POWER_DATE}"/"钢铁格式图表"
    steel_bar_line(fire_all,REGION,"全国","重点省份火电量（万亿千瓦时）","全国火电量（万亿千瓦时）","全国火电量",power_prod/"全国及重点省份火力发电量_最新基准情景.png","火力发电量（2025年省级分配为估算）")
    supply=wide(power_prod.parent/"分省电煤供给_基准情景.xlsx")/1e8
    steel_bar_line(supply,REGION,"全国","重点省份供给（亿吨标准煤）","全国供给（亿吨标准煤）","全国情景供给",power_prod/"全国及重点省份电煤供给_情景代理.png","电煤供给情景代理（非官方原煤产量）",years=np.arange(2025,2036))
    elec=wide(base/"分省电力消费量数据(2005-2035).xlsx")/1e4
    dem_charts=dem/"钢铁格式图表"
    steel_bar_line(elec,REGION,"全国","重点省份用电量（万亿千瓦时）","全国用电量（万亿千瓦时）","全国用电需求",dem_charts/"全国及重点省份用电需求_最新基准情景.png","全国及重点省份用电需求（基准情景）")
    steel_bar_line(coal,REGION,"全国","重点省份需求（亿吨标准煤）","全国需求（亿吨标准煤）","全国电煤需求",dem_charts/"全国及重点省份电煤需求_最新基准情景.png","电煤需求（历史段按2024煤耗基线折算）")
    (dem_charts/"图表口径说明.md").write_text("火电历史采用国家统计局分省实绩；2024年为研究基年，2025年起为预测；2025公报仅事后核验。历史电煤用2024年煤耗基线折算，非逐年实际耗煤。未来煤耗下降为情景假设。电煤需求和运输均为标准煤，供给为情景代理，非实际原煤产量。",encoding="utf-8")
    assert abs(fire_all.loc[2024,"全国"]-6.374265)<1e-8
    steel_bar_line(coal,REGION,"全国","重点省份电煤需求（亿吨标准煤）","全国电煤需求（亿吨标准煤）","全国电煤需求",out/"全国及重点省份电煤需求_最新基准情景.png","全国及重点省份电煤需求（最新基准情景）")
    bal=pd.read_csv(ROOT/"output"/"4.运量"/"电力行业电煤货运区域匹配_情景修正版_20261022"/"数据"/"2025-2035省级电煤供需与货运汇总.csv")
    bal["省际电煤货运量_亿吨"]=(bal["省际流入量_吨"]+bal["省际流出量_吨"])/1e8
    fyears=np.arange(2025,2036);pv=bal.pivot(index="年份",columns="省份",values="省际电煤货运量_亿吨").reindex(index=fyears,columns=REGION).fillna(0)
    annual=pd.read_csv(ROOT/"output"/"4.运量"/"电力行业电煤货运区域匹配_情景修正版_20261022"/"数据"/"2025-2035全国电力与电煤汇总.csv").set_index("年份")
    pv["全国"]=annual["省际电煤运输量_亿吨标煤"].reindex(fyears)
    steel_bar_line(pv,REGION,"全国","重点省份电煤货运量（亿吨标准煤）","全国省际电煤运输量（亿吨标准煤）","全国省际运输量",out/"全国及重点省份电煤运输量_最新基准情景.png","全国及重点省份电煤运输量（最新基准情景）",years=fyears)
    return {"2035发电量_万亿千瓦时":float(gen.loc[2035,"全国"]),"2035电煤需求_亿吨":float(coal.loc[2035,"全国"]),"2035省际电煤运输量_亿吨标煤":float(pv.loc[2035,"全国"])}

if __name__=="__main__":
    result={**cement(),**power()}
    (ROOT/"output"/"图表更新核验_20261022.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
