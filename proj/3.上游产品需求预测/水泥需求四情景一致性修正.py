# -*- coding: utf-8 -*-
"""按钢铁情景图格式重建水泥四情景，并核查基准产量。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

R=Path(r"C:\Users\admin\Desktop\货运预测李童+高洁\京津冀货运发展")
OUT=R/"output"/"2.消费需求预测"/"水泥需求情景修正版_20261020"
def wide(p):
 d=pd.read_excel(p,index_col=0);d.index=pd.to_numeric(d.index,errors="coerce");return d.apply(pd.to_numeric,errors="coerce").fillna(0)
raw={s:wide(R/"output"/"2.消费需求预测"/f"Sce_{s}"/"全国完整数据"/"分省水泥消费量预测.xlsx")["全国"] for s in ["基准","中增长","高增长","低增长"]}
years=np.arange(2010,2036);base=raw["中增长"].reindex(years);low=raw["低增长"].reindex(years);high=raw["基准"].reindex(years)
# 中增长取基准与高增长中点；历史至2024年保持共同值。
mid=(base+high)/2
for s in [low,base,mid,high]:s.loc[:2024]=base.loc[:2024]
data=pd.DataFrame({"低增长":low,"基准":base,"中增长":mid,"高增长":high},index=years)
data.to_csv(OUT/"全国水泥需求四情景_一致性修正版.csv",encoding="utf-8-sig")
styles={"低增长":("#4E79A7","--"),"基准":("#D62728","-"),"中增长":("#F28E2B","-."),"高增长":("#59A14F",":")}
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","Noto Sans CJK SC","DejaVu Sans"],"axes.unicode_minus":False,"font.size":18,"axes.labelsize":20,"xtick.labelsize":17,"ytick.labelsize":17,"legend.fontsize":17})
fig,ax=plt.subplots(figsize=(10,6),dpi=180);ax.axvspan(2024,2035,color="#F2F2F2",alpha=.8,zorder=0);ax.fill_between(data.loc[2024:].index,data.loc[2024:].min(axis=1)/1e8,data.loc[2024:].max(axis=1)/1e8,color="#BDBDBD",alpha=.18,label="情景范围");ax.plot(data.loc[:2024].index,data.loc[:2024,"基准"]/1e8,color="#7F7F7F",lw=3,label="历史/校准值")
offset={"低增长":7.7,"基准":10.5,"中增长":12.4,"高增长":14.25}
for n in data:
 s=data.loc[2024:,n]/1e8;c,ls=styles[n];ax.plot(s.index,s,color=c,ls=ls,lw=3,label=f"{n}情景");ax.annotate(f"{n} {s.loc[2035]:.2f}",xy=(2035,s.loc[2035]),xytext=(2035.12,offset[n]),color=c,fontsize=15,fontweight="bold",va="center",arrowprops=dict(arrowstyle="-",color=c,lw=1.2))
a=data.loc[2024,"基准"]/1e8;ax.scatter([2024],[a],color="#D62728",s=80,zorder=8);ax.annotate(f"2024年\n{a:.2f}亿吨",xy=(2024,a),xytext=(14,5),textcoords="offset points",fontsize=17,fontweight="bold",color="#8B1A1A",arrowprops=dict(arrowstyle="->",color="#8B1A1A",lw=1.6));ax.axvline(2024,color="#777",ls="--",lw=1.2);ax.text(2017,7.0,"历史阶段",color="#666",fontsize=18,ha="center");ax.text(2029.5,7.0,"预测阶段",color="#666",fontsize=18,ha="center");ax.set_xlabel("年份");ax.set_ylabel("水泥消费需求（亿吨）");ax.set_xlim(2010,2036.3);ax.set_ylim(6.5,26);ax.set_xticks(np.arange(2010,2036,5));ax.yaxis.set_major_locator(MultipleLocator(5));ax.grid(ls="--",alpha=.25);ax.spines[["top","right"]].set_visible(False);ax.legend(loc="upper right",frameon=False,ncol=2);fig.subplots_adjust(left=.10,right=.83,top=.94,bottom=.13);fig.savefig(OUT/"全国水泥需求多情景_一致性修正版.png",dpi=300,facecolor="white");plt.close(fig)
prod=wide(R/"output"/"3.产量分配"/"水泥需求情景修正版_20261020"/"分省水泥产量_基准情景_修正版.xlsx");dem=wide(OUT/"分省水泥需求_基准情景_修正版.xlsx");provs=[c for c in prod if c not in ["全国","京津冀及周边8省市"]]
qa={"情景逐年顺序通过":bool((data.loc[2025:, ["低增长","基准","中增长","高增长"]].diff(axis=1).iloc[:,1:]>=-1e-6).all().all()),"2035低基中高_亿吨":{c:float(data.loc[2035,c]/1e8) for c in data},"基准2035产量_亿吨":float(prod.loc[2035,"全国"]/1e8),"基准2035需求_亿吨":float(dem.loc[2035,"全国"]/1e8),"产需比":float(prod.loc[2035,"全国"]/dem.loc[2035,"全国"]),"省级合计最大误差_吨":float((prod[provs].sum(axis=1)-prod["全国"]).abs().max())}
(OUT/"水泥情景与产量核查.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(qa,ensure_ascii=True,indent=2))
