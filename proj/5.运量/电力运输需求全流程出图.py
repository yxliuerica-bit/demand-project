from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(r'C:\Users\admin\Desktop\货运预测李童+高洁\京津冀货运发展')
PACK=Path(r'C:\Users\admin\Desktop\货运预测李童+高洁\电力需求预测图表脚本包')
PIPE=ROOT/'output'/'4.运量'/'电力行业电煤货运区域匹配_情景修正版_20261022'
DEM=ROOT/'output'/'2.消费需求预测'/'电力与电煤需求情景修正版_20261022'
OUT=PIPE/'电力运输需求全流程图表'
OUT.mkdir(parents=True,exist_ok=True)
REGION=['北京市','天津市','河北省','河南省','山东省','山西省','内蒙古自治区','陕西省']
COLORS=['#DCEAF5','#AFC8E2','#DDEEC8','#B5D79D','#F7D4D7','#EB9BA3','#F8E2BE','#F5C794']
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':15,'axes.labelsize':17,'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':13})

def save(fig,name):
    fig.savefig(OUT/name,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)

def wide_xlsx(path):
    d=pd.read_excel(path);d=d.rename(columns={d.columns[0]:'年份'}).set_index('年份');return d.apply(pd.to_numeric,errors='coerce')

# Current model inputs/results
scen=pd.read_csv(DEM/'全国电力需求情景_亿千瓦时.csv',index_col=0);scen.index=pd.to_numeric(scen.index)
fire=pd.read_csv(DEM/'全国火电发电量情景_MWh.csv',index_col=0);fire.index=pd.to_numeric(fire.index)
coal=pd.read_csv(DEM/'全国电煤需求情景_吨标煤.csv',index_col=0);coal.index=pd.to_numeric(coal.index)
ann=pd.read_csv(PIPE/'数据'/'2025-2035全国电力与电煤汇总.csv').set_index('年份')
bal=pd.read_csv(PIPE/'数据'/'2025-2035省级电煤供需与货运汇总.csv')
od=pd.read_csv(PIPE/'数据'/'2025-2035省际电煤OD预测.csv')
base=ROOT/'output'/'2.消费需求预测'/'Sce_中增长'/'全国完整数据'
elec_prov=wide_xlsx(base/'分省电力消费量数据(2005-2035).xlsx')

# 1. PPT vs current baseline; PPT is source material dated 2026-04-26, current pipeline updated 2026-10-22.
ppt=pd.Series({2020:7.762,2023:9.439,2025:11.510,2030:12.942,2035:14.154},name='PPT结果')
current=(elec_prov['全国']/1e4).reindex(ppt.index);current.name='当前项目基准'
cmp=pd.concat([ppt,current],axis=1);cmp['差值_当前减PPT']=cmp['当前项目基准']-cmp['PPT结果'];cmp['相对差异']=cmp['差值_当前减PPT']/cmp['PPT结果'];cmp.to_csv(OUT/'01_PPT与当前项目用电量对比.csv',encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(11.8,5.3));ax.plot(cmp.index,cmp['PPT结果'],marker='o',lw=2.4,color='#9E9E9E',label='PPT分部门预测');ax.plot(cmp.index,cmp['当前项目基准'],marker='o',lw=2.8,color='#D62728',label='当前项目基准')
for x,y in cmp['当前项目基准'].items():ax.text(x,y+.18,f'{y:.2f}',ha='center',color='#B22222',fontweight='bold')
ax.set_ylabel('全社会用电量（万亿千瓦时）');ax.set_xlabel('年份');ax.set_xticks(cmp.index);ax.grid(axis='y',ls='--',alpha=.3);ax.legend(frameon=False,ncol=2);save(fig,'01_PPT与当前项目用电量对比.png')

# 2. Current electricity scenarios
fig,ax=plt.subplots(figsize=(11.8,5.3));
for col,color in zip(scen.columns,['#D62728','#ED7D31','#4472C4']):ax.plot(scen.index,scen[col]/1e4,marker='o',lw=2.5,label=col,color=color)
ax.set_ylabel('全国用电量（万亿千瓦时）');ax.set_xlabel('年份');ax.set_xticks([2025,2030,2035]);ax.grid(ls='--',alpha=.3);ax.legend(frameon=False,ncol=3);save(fig,'02_全国电力需求情景.png')

# 3. Conversion chain
x=ann.index
fig,axs=plt.subplots(3,1,figsize=(11.8,10),sharex=True)
series=[(ann['全国用电量_万亿千瓦时'],'全社会用电量','万亿千瓦时','#2F5597'),(ann['全国火电量_万亿千瓦时'],'火力发电量','万亿千瓦时','#ED7D31'),(ann['全国电煤需求_亿吨标煤'],'电煤需求','亿吨标准煤','#7F6000')]
for ax,(s,title,unit,color) in zip(axs,series):
 ax.plot(x,s,marker='o',lw=2.5,color=color);ax.fill_between(x,s,alpha=.12,color=color);ax.set_ylabel(unit);ax.text(.01,.87,title,transform=ax.transAxes,fontweight='bold',fontsize=17);ax.text(x[-1],s.iloc[-1],f'  {s.iloc[-1]:.2f}',va='center',fontweight='bold',color=color);ax.grid(axis='y',ls='--',alpha=.25)
axs[-1].set_xlabel('年份');axs[-1].set_xticks([2025,2030,2035]);fig.subplots_adjust(hspace=.12);save(fig,'03_用电需求到电煤需求换算链.png')

# 4. JJJ 8 provincial coal demand
z=bal[bal['省份'].isin(REGION)].pivot(index='年份',columns='省份',values='电煤需求_吨').reindex(columns=REGION)/1e8
fig,ax=plt.subplots(figsize=(12.8,5.4));bottom=np.zeros(len(z))
for p,c in zip(REGION,COLORS):ax.bar(z.index,z[p],bottom=bottom,color=c,edgecolor='white',linewidth=.3,label=p);bottom+=z[p].to_numpy()
ax.set_ylabel('八省市电煤需求（亿吨标准煤）');ax.set_xlabel('年份');ax.set_xticks([2025,2030,2035]);ax.grid(axis='y',ls='--',alpha=.3);ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=False);save(fig,'04_京津冀8省市电煤需求.png')

# 5. Demand and interprovincial freight
fig,ax=plt.subplots(figsize=(11.8,5.4));ax.plot(ann.index,ann['全国电煤需求_亿吨标煤'],marker='o',lw=2.7,color='#7F6000',label='全国电煤需求');ax.plot(ann.index,ann['省际电煤运输量_亿吨标煤'],marker='o',lw=2.7,color='#4472C4',label='省际电煤运输量');ax.set_ylabel('亿吨标准煤');ax.set_xlabel('年份');ax.set_xticks([2025,2030,2035]);ax.grid(ls='--',alpha=.3);ax.legend(frameon=False,ncol=2)
ax2=ax.twinx();ratio=ann['省际电煤运输量_亿吨标煤']/ann['全国电煤需求_亿吨标煤'];ax2.plot(ann.index,ratio,lw=1.8,ls='--',color='#70AD47',label='省际运输比例');ax2.set_ylabel('省际运输量/电煤需求');ax2.yaxis.set_major_formatter(PercentFormatter(1));save(fig,'05_电煤需求与省际运输量.png')

# 6. Mode + top OD
mode=pd.read_csv(PIPE/'数据'/'2025-2035省际电煤OD运输方式匹配.csv');m=mode[mode['年份']==2035][['公路_吨','铁路_吨','水运_吨']].sum()/1e8;m.index=['公路','铁路','水运']
top=od[od['年份']==2035].nlargest(10,'省际电煤预测量_吨').copy();top['流向']=top['起点省份'].str.replace('省|市|自治区|壮族|回族|维吾尔','',regex=True)+'—'+top['终点省份'].str.replace('省|市|自治区|壮族|回族|维吾尔','',regex=True);top['亿吨标准煤']=top['省际电煤预测量_吨']/1e8
fig,(a1,a2)=plt.subplots(1,2,figsize=(14.5,6),gridspec_kw={'width_ratios':[.8,1.7]});bars=a1.bar(m.index,m.values,color=['#ED7D31','#4472C4','#5B9BD5']);a1.set_ylabel('2035年运输量（亿吨标准煤）');a1.grid(axis='y',ls='--',alpha=.25)
for b,v in zip(bars,m):a1.text(b.get_x()+b.get_width()/2,v+.08,f'{v:.2f}\n{v/m.sum():.1%}',ha='center',fontweight='bold')
top=top.sort_values('亿吨标准煤');a2.barh(top['流向'],top['亿吨标准煤'],color='#AFC8E2');a2.set_xlabel('2035年省际电煤运输量（亿吨标准煤）');a2.grid(axis='x',ls='--',alpha=.25);fig.subplots_adjust(wspace=.35);save(fig,'06_运输方式与主要省际流向.png')

summary={
 'PPT_2035用电量_万亿千瓦时':float(ppt.loc[2035]),
 '当前_2035用电量_万亿千瓦时':float(ann.loc[2035,'全国用电量_万亿千瓦时']),
 '当前_2035火电量_万亿千瓦时':float(ann.loc[2035,'全国火电量_万亿千瓦时']),
 '当前_2035电煤需求_亿吨标准煤':float(ann.loc[2035,'全国电煤需求_亿吨标煤']),
 '当前_2035省际电煤运输_亿吨标准煤':float(ann.loc[2035,'省际电煤运输量_亿吨标煤']),
 '当前_2035省际运输比例':float(ratio.loc[2035]),
 '当前_铁路占比':float(m['铁路']/m.sum())}
(OUT/'数据口径与图表说明.md').write_text(f'''# 电力运输需求全流程图表说明\n\n## 版本判断\n\n- 《电力需求预测_260426.pptx》及图表脚本包属于分部门用电需求方案，2035年全社会用电量为{ppt.loc[2035]:.3f}万亿千瓦时。\n- 当前货运项目的基准情景来自 `Sce_中增长` 分省模型，2035年用电量为{summary['当前_2035用电量_万亿千瓦时']:.3f}万亿千瓦时。该结果已经进入电煤需求、供需匹配、OD和运输方式流程，因此全流程展示以当前项目结果为准。\n- PPT分部门结果可用于解释用电需求来源，但不能与当前电煤货运数值直接拼接。若要使用PPT的行业结构图，需要先把各部门预测统一校准到当前全国用电总量。\n\n## 当前换算链\n\n全社会用电需求 → 火力发电量 → 单位供电煤耗 → 电煤需求 → 分省供需匹配 → 省际OD → 公路/铁路/水运。\n\n2035年：用电量{summary['当前_2035用电量_万亿千瓦时']:.3f}万亿千瓦时，火电量{summary['当前_2035火电量_万亿千瓦时']:.3f}万亿千瓦时，电煤需求{summary['当前_2035电煤需求_亿吨标准煤']:.3f}亿吨标准煤，省际电煤运输{summary['当前_2035省际电煤运输_亿吨标准煤']:.3f}亿吨标准煤。\n\n## 口径修正\n\n本次已将电煤计算改为读取 `分省火力发电量数据(2005-2035).xlsx`，并按“1亿千瓦时=10万兆瓦时”换算后乘以煤耗。旧脚本误把已经包含煤耗换算的 `分省火电消费量预测.xlsx` 再次乘以煤耗。\n''',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))

(OUT/"口径补充.md").write_text("电煤需求和运输量均为标准煤，非实际原煤吨数。省级供给为情景代理，非官方原煤产量或实际可外运量。煤耗2024年302.5克标煤/千瓦时，2035年285为情景假设；未加3%供给余量。",encoding="utf-8")
