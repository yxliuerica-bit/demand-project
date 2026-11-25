"""三张电力专题图；行业图为按用电结构分摊的发电量估算。"""
from pathlib import Path
import runpy,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output'/'2.消费需求预测'
DEM=BASE/'电力与电煤需求情景修正版_20261022'
OUT=DEM/'三张电力专题图'
OUT.mkdir(parents=True,exist_ok=True)
YEARS=np.arange(2024,2036)
REGION=['北京市','天津市','河北省','河南省','山东省','山西省','内蒙古自治区','陕西省']
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False,'font.size':15,'axes.labelsize':17,'axes.titlesize':20,'legend.fontsize':13})
def wide(scenario):
    label={'高增长':'高增长','中增长':'基准','低增长':'低增长'}[scenario]
    d=pd.read_csv(DEM/f'分省总发电量_历史及预测_{label}_亿千瓦时.csv',index_col=0)
    return d.loc[YEARS]/1e4
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)

gen={label:wide(source) for label,source in [('高增长','高增长'),('基准','中增长'),('低增长','低增长')]}
scenarios=pd.DataFrame({label:d['全国'] for label,d in gen.items()})
scenarios.index.name='年份'
scenarios.to_csv(OUT/'01_全国发电量不同情景_万亿千瓦时.csv',encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(13.5,6.5))
for (label,s),color in zip(scenarios.items(),['#D34A42','#E8A044','#4778B5']):
    ax.plot(YEARS,s,marker='o',lw=2.7,label=label,color=color)
    ax.annotate(f'{s.iloc[-1]:.2f}',(2035,s.iloc[-1]),xytext=(8,0),textcoords='offset points',color=color,va='center',fontweight='bold')
ax.set(title='全国发电量不同情景预测（2024基年）',xlabel='年份',ylabel='发电量（万亿千瓦时）',xticks=[2024,2030,2035],xlim=(2023.6,2036))
ax.grid(axis='y',ls='--',alpha=.25);ax.legend(frameon=False,ncol=3);ax.spines[['top','right']].set_visible(False)
fig.tight_layout();save(fig,'01_全国发电量不同情景预测')

# 行业原始预测来自已有PPT整理数据；只使用结构，重新归一后匹配项目总发电量。
pack=ROOT.parent/'电力需求预测图表脚本包'/'chart_data.py'
sector_source=runpy.run_path(str(pack))
raw=pd.DataFrame({**sector_source['INDUSTRY'],**sector_source['NONINDUSTRY']},index=[int(y) for y in sector_source['YEARS']])
raw=raw.reindex(range(2020,2036)).interpolate(method='index').loc[YEARS]
shares=raw.div(raw.sum(axis=1),axis=0)
allocated=shares.mul(scenarios['基准'],axis=0)
assert np.allclose(allocated.sum(axis=1),scenarios['基准'])
allocated.index.name='年份'
allocated.to_csv(OUT/'02_分行业分摊发电量_万亿千瓦时.csv',encoding='utf-8-sig')
shares.to_csv(OUT/'02_分行业用电结构占比.csv',encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(17,10))
bottom=np.zeros(len(YEARS));colors=plt.get_cmap('tab20').colors
short={'黑色金属冶炼及压延加工业':'黑色金属冶炼及压延','有色金属冶炼及压延加工业':'有色金属冶炼及压延','化学原料及化学制品制造业':'化学原料及制品','电力、热力、燃气及水生产供应业':'电力热力燃气及水供应','互联网数据服务业及移动通信基站':'互联网数据及通信基站'}
for i,c in enumerate(allocated):
    ax.bar(YEARS,allocated[c],bottom=bottom,width=.65,label=short.get(c,c),color=colors[i],edgecolor='white',linewidth=.25)
    bottom+=allocated[c].to_numpy()
for y,total in zip(YEARS,bottom):ax.text(y,total+.12,f'{total:.2f}',ha='center',fontsize=12)
ax.set(title='下游行业对应发电量预测（基准情景）',xlabel='年份',ylabel='分摊发电量（万亿千瓦时）',xticks=YEARS,ylim=(0,bottom.max()*1.1))
ax.grid(axis='y',ls='--',alpha=.25);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False)
fig.subplots_adjust(left=.08,right=.73,bottom=.15,top=.91)
fig.text(.08,.045,'按已有分行业用电预测结构分摊项目总发电量；非行业直接发电量。缺失年份占比由原始预测值线性插值得到。',fontsize=13,color='#555')
save(fig,'02_下游行业分摊发电量堆叠柱状图')

fire=pd.read_csv(DEM/'分省火力发电量_历史及预测_基准_亿千瓦时.csv',index_col=0)/1e4
history=pd.read_csv(DEM/'分省火力发电量_国家统计局历史实绩.csv',index_col=0)/1e4
fire=fire.combine_first(history.loc[[2024]])
total=gen['基准'][REGION]
assert not total.isna().any().any() and not fire[REGION].isna().any().any()
fig,axes=plt.subplots(2,4,figsize=(18,9),sharex=True,sharey=True)
for ax,p in zip(axes.flat,REGION):
    ax.plot(YEARS,total[p],color='#2F65A0',marker='o',markersize=4,lw=2.3,label='总发电量')
    ax.plot(YEARS,fire.loc[YEARS,p],color='#DA8040',marker='s',markersize=3.8,lw=2.3,label='火力发电量')
    ax.set_title(p.replace('自治区','').replace('省','').replace('市',''),fontsize=18)
    ax.set_xticks([2025,2030,2035]);ax.set_ylim(0,max(total.max().max(),fire.loc[YEARS,REGION].max().max())*1.1);ax.grid(axis='y',ls='--',alpha=.25);ax.spines[['top','right']].set_visible(False)
fig.suptitle('京津冀及周边八省市总发电量与火电量预测（基准情景）',fontsize=23,y=.98)
fig.supylabel('发电量（万亿千瓦时）',x=.015,fontsize=18);fig.supxlabel('年份',y=.08,fontsize=18)
fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.935),ncol=2,frameon=False,fontsize=16)
fig.subplots_adjust(left=.065,right=.985,bottom=.16,top=.84,hspace=.28,wspace=.12)
fig.text(.065,.025,'总发电量按2024基年实绩校准；火电按总量减非火电供给估算。省级结构按历史趋势分配，未模拟省级调度。',fontsize=13,color='#555')
save(fig,'03_八省市总发电量与火电量对比')
comparison=pd.concat({'总发电量':total,'火力发电量':fire.loc[YEARS,REGION]},axis=1)
comparison.to_csv(OUT/'03_八省市发电量对比_万亿千瓦时.csv',encoding='utf-8-sig')
notes='''# 三张电力专题预测图

- 01：全国总发电量，采用原高增长、中增长、低增长模型，单位万亿千瓦时；与全社会用电量口径不同。
- 02：已有PPT整理的14类下游部门用电预测，经逐年结构归一后分配到当前基准情景全国总发电量。2025、2030、2035为来源节点，中间年份线性插值。这是按用电结构分摊的发电需求估算，不是行业自身发电量，也不是已独立重估的行业预测。
- 03：八省市逐省对比总发电量与校准后的火电量，所有面板使用相同纵轴，便于省间比较。此处采用生产端发电量，并非消费端用电量；未对省际送受电或线损作额外换算。
- 火电省级预测按国家统计局历史结构及趋势分配，2024年为基年，2025公报数据不参与校准。
'''
(OUT/'图表口径说明.md').write_text(notes,encoding='utf-8')
index=ROOT/'output'/'水泥电力图表更新索引_20261022.md'
text=index.read_text(encoding='utf-8');marker='## 新增：三张电力专题预测图'
text=text.split(marker)[0].rstrip()+'\n\n'+marker+'\n\n'
for p in sorted(OUT.glob('*.png')):text+=f'- [{p.stem}]({p.relative_to(ROOT/"output").as_posix()})\n'
index.write_text(text,encoding='utf-8')
print(json.dumps({'output':str(OUT),'行业分摊加总最大误差':float((allocated.sum(axis=1)-scenarios['基准']).abs().max()),'火电超过总发电量的省年数':int((fire.loc[YEARS,REGION]>total).sum().sum())},ensure_ascii=False,indent=2))
