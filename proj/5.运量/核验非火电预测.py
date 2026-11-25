from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from 电力非火电供给情景 import predict
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output'/'2.消费需求预测'
DEM=BASE/'电力与电煤需求情景修正版_20261022'
OUT=DEM/'三张电力专题图'
raw={}
for label,old in [('高增长','高增长'),('基准','中增长'),('低增长','低增长')]:
    raw[label]=pd.read_excel(BASE/f'Sce_{old}'/'全国完整数据'/'分省发电量数据(2005-2035).xlsx',index_col=0)['全国']
total,fire=predict(raw,DEM)
nonfire=pd.read_csv(DEM/'全国非火电分电源供给_亿千瓦时.csv',index_col=0)
assert np.allclose(fire.add(nonfire.sum(axis=1),axis=0),total)
assert np.allclose(fire.loc[2024],63742.65)
assert np.allclose(total.loc[2024],100868.8)
coal=pd.read_csv(DEM/'全国电煤需求情景_吨标煤.csv',index_col=0)
assert np.allclose(coal.loc[2035],fire.loc[2035]*1e5*.285)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False,'font.size':14})
fig,ax=plt.subplots(figsize=(13.5,7))
bottom=np.zeros(len(nonfire))
for c,color in zip(['水电','核电','风电','太阳能'],['#6FA8DC','#A890C0','#6BB99F','#EBC56B']):
    v=nonfire[c]/1e4;ax.bar(nonfire.index,v,bottom=bottom,label=c,color=color);bottom+=v.to_numpy()
v=fire['基准']/1e4;ax.bar(fire.index,v,bottom=bottom,label='剩余火电需求',color='#C88F72')
ax.plot(total.index,total['基准']/1e4,c='#333',marker='o',label='总发电量（基准）',lw=2)
ax.set(xlabel='年份',ylabel='发电量（万亿千瓦时）',title='非火电供给与剩余火电需求（基准情景）')
ax.legend(ncol=3,frameon=False,loc='upper left');ax.set_ylim(0,total['基准'].max()/1e4*1.22);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(OUT/'05_分电源供给与剩余火电需求.png',dpi=300);plt.close(fig)
old=DEM/'火电占比共同情景假设.csv'
if old.exists():
    archive=DEM/'已停用_火电占比共同情景假设.csv'
    old.replace(archive)
summary=fire.loc[[2025,2030,2035]]/1e4
summary.to_csv(DEM/'火电需求重预测核验摘要.csv',encoding='utf-8-sig')
print(summary.to_string())
print('能量平衡、2024基年控制值、2035煤耗换算核验通过；2035非火电供给：',nonfire.loc[2035].sum()/1e4)
