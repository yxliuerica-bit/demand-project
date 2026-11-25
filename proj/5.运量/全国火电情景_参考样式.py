"""按参考图绘制火电情景，保留模型情景原标签。"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output'/'2.消费需求预测'
DEM=BASE/'电力与电煤需求情景修正版_20261022'
OUT=DEM/'三张电力专题图'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei'],'axes.unicode_minus':False,'font.size':15})
fire=pd.read_csv(DEM/'全国剩余火电需求_亿千瓦时.csv',index_col=0)/1e4
fire.index=pd.to_numeric(fire.index)
hist=pd.read_csv(DEM/'分省火力发电量_国家统计局历史实绩.csv',index_col=0)
hist.index=pd.to_numeric(hist.index)
hist=hist.sum(axis=1).sort_index()/1e4
gen={}
for label,source in [('高增长','高增长'),('基准','中增长'),('低增长','低增长')]:
    d=pd.read_excel(BASE/f'Sce_{source}'/'全国完整数据'/'分省发电量数据(2005-2035).xlsx',index_col=0)
    d.index=pd.to_numeric(d.index)
    gen[label]=pd.read_csv(DEM/'全国总发电量情景_亿千瓦时.csv',index_col=0)[label]/1e4
fig,ax=plt.subplots(figsize=(10.8,7.2))
ax.plot(hist.loc[2020:2024].index,hist.loc[2020:2024],color='#B96800',marker='o',lw=2.5,label='历史实绩（分省加总）')
colors={'高增长':'#C77500','基准':'#FF9700','低增长':'#E5C900'}
rows=[]
for label in fire:
    s=fire[label].sort_index()
    ax.plot(s.index,s.values,marker='o',markersize=5.8,lw=2.4,color=colors[label],label=label)
    value=float(fire.loc[2035,label]);share=value/gen[label].loc[2035]
    rows.append((label,value,share))
# 按终点值排布标签，避免曲线接近时文本相互遮挡。
ordered=sorted(rows,key=lambda x:x[1])
positions=[]
for _,v,_ in ordered:positions.append(max(v,positions[-1]+.48) if positions else v)
for (label,v,share),label_y in zip(ordered,positions):
    ax.annotate(f'{label}：{v:.2f}万亿千瓦时\n占总发电量{share:.1%}',xy=(2035,v),xytext=(2036.2,label_y),ha='left',va='center',fontsize=14,color=colors[label],fontweight='bold',annotation_clip=False,arrowprops={'arrowstyle':'-','color':colors[label],'lw':1})
ax.set_xlim(2019.5,2035.5)
ax.set_ylim(min(hist.loc[2020:2024].min(),fire.min().min())-.5,max(max(positions)+.3,fire.max().max()+.25))
ax.set_xticks([2020,2025,2030,2035]);ax.tick_params(axis='x',labelsize=12)
ax.set_ylabel('火电发电量（万亿千瓦时）',fontsize=17);ax.set_xlabel('年份',fontsize=16)
ax.set_title('全国火电发电量不同情景预测',fontsize=20,pad=15)
ax.grid(ls='--',alpha=.28);ax.legend(loc='upper left',fontsize=12,frameon=False)
fig.subplots_adjust(left=.11,right=.69,top=.9,bottom=.21)
fig.text(.11,.07,'火电需求＝总发电量－水电、核电、风电及太阳能供给；各情景共用非火电供给路径。\n2024年为研究基年，2025年起为预测；2025公报值仅事后核验。',fontsize=11,color='#555')
png=OUT/'04_全国火电发电量不同情景_参考图样式.png'
fig.savefig(png,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
pd.DataFrame(rows,columns=['情景','2035火电量_万亿千瓦时','占对应总发电量比例']).to_csv(OUT/'04_火电情景终点与占比.csv',index=False,encoding='utf-8-sig')
index=ROOT/'output'/'水泥电力图表更新索引_20261022.md'
text=index.read_text(encoding='utf-8')
if png.stem not in text:text+=f'\n- [{png.stem}]({png.relative_to(ROOT/"output").as_posix()})\n'
index.write_text(text,encoding='utf-8')
print(pd.DataFrame(rows,columns=['情景','2035火电量_万亿千瓦时','占比']).to_string(index=False))
