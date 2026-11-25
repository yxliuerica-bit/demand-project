"""分省总发电量、火电量和非火电量的统一数据出口。"""
import numpy as np
import pandas as pd

def distribute(total,weights,caps):
    x=pd.Series(0.,index=weights.index)
    for _ in range(len(x)+2):
        left=total-x.sum()
        if left<1e-7:break
        room=(caps-x).clip(lower=0);w=weights.where(room>1e-8,0).clip(lower=0)
        if w.sum()==0:w=room
        x+=np.minimum(left*w/w.sum(),room)
    assert abs(x.sum()-total)<1e-4
    return x

def export(root,out,provinces,region,mapping,national,fire_national,fire_shares,fire_history,wide,norm):
    src=pd.read_excel(root/'data'/'预测原始数据'/'发电量分省年度数据.xls',header=3)
    src['地区']=src['地区'].map(norm)
    yc={int(str(c)[:4]):c for c in src if str(c)[:4].isdigit()}
    history=src.set_index('地区').reindex(provinces)[list(yc.values())].T
    history.index=list(yc);history=history.apply(pd.to_numeric,errors='coerce').sort_index()
    assert not history.isna().any().any()
    result={};rows=[]
    for label,old in mapping.items():
        raw=wide(root/'output'/'2.消费需求预测'/f'Sce_{old}'/'全国完整数据'/'分省发电量数据(2005-2035).xlsx')
        gen=history.copy();fire=fire_history.copy().sort_index()
        for year in range(2025,2036):
            # 省级增长信号锚定官方2024结构，再归一到新版全国控制量。
            signal=history.loc[2024]*raw.loc[year,provinces].clip(lower=0)/raw.loc[2024,provinces].clip(lower=1e-8)
            gen.loc[year]=signal/signal.sum()*national.loc[year,label]
            fire.loc[year]=distribute(fire_national.loc[year,label],fire_shares.loc[year],gen.loc[year])
            rows.extend([{'年份':year,'情景':label,'省份':p,'总发电量_亿千瓦时':gen.loc[year,p],'火电量_亿千瓦时':fire.loc[year,p],'非火电量_亿千瓦时':gen.loc[year,p]-fire.loc[year,p],'数据属性':'2024基年情景预测；省级非火电为余额，非独立装机模型'} for p in provinces])
        for df,name in [(gen,'总发电量'),(fire,'火力发电量')]:
            df['全国']=df[provinces].sum(axis=1);df['京津冀及周边8省市']=df[region].sum(axis=1)
            df.index.name='年份';df.to_csv(out/f'分省{name}_历史及预测_{label}_亿千瓦时.csv',encoding='utf-8-sig')
        result[label]=(gen,fire)
    pd.DataFrame(rows).to_csv(out/'分省电力供需预测_三情景.csv',index=False,encoding='utf-8-sig')
    (out/'分省预测衔接说明.md').write_text('2024年及以前采用官方分省历史数据。2025—2035年总发电量将各省相对2024年的原模型增长信号作用于官方2024实绩，并归一到新版全国总量。火电按官方历史份额趋势分配，同时约束火电不超过本省总发电量；超额在仍有空间的省份间再分配。非火电为总量减火电的余额，不是省级装机或电网调度独立预测。原Sce目录作为旧模型输入保留，下游使用本目录新版CSV。',encoding='utf-8')
    return result
