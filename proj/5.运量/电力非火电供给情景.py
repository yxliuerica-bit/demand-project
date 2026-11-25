"""装机驱动非火电供给与剩余火电需求；未来参数均为项目假设。"""
import numpy as np
import pandas as pd

SOURCE='https://www.stats.gov.cn/xxgk/sjfb/tjgb2020/202502/t20250228_1958817.html'
POLICY='https://www.nea.gov.cn/20260130/b5b729cb7ad74723bcd614b663c75da6/c.html'
def predict(raw_generation,out):
    years=np.arange(2024,2036)
    # 单位：亿千瓦时；全口径公报，同一来源闭合，0.4为分项舍入残差。
    actual={'水电':14256.8,'核电':4508.5,'风电':9970.4,'太阳能':8390.4}
    base_total=100868.8;base_fire=63742.65
    total=pd.DataFrame({s:d.loc[years]/d.loc[2024]*base_total for s,d in raw_generation.items()},index=years)
    total.to_csv(out/'原始2024归一总发电量_未作顺序约束.csv',encoding='utf-8-sig')
    # 原模型2025高增长增幅略低于基准；仅对违反情景顺序的点作包络约束。
    total['高增长']=np.maximum(total['高增长'],total['基准'])
    total['低增长']=np.minimum(total['低增长'],total['基准'])
    # 容量单位亿千瓦；未来风光每年净增合计2亿千瓦为项目路径，2035合计38.4。
    capacities={'风电':(5.2068,1.18,12.8),'太阳能':(8.8666,1.452,25.6),'核电':(.6083,1.069,1.3)}
    nonfire=pd.DataFrame(index=years);params=[]
    for name,(start,growth,end) in capacities.items():
        cap=pd.Series(np.linspace(start,end,len(years)),index=years)
        previous=start/growth
        avg=(cap+cap.shift(1).fillna(previous))/2
        base_avg=(start+previous)/2
        # 实绩/年平均容量得到等效小时；已含历史消纳效率，不再重复扣弃风弃光率。
        hours=actual[name]/base_avg
        nonfire[name]=avg*hours
        params.append({'电源':name,'2024年末容量_亿千瓦':start,'2023年末容量_亿千瓦':previous,'2035年末容量_亿千瓦':end,'基年等效小时':hours,'未来参数性质':'容量线性增长、等效小时固定，项目假设','实绩来源':SOURCE})
    nonfire['水电']=actual['水电']*1.015**(years-2024)
    nonfire['统计舍入差额']=base_total-base_fire-sum(actual.values())
    nonfire=nonfire[['水电','核电','风电','太阳能','统计舍入差额']]
    available=nonfire.sum(axis=1)
    fire=total.sub(available,axis=0).clip(lower=0)
    used=pd.DataFrame({s:np.minimum(available,total[s]) for s in total},index=years)
    surplus=pd.DataFrame({s:np.maximum(available-total[s],0) for s in total},index=years)
    assert np.allclose(fire+used,total)
    assert np.allclose(fire.loc[2024],base_fire)
    assert (fire['高增长']>=fire['基准']-1e-6).all() and (fire['基准']>=fire['低增长']-1e-6).all()
    for df,name in [(total,'全国总发电量情景_亿千瓦时'),(nonfire,'全国非火电分电源供给_亿千瓦时'),(fire,'全国剩余火电需求_亿千瓦时'),(surplus,'非火电潜在富余_亿千瓦时')]:
        df.index.name='年份';df.to_csv(out/(name+'.csv'),encoding='utf-8-sig')
    pd.DataFrame(params).to_csv(out/'非火电装机与效率假设.csv',index=False,encoding='utf-8-sig')
    sensitivity=[]
    for factor in [.9,1.,1.1]:
        nonfire_2035=nonfire.loc[2035,'水电']+nonfire.loc[2035,'统计舍入差额']+nonfire.loc[2035,['风电','太阳能','核电']].sum()*factor
        for scenario in total:
            sensitivity.append({'情景':scenario,'风光核等效小时系数':factor,'2035剩余火电量_万亿千瓦时':max(total.loc[2035,scenario]-nonfire_2035,0)/1e4})
    pd.DataFrame(sensitivity).to_csv(out/'非火电效率敏感性_2035.csv',index=False,encoding='utf-8-sig')
    report=f'''# 2024基年：非火电供给驱动的火电需求预测

2024年为统一研究基年，2025—2035年为预测期。全国总发电量基准100868.8亿千瓦时；火电基准63742.65亿千瓦时，直接采用项目官方分省历史表加总（与公报63742.6有0.05亿千瓦时舍入差）。水电14256.8、核电4508.5、风电9970.4、太阳能8390.4亿千瓦时采用2024年统计公报。分项舍入差单列，不作为新增能源。

全国总发电量=原各情景相对2024年的增长轨迹×2024全国实绩。2025年公报数据只用于事后核验，不再校准模型。原模型2025年高增长增幅略低于基准，对该点作高增长不低于基准的包络约束；原始归一数据单独保留，不交换情景标签。

风光核供给=年平均装机×2024年等效发电小时，2023年末容量由2024公布容量与增速反推。水电按年增1.5%项目假设；不把抽蓄容量作为新增一次能源。2035年末风电12.8、太阳能25.6、核电1.3亿千瓦为保留的项目情景假设，非官方预测；从2024年末线性增长至端点。未来效率固定并提供±10%敏感性。后续政策仅为情景参考，并非宣称模型只使用2024年以前可得信息。

火电=max(总发电量-非火电供给,0)。三增长情景共用非火电供给路径；未强制火电下降，未模拟逐时调度、电网约束与省际送受电。省级火电按截至2024年的官方历史份额及趋势分配。

电煤仍为全部火电乘煤耗的标准煤代理，未剥离燃气、生物质等非煤火电，不是实际煤电耗煤或原煤运输量。煤耗以2024年302.5克标准煤/千瓦时为基线，2035年285为假设。

2024公报：{SOURCE}
政策参考：{POLICY}
'''
    (out/'火电情景修正说明.md').write_text(report,encoding='utf-8')
    pd.DataFrame({'2025预测总发电量_亿千瓦时':total.loc[2025],'2025实绩总发电量_亿千瓦时':105752.5,'2025预测火电量_亿千瓦时':fire.loc[2025],'2025实绩火电量_亿千瓦时':63271.5}).assign(火电相对误差=lambda d:d['2025预测火电量_亿千瓦时']/d['2025实绩火电量_亿千瓦时']-1).to_csv(out/'2025事后核验_不参与校准.csv',encoding='utf-8-sig')
    return total,fire
