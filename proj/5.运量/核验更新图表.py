from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output'
groups=[OUT/'2.消费需求预测'/'水泥需求情景修正版_20261022',OUT/'2.消费需求预测'/'电力与电煤需求情景修正版_20261022',OUT/'3.产量分配'/'水泥需求情景修正版_20261022',OUT/'3.产量分配'/'电煤供给情景修正版_20261022',OUT/'4.运量'/'水泥货运区域匹配_情景修正版_20261022',OUT/'4.运量'/'电力行业电煤货运区域匹配_情景修正版_20261022']
checks={}
dem=groups[1]
prov=pd.read_csv(dem/'分省电力供需预测_三情景.csv')
national=pd.read_csv(dem/'全国总发电量情景_亿千瓦时.csv',index_col=0)
national_fire=pd.read_csv(dem/'全国剩余火电需求_亿千瓦时.csv',index_col=0)
for label,z in prov.groupby('情景'):
    assert z.groupby('年份')['省份'].nunique().eq(31).all()
    assert (z['火电量_亿千瓦时']<=z['总发电量_亿千瓦时']+1e-7).all()
    assert (z[['总发电量_亿千瓦时','火电量_亿千瓦时','非火电量_亿千瓦时']]>=-1e-7).all().all()
    by_year=z.groupby('年份').sum(numeric_only=True)
    assert np.allclose(by_year['总发电量_亿千瓦时'],national.loc[by_year.index,label])
    assert np.allclose(by_year['火电量_亿千瓦时'],national_fire.loc[by_year.index,label])
checks['分省电力三情景']='31省×11年×3情景完整；总量、火电全国控制值及省级火电上限通过'
for industry,folder,value in [('水泥',groups[-2],'省际水泥预测量_吨'),('电煤',groups[-1],'省际电煤预测量_吨')]:
    od=pd.read_csv(next((folder/'数据').glob('*省际*OD预测.csv')))
    mode=pd.read_csv(next((folder/'数据').glob('*OD运输方式匹配.csv')))
    diff=(mode[['公路_吨','铁路_吨','水运_吨']].sum(axis=1)-mode[value]).abs().max()
    assert diff<1e-4
    assert (od[value]>=0).all()
    assert np.allclose(od.groupby('年份')[value].sum(),mode.groupby('年份')[value].sum())
    checks[industry]={'全期方式逐行加总最大误差_吨':float(diff),'年份数':int(od['年份'].nunique())}
prod=pd.read_excel(groups[2]/'分省水泥产量_基准情景_修正版.xlsx',index_col=0)
assert abs(prod.loc[2024,'全国']/1e8-18.25239)<1e-8
assert abs(prod.loc[2025,'全国']/1e8-16.9)<1e-8
checks['水泥产量控制值核验']='2024年18.25239亿吨；2025年16.9亿吨，通过'
(OUT/'图表全期核验_20261022.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 水泥、电力图表更新（2026-10-22）','','水泥下游行业结构以中增长模型分项占比校准至新版需求总量；历史需求为模型值，不等同统计局产量。水泥产量历史采用统计局分省实绩，2025年省级分配为估算。','','电煤需求与货运均为标准煤，供给为情景代理。历史电煤图按2024年煤耗基线折算，不是逐年实测耗煤。旧日期目录保留，本索引只列10月22日版本。','']
files=[]
for folder in groups:
    lines+=['## '+folder.name,'']
    for p in sorted(folder.rglob('*.png')):
        lines.append(f'- [{p.stem}]({p.relative_to(OUT).as_posix()})')
        files.append(p)
    lines.append('')
(OUT/'水泥电力图表更新索引_20261022.md').write_text('\n'.join(lines),encoding='utf-8')
with (OUT/'水泥电力图表更新索引_20261022.md').open('a',encoding='utf-8') as f:
    f.write('\n## 新版分省预测数据（2024基年）\n\n')
    for p in sorted(dem.glob('分省*情景.csv'))+sorted(dem.glob('分省*历史及预测*.csv'))+sorted(dem.glob('分省电煤折算需求*.csv')):
        f.write(f'- [{p.stem}]({p.relative_to(OUT).as_posix()})\n')
    f.write(f'\n- [分省预测衔接说明]({(dem/"分省预测衔接说明.md").relative_to(OUT).as_posix()})\n')
# 缩略图仅用于检查画布、图例与标签。
for start in range(0,len(files),12):
    batch=files[start:start+12]
    canvas=Image.new('RGB',(1800,420*((len(batch)+2)//3)),'#eeeeee')
    draw=ImageDraw.Draw(canvas)
    for i,p in enumerate(batch):
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            thumb=ImageOps.contain(im.convert('RGB'),(590,390))
            x=(i%3)*600;y=(i//3)*420
            canvas.paste(thumb,(x+(600-thumb.width)//2,y+(390-thumb.height)//2))
            draw.text((x+10,y+395),str(start+i+1),fill='black')
    canvas.save(OUT/f'图表检查缩略图_{start//12+1}.jpg')
print(json.dumps({'图表数量':len(files),'checks':checks},ensure_ascii=False,indent=2))
