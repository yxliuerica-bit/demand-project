# -*- coding: utf-8 -*-
"""
绘制“京津冀及周边8省市”钢铁/水泥消费结构图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 基本参数 ----------
sce = 'Sce_Pop_BAU_GDP_BAU'
base_dir = '/lustre/home/litong/intern/京津冀货运发展/output/predict_results/' + sce + '/12_跨省应用模型预测/分省完整数据/'

steel_file  = os.path.join(base_dir, '下游行业用钢需求预测.xlsx')
cement_file = os.path.join(base_dir, '下游行业水泥需求预测.xlsx')

steel_stream  = ['新增建筑面积','汽车产量','新增铁路里程','洗衣机产量','电冰箱产量','空调产量']
cement_stream = ['新增建筑面积','新增高速公路里程','新增其他等级公路里程','新增铁路里程','农村水电基础设施投资']

# 京津冀及周边 8 省市
provlist_bthsa = ['北京市','天津市','河北省','河南省','山东省','山西省','内蒙古自治区','陕西省']

years_plot = range(2010, 2036)
save_png   = True
dpi        = 300

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

steel_colors  = ['#bdbdbd','#ef5350','#fbc02d','#7e57c2','#1e88e5','#43a047']
cement_colors = ['#bdbdbd','#1e88e5','#90caf9','#7e57c2','#26a69a']

# ---------- 工具函数 ----------
def read_sector_regional_series(xlsx_path: str, sector: str, suffix: str, provinces: list) -> pd.Series:
    """读取行业中间结果，并按指定省份汇总（京津冀+周边8省市）。"""
    sheet_name = f'{sector}_{suffix}'
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, index_col=0)
    numeric_df = df.select_dtypes(include='number')
    # 只保留我们关心的省份列
    cols_exist = [c for c in provinces if c in numeric_df.columns]
    if not cols_exist:
        raise RuntimeError(f"{sheet_name} 中不含 8 省市列")
    reg = numeric_df[cols_exist].sum(axis=1)
    reg.name = sector
    reg = reg.reindex(years_plot).fillna(0).clip(lower=0)
    return reg

def build_stacked_df(xlsx_path: str, sectors: list, suffix: str, provinces: list) -> pd.DataFrame:
    series_list = []
    for s in sectors:
        try:
            series_list.append(read_sector_regional_series(xlsx_path, s, suffix, provinces))
        except Exception as e:
            print(f'警告：读取 {xlsx_path} -> {s}_{suffix} 失败：{e}')
    if not series_list:
        raise RuntimeError('没有任何行业数据可用于绘图。')
    df = pd.concat(series_list, axis=1).fillna(0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

def plot_structure(df: pd.DataFrame, title: str, colors: list, ylabel: str, out_png: str = None):
    order = df.mean().sort_values(ascending=True).index.tolist()
    df_plot = df[order]

    x = df_plot.index.values
    y = df_plot.T.values

    if len(colors) < df_plot.shape[1]:
        repeats = int(np.ceil(df_plot.shape[1] / len(colors)))
        colors = (colors * repeats)[:df_plot.shape[1]]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.stackplot(x, y, labels=df_plot.columns.tolist(), colors=colors, alpha=0.95, linewidth=0.2)

    ax.set_xlim(min(x), max(x))
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    leg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    for t in leg.get_texts():
        t.set_fontsize(9)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
        print(f'已保存图片：{out_png}')
    plt.show()
    plt.close()

# ---------- 钢铁 ----------
if os.path.exists(steel_file):
    steel_df = build_stacked_df(
        xlsx_path=steel_file,
        sectors=steel_stream,
        suffix='用钢中间结果',
        provinces=provlist_bthsa
    )
    plot_structure(
        df=steel_df,
        title='京津冀及周边8省市钢铁消费结构',
        colors=steel_colors,
        ylabel='Steel demand (million tons)',
        out_png=os.path.join(base_dir, '京津冀及周边8省市_钢铁消费结构.png') if save_png else None
    )

# ---------- 水泥 ----------
if os.path.exists(cement_file):
    cement_df = build_stacked_df(
        xlsx_path=cement_file,
        sectors=cement_stream,
        suffix='水泥中间结果',
        provinces=provlist_bthsa
    )
    plot_structure(
        df=cement_df,
        title='京津冀及周边8省市水泥消费结构',
        colors=cement_colors,
        ylabel='Cement demand (million tons)',
        out_png=os.path.join(base_dir, '京津冀及周边8省市_水泥消费结构.png') if save_png else None
    )
