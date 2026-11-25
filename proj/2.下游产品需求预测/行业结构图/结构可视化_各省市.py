# -*- coding: utf-8 -*-
"""
为每个省份绘制：按下游行业分解的钢铁/水泥“需求结构”堆叠面积图
数据来源：各产品“中间结果”工作表（省级列 + 年份行）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 路径与基础配置 ----------
sce = 'Sce_Pop_BAU_GDP_BAU'
base_dir = f'/lustre/home/litong/intern/京津冀货运发展/output/predict_results/{sce}/11_全国建模回填无法建模省份历史数据/分省完整数据/'

steel_file  = os.path.join(base_dir, '下游行业用钢需求预测.xlsx')
cement_file = os.path.join(base_dir, '下游行业水泥需求预测.xlsx')

# 下游行业清单
steel_stream  = ['新增建筑面积','汽车产量','新增铁路里程','洗衣机产量','电冰箱产量','空调产量']
cement_stream = ['新增建筑面积','新增高速公路里程','新增其他等级公路里程','新增铁路里程','农村水电基础设施投资']

# 年份范围与图像输出
years_plot = range(2010, 2036)
out_root   = os.path.join(base_dir, '省份需求结构图')
out_steel  = os.path.join(out_root, '钢铁')
out_cement = os.path.join(out_root, '水泥')
os.makedirs(out_steel, exist_ok=True)
os.makedirs(out_cement, exist_ok=True)

# 画图设置
save_png = True
dpi = 300
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
steel_colors  = ['#bdbdbd','#ef5350','#fbc02d','#7e57c2','#1e88e5','#43a047']
cement_colors = ['#bdbdbd','#1e88e5','#90caf9','#7e57c2','#26a69a']

# 省份列表（也可从任意一个中间结果表推断）
PROVINCES = [
    '北京市','天津市','河北省','山西省','内蒙古自治区','辽宁省','吉林省','黑龙江省',
    '上海市','江苏省','浙江省','安徽省','福建省','江西省','山东省','河南省','湖北省',
    '湖南省','广东省','广西壮族自治区','海南省','重庆市','四川省','贵州省','云南省',
    '西藏自治区','陕西省','甘肃省','青海省','宁夏回族自治区','新疆维吾尔自治区'
]

# ---------- 工具函数 ----------
# ---------- 工具函数 ----------
def read_sector_sheet_for_province(xlsx_path: str, sector: str, suffix: str, province: str) -> pd.Series:
    """
    读取逻辑保持不变，确保输出原始 Series
    """
    sheet_name = f'{sector}_{suffix}'
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, index_col=0)
    df.index = pd.to_numeric(df.index, errors='coerce')
    df = df.loc[years_plot].copy()

    df = df.select_dtypes(include='number')
    if province not in df.columns:
        s = pd.Series(0.0, index=pd.Index(years_plot, name='年份'), name=sector)
    else:
        s = df[province].copy()
        s.index = pd.Index(years_plot, name='年份')
        s.name = sector
    return s

def build_province_stack_df(xlsx_path: str, sectors: list, suffix: str, province: str) -> pd.DataFrame:
    """
    核心修改点：为某个省份整合数据，并对 0 值进行插值处理
    """
    series_list = []
    for sec in sectors:
        try:
            s = read_sector_sheet_for_province(xlsx_path, sec, suffix, province)
            
            # --- 执行插值逻辑 ---
            # 1. 将 0 值替换为 NaN，以便 interpolate 识别
            s_temp = s.replace(0.0, np.nan)
            
            # 2. 如果序列不全是 NaN，则执行线性插值
            if s_temp.notna().any():
                # limit_direction='both' 确保首尾的 0 也能根据最近的值向外填充
                s_temp = s_temp.interpolate(method='linear', limit_direction='both')
            
            # 3. 将插值后的结果填回序列，若仍有 NaN 则填充为 0
            s = s_temp.fillna(0.0)
            series_list.append(s)
            
        except Exception as e:
            print(f'警告：{os.path.basename(xlsx_path)} -> {sec}_{suffix} 读取/插值失败。原因：{e}')
            series_list.append(pd.Series(0.0, index=pd.Index(years_plot, name='年份'), name=sec))
            
    df = pd.concat(series_list, axis=1).fillna(0.0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # 最终确保没有负值
    df[df < 0] = 0.0
    return df

# ---------- 绘图函数微调（增加边缘平滑处理） ----------
def plot_stack_for_province(df: pd.DataFrame, province: str, title_suffix: str, colors: list, ylabel: str, out_png: str):
    """
    绘图逻辑保持不变，但增加对空 DataFrame 的判断
    """
    if df.empty or (df.sum().sum() == 0):
        print(f'[跳过] {province} 数据全为 0，不生成图片')
        return

    # 行业排序
    order = df.mean().sort_values(ascending=True).index.tolist()
    df_plot = df[order]

    x = df_plot.index.values
    y = df_plot.T.values

    # 自动循环颜色
    if len(colors) < df_plot.shape[1]:
        repeats = int(np.ceil(df_plot.shape[1] / len(colors)))
        colors = (colors * repeats)[:df_plot.shape[1]]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    # 使用 stackplot 绘制，此时 y 已经是插值后的平滑数据
    ax.stackplot(x, y, labels=df_plot.columns.tolist(), colors=colors, alpha=0.95, linewidth=0.1)

    ax.set_xlim(min(x), max(x))
    ax.set_xlabel('年份')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{province}{title_suffix}')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    leg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)

    plt.tight_layout()
    if save_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=dpi, bbox_inches='tight')
        print(f'[OK] 保存插值后的图像：{out_png}')
    plt.close()

# ---------- 主流程：逐省绘图 ----------
# 钢铁
if os.path.exists(steel_file):
    for prov in PROVINCES:
        df_prov = build_province_stack_df(
            xlsx_path=steel_file,
            sectors=steel_stream,
            suffix='用钢中间结果',
            province=prov
        )
        out_png = os.path.join(out_steel, f'{prov}_钢铁需求结构.png')
        plot_stack_for_province(
            df=df_prov,
            province=prov,
            title_suffix='钢铁需求结构（按下游行业）',
            colors=steel_colors,
            ylabel='Steel demand',
            out_png=out_png
        )
else:
    print(f'[跳过] 未找到钢铁文件：{steel_file}')

# 水泥
if os.path.exists(cement_file):
    for prov in PROVINCES:
        df_prov = build_province_stack_df(
            xlsx_path=cement_file,
            sectors=cement_stream,
            suffix='水泥中间结果',
            province=prov
        )
        out_png = os.path.join(out_cement, f'{prov}_水泥需求结构.png')
        plot_stack_for_province(
            df=df_prov,
            province=prov,
            title_suffix='水泥需求结构（按下游行业）',
            colors=cement_colors,
            ylabel='Cement demand',
            out_png=out_png
        )
else:
    print(f'[跳过] 未找到水泥文件：{cement_file}')
