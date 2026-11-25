# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# ======================== 1. 基础配置 ========================

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False

sce = '基准'
path_down_stream = f'/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_{sce}/全国完整数据/'
path_intensity = '/lustre/home/litong/intern/京津冀货运发展/data/单位用料强度/单位发电煤耗.xlsx'
path_history_coal = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史煤炭消费量年度数据.xlsx'

provinces = [
    '北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑龙江省',
    '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省',
    '湖南省', '广东省', '广西壮族自治区', '海南省', '重庆市', '四川省', '贵州省', '云南省',
    '西藏自治区', '陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区'
]
provlist_bthsa = ['北京市','天津市','河北省','河南省','山东省','山西省','内蒙古自治区','陕西省']

# ======================== 2. 辅助函数 ========================

def load_reformed_data(path, factor):
    file_path = os.path.join(path, f'分省{factor}数据(2005-2035).xlsx')
    if not os.path.exists(file_path): return None
    df = pd.read_excel(file_path, index_col=0)
    df = df.loc[2010:2035].copy().fillna(0)
    df[df < 0] = 0
    return df.reindex(columns=provinces, fill_value=0)

def load_national_series_from_wide_excel(path, sheet_name):
    if not os.path.exists(path): return None
    df0 = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_row = 0
    for i in range(min(len(df0), 50)):
        if str(df0.iat[i, 0]).strip() == '地区':
            header_row = i
            break
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    year_cols = [c for c in df.columns if re.fullmatch(r'\d{4}年', str(c))]
    df_years = df[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    nat_ton = df_years.sum(axis=0) * 10000.0 # 万吨 -> 吨
    nat_ton.index = nat_ton.index.map(lambda x: int(str(x).replace('年', '')))
    return nat_ton.sort_index()

def get_power_intensity_series(base_val_2024, years_index):
    yrs = pd.Index(range(2010, 2036))
    factors = pd.Series(1.0, index=yrs, dtype=float)
    # 2024年系数为1.0，历史效能低(耗煤多)，未来效能高(耗煤少)
    factors.loc[2010:2024] = np.linspace(1.10, 1.0, len(range(2010, 2025)))
    factors.loc[2024:2035] = np.linspace(1.0, 0.90, len(range(2024, 2036)))
    return factors.reindex(years_index) * float(base_val_2024)

# ======================== 3. 计算与绘图 ========================

def run_all():
    print(f"🚀 开始火电任务 (基准: 2024)...")
    years = pd.Index(range(2010, 2036), name='year')
    
    # --- 1. 计算阶段 ---
    df_gen = load_reformed_data(path_down_stream, '火力发电量')
    if df_gen is None: return

    df_int = pd.read_excel(path_intensity)
    target_row = df_int[df_int['标识'].astype(str).str.contains('火力发电量', na=False)]
    base_intensity_2024 = target_row['发电煤耗'].values[0]
    unit_scale = target_row['unit'].values[0]

    intensity_ser = get_power_intensity_series(base_intensity_2024, years)
    df_power_raw = df_gen.mul(intensity_ser, axis=0) * unit_scale
    
    # 历史校准
    real_nat = load_national_series_from_wide_excel(path_history_coal, '分省历史煤炭消费量年度数据')
    if real_nat is not None and 2019 in real_nat.index:
        scale_ratio = real_nat.loc[2019] / df_power_raw.sum(axis=1).at[2019]
        df_power_final = df_power_raw * scale_ratio
        print(f"⚖️ 校准系数: {scale_ratio:.4f}")
    else:
        df_power_final = df_power_raw

    df_power_final['全国'] = df_power_final[provinces].sum(axis=1)
    df_power_final['京津冀及周边8省市'] = df_power_final[provlist_bthsa].sum(axis=1)
    
    # 保存数据
    output_xlsx = os.path.join(path_down_stream, '分省火电消费量预测.xlsx')
    df_power_final.to_excel(output_xlsx)
    print(f"✅ 数据已保存: {output_xlsx}")

    # --- 2. 绘图阶段 ---
    print("🎨 开始生成火电需求图表...")
    try:
        # 单位转换: 吨 -> 亿吨煤
        df_plot = df_power_final.loc[range(2010, 2036)].fillna(0) / 1e8
        
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
        colors = ['#DEECF6','#AFC8E2','#E2F2CD','#B6DAA7','#F9D5D5','#EF98A1','#FBE3C0','#FBC99A']
        
        # 堆叠柱状图
        bottom = np.zeros(len(df_plot))
        for i, prov in enumerate(provlist_bthsa):
            ax1.bar(df_plot.index, df_plot[prov], bottom=bottom, color=colors[i], label=prov, width=0.7)
            bottom += df_plot[prov]
            
        ax1.set_ylabel('分省火电生产需求 (亿吨煤)')
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 全国折线图
        ax2 = ax1.twinx()
        line, = ax2.plot(df_plot.index, df_plot['全国'], color='#d62728', marker='o', markersize=3, label='全国总计')
        ax2.set_ylabel('全国火电生产需求 (亿吨煤)')
        ax2.set_ylim(0, df_plot['全国'].max() * 1.3)
        
        # 标注特定年份 (5倍数年 和 2024基准年)
        for yr in df_plot.index:
            if yr % 5 == 0 or yr == 2024:
                val = df_plot.loc[yr, '全国']
                is_2024 = (yr == 2024)
                ax2.text(yr, val + 0.05, f'{val:.3f}', ha='center', fontsize=7, 
                         fontweight='bold' if is_2024 else 'normal',
                         color='#d62728' if is_2024 else 'black')

        ax1.legend(loc='upper left', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        
        save_png = os.path.join(path_down_stream, f'京津冀及周边8省份火电消费量_{sce}.png')
        plt.savefig(save_png, bbox_inches='tight')
        plt.close()
        print(f"📊 图表已生成: {save_png}")

    except Exception as e:
        print(f"❌ 绘图失败: {e}")

if __name__ == "__main__":
    run_all()