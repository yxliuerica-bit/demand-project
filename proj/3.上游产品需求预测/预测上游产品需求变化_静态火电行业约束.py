# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# ======================== 基础配置 ========================

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False

provinces = [
    '北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑_龙江省',
    '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省',
    '湖南省', '广东省', '广西壮族自治区', '海南省', '重庆市', '四川省', '贵州省', '云南省',
    '西藏自治区', '陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区'
]
provlist_bthsa = ['北京市','天津市','河北省','河南省','山东省','山西省','内蒙古自治区','陕西省']

steel_stream = ['新增建筑面积','汽车产量','新增铁路里程','洗衣机产量','电冰箱产量','空调产量']
cement_stream = ['新增建筑面积','新增高速公路里程','新增其他等级公路里程','新增铁路里程','农村水电基础设施投资']
power_stream = ['火力发电量']

sce = '高增长'
path_down_stream = f'/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_{sce}/全国完整数据/'

do_steel, do_cement, do_power = 1, 1, 1
constraint_steel, constraint_cement, constraint_power = 1, 1, 1
plot_predict = 1

# ======================== 通用辅助函数 ========================

def load_reformed_data(path, factor):
    file_path = os.path.join(path, f'分省{factor}数据(2005-2035).xlsx')
    if not os.path.exists(file_path): return None
    try:
        df = pd.read_excel(file_path, index_col=0)
        df = df.loc[2010:2035].copy().fillna(0)
        df[df < 0] = 0
        return df.reindex(columns=provinces, fill_value=0)
    except: return None

def load_national_series_from_wide_excel(path, sheet_name):
    """读取旧版分省历史产量宽表 (火电使用)"""
    df0 = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_row = 0
    for i in range(min(len(df0), 50)):
        if str(df0.iat[i, 0]).strip() == '地区':
            header_row = i
            break
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    year_cols = [c for c in df.columns if re.fullmatch(r'\d{4}年', str(c))]
    df_years = df[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    nat_ton = df_years.sum(axis=0) * 10000.0
    nat_ton.index = nat_ton.index.map(lambda x: int(str(x).replace('年', '')))
    return nat_ton.sort_index()

def load_national_consumption_recalculated(file_path):
    """读取新版核算消费量长表 (钢铁、水泥使用)"""
    if not os.path.exists(file_path):
        print(f"❌ 找不到核算文件: {file_path}")
        return None
    try:
        df = pd.read_excel(file_path)
        df.columns = [str(c).strip() for c in df.columns]
        # 提取年份和消费量 (万吨 -> 吨)
        target_col = "历史消费量(万吨)"
        if target_col not in df.columns:
            target_col = [c for c in df.columns if "历史消费量" in c][0]
        series = df.set_index("年份")[target_col] * 10000.0
        return series.sort_index()
    except Exception as e:
        print(f"读取核算文件失败: {e}")
        return None

def get_intensity_series(product, years_index, base_intensity, industry_type):
    yrs = pd.Index(range(2010, 2036))
    factors = pd.Series(1.0, index=yrs, dtype=float)
    def linear_segment(y0, f0, y1, f1):
        seg_yrs = np.arange(y0, y1 + 1)
        return dict(zip(seg_yrs, np.linspace(f0, f1, len(seg_yrs))))
    mapping = {}
    if industry_type == 'steel':
        if product == '汽车产量':
            mapping.update(linear_segment(2010, 1.0, 2020, 0.8))
            mapping.update(linear_segment(2020, 0.8, 2030, 0.72))
        elif product in ['洗衣机产量', '电冰箱产量', '空调产量']:
            mapping.update(linear_segment(2010, 1.0, 2020, 0.9))
            mapping.update(linear_segment(2020, 0.9, 2030, 0.81))
    elif industry_type == 'power':
        mapping.update(linear_segment(2010, 1.0, 2020, 0.95))
        mapping.update(linear_segment(2020, 0.95, 2030, 0.9025))
    if mapping:
        for y in yrs:
            if y in mapping: factors.loc[y] = mapping[y]
            elif y < min(mapping.keys()): factors.loc[y] = list(mapping.values())[0]
            else: factors.loc[y] = list(mapping.values())[-1]
    return factors.reindex(years_index) * float(base_intensity)

# ======================== 行业计算逻辑 ========================

base_data = {f: load_reformed_data(path_down_stream, f) for f in set(steel_stream + cement_stream + power_stream)}

def calculate_industry_demand(industry_name, stream_list, intensity_file, intensity_col, industry_type):
    print(f"执行 {industry_name} 计算...")
    years = pd.Index(range(2010, 2036), name='year')
    result_df = pd.DataFrame(0.0, index=years, columns=provinces)
    try:
        df_intensity = pd.read_excel(intensity_file)
        name_col = next(c for c in ['标识', '指标', '项目'] if c in df_intensity.columns)
        df_intensity = df_intensity.rename(columns={name_col: 'ID'})
        with pd.ExcelWriter(path_down_stream + f'下游行业{industry_name}需求预测.xlsx') as writer:
            for prod in stream_list:
                if prod in base_data and base_data[prod] is not None:
                    row = df_intensity[df_intensity['ID'] == prod]
                    if not row.empty:
                        base_val = row[intensity_col].values[0]
                        unit = row['unit'].values[0]
                        intensity_ser = get_intensity_series(prod, years, base_val, industry_type)
                        df_prod = base_data[prod].mul(intensity_ser, axis=0) * unit
                        result_df = result_df.add(df_prod, fill_value=0)
                        df_prod.to_excel(writer, sheet_name=f'{prod}_中间结果')
            result_df['全国'] = result_df[provinces].sum(axis=1)
            result_df['京津冀及周边8省市'] = result_df[provlist_bthsa].sum(axis=1)
            result_df.to_excel(writer, sheet_name=f'下游行业{industry_name}需求')
            return result_df
    except Exception as e:
        print(f"计算失败: {e}"); return None

if do_steel: calculate_industry_demand('用钢', steel_stream, '/lustre/home/litong/intern/京津冀货运发展/data/单位用料强度/单位用钢强度.xlsx', '用钢强度', 'steel')
if do_cement: calculate_industry_demand('水泥', cement_stream, '/lustre/home/litong/intern/京津冀货运发展/data/单位用料强度/单位水泥强度.xlsx', '水泥强度', 'cement')
if do_power: calculate_industry_demand('火电', power_stream, '/lustre/home/litong/intern/京津冀货运发展/data/单位用料强度/单位发电煤耗.xlsx', '发电煤耗', 'power')

# ======================== 约束处理逻辑 ========================

def apply_industry_constraint_v2(industry_label, forecast_file, history_file, history_sheet, output_file, use_recalculated=False):
    try:
        df_pred = pd.read_excel(forecast_file, sheet_name=f'下游行业{industry_label}需求', index_col=0).fillna(0.0)
        
        # 区分读取逻辑
        if use_recalculated:
            real_nat = load_national_consumption_recalculated(history_file)
            limit_year = 2025 # 包含 2025
        else:
            real_nat = load_national_series_from_wide_excel(history_file, history_sheet)
            limit_year = 2024 # 包含 2024

        years_all = pd.Index(range(2010, 2036), name='year')
        df_pred = df_pred.reindex(index=years_all, fill_value=0.0)
        pred_nat = df_pred['全国']
        scale_factors = {}
        last_scale = 1.0
        
        # 历史约束
        for y in range(2010, limit_year + 1):
            if y in real_nat.index and pred_nat.get(y, 0) > 0:
                last_scale = np.clip(real_nat.loc[y] / pred_nat.get(y), 0.1, 10.0)
            scale_factors[y] = last_scale
        
        # 未来年份延续
        for y in range(limit_year + 1, 2036):
            scale_factors[y] = last_scale

        if industry_label == '火电':
            target_ratio = scale_factors.get(2019, last_scale)
            for y in years_all: scale_factors[y] = target_ratio

        df_out = df_pred.copy()
        for y, s in scale_factors.items():
            if y in df_out.index: df_out.loc[y, provinces] *= s
            
        df_out['全国'] = df_out[provinces].sum(axis=1)
        df_out['京津冀及周边8省市'] = df_out[provlist_bthsa].sum(axis=1)
        df_out.to_excel(path_down_stream + output_file)
        print(f"任务 {industry_label} 约束完成。")
    except Exception as e: print(f"约束失败: {e}")

path_recalc = "/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/历史消费需求核算结果/"

if constraint_steel:
    apply_industry_constraint_v2('用钢', path_down_stream+'下游行业用钢需求预测.xlsx', 
                                 os.path.join(path_recalc, "历史粗钢表观消费量核算结果.xlsx"), 
                                 None, '分省钢铁消费量预测.xlsx', use_recalculated=True)

if constraint_cement:
    apply_industry_constraint_v2('水泥', path_down_stream+'下游行业水泥需求预测.xlsx', 
                                 os.path.join(path_recalc, "历史水泥表观消费量核算结果.xlsx"), 
                                 None, '分省水泥消费量预测.xlsx', use_recalculated=True)

if constraint_power:
    apply_industry_constraint_v2('火电', path_down_stream+'下游行业火电需求预测.xlsx', 
                                 '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史煤炭消费量年度数据.xlsx', 
                                 '分省历史煤炭消费量年度数据', '分省火电消费量预测.xlsx', use_recalculated=False)

# ======================== 绘图逻辑 ========================

if plot_predict:
    # 定义数值精度
    precision_map = {'钢铁': '.1f', '水泥': '.1f', '火电': '.3f'}

    for prod_name, file_key, unit_label in [('钢铁', '钢铁', '亿吨'), ('水泥', '水泥', '亿吨'), ('火电', '火电', '亿吨煤')]:
        file_path = path_down_stream + f'分省{file_key}消费量预测.xlsx'
        if not os.path.exists(file_path): continue
        
        # --- 新增逻辑：根据行业确定红色标注的特殊年份 ---
        special_year = 2024 if prod_name == '火电' else 2025
        
        try:
            fig, ax1 = plt.subplots(1, 1, figsize=(7, 3.5), dpi=300)
            df = pd.read_excel(file_path, index_col=0)
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0) / 1e8
            years_arr = np.arange(2010, 2036, 1)
            df_subset = df.loc[years_arr].fillna(0)
            df_provinces = df_subset[provlist_bthsa]
            
            colors = ['#DEECF6','#AFC8E2','#E2F2CD','#B6DAA7','#F9D5D5','#EF98A1','#FBE3C0','#FBC99A']
            bottom = np.zeros(len(years_arr))
            for i, province in enumerate(provlist_bthsa):
                province_data = df_provinces[province].clip(lower=0)
                ax1.bar(years_arr, province_data, bottom=bottom, color=colors[i], width=0.7, alpha=0.9, label=province)
                bottom += province_data
            
            ax1.set_ylabel(f'分省{prod_name}生产需求 ({unit_label})')
            ax1.grid(axis='y', linestyle='--', alpha=0.5)
            
            ax2 = ax1.twinx()
            national_data = df_subset['全国'].clip(lower=0)
            line_plot, = ax2.plot(years_arr, national_data, color='#d62728', marker='o', markersize=4, linewidth=1.2, label='全国生产需求', zorder=10)
            ax2.set_ylabel(f'全国{prod_name}生产需求 ({unit_label})')
            
            ax2.set_ylim(0, national_data.max() * 1.25)
            
            # --- 核心修改部分 ---
            fmt = precision_map.get(prod_name, '.1f')
            for i, yr in enumerate(years_arr):
                val = national_data.iloc[i]
                if val > 0:
                    # 显示条件：是5的倍数 或者是 该行业的特殊年份
                    if (yr % 5 == 0) or (yr == special_year):
                        # 判断是否为当前行业的特殊年份，以应用红色加粗突出
                        is_special = (yr == special_year)
                        ax2.text(yr, val + (national_data.max() * 0.03), 
                                 f'{val:{fmt}}', 
                                 ha='center', va='bottom', 
                                 fontsize=7, 
                                 fontweight='bold' if is_special else 'normal',
                                 color='#d62728' if is_special else 'black')
            
            ax1.set_xticks(np.arange(2010, 2036, 5))
            ax2.grid(visible=False)
            handles_bar, labels_bar = ax1.get_legend_handles_labels()
            ax1.legend(handles_bar + [line_plot], labels_bar + ['全国'], loc='upper left', bbox_to_anchor=(1.1, 1), fontsize=8)
            
            plt.tight_layout()
            save_name = f'京津冀及周边8省份{prod_name}消费量_优化版_{sce}.png'
            plt.savefig(path_down_stream + save_name, bbox_inches='tight')
            plt.close()
            print(f"已生成图表: {save_name}")
        except Exception as e:
            print(f"绘图失败 ({prod_name}): {e}")

print("绘图任务执行完毕。")