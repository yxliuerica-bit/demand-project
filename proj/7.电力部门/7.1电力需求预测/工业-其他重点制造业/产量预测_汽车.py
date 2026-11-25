# -*- coding: utf-8 -*-
"""
制造业产量预测-宽表合并版
1. 逻辑：真值优先 + 比例放缩锚定
2. 导出：每个产品生成一个宽表CSV，包含全国和区域合计
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# ------------------- 配置 -------------------
HISTORY_START_YEAR = 2010
HISTORY_END_YEAR = 2024
FUTURE_YEARS = list(range(2025, 2036))
X_COLS = ['GDP', '人口', '第三产业GDP', '城镇人口']
SCENARIO_NAME = "基准"

# 区域定义
JJJ_REGION = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '山东省', '河南省']

# ------------------- 路径配置 -------------------
BASE_PATH = '/lustre/home/litong/intern/京津冀货运发展'
FACTOR_DATA_DIR = f'{BASE_PATH}/data/预测原始数据/电力部门相关数据/其他重点制造业产量数据'
MACRO_DATA_DIR = f'{BASE_PATH}/data/预测原始数据'
FUTURE_SCENARIO_DIR = f'{BASE_PATH}/output/scenarios'
OUTPUT_BASE = f'{BASE_PATH}/output/7.电力部门/7.1电力需求预测/工业-其他重点制造业'

# ======================================================
# 工具函数
# ======================================================

def clean_and_melt(df, val_name):
    df.columns = [str(c).replace('\t', '').replace(' ', '').strip() for c in df.columns]
    melted = df.melt(id_vars='地区', var_name='年份', value_name=val_name)
    melted['年份'] = melted['年份'].astype(str).str.replace('年', '', regex=False).str.replace('\t', '', regex=False).str.strip()
    melted['年份'] = pd.to_numeric(melted['年份'], errors='coerce')
    
    if melted[val_name].dtype == object:
        melted[val_name] = melted[val_name].astype(str).str.replace('\t', '', regex=False).str.replace(',', '', regex=False).str.strip()
    melted[val_name] = pd.to_numeric(melted[val_name], errors='coerce')
    melted['地区'] = melted['地区'].astype(str).str.replace('\t', '').str.strip()
    return melted.dropna(subset=['年份'])

def read_flexible_csv(p, skip=0):
    try:
        return pd.read_csv(p, encoding='gb18030', skiprows=skip, sep=None, engine='python')
    except Exception:
        return pd.read_csv(p, encoding='utf-8', skiprows=skip, sep=None, engine='python')

# ======================================================
# 数据加载逻辑
# ======================================================

def load_macro_data():
    gdp = clean_and_melt(read_flexible_csv(f'{MACRO_DATA_DIR}/分省历史GDP数据.csv'), 'GDP')
    pop = clean_and_melt(read_flexible_csv(f'{MACRO_DATA_DIR}/分省历史人口数据.csv'), '人口')
    third = clean_and_melt(read_flexible_csv(f'{MACRO_DATA_DIR}/分省历史第三产业GDP数据.csv'), '第三产业GDP')
    urban = clean_and_melt(read_flexible_csv(f'{MACRO_DATA_DIR}/分省历史城镇人口数据.csv'), '城镇人口')
    merged = gdp.merge(pop, on=['地区', '年份']).merge(third, on=['地区', '年份']).merge(urban, on=['地区', '年份'])
    return merged[(merged['年份'] >= HISTORY_START_YEAR) & (merged['年份'] <= HISTORY_END_YEAR)]

def load_future_macro():
    def to_long(f, val_name):
        df = pd.read_csv(f).rename(columns={'Unnamed: 0': '年份'})
        return df.melt(id_vars='年份', var_name='地区', value_name=val_name)
    gdp_f = to_long(f"{FUTURE_SCENARIO_DIR}/gdp_predictions_{SCENARIO_NAME}_完整年份.csv", 'GDP')
    pop_f = to_long(f"{FUTURE_SCENARIO_DIR}/population_predictions_{SCENARIO_NAME}_完整年份.csv", '人口')
    urban_f = to_long(f"{FUTURE_SCENARIO_DIR}/urban_population_predictions_{SCENARIO_NAME}_完整年份.csv", '城镇人口')
    third_f = to_long(f"{FUTURE_SCENARIO_DIR}/third_gdp_predictions_{SCENARIO_NAME}_完整年份.csv", '第三产业GDP')
    future = gdp_f.merge(pop_f, on=['地区', '年份']).merge(urban_f, on=['地区', '年份']).merge(third_f, on=['地区', '年份'])
    return future[future['年份'].isin(FUTURE_YEARS)]

# ======================================================
# 核心预测引擎
# ======================================================

def predict_single_factor(hist_macro, future_macro, factor_file, factor_name):
    raw_df = read_flexible_csv(factor_file, skip=2)
    hist_target = clean_and_melt(raw_df, 'actual_val')
    combined_hist = hist_macro.merge(hist_target, on=['地区', '年份'], how='left')
    
    provinces = combined_hist['地区'].unique()
    all_prov_data = []

    for prov in provinces:
        p_data = combined_hist[combined_hist['地区'] == prov].copy().sort_values('年份')
        f_data = future_macro[future_macro['地区'] == prov].copy().sort_values('年份')
        
        train_df = p_data.dropna(subset=X_COLS + ['actual_val'])
        train_df = train_df[(train_df[X_COLS] > 0).all(axis=1) & (train_df['actual_val'] > 0)]
        
        if len(train_df) < 3: continue
        
        model = Ridge(alpha=1.0).fit(np.log1p(train_df[X_COLS]), np.log1p(train_df['actual_val']))
        all_X = pd.concat([p_data[X_COLS], f_data[X_COLS]])
        base_preds = np.expm1(model.predict(np.log1p(all_X.values.astype(float))))
        
        # 饱和修正
        urban_rate = f_data['城镇人口'].values / f_data['人口'].values
        for i, u in enumerate(urban_rate):
            if u > 0.80:
                decay = np.clip((0.98 - u) / 0.18, 0.6, 1.0)
                base_preds[len(p_data) + i] *= (decay ** 0.3)

        # 系数锚定
        p_data['pred_raw'] = base_preds[:len(p_data)]
        p_data['ratio'] = p_data['actual_val'] / p_data['pred_raw']
        valid_ratios = p_data.dropna(subset=['ratio'])
        latest_ratio = valid_ratios.iloc[-1]['ratio'] if not valid_ratios.empty else 1.0

        # 填补与预测
        p_data[factor_name] = [row['actual_val'] if not np.isnan(row['actual_val']) else row['pred_raw'] * latest_ratio for _, row in p_data.iterrows()]
        f_data[factor_name] = base_preds[len(p_data):] * latest_ratio
        
        res = pd.concat([p_data[['年份', '地区', factor_name]], f_data[['年份', '地区', factor_name]]])
        all_prov_data.append(res)

    # 合并为宽表
    df_long = pd.concat(all_prov_data)
    df_wide = df_long.pivot(index='年份', columns='地区', values=factor_name).reset_index()
    df_wide.columns.name = None
    
    # 计算合计列
    prov_cols = [c for c in df_wide.columns if c != '年份']
    df_wide['全国'] = df_wide[prov_cols].sum(axis=1)
    
    jjj_actual = [c for c in JJJ_REGION if c in df_wide.columns]
    df_wide['京津冀及周边8省市'] = df_wide[jjj_actual].sum(axis=1)
    
    # 按照要求的顺序重排列表格（如果省份不存在填0）
    target_cols = ['年份', '北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑龙江省', '上海市', 
                   '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省', '湖南省', '广东省', 
                   '广西壮族自治区', '海南省', '重庆市', '四川省', '贵州省', '云南省', '西藏自治区', '陕西省', '甘肃省', 
                   '青海省', '宁夏回族自治区', '新疆维吾尔自治区', '全国', '京津冀及周边8省市']
    
    for col in target_cols:
        if col not in df_wide.columns:
            df_wide[col] = 0
            
    return df_wide[target_cols].rename(columns={'年份': 'Year'})

# ======================================================
# 执行
# ======================================================

if __name__ == '__main__':
    indicators = {
        # '移动通信手持机产量': '移动通信手持机产量分省年度数据.csv',
        # '微型电子计算机产量': '微型电子计算机产量分省年度数据.csv',
        # '集成电路产量': '集成电路产量分省年度数据.csv',
        # '布产量': '布产量分省年度数据.csv',
        # '塑料产量': '塑料产量分省年度数据.csv',
        '汽车产量': '汽车产量分省年度数据.csv'
    }

    hist_macro = load_macro_data()
    future_macro = load_future_macro()
    save_dir = OUTPUT_BASE
    os.makedirs(save_dir, exist_ok=True)

    for factor_name, file_name in indicators.items():
        print(f">>> 正在生成宽表: {factor_name}")
        file_path = os.path.join(FACTOR_DATA_DIR, file_name)
        if os.path.exists(file_path):
            result_wide = predict_single_factor(hist_macro, future_macro, file_path, factor_name)
            save_path = os.path.join(save_dir, f"{factor_name}预测结果.csv")
            result_wide.to_csv(save_path, index=False, encoding='utf-8-sig')
            
    print("\n[任务完成] 所有产品宽表已保存至 '指标汇总预测结果' 文件夹。")