# -*- coding: utf-8 -*-
"""
最终稳健优化版：
1. 修复：彻底解决 'float object has no attribute log1p' 的类型报错。
2. 逻辑：末年锚定因子 + 城镇化饱和衰减修正。
3. 导出：保存指定路径的完整 CSV。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# ------------------- 调试配置 -------------------
DEBUG_MODE = True 
# DEBUG_PROVINCES = ['北京市', '山东省', '天津市', '河北省', '上海市'] 
DEBUG_PROVINCES = ['山东省'] 

# ------------------- Matplotlib 字体 -------------------
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# ------------------- 常量 -------------------
HISTORY_START_YEAR = 2010
HISTORY_END_YEAR = 2024
FUTURE_YEARS = list(range(2025, 2036))
X_COLS = ['GDP', '人口', '第三产业GDP', '城镇人口']
scenario_name = "基准"
SMOOTH_WINDOW = 3

# ======================================================
# 1) 数据加载
# ======================================================

def load_and_preprocess_data(fi_predict_factor, factor_name):
    def read_gbk(p):
        df = pd.read_csv(p, encoding='gbk')
        if 'Unnamed' in df.columns[-1]: df = df.iloc[:, :-1]
        return df.melt(id_vars='地区', var_name='年份', value_name='val')

    base_p = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/'
    
    gdp = read_gbk(base_p + '分省历史GDP数据.csv').rename(columns={'val': 'GDP'})
    pop = read_gbk(base_p + '分省历史人口数据.csv').rename(columns={'val': '人口'})
    third = pd.read_csv(base_p + '分省历史第三产业GDP数据.csv', encoding='gbk').melt(id_vars='地区', var_name='年份', value_name='第三产业GDP')
    urban = pd.read_csv(base_p + '分省历史城镇人口数据.csv', encoding='gbk').melt(id_vars='地区', var_name='年份', value_name='城镇人口')
    
    prod_pre = pd.read_excel(fi_predict_factor, sheet_name='分省年度数据', skiprows=3).melt(id_vars='地区', var_name='年份', value_name=factor_name)

    for df in [gdp, pop, third, urban, prod_pre]:
        df['年份'] = df['年份'].astype(str).str.replace('年', '')
        df['年份'] = pd.to_numeric(df['年份'], errors='coerce')
    
    merged = gdp.merge(pop, on=['地区', '年份']).merge(third, on=['地区', '年份']).merge(urban, on=['地区', '年份']).merge(prod_pre, on=['地区', '年份'])
    merged = merged.dropna(subset=['年份'])
    merged['年份'] = merged['年份'].astype(int)
    merged = merged[(merged['年份'] >= HISTORY_START_YEAR) & (merged['年份'] <= HISTORY_END_YEAR)]
    
    if DEBUG_MODE and DEBUG_PROVINCES:
        merged = merged[merged['地区'].isin(DEBUG_PROVINCES)]
    return merged.sort_values(['地区', '年份']).reset_index(drop=True)

def load_future_province_data():
    fipath_future = '/lustre/home/litong/intern/京津冀货运发展/output/scenarios'
    def to_long(f, val_name):
        df = pd.read_csv(f).rename(columns={'Unnamed: 0': '年份'})
        return df.melt(id_vars='年份', var_name='地区', value_name=val_name)

    gdp_f = to_long(f"{fipath_future}/gdp_predictions_{scenario_name}_完整年份.csv", 'GDP')
    pop_f = to_long(f"{fipath_future}/population_predictions_{scenario_name}_完整年份.csv", '人口')
    urban_f = to_long(f"{fipath_future}/urban_population_predictions_{scenario_name}_完整年份.csv", '城镇人口')
    third_f = to_long(f"{fipath_future}/third_gdp_predictions_{scenario_name}_完整年份.csv", '第三产业GDP')

    future_data = gdp_f.merge(pop_f, on=['地区', '年份']).merge(urban_f, on=['地区', '年份']).merge(third_f, on=['地区', '年份'])
    return future_data[future_data['年份'].isin(FUTURE_YEARS)].sort_values(['地区', '年份'])

# ======================================================
# 2) 核心模型：解决类型报错 + 饱和对齐
# ======================================================

def stageB_predict_with_smooth_saturation(smoothed_hist, future_data, factor_name):
    provinces = sorted(smoothed_hist['地区'].unique())
    trained_models = {}
    
    for prov in provinces:
        df = smoothed_hist[smoothed_hist['地区'] == prov].copy()
        y_col = f'历史_平滑_{factor_name}'
        
        # --- 强力类型检查与清洗 ---
        for col in X_COLS + [y_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        train_df = df.dropna(subset=X_COLS + [y_col])
        # 确保 log1p 接收的是正数且类型为 float
        train_df = train_df[(train_df[X_COLS] > 0).all(axis=1) & (train_df[y_col] > 0)]
        
        if len(train_df) < 3: continue
            
        # 使用 .values.astype(float) 确保 NumPy 兼容性
        X_train_vals = train_df[X_COLS].values.astype(float)
        y_train_vals = train_df[y_col].values.astype(float)
        
        X_train_log = np.log1p(X_train_vals) 
        y_train_log = np.log1p(y_train_vals)
        
        alpha_val = 10.0 if prov in ['北京市', '天津市', '上海市'] else 1.0
        model = Ridge(alpha=alpha_val).fit(X_train_log, y_train_log)
        trained_models[prov] = {'model': model}

    preds_list = []
    for prov in provinces:
        if prov not in trained_models: continue
        fut = future_data[future_data['地区'] == prov].copy()
        if fut.empty: continue
            
        # 预测特征也要强制转 float
        for col in X_COLS: fut[col] = pd.to_numeric(fut[col], errors='coerce')
        fut = fut.dropna(subset=X_COLS)
        X_fut_vals = fut[X_COLS].values.astype(float)
        
        y_pred = np.expm1(trained_models[prov]['model'].predict(np.log1p(X_fut_vals)))
        
        # 饱和度衰减
        urban_rate = pd.to_numeric(fut['城镇人口'], errors='coerce') / pd.to_numeric(fut['人口'], errors='coerce')
        threshold, saturation_max = 0.80, 0.98 
        for i in range(len(y_pred)):
            u = urban_rate.iloc[i]
            if u > threshold:
                decay = np.clip((saturation_max - u) / (saturation_max - threshold), 0.6, 1.0)
                y_pred[i] *= (decay ** 0.3)
        
        # 历史末年对齐
        last_hist_series = smoothed_hist[smoothed_hist['地区'] == prov][f'历史_平滑_{factor_name}']
        if not last_hist_series.empty:
            last_hist_val = last_hist_series.iloc[-1]
            scale_factor = last_hist_val / y_pred[0] if y_pred[0] != 0 else 1.0
            y_pred = y_pred * scale_factor * 0.98 
        
        fut[f'预测_平滑优先_{factor_name}'] = y_pred
        fut[f'预测_原始数据_{factor_name}'] = y_pred
        preds_list.append(fut)

    return pd.concat(preds_list) if preds_list else pd.DataFrame()

# ======================================================
# 3) 执行与导出
# ======================================================

def smooth_history(df, factor_name):
    out = df.copy().sort_values(['地区', '年份'])
    # 确保数值列是 float 再进行滚动
    out[factor_name] = pd.to_numeric(out[factor_name], errors='coerce')
    out[f'历史_平滑_{factor_name}'] = out.groupby('地区')[factor_name].transform(
        lambda s: s.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
    ).clip(lower=0)
    return out

def plot_debug(hist, pred, factor_name, region, fopath):
    plt.figure(figsize=(12, 6))
    plt.plot(hist['年份'], hist[factor_name], 'o-', label='历史原始')
    plt.plot(hist['年份'], hist[f'历史_平滑_{factor_name}'], '--', label='历史平滑')
    if not pred.empty:
        plt.plot(pred['年份'], pred[f'预测_平滑优先_{factor_name}'], 's-.', color='green', label='锚定修正预测')
    plt.title(f"{region} - {factor_name} 修正调试", fontsize=14)
    plt.axvline(x=2024.5, color='gray', linestyle=':')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_dir = os.path.join(fopath, '省份指标趋势图')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{region}_{factor_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    factor_mapping = {'新增建筑面积': '建筑业房屋竣工面积',
                        '汽车产量': '汽车产量',
                        '洗衣机产量': '家用洗衣机产量',
                        '电冰箱产量': '家用电冰箱产量',
                        '空调产量': '房间空调器',
                        '农村水电基础设施投资': '农村水电建设本年完成投资额',
                        '新增铁路里程': '新增铁路营业里程',
                        '新增高速公路里程': '新增高速等级公路里程',
                        '新增其他等级公路里程': '新增除高速外等级公路里程',
                        '火力发电量': '火力发电量'} 
    base_fopath = f'/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_{scenario_name}'
    csv_save_dir = os.path.join(base_fopath, '省份完整数据')
    os.makedirs(csv_save_dir, exist_ok=True)

    for factor_name, input_name in factor_mapping.items():
        print(f">>> 正在处理: {factor_name}")
        fi = f"/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/{input_name}分省年度数据.xls"
        
        hist_data = load_and_preprocess_data(fi, factor_name)
        future_feat = load_future_province_data()
        hist_smoothed = smooth_history(hist_data, factor_name)
        
        predictions = stageB_predict_with_smooth_saturation(hist_smoothed, future_feat, factor_name)
        
        for prov in hist_smoothed['地区'].unique():
            p_hist = hist_smoothed[hist_smoothed['地区'] == prov].copy()
            p_pred = predictions[predictions['地区'] == prov].copy() if not predictions.empty else pd.DataFrame()

            # 合并导出
            combined = pd.concat([p_hist, p_pred], ignore_index=True)
            csv_path = os.path.join(csv_save_dir, f"{prov}{factor_name}完整数据.csv")
            combined.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # 绘图
            plot_debug(p_hist, p_pred, factor_name, prov, base_fopath)
            
    print("\n任务完成！")