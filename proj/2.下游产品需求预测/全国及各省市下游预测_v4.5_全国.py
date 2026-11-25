# -*- coding: utf-8 -*-
"""
最终完整生产版 (v4.6_饱和度修正)
1. 修复：彻底解决 Excel 底部“备注”导致 int() 转换失败的问题。
2. 模型：对数岭回归 (Log-Log Ridge) + 城镇化饱和度修正。
3. 结构：支持阶段A回填、阶段B跨省映射、全国/区域汇总导出。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression

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

# 阶段B“跨省模型引用”映射 (根据需求开启)
CROSS_PROV_MODEL = {
    # '北京市': '天津市',
    # '山东省': '河北省',
}

# ======================================================
# 1) 强力数据加载与清洗 (解决“备注”报错的关键)
# ======================================================

def load_and_preprocess_data(fi_predict_factor, factor_name):
    def read_and_melt(p, val_col):
        df = pd.read_csv(p, encoding='gbk')
        if 'Unnamed' in df.columns[-1]: df = df.iloc[:, :-1]
        df = df.melt(id_vars='地区', var_name='年份', value_name=val_col)
        # 清洗：去掉“年”字，强转数字，非法字符变NaN并剔除
        df['年份'] = df['年份'].astype(str).str.replace('年', '')
        df['年份'] = pd.to_numeric(df['年份'], errors='coerce')
        df = df.dropna(subset=['年份'])
        df['年份'] = df['年份'].astype(int)
        # 剔除地区列中的“备注”行
        df = df[~df['地区'].astype(str).str.contains('备注|来源|注：')]
        return df

    base_p = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/'
    
    gdp = read_and_melt(base_p + '分省历史GDP数据.csv', 'GDP')
    pop = read_and_melt(base_p + '分省历史人口数据.csv', '人口')
    third = read_and_melt(base_p + '分省历史第三产业GDP数据.csv', '第三产业GDP')
    urban = read_and_melt(base_p + '分省历史城镇人口数据.csv', '城镇人口')
    
    # 读取 Excel 指标
    prod_pre = pd.read_excel(fi_predict_factor, sheet_name='分省年度数据', skiprows=3)
    prod_pre = prod_pre.melt(id_vars='地区', var_name='年份', value_name=factor_name)
    prod_pre['年份'] = prod_pre['年份'].astype(str).str.replace('年', '')
    prod_pre['年份'] = pd.to_numeric(prod_pre['年份'], errors='coerce')
    prod_pre = prod_pre.dropna(subset=['年份'])
    prod_pre['年份'] = prod_pre['年份'].astype(int)

    # 合并
    merged = gdp.merge(pop, on=['地区', '年份']).merge(third, on=['地区', '年份']).merge(urban, on=['地区', '年份']).merge(prod_pre, on=['地区', '年份'])
    merged = merged[(merged['年份'] >= HISTORY_START_YEAR) & (merged['年份'] <= HISTORY_END_YEAR)].dropna(subset=X_COLS)
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
# 2) 阶段 A：历史回填
# ======================================================

def stageA_fill_history(historical_data, factor_name):
    provinces = sorted(historical_data['地区'].unique())
    filled = historical_data.copy()
    filled[f'模型回填_{factor_name}'] = np.nan
    model_rows = []

    # 全局基准模型（用于极度缺失的省份）
    nat_train = historical_data[historical_data[factor_name].notna()]
    nat_model = LinearRegression().fit(nat_train[X_COLS], nat_train[factor_name]) if len(nat_train) > 5 else None

    for prov in provinces:
        df = historical_data[historical_data['地区'] == prov].copy()
        train_df = df[df[factor_name].notna()]
        
        if len(train_df) >= 3:
            m = LinearRegression().fit(train_df[X_COLS], train_df[factor_name])
            miss = df[factor_name].isna()
            if miss.any():
                filled.loc[(filled['地区'] == prov) & miss, f'模型回填_{factor_name}'] = np.clip(m.predict(df.loc[miss, X_COLS]), 0, None)
            model_rows.append({'层级': '省级', '对象': prov, '训练R²': m.score(train_df[X_COLS], train_df[factor_name])})
        elif nat_model:
            miss = df[factor_name].isna()
            if miss.any():
                filled.loc[(filled['地区'] == prov) & miss, f'模型回填_{factor_name}'] = np.clip(nat_model.predict(df.loc[miss, X_COLS]), 0, None)
            model_rows.append({'层级': '全国', '对象': prov, '说明': '回填样本不足，引用全国模型'})

    filled['历史_融合值'] = filled[factor_name].combine_first(filled[f'模型回填_{factor_name}'])
    return filled, pd.DataFrame(model_rows)

# ======================================================
# 3) 阶段 B：平滑 + 对数岭回归 + 饱和度修正
# ======================================================

def smooth_history(filled_hist, factor_name):
    out = filled_hist.copy().sort_values(['地区', '年份'])
    out[f'历史_平滑_{factor_name}'] = out.groupby('地区')['历史_融合值'].transform(
        lambda s: s.rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
    ).clip(lower=0)
    return out

def stageB_predict_future(smoothed_hist, future_data, factor_name):
    provinces = sorted(smoothed_hist['地区'].unique())
    trained = {}
    
    # 训练阶段
    for prov in provinces:
        df = smoothed_hist[smoothed_hist['地区'] == prov].copy()
        y_col = f'历史_平滑_{factor_name}'
        
        # --- 核心修复：强制类型转换 ---
        # 确保参与计算的所有列都是 float，无法转换的变 NaN
        for col in X_COLS + [y_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 剔除包含 NaN 或 <= 0 的行（log 必须要求正数）
        train_df = df.dropna(subset=X_COLS + [y_col])
        train_df = train_df[(train_df[X_COLS] > 0).all(axis=1) & (train_df[y_col] > 0)]
        
        if len(train_df) < 3:
            print(f"警告: {prov} 有效样本量不足({len(train_df)})，跳过训练。")
            continue
        
        # 显式转换为 float 数组后再传给 numpy
        X_train_vals = train_df[X_COLS].values.astype(float)
        y_train_vals = train_df[y_col].values.astype(float)
        
        X_log = np.log1p(X_train_vals)
        y_log = np.log1p(y_train_vals)
        
        # 针对大城市加强约束
        alpha_val = 15.0 if prov in ['北京市', '上海市', '天津市'] else 1.0
        model = Ridge(alpha=alpha_val).fit(X_log, y_log)
        trained[prov] = {'model': model, 'r2': model.score(X_log, y_log)}

    # 预测阶段
    preds_list = []
    model_summary = []
    
    for prov in provinces:
        donor = CROSS_PROV_MODEL.get(prov, prov)
        m_info = trained.get(donor) if donor in trained else trained.get(prov)
        if not m_info: continue
        
        fut = future_data[future_data['地区'] == prov].copy()
        if fut.empty: continue
        
        # --- 预测时的类型转换 ---
        for col in X_COLS:
            fut[col] = pd.to_numeric(fut[col], errors='coerce')
        fut = fut.dropna(subset=X_COLS)
        
        X_fut_vals = fut[X_COLS].values.astype(float)
        y_pred = np.expm1(m_info['model'].predict(np.log1p(X_fut_vals)))
        
        # 城镇化率饱和度修正
        # 确保城镇人口和总人口也是 float
        u_pop = pd.to_numeric(fut['城镇人口'], errors='coerce')
        t_pop = pd.to_numeric(fut['人口'], errors='coerce')
        u_rate = u_pop / t_pop
        
        for i in range(len(y_pred)):
            if u_rate.iloc[i] > 0.80:
                decay = np.clip((0.96 - u_rate.iloc[i]) / (0.96 - 0.80), 0.1, 1.0)
                y_pred[i] *= (decay ** 0.65)
        
        fut[f'预测_原始数据_{factor_name}'] = y_pred
        fut[f'预测_平滑优先_{factor_name}'] = y_pred
        preds_list.append(fut)
        model_summary.append({'对象': prov, '模型来源': donor, '训练R²': m_info['r2']})

    return (pd.concat(preds_list) if preds_list else pd.DataFrame()), pd.DataFrame(model_summary)

# ======================================================
# 4) 绘图、汇总与保存
# ======================================================

def aggregate_regional_results(prov_preds, hist_smoothed, factor_name):
    jjj_list = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '山东省', '河南省', '辽宁省', '陕西省']
    regions = {'全国': prov_preds['地区'].unique(), '京津冀及周边8省市': jjj_list}
    
    results = {}
    for r_name, r_list in regions.items():
        # 汇总预测
        p_sub = prov_preds[prov_preds['地区'].isin(r_list)]
        p_agg = p_sub.groupby('年份').agg({c: 'sum' for c in X_COLS + [f'预测_原始数据_{factor_name}', f'预测_平滑优先_{factor_name}']}).reset_index()
        p_agg['地区'] = r_name
        # 汇总历史
        h_sub = hist_smoothed[hist_smoothed['地区'].isin(r_list)]
        h_agg = h_sub.groupby('年份').agg({c: 'sum' for c in X_COLS + ['历史_融合值', f'历史_平滑_{factor_name}']}).reset_index()
        h_agg['地区'] = r_name
        results[r_name] = (h_agg, p_agg)
    return results

def save_and_plot(hist, pred, factor_name, region, fopath, is_prov=True):
    plt.figure(figsize=(12, 6))
    plt.plot(hist['年份'], hist['历史_融合值'], 'o-', label='历史融合')
    plt.plot(hist['年份'], hist[f'历史_平滑_{factor_name}'], '--', label='历史平滑')
    if not pred.empty:
        plt.plot(pred['年份'], pred[f'预测_平滑优先_{factor_name}'], 's-.', color='green', label='饱和修正预测')
    plt.axvline(x=2024.5, color='gray', linestyle=':')
    plt.title(f"{region} - {factor_name} 趋势与预测", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 路径
    p_type = '省份' if is_prov else '区域'
    img_dir = os.path.join(fopath, f'{p_type}指标趋势图')
    csv_dir = os.path.join(fopath, f'{p_type}完整数据')
    for d in [img_dir, csv_dir]: os.makedirs(d, exist_ok=True)
    
    plt.savefig(f"{img_dir}/{region}_{factor_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    pd.concat([hist, pred], ignore_index=True).to_csv(f"{csv_dir}/{region}{factor_name}完整数据.csv", index=False)

# ======================================================
# 5) 执行
# ======================================================

if __name__ == '__main__':
    fopath = f'/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_{scenario_name}/'
    os.makedirs(fopath, exist_ok=True)
    
    predict_to_input = {
        '新增建筑面积': '建筑业房屋竣工面积', '汽车产量': '汽车产量',
        '火力发电量': '火力发电量', '空调产量': '房间空调器'
    }

    future_feat = load_future_province_data()

    for factor_name, input_name in predict_to_input.items():
        print(f"\n>>> 处理指标: {factor_name}")
        fi = f"/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/{input_name}分省年度数据.xls"
        
        hist_raw = load_and_preprocess_data(fi, factor_name)
        hist_filled, resA = stageA_fill_history(hist_raw, factor_name)
        hist_smoothed = smooth_history(hist_filled, factor_name)
        prov_preds, resB = stageB_predict_future(hist_smoothed, future_feat, factor_name)
        
        # 保存中间结果
        resA.to_csv(f"{fopath}/各省{factor_name}_阶段A_结果.csv", index=False)
        resB.to_csv(f"{fopath}/各省{factor_name}_阶段B_结果.csv", index=False)

        # 省级保存
        for prov in hist_smoothed['地区'].unique():
            save_and_plot(hist_smoothed[hist_smoothed['地区']==prov], prov_preds[prov_preds['地区']==prov], factor_name, prov, fopath)

        # 区域汇总保存
        regional_data = aggregate_regional_results(prov_preds, hist_smoothed, factor_name)
        for r_name, (h_agg, p_agg) in regional_data.items():
            save_and_plot(h_agg, p_agg, factor_name, r_name, fopath, is_prov=False)

    print("\n所有省份及区域预测完成，报错已修复。")