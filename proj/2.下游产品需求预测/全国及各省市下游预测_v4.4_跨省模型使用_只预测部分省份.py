# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ------------------- 1. 核心配置与过滤器 -------------------
TARGET_PROVINCES = ['北京市', '山东省', '陕西省']
JJJ_8_PROVINCES = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '山东省', '河南省', '辽宁省', '陕西省']

CROSS_PROV_MODEL = {
    '北京市': '北京市',
    '陕西省': '京津冀及周边8省市',
    '山东省': '山东省', 
}

scenario_name = "基准"
HISTORY_START_YEAR = 2010
HISTORY_END_YEAR = 2024
FUTURE_YEARS = list(range(2025, 2036))
X_COLS = ['GDP', '人口', '第三产业GDP', '城镇人口']
SMOOTH_WINDOW = 3

# 【新增控制开关】设定是否需要乘以校准系数
USE_CALIBRATION = True 

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# ======================================================
# 2) 核心工具函数 (加载与模型拟合)
# ======================================================

def load_and_preprocess_data(fi_predict_factor, factor_name):
    def read_csv_standard(path, val_name):
        df = pd.read_csv(path, encoding='gbk')
        df = df.iloc[:, :-1].melt(id_vars='地区', var_name='年份', value_name=val_name)
        df['年份'] = df['年份'].str.replace('年', '').astype(int)
        return df

    gdp = read_csv_standard('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史GDP数据.csv', 'GDP')
    pop = read_csv_standard('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史人口数据.csv', '人口')
    third_gdp = read_csv_standard('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史第三产业GDP数据.csv', '第三产业GDP')
    urban_pop = read_csv_standard('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史城镇人口数据.csv', '城镇人口')

    prod_pre = pd.read_excel(fi_predict_factor, sheet_name='分省年度数据', skiprows=3)
    prod_pre = prod_pre.melt(id_vars='地区', var_name='年份', value_name=factor_name)
    prod_pre['年份'] = prod_pre['年份'].str.replace('年', '').astype(int)

    merged = gdp.merge(pop, on=['地区', '年份']).merge(third_gdp, on=['地区', '年份']).merge(urban_pop, on=['地区', '年份']).merge(prod_pre, on=['地区', '年份'])
    required_provinces = set(TARGET_PROVINCES) | set(JJJ_8_PROVINCES) | set(CROSS_PROV_MODEL.values())
    required_provinces = {p for p in required_provinces if p != '京津冀及周边8省市'}
    merged = merged[merged['地区'].isin(required_provinces)].copy()
    merged = merged.dropna(subset=X_COLS)
    merged = merged[(merged['年份'] >= HISTORY_START_YEAR) & (merged['年份'] <= HISTORY_END_YEAR)]
    return merged.sort_values(['地区', '年份']).reset_index(drop=True)

def load_future_province_data(gdp_f, pop_f, urban_f, third_f):
    def to_long(path, val_name):
        df = pd.read_csv(path).rename(columns={'Unnamed: 0': '年份'})
        return df.melt(id_vars='年份', var_name='地区', value_name=val_name)
    future = to_long(gdp_f, 'GDP').merge(to_long(pop_f, '人口'), on=['地区', '年份']).merge(to_long(third_f, '第三产业GDP'), on=['地区', '年份']).merge(to_long(urban_f, '城镇人口'), on=['地区', '年份'])
    future = future[future['地区'].isin(TARGET_PROVINCES)]
    return future[future['年份'].isin(FUTURE_YEARS)].sort_values(['地区', '年份']).reset_index(drop=True)

def fit_model(X, y, max_degree=3):
    X, y = np.asarray(X), np.asarray(y)
    mask = np.isfinite(y); X, y = X[mask], y[mask]
    best_score, best_degree, best_model = -np.inf, 1, None
    max_deg = min(max_degree, len(X) - 1) if len(X) > 1 else 1
    for deg in range(1, max_deg + 1):
        model = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False), LinearRegression())
        cv = min(5, len(X)) if len(X) >= 2 else 1
        score = np.mean(cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
        if score > best_score:
            best_score, best_degree, best_model = score, deg, model
    best_model.fit(X, y)
    return best_model, best_degree

# ======================================================
# 3) 预测与输出核心函数
# ======================================================

def stageA_fill_history(hist_data, factor_name):
    provinces = hist_data['地区'].unique()
    filled = hist_data.copy()
    filled[f'模型回填_{factor_name}'] = np.nan
    nat_train = hist_data[hist_data[factor_name].notna()]
    nat_model = fit_model(nat_train[X_COLS], nat_train[factor_name])[0]
    for prov in provinces:
        df = hist_data[hist_data['地区'] == prov].copy()
        train_df = df[df[factor_name].notna()]
        model = fit_model(train_df[X_COLS], train_df[factor_name])[0] if len(train_df) >= 3 else nat_model
        miss_mask = df[factor_name].isna()
        if miss_mask.any():
            filled.loc[(filled['地区'] == prov) & miss_mask, f'模型回填_{factor_name}'] = np.clip(model.predict(np.asarray(df.loc[miss_mask, X_COLS])), 0, None)
    filled['历史_融合值'] = filled[factor_name].combine_first(filled[f'模型回填_{factor_name}'])
    return filled

def stageB_predict_future(smoothed_hist, future_data, factor_name, use_k=True):
    trained_models = {}
    for prov in smoothed_hist['地区'].unique():
        df = smoothed_hist[smoothed_hist['地区'] == prov]
        y_val = df[f'历史_平滑_{factor_name}']
        if len(y_val.dropna()) >= 3:
            model, _ = fit_model(df[X_COLS], y_val)
            trained_models[prov] = {'model': model}

    jjj_df = smoothed_hist[smoothed_hist['地区'].isin(JJJ_8_PROVINCES)]
    jjj_y = jjj_df[f'历史_平滑_{factor_name}']
    if len(jjj_y.dropna()) >= 3:
        jjj_model, _ = fit_model(jjj_df[X_COLS], jjj_y)
        trained_models['京津冀及周边8省市'] = {'model': jjj_model}

    preds = []
    for prov in TARGET_PROVINCES:
        donor = CROSS_PROV_MODEL.get(prov, prov)
        info = trained_models.get(donor, trained_models.get(prov))
        if info is None: continue
        
        fut = future_data[future_data['地区'] == prov].copy()
        raw_preds = info['model'].predict(np.asarray(fut[X_COLS]))
        
        # --- 【核心修改逻辑：分省对齐校准】 ---
        if use_k:
            # 找到当前省份 2024 年的数据
            hist_2024 = smoothed_hist[(smoothed_hist['地区'] == prov) & (smoothed_hist['年份'] == 2024)]
            if not hist_2024.empty:
                real_2024 = hist_2024['历史_融合值'].values[0]
                # 用引用的模型（如上海模型）预测该省份 2024 年值
                feat_2024 = hist_2024[X_COLS].values
                pred_2024 = info['model'].predict(feat_2024)[0]
                
                # 计算系数 k = 真实 / 预测
                k = real_2024 / pred_2024 if pred_2024 != 0 else 1.0
                raw_preds = raw_preds * k
                print(f"   [校准开启] {prov} 引用 {donor} 模型, 2024比率 k={k:.4f}")
            else:
                print(f"   [警告] {prov} 缺失2024年历史数据，校准跳过")

        fut[f'预测_平滑优先_{factor_name}'] = np.clip(raw_preds, 0, None)
        preds.append(fut)
    return pd.concat(preds) if preds else pd.DataFrame()

# (calibrate_predictions, plot_region_indicator, save_complete_data 函数保持不变)
def calibrate_predictions(predictions, hist_filled, factor_name, max_growth=0.05):
    for prov in predictions['地区'].unique():
        hist_prov = hist_filled[hist_filled['地区'] == prov]
        base_val = hist_prov['历史_融合值'].dropna().iloc[-1]
        base_yr = int(hist_prov['年份'].max())
        mask = (predictions['地区'] == prov)
        predictions.loc[mask, f'预测_平滑优先_{factor_name}'] = predictions.loc[mask].apply(
            lambda r: min(r[f'预测_平滑优先_{factor_name}'], base_val * (1 + max_growth)**(r['年份'] - base_yr)), axis=1
        )
    return predictions

def plot_region_indicator(historical, predictions, factor_name, region, save_path):
    hist_region = historical.sort_values('年份').copy()
    pred_region = predictions.sort_values('年份').copy() if not predictions.empty else pd.DataFrame()
    plt.figure(figsize=(10, 6))
    plt.plot(hist_region['年份'], hist_region['历史_融合值'], label='历史数据', color='gray', linewidth=2)
    if f'历史_平滑_{factor_name}' in hist_region.columns:
        plt.plot(hist_region['年份'], hist_region[f'历史_平滑_{factor_name}'], '--', label='历史平滑', color='blue', alpha=0.7)
    if not pred_region.empty:
        plt.plot(pred_region['年份'], pred_region[f'预测_平滑优先_{factor_name}'], '-.', label='未来预测', color='red', linewidth=2)
    plt.title(f"{region} - {factor_name} 趋势预测", fontsize=14)
    plt.legend()
    img_dir = os.path.join(save_path, '省份指标趋势图')
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(f"{img_dir}/{region}_{factor_name}.png", dpi=300)
    plt.close()

def save_complete_data(historical, predictions, factor_name, region, save_path):
    hist_out = historical[['地区', '年份'] + X_COLS].copy()
    hist_out[f'原始_{factor_name}'] = historical[factor_name]
    for col in [f'模型回填_{factor_name}', '历史_融合值', f'历史_平滑_{factor_name}']:
        hist_out[col] = historical.get(col, np.nan)
    pred_out = predictions[['地区', '年份'] + X_COLS].copy()
    for col in [f'预测_平滑优先_{factor_name}']:
        pred_out[col] = predictions.get(col, np.nan)
    full = pd.concat([hist_out, pred_out], ignore_index=True).sort_values('年份')
    data_dir = os.path.join(save_path, '省份完整数据')
    os.makedirs(data_dir, exist_ok=True)
    file_name = f"{region}{factor_name}完整数据.csv"
    full.to_csv(os.path.join(data_dir, file_name), index=False, encoding='utf-8-sig')

# ======================================================
# 4) 主程序执行
# ======================================================

if __name__ == '__main__':
    fip = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/'
    fop_root = f'/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_{scenario_name}/'
    
    predict_to_input = {
        '新增建筑面积': '建筑业房屋竣工面积',
        '汽车产量': '汽车产量',
        '洗衣机产量': '家用洗衣机产量',
        '电冰箱产量': '家用电冰箱产量',
        '空调产量': '房间空调器',
        '农村水电基础设施投资': '农村水电建设本年完成投资额',
        '新增铁路里程': '新增铁路营业里程',
        '新增高速公路里程': '新增高速等级公路里程',
        '新增其他等级公路里程': '新增除高速外等级公路里程',
        '火力发电量': '火力发电量'
    }

    fip_fut = '/lustre/home/litong/intern/京津冀货运发展/output/scenarios'
    future_data = load_future_province_data(
        f"{fip_fut}/gdp_predictions_{scenario_name}_完整年份.csv",
        f"{fip_fut}/population_predictions_{scenario_name}_完整年份.csv",
        f"{fip_fut}/urban_population_predictions_{scenario_name}_完整年份.csv",
        f"{fip_fut}/third_gdp_predictions_{scenario_name}_完整年份.csv"
    )

    for factor, input_f in predict_to_input.items():
        print(f"\n>>> 正在处理指标: {factor}")
        hist_raw = load_and_preprocess_data(f"{fip}{input_f}分省年度数据.xls", factor)
        hist_filled = stageA_fill_history(hist_raw, factor)
        
        hist_smooth = hist_raw.copy()
        hist_smooth[f'历史_平滑_{factor}'] = hist_filled.groupby('地区')['历史_融合值'].transform(
            lambda s: s.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
        ).clip(0)
        hist_smooth['历史_融合值'] = hist_filled['历史_融合值']
        
        # 传入 USE_CALIBRATION 参数控制校准逻辑
        preds = stageB_predict_future(hist_smooth, future_data, factor, use_k=USE_CALIBRATION)
        if preds.empty: continue
        preds = calibrate_predictions(preds, hist_filled, factor)

        for prov in TARGET_PROVINCES:
            print(f"   - 正在保存省份结果: {prov}")
            h_p = hist_smooth[hist_smooth['地区'] == prov]
            p_p = preds[preds['地区'] == prov]
            plot_region_indicator(h_p, p_p, factor, prov, fop_root)
            save_complete_data(h_p, p_p, factor, prov, fop_root)