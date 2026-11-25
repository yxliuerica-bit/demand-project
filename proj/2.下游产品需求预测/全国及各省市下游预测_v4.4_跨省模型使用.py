# -*- coding: utf-8 -*-
"""
两阶段流程（全国模型回填改造版 + 阶段B跨省模型引用）：

变更点（相对你上一版）：
- 阶段B预测时支持“用外省模型预测本省”，映射如下：
    北京市 <- 天津市
    山东省 <- 河北省
    浙江省 <- 江苏省
  训练仍对每个省各自训练；预测阶段选择映射的“模型来源省”的已训练模型，使用“被预测省”的未来特征做预测。

其他逻辑保持不变：阶段A不可建模省用全国模型回填历史；历史平滑；温和上限+非负保护；区域求和。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ------------------- Matplotlib 字体 -------------------
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# ------------------- 常量 -------------------
SHOW_REFERENCE_ORIGINAL = False

HISTORY_START_YEAR = 2010
HISTORY_END_YEAR = 2024
FUTURE_YEARS = list(range(2025, 2036))
X_COLS = ['GDP', '人口', '第三产业GDP', '城镇人口']

#情景
scenario_name = "基准"

# 历史平滑参数（中心移动平均）
SMOOTH_WINDOW = 3

# 阶段B“跨省模型引用”映射（key=被预测省，value=模型来源省）
CROSS_PROV_MODEL = {
    #'北京市': '天津市',
    # '山东省': '辽宁省',
    # '陕西省': '江苏省',
}

# ======================================================
# 1) 数据加载
# ======================================================

def load_and_preprocess_data(fi_predict_factor, factor_name):
    gdp = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史GDP数据.csv', encoding='gbk')
    gdp = gdp.iloc[:, :-1].melt(id_vars='地区', var_name='年份', value_name='GDP')
    gdp['年份'] = gdp['年份'].str.replace('年', '').astype(int)

    pop = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史人口数据.csv', encoding='gbk')
    pop = pop.iloc[:, :-1].melt(id_vars='地区', var_name='年份', value_name='人口')
    pop['年份'] = pop['年份'].str.replace('年', '').astype(int)

    third_gdp = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史第三产业GDP数据.csv', encoding='gbk')
    third_gdp = third_gdp.melt(id_vars='地区', var_name='年份', value_name='第三产业GDP')
    third_gdp['年份'] = third_gdp['年份'].str.replace('年', '').astype(int)

    urban_pop = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史城镇人口数据.csv', encoding='gbk')
    urban_pop = urban_pop.melt(id_vars='地区', var_name='年份', value_name='城镇人口')
    urban_pop['年份'] = urban_pop['年份'].str.replace('年', '').astype(int)

    prod_pre = pd.read_excel(fi_predict_factor, sheet_name='分省年度数据', skiprows=3)
    prod_pre = prod_pre.melt(id_vars='地区', var_name='年份', value_name=factor_name)
    prod_pre['年份'] = prod_pre['年份'].str.replace('年', '').astype(int)

    merged = pd.merge(gdp, pop, on=['地区', '年份'], how='inner')
    merged = pd.merge(merged, third_gdp, on=['地区', '年份'], how='inner')
    merged = pd.merge(merged, urban_pop, on=['地区', '年份'], how='inner')
    merged = pd.merge(merged, prod_pre, on=['地区', '年份'], how='inner')

    merged = merged.dropna(subset=X_COLS)
    merged = merged[(merged['年份'] >= HISTORY_START_YEAR) & (merged['年份'] <= HISTORY_END_YEAR)].copy()
    merged.sort_values(['地区', '年份'], inplace=True)
    return merged.reset_index(drop=True)

# ======================================================
# 2) 未来特征数据
# ======================================================

def load_future_province_data(gdp_future_file, pop_future_file, urban_pop_future_file, third_gdp_future_file):
    def to_long(path, value_name):
        df = pd.read_csv(path)
        df = df.rename(columns={'Unnamed: 0': '年份'})
        return df.melt(id_vars='年份', var_name='地区', value_name=value_name)

    gdp_long = to_long(gdp_future_file, 'GDP')
    pop_long = to_long(pop_future_file, '人口')
    third_gdp_long = to_long(third_gdp_future_file, '第三产业GDP')
    urban_pop_long = to_long(urban_pop_future_file, '城镇人口')

    future_data = gdp_long.merge(pop_long, on=['地区', '年份'], how='inner') \
                          .merge(third_gdp_long, on=['地区', '年份'], how='inner') \
                          .merge(urban_pop_long, on=['地区', '年份'], how='inner')
    future_data = future_data[future_data['年份'].isin(FUTURE_YEARS)].copy()
    future_data.sort_values(['地区', '年份'], inplace=True)
    return future_data.reset_index(drop=True)

# ======================================================
# 3) 多项式回归（CV 选优）
# ======================================================

def fit_model(X, y, max_degree=3, min_samples=3):
    y = np.asarray(y)
    mask = np.isfinite(y)
    if hasattr(X, 'iloc'):
        X = X[mask]
    else:
        X = np.asarray(X)[mask]
    y = y[mask]
    if len(y) < min_samples:
        raise ValueError(f'有效样本不足: {len(y)} < {min_samples}')

    best_score, best_degree, best_model = -np.inf, 1, None
    max_possible_degree = min(max_degree, len(X) - 1) if len(X) > 1 else 1
    for degree in range(1, max_possible_degree + 1):
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
        cv_folds = min(5, len(X)) if len(X) >= 2 else 1
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        avg_score = float(np.mean(scores))
        if avg_score > best_score:
            best_score, best_degree, best_model = avg_score, degree, model
    best_model.fit(X, y)
    return best_model, best_degree, best_score

# ======================================================
# 4) 阶段A：历史回填（可建模=省级模型；不可建模=全国模型）
# ======================================================

def stageA_fill_history(historical_data, factor_name):
    provinces = sorted(historical_data['地区'].unique())
    modelable_set, non_modelable_set = set(), set()

    for prov in provinces:
        n = int(historical_data[(historical_data['地区'] == prov) & (historical_data[factor_name].notna())].shape[0])
        (modelable_set if n >= 3 else non_modelable_set).add(prov)

    filled = historical_data.copy()
    filled[f'模型回填_{factor_name}'] = np.nan

    model_rows = []

    # 省级可建模：本省模型回填
    for prov in sorted(modelable_set):
        df = historical_data[historical_data['地区'] == prov].copy()
        train_df = df[df[factor_name].notna()][X_COLS + [factor_name]]
        try:
            model, degree, _ = fit_model(train_df[X_COLS], train_df[factor_name])
        except Exception:
            non_modelable_set.add(prov)
            continue
        miss_mask = df[factor_name].isna()
        if miss_mask.any():
            y_fill = model.predict(df.loc[miss_mask, X_COLS])
            filled.loc[(filled['地区'] == prov) & miss_mask, f'模型回填_{factor_name}'] = np.clip(y_fill, 0, None)
        model_rows.append({
            '层级': '省级', '对象': prov,
            '历史_原始样本数': len(train_df), '最佳阶数': degree,
            '训练R²': model.score(train_df[X_COLS], train_df[factor_name])
        })

    # 全国模型：用于不可建模省份
    # 显式要求：全国所有省份、2010年及以后、目标指标不为空
    national_train = historical_data[
        (historical_data['年份'] >= 2010) & 
        (historical_data[factor_name].notna())
    ][X_COLS + [factor_name]]
    national_model = None
    if len(national_train) >= 3:
        try:
            national_model, national_degree, _ = fit_model(national_train[X_COLS], national_train[factor_name])
            national_r2 = national_model.score(national_train[X_COLS], national_train[factor_name])
            model_rows.append({'层级': '全国', '对象': '全国模型', '历史_原始样本数': len(national_train),
                               '最佳阶数': national_degree, '训练R²': national_r2})
        except Exception as e:
            print(f'全国模型拟合失败：{e}')

    if national_model is None and len(non_modelable_set) > 0:
        print('警告：全国模型无法训练，以下省份的历史缺口无法回填：', sorted(non_modelable_set))

    for prov in sorted(non_modelable_set):
        if national_model is None:
            continue
        df = historical_data[historical_data['地区'] == prov].copy()
        miss_mask = df[factor_name].isna()
        if miss_mask.any():
            y_fill = national_model.predict(df.loc[miss_mask, X_COLS])
            filled.loc[(filled['地区'] == prov) & miss_mask, f'模型回填_{factor_name}'] = np.clip(y_fill, 0, None)

    filled['历史_融合值'] = filled[factor_name].combine_first(filled[f'模型回填_{factor_name}'])
    model_summary = pd.DataFrame(model_rows)
    return filled, modelable_set, non_modelable_set, model_summary

# ======================================================
# 5) 历史平滑
# ======================================================

def smooth_history(filled_hist, factor_name, window=SMOOTH_WINDOW):
    out = filled_hist.copy().sort_values(['地区', '年份'])
    out[f'历史_平滑_{factor_name}'] = (
        out.groupby('地区')['历史_融合值']
           .transform(lambda s: s.rolling(window=window, center=True, min_periods=1).mean())
    ).clip(lower=0)
    return out

# ======================================================
# 6) 阶段B：平滑历史再建模，预测未来（支持跨省模型引用）
# ======================================================

def stageB_predict_future(smoothed_hist, future_data, factor_name):
    provinces = sorted(smoothed_hist['地区'].unique())

    # 先对每个省“各自训练”模型，缓存下来
    trained = {}  # prov -> {'model':..., 'degree':..., 'n':..., 'r2':...}
    for prov in provinces:
        df = smoothed_hist[smoothed_hist['地区'] == prov].copy()
        y_col = f'历史_平滑_{factor_name}'
        train_df = df[[*X_COLS, y_col]].dropna()
        if len(train_df) < 3:
            continue
        model, degree, _ = fit_model(train_df[X_COLS], train_df[y_col])
        trained[prov] = {
            'model': model,
            'degree': degree,
            'n': len(train_df),
            'r2': model.score(train_df[X_COLS], train_df[y_col]),
        }

    preds_list, model_rows = [], []

    # 逐省预测：若在映射中，则用“来源省”的模型；否则用本省模型
    for prov in provinces:
        donor = CROSS_PROV_MODEL.get(prov, prov)  # 模型来源省
        donor_model_info = trained.get(donor)
        # 若来源省模型不存在，退回到本省模型；仍无则跳过
        if donor_model_info is None:
            donor_model_info = trained.get(prov)
        if donor_model_info is None:
            continue

        # 未来特征：使用“被预测省”的特征
        fut = future_data[future_data['地区'] == prov].copy()
        if fut.empty:
            # 若未来特征缺，则从历史末年外推
            df_hist = smoothed_hist[smoothed_hist['地区'] == prov].copy()
            last_year = int(df_hist['年份'].max())
            fut = pd.DataFrame({'年份': FUTURE_YEARS})
            fut['地区'] = prov
            last_row = df_hist.loc[df_hist['年份'] == last_year].iloc[0]
            for c in X_COLS:
                fut[c] = float(last_row[c]) * (1.02 ** (fut['年份'] - last_year))

        # 用“donor”来源省的模型做预测
        model = donor_model_info['model']
        fut[f'预测_原始数据_{factor_name}'] = np.clip(model.predict(fut[X_COLS]), 0, None)
        fut[f'预测_平滑优先_{factor_name}'] = fut[f'预测_原始数据_{factor_name}']
        preds_list.append(fut)

        # 记录摘要（清楚标识是否“跨省引用”）
        used_cross = (prov in CROSS_PROV_MODEL)
        model_rows.append({
            '层级': '省级', '对象': prov,
            '阶段B_样本数(来源省)': donor_model_info['n'],
            '阶段B_最佳阶数(来源省)': donor_model_info['degree'],
            '阶段B_R²(来源省)': donor_model_info['r2'],
            '是否跨省引用模型': '是' if used_cross else '否',
            '模型来源省': donor if used_cross else prov
        })

    provincial_predictions = pd.concat(preds_list, ignore_index=True) if preds_list else pd.DataFrame()
    model_summaryB = pd.DataFrame(model_rows)
    return provincial_predictions, model_summaryB

# ======================================================
# 7) 非负保护 + 上限校准
# ======================================================

def ensure_non_negative(predictions, factor_name):
    predictions = predictions.copy()
    for col in [f'预测_原始数据_{factor_name}', f'预测_平滑优先_{factor_name}']:
        if col in predictions.columns:
            predictions[col] = predictions[col].where(predictions[col].isna(), predictions[col].clip(lower=0))
    return predictions

def calibrate_predictions(predictions, hist_filled, factor_name, max_growth=0.05):
    predictions = predictions.copy()
    for prov in predictions['地区'].unique():
        preds = predictions[predictions['地区'] == prov].copy()
        hist = hist_filled[hist_filled['地区'] == prov].sort_values('年份')
        if hist.empty:
            continue
        base_series = hist['历史_融合值'].dropna()
        if base_series.empty:
            continue
        base_value = float(base_series.iloc[-1])
        base_year = int(hist['年份'].iloc[-1])
        for _, row in preds.iterrows():
            year = int(row['年份'])
            upper = base_value * (1 + max_growth) ** (year - base_year)
            ix = predictions[(predictions['地区'] == prov) & (predictions['年份'] == year)].index
            if len(ix) > 0:
                for col in [f'预测_原始数据_{factor_name}', f'预测_平滑优先_{factor_name}']:
                    if col in predictions.columns and not pd.isna(predictions.at[ix[0], col]):
                        predictions.at[ix[0], col] = min(predictions.at[ix[0], col], upper)
    return predictions

# ======================================================
# 8) 区域汇总
# ======================================================

def aggregate_regional_historical_data(provincial_data, region, factor_name):
    region_provinces = {
        '全国': provincial_data['地区'].unique(),
        '京津冀及周边8省市': ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '山东省', '河南省', '辽宁省', '陕西省']
    }
    if region not in region_provinces:
        raise ValueError(f'不支持的区域: {region}')

    region_list = region_provinces[region]
    region_data = provincial_data[provincial_data['地区'].isin(region_list)].copy()

    aggregated = region_data.groupby('年份').agg({
        'GDP': 'sum', '人口': 'sum', '第三产业GDP': 'sum', '城镇人口': 'sum',
        factor_name: 'sum', f'模型回填_{factor_name}': 'sum', '历史_融合值': 'sum', f'历史_平滑_{factor_name}': 'sum'
    }).reset_index()
    aggregated['地区'] = region

    pre_2025_data = aggregated[aggregated['年份'] < 2025]
    print(f"生成{region}历史数据: 共{len(aggregated)}条记录，其中2025年前{len(pre_2025_data)}条，年份范围: {aggregated['年份'].min()} - {aggregated['年份'].max()}")
    return aggregated

def aggregate_regional_predictions(provincial_predictions, factor_name):
    jingjinji_regions = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '山东省', '河南省', '辽宁省', '陕西省']

    def agg(df):
        out = df.groupby('年份').agg({
            'GDP': 'sum', '人口': 'sum', '第三产业GDP': 'sum', '城镇人口': 'sum',
            f'预测_原始数据_{factor_name}': 'sum', f'预测_平滑优先_{factor_name}': 'sum'
        }).reset_index()
        return out

    national_pred = agg(provincial_predictions); national_pred['地区'] = '全国'
    jjj_pred = agg(provincial_predictions[provincial_predictions['地区'].isin(jingjinji_regions)]); jjj_pred['地区'] = '京津冀及周边8省市'
    return {'全国': national_pred, '京津冀及周边8省市': jjj_pred}

# ======================================================
# 9) 绘图
# ======================================================

def plot_region_indicator(historical, predictions, factor_name, region, save_path, is_province=False, show_smoothed=True):
    if historical.empty:
        print(f'警告: {region}没有历史数据，无法绘制趋势图')
        return

    hist_region = historical.sort_values('年份').copy()
    pred_region = predictions.sort_values('年份').copy() if not predictions.empty else pd.DataFrame()

    hist_fused = hist_region['历史_融合值'].clip(lower=0)

    plt.figure(figsize=(14, 8))
    plt.plot(hist_region['年份'], hist_fused, linewidth=2, label='历史（原始→回填融合）')

    if show_smoothed and f'历史_平滑_{factor_name}' in hist_region.columns:
        plt.plot(hist_region['年份'], hist_region[f'历史_平滑_{factor_name}'], linestyle='--', linewidth=2, label='历史（平滑）')

    if not pred_region.empty:
        col_pref = f'预测_平滑优先_{factor_name}'
        if col_pref in pred_region.columns:
            plt.plot(pred_region['年份'], pred_region[col_pref], linestyle='-.', linewidth=2, label='预测')

    title_prefix = '省份' if is_province else '区域'
    plt.title(f'{title_prefix} {region}{factor_name}历史趋势与预测', fontsize=16, pad=15)
    plt.xlabel('年份', fontsize=14)
    unit = '（万平方米）' if factor_name == '新增建筑面积' else ('（万辆）' if factor_name == '汽车产量' else '')
    plt.ylabel(f'{factor_name}{unit}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)

    y_candidates = [np.nanmax(hist_fused.values) if len(hist_fused) else 0]
    for col in [f'历史_平滑_{factor_name}', f'预测_平滑优先_{factor_name}', f'预测_原始数据_{factor_name}']:
        if col in hist_region.columns:
            y_candidates.append(np.nanmax(hist_region[col].values))
        if not pred_region.empty and col in pred_region.columns:
            y_candidates.append(np.nanmax(pred_region[col].values))
    y_top = float(np.nanmax(y_candidates)) if y_candidates else 0.0
    if not np.isfinite(y_top) or y_top <= 0:
        y_top = 1.0
    plt.ylim(0, y_top * 1.05)

    all_years = hist_region['年份']
    if not pred_region.empty:
        all_years = pd.concat([all_years, pred_region['年份']])
    plt.xlim(min(all_years.min(), HISTORY_START_YEAR) - 1, all_years.max() + 1)

    plt.axvline(x=2025, color='gray', linestyle=':', linewidth=1.5)

    plt.tight_layout()
    save_dir = os.path.join(save_path, '省份指标趋势图' if is_province else '区域指标趋势图')
    os.makedirs(save_dir, exist_ok=True)
    save_file = f"{save_dir}/{region}_{factor_name}_趋势与预测.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'已保存{region}趋势图: {save_file}')

# ======================================================
# 10) 导出
# ======================================================

def save_complete_data(historical, predictions, factor_name, region, save_path, is_province=False):
    if not historical.empty:
        hist_out = historical[['地区', '年份'] + X_COLS].copy()
        hist_out[f'原始_{factor_name}'] = historical[factor_name]
        for col in [f'模型回填_{factor_name}', '历史_融合值', f'历史_平滑_{factor_name}']:
            hist_out[col] = historical.get(col, np.nan)
    else:
        hist_out = pd.DataFrame()

    if not predictions.empty:
        pred_out = predictions[['地区', '年份'] + X_COLS].copy()
        for col in [f'预测_原始数据_{factor_name}', f'预测_平滑优先_{factor_name}']:
            pred_out[col] = predictions.get(col, np.nan)
    else:
        pred_out = pd.DataFrame()

    dfs = []
    if not hist_out.empty: dfs.append(hist_out)
    if not pred_out.empty: dfs.append(pred_out)
    if dfs:
        full_output = pd.concat(dfs, ignore_index=True).sort_values(by=['地区', '年份'])
        save_dir = os.path.join(save_path, '省份完整数据' if is_province else '区域完整数据')
        os.makedirs(save_dir, exist_ok=True)
        save_file = f"{save_dir}/{region}{factor_name}完整数据.csv"
        full_output.to_csv(save_file, index=False)
        print(f"已保存完整数据（含历史回填/平滑 + 未来预测）: {save_file}")

# ======================================================
# 11) 主程序
# ======================================================

if __name__ == '__main__':
    fipath = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/'
    fopath = f'/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_{scenario_name}/'
    os.makedirs(fopath, exist_ok=True)

    regions = ['全国', '京津冀及周边8省市']

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

    fipath_future = '/lustre/home/litong/intern/京津冀货运发展/output/scenarios'
    gdp_future_file = f"{fipath_future}/gdp_predictions_{scenario_name}_完整年份.csv"
    pop_future_file = f"{fipath_future}/population_predictions_{scenario_name}_完整年份.csv"
    urban_pop_future_file = f"{fipath_future}/urban_population_predictions_{scenario_name}_完整年份.csv"
    third_gdp_future_file = f"{fipath_future}/third_gdp_predictions_{scenario_name}_完整年份.csv"

    for factor_name, input_name in predict_to_input.items():
        print(f"\n{'='*30} 处理指标: {factor_name} {'='*30}")
        fi_predict_factor = f"{fipath}{input_name}分省年度数据.xls"

        print('加载历史数据并进行预处理…')
        historical_provincial = load_and_preprocess_data(fi_predict_factor, factor_name)
        if historical_provincial.empty:
            print(f'历史数据为空，跳过：{factor_name}')
            continue

        print('加载未来特征数据…')
        future_provincial_data = load_future_province_data(
            gdp_future_file, pop_future_file, urban_pop_future_file, third_gdp_future_file
        )

        print('阶段A：省级/全国模型回填历史…')
        hist_filled, modelable_set, non_modelable_set, model_results_A = stageA_fill_history(
            historical_provincial, factor_name
        )

        print('历史平滑（中心移动平均）…')
        hist_smoothed = smooth_history(hist_filled, factor_name, window=SMOOTH_WINDOW)

        print('阶段B：历史平滑后再建模进行未来预测…（含跨省模型引用）')
        provincial_predictions, model_results_B = stageB_predict_future(
            hist_smoothed, future_provincial_data, factor_name
        )

        provincial_predictions = calibrate_predictions(provincial_predictions, hist_filled, factor_name)
        provincial_predictions = ensure_non_negative(provincial_predictions, factor_name)

        if not model_results_A.empty:
            model_results_A.to_csv(f"{fopath}/各省{factor_name}_阶段A_历史回填模型结果.csv", index=False)
        if not model_results_B.empty:
            model_results_B.to_csv(f"{fopath}/各省{factor_name}_阶段B_平滑再建模结果.csv", index=False)


        print('绘图与导出各省结果…')
        for province in hist_smoothed['地区'].unique():
            hist_prov = hist_smoothed[hist_smoothed['地区'] == province].copy()
            pred_prov = provincial_predictions[provincial_predictions['地区'] == province].copy()
            plot_region_indicator(hist_prov, pred_prov, factor_name, province, fopath, is_province=True, show_smoothed=True)
            save_complete_data(hist_prov, pred_prov, factor_name, province, fopath, is_province=True)

        print('从省份数据汇总区域历史数据和预测数据…')
        regional_predictions = aggregate_regional_predictions(provincial_predictions, factor_name)

        for region in regions:
            print(f"\n----- 处理区域: {region} -----")
            region_hist = aggregate_regional_historical_data(hist_smoothed, region, factor_name)
            region_pred = regional_predictions.get(region, pd.DataFrame())
            plot_region_indicator(region_hist, region_pred, factor_name, region, fopath, is_province=False, show_smoothed=True)
            save_complete_data(region_hist, region_pred, factor_name, region, fopath, is_province=False)

    print('\n所有指标、省份和区域处理完成！')
