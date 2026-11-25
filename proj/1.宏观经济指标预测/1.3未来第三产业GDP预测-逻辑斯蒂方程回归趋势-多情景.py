# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 12:38:57 2025

@author: Yuxi Liu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 逻辑斯蒂方程
def logistic_function(t, K, P0, r):
    return K / (1 + (K/P0 - 1) * np.exp(-r * t))

# 分省gdp预测函数
def predict_province_gdp(province_data, base_year, predict_to):
    years = province_data.index.astype(int)
    gdp = province_data.values.astype(float)
    t = years - base_year

    P0_est = gdp[0]
    K_est = max(gdp) * 1.5
    r_est = 0.01

    bounds = ([gdp.min(), gdp.min()*0.8, -0.1],
              [gdp.max()*10, gdp.max()*1.2, 0.2])

    try:
        params, cov = curve_fit(
            logistic_function, t, gdp,
            p0=[K_est, P0_est, r_est],
            bounds=bounds, maxfev=5000
        )
        K, P0, r = params

        fitted = logistic_function(t, K, P0, r)
        r_squared = 1 - np.sum((gdp - fitted)**2) / np.sum((gdp - np.mean(gdp))**2)

        future_years = np.arange(base_year, predict_to + 1)
        t_future = future_years - base_year
        predicted = logistic_function(t_future, K, P0, r)

        if r > 0:
            inflection_point = np.log((K/P0)-1) / r
            peak_year = base_year + inflection_point
            peak_gdp = K / 2
        else:
            peak_year = None
            peak_gdp = None

        return {
            'params': {'K': K, 'P0': P0, 'r': r},
            'r_squared': r_squared,
            'predicted': pd.Series(predicted, index=future_years),
            'peak_year': peak_year,
            'peak_gdp': peak_gdp
        }
    except Exception as e:
        print(f"拟合{province_data.name}时出错: {str(e)}")
        return None

# 全国GDP增速约束函数（保留）
def apply_national_growth_constraint(province_predictions, base_year, target_growth_rates):
    national_gdp = province_predictions.sum(axis=1)
    target_national = pd.Series(index=province_predictions.index, dtype=float)
    target_national[base_year] = national_gdp[base_year]
    for year in sorted(target_growth_rates.keys()):
        growth_rate = target_growth_rates[year]
        target_national[year] = target_national[year-1] * (1 + growth_rate)
    adjustment_factors = target_national / national_gdp
    adjusted_predictions = province_predictions.copy()
    for province in adjusted_predictions.columns:
        adjusted_predictions[province] = adjusted_predictions[province] * adjustment_factors
    return adjusted_predictions, adjustment_factors, target_national

# ========= 主程序 =========

# 1. 加载第三产业历史数据（保留全部历史年份）
df = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史第三产业GDP数据.csv', encoding='gbk')

# 1.1 预处理
# 若有“备注”列可删除： if '备注' in df.columns: df = df.drop(columns=['备注'])
df = df.set_index('地区')
df_transposed = df.T
df_transposed.index = df_transposed.index.str.replace('年', '').astype(int)
df_transposed = df_transposed.sort_index(ascending=True)

years = df_transposed.index
provinces = df_transposed.columns
data = pd.DataFrame(index=years, columns=provinces)
for year in years:
    for province in provinces:
        data.loc[year, province] = df_transposed.loc[year, province]
data = data.apply(pd.to_numeric, errors='coerce')

# 另存转置历史数据（可选）
data.to_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史第三产业GDP数据转置.csv', encoding='utf-8-sig')
print("数据格式转换完成！")
print(f"数据形状: {data.shape}")
print(f"年份范围: {data.index.min()} - {data.index.max()}")
print(f"省份数量: {len(data.columns)}")

# 2. 读取情景约束（低/中/高）
df_rates = pd.read_excel('/lustre/home/litong/intern/京津冀货运发展/data/constraints/第三产业GDP预测约束.xlsx')
scenario_rates = {
    '基准': None,
    '低增长': df_rates.loc[df_rates['情景']=='低', np.arange(2025,2036).tolist()].values.tolist()[0],
    '中增长': df_rates.loc[df_rates['情景']=='中', np.arange(2025,2036).tolist()].values.tolist()[0],
    '高增长': df_rates.loc[df_rates['情景']=='高', np.arange(2025,2036).tolist()].values.tolist()[0]
}

# 3. 年份设置（拟合窗口 vs 预测区间）
history_syear = 2015   # 拟合窗口起年
history_eyear = 2025   # 拟合窗口止年（不含）
predict_syear = 2025
predict_eyear = 2035

# —— 关键：不再裁剪 data；保留全部历史到 data_all，只在拟合时用窗口 data_hist ——
data_all = data.copy()                                    # 全部历史
fit_years = np.arange(history_syear, history_eyear, 1)    # 2015-2024
data_hist = data_all.loc[fit_years]                       # 用于拟合

# 4. 基准情景预测（模型期 2015–2035）
base_predictions = pd.DataFrame(index=np.arange(history_syear, predict_eyear+1))
base_params = pd.DataFrame(columns=['K', 'P0', 'r', 'r_squared', 'peak_year', 'peak_gdp'])

plt.figure(figsize=(15, 20))
n_cols = 4
n_rows = int(np.ceil(len(data_all.columns) / n_cols))

for i, province in enumerate(data_all.columns):
    print(f"处理省份: {province}...")
    province_data = data_hist[province]
    province_data.name = province

    result = predict_province_gdp(province_data, history_syear, predict_eyear)
    if result is not None:
        base_predictions[province] = result['predicted']
        params = result['params']
        params['r_squared'] = result['r_squared']
        params['peak_year'] = result['peak_year']
        params['peak_gdp'] = result['peak_gdp']
        base_params.loc[province] = params

        # ---------------------------------------------------------
        # 绘图部分修改：实现无缝连接
        # ---------------------------------------------------------
        plt.subplot(n_rows, n_cols, i+1)
        
        # 1. 历史原始数据：改为 'bo' (纯蓝点)，markersize设为3，避免遮挡拟合线
        plt.plot(province_data.index, province_data, 'bo', markersize=4, label='历史数据')
        
        # 2. 拟合曲线 (绿线)：延伸到 2024 年底 (+1 保证包含 2024)
        history_years_full = np.arange(history_syear, history_eyear + 1)
        t_history_full = history_years_full - history_syear
        fitted_full = logistic_function(t_history_full, result['params']['K'],
                                       result['params']['P0'],
                                       result['params']['r'])
        plt.plot(history_years_full, fitted_full, 'g-', alpha=0.8, label='拟合曲线')
        
        # 3. 预测曲线 (红线)：从 2024 年开始接力画起
        # 这样 2024-2025 之间就不会有白色的断裂点
        future_years_with_bridge = np.arange(history_eyear, predict_eyear + 1)
        # 从 predicted 序列中截取从 2024 开始的部分
        predicted_with_bridge = result['predicted'].loc[future_years_with_bridge]
        plt.plot(future_years_with_bridge, predicted_with_bridge, 'r-', linewidth=1.5, label='预测')

        # 标记拐点（峰值）
        if result['peak_year'] and history_syear < result['peak_year'] < predict_eyear:
            plt.axvline(x=result['peak_year'], color='gray', linestyle=':', alpha=0.7)
            plt.text(result['peak_year'], result['peak_gdp']*0.8,
                     f'峰值: {result["peak_year"]:.0f}\n{result["peak_gdp"]:.1f}亿',
                     fontsize=8, ha='center')

        plt.title(province)
        plt.xlabel('年份')
        plt.ylabel('第三产业gdp(亿元)')
        plt.grid(alpha=0.3)
        plt.annotate(f'R²={result["r_squared"]:.3f}',
                     xy=(0.05, 0.85), xycoords='axes fraction',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

plt.tight_layout()
plt.suptitle(f'中国各省第三产业gdp预测({history_syear}-{predict_eyear})', fontsize=16, y=1.02)
out_dir = '/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/1.3第三产业GDP预测'
os.makedirs(out_dir, exist_ok=True)
plt.savefig(f'{out_dir}/province_third_gdp_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# 基准情景：全国与区域汇总（基于省份列）
bthsa_8prov = ['北京市','天津市','河北省','河南省','山东省','山西省','内蒙古自治区','陕西省']
base_predictions['全国'] = base_predictions[data_all.columns].sum(axis=1)
base_predictions['京津冀及周边8省市'] = base_predictions[bthsa_8prov].sum(axis=1)

# ===== 合并“全部历史 + 预测”到一张表 =====
provinces_list = list(data_all.columns)
full_index = sorted(set(data_all.index.tolist()) | set(range(predict_syear, predict_eyear+1)))

full_predictions_base = pd.DataFrame(index=full_index, columns=provinces_list)
# 填入全部历史年份
full_predictions_base.loc[data_all.index, provinces_list] = data_all[provinces_list]
# 覆盖/追加 2025–2035 的预测
full_predictions_base.loc[predict_syear:predict_eyear, provinces_list] = \
    base_predictions.loc[predict_syear:predict_eyear, provinces_list]
# 汇总列
full_predictions_base['全国'] = full_predictions_base[provinces_list].sum(axis=1, min_count=1)
full_predictions_base['京津冀及周边8省市'] = full_predictions_base[bthsa_8prov].sum(axis=1, min_count=1)

# 保存“基准_完整年份”CSV
full_base_path = f'{out_dir}/third_gdp_predictions_基准_完整年份.csv'
full_predictions_base.to_csv(full_base_path, encoding='utf-8-sig')
print(f"已保存 基准 情景完整年份至: {full_base_path}")

# 保存容器
all_scenario_predictions = {'基准': base_predictions}
all_scenario_params = {'基准': base_params}

# 5. 约束情景预测
for scenario_name, growth_rates in scenario_rates.items():
    if scenario_name == '基准':
        continue
    print("="*50)
    print(f"正在进行 {scenario_name} 情景预测...")
    print(f"增长率约束: {growth_rates}")
    print("="*50)

    scenario_predictions = base_predictions.copy()
    scenario_params = base_params.copy()

    base_national_gdp = base_predictions['全国'].copy()
    base_2024 = base_national_gdp.loc[2024]

    constrained_national = pd.Series(index=np.arange(2025, 2036), dtype=float)
    current_gdp = base_2024
    for year, growth_rate in zip(range(2025, 2036), growth_rates):
        current_gdp *= (1 + growth_rate)
        constrained_national.loc[year] = current_gdp

    adjustment_factors = constrained_national / base_national_gdp.loc[2025:2035]

    for year in range(2025, 2036):
        factor = adjustment_factors.loc[year]
        for province in data_all.columns:
            scenario_predictions.loc[year, province] = base_predictions.loc[year, province] * factor

    scenario_predictions['全国'] = scenario_predictions[data_all.columns].sum(axis=1)
    scenario_predictions['京津冀及周边8省市'] = scenario_predictions[bthsa_8prov].sum(axis=1)

    scenario_params['scenario'] = scenario_name
    scenario_params['growth_constraint'] = str(growth_rates)

    all_scenario_predictions[scenario_name] = scenario_predictions
    all_scenario_params[scenario_name] = scenario_params

    # ----- 针对该情景保存“完整年份” CSV -----
    full_predictions_scenario = pd.DataFrame(index=full_index, columns=provinces_list)
    # 填入全部历史年份
    full_predictions_scenario.loc[data_all.index, provinces_list] = data_all[provinces_list]
    # 覆盖/追加 2025–2035 的预测
    full_predictions_scenario.loc[predict_syear:predict_eyear, provinces_list] = \
        scenario_predictions.loc[predict_syear:predict_eyear, provinces_list]
    # 汇总列
    full_predictions_scenario['全国'] = full_predictions_scenario[provinces_list].sum(axis=1, min_count=1)
    full_predictions_scenario['京津冀及周边8省市'] = full_predictions_scenario[bthsa_8prov].sum(axis=1, min_count=1)
    
    full_scenario_path = f'{out_dir}/third_gdp_predictions_{scenario_name}_完整年份.csv'
    full_predictions_scenario.to_csv(full_scenario_path, encoding='utf-8-sig')
    print(f"已保存 {scenario_name} 情景完整年份至: {full_scenario_path}")
    # ---------------------------------------

    plt.figure(figsize=(12, 8))
    plt.plot(base_predictions.index, base_predictions['全国']/1e4, 'b-', label='基准情景')
    plt.plot(scenario_predictions.index, scenario_predictions['全国']/1e4, 'r--', label=f'{scenario_name}情景')
    plt.title(f'全国第三产业GDP预测对比: 基准 vs {scenario_name}')
    plt.xlabel('年份')
    plt.ylabel('全国第三产业GDP(万亿元)')
    plt.grid(True)
    plt.legend()
    plt.xticks(np.arange(2015, 2036, 2))
    plt.savefig(f'{out_dir}/national_third_gdp_comparison_{scenario_name}.png')
    plt.show()

# 6. 保存（各情景模型期 & 参数）
os.makedirs(out_dir, exist_ok=True)
for scenario_name, predictions in all_scenario_predictions.items():
    file_path = f'{out_dir}/third_gdp_predictions_{scenario_name}.csv'
    predictions.to_csv(file_path, encoding='utf-8-sig')
    print(f"已保存 {scenario_name} 情景预测结果至: {file_path}")

    params = all_scenario_params[scenario_name]
    param_path = f'{out_dir}/third_gdp_params_{scenario_name}.csv'
    params.to_csv(param_path, encoding='utf-8-sig')
    print(f"已保存 {scenario_name} 情景参数至: {param_path}")

# 7. 多情景对比分析
plt.figure(figsize=(14, 8))
for scenario_name, predictions in all_scenario_predictions.items():
    gdp_data = predictions.loc[2025:2035, '全国'] / 1e4
    plt.plot(gdp_data.index, gdp_data.values, label=scenario_name)
plt.title('多情景全国第三产业GDP预测对比 (2025-2035)', fontsize=16)
plt.xlabel('年份', fontsize=12)
plt.ylabel('第三产业GDP (万亿元)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.xticks(np.arange(2025, 2036, 2))
plt.tight_layout()
plt.savefig(f'{out_dir}/national_third_gdp_scenarios_comparison.png', dpi=300)
plt.show()

province_gdp_2035 = pd.DataFrame()
for scenario_name, predictions in all_scenario_predictions.items():
    province_gdp_2035[scenario_name] = predictions.loc[2035, data_all.columns] 
province_gdp_2035 = province_gdp_2035.sort_values(by='基准', ascending=False)

plt.figure(figsize=(16, 10))
province_gdp_2035.plot(kind='bar', width=0.8)
plt.title('2035年各省第三产业GDP预测多情景对比', fontsize=16)
plt.xlabel('省份', fontsize=12)
plt.ylabel('第三产业GDP (亿元)', fontsize=12)
plt.xticks(rotation=80, fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{out_dir}/province_third_gdp_2035_comparison.png', dpi=300)
plt.show()

print("所有情景预测完成！")