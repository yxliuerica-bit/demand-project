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
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']  # 使用系统已有的中文字体
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# -----------------------------
# 逻辑斯蒂方程
# -----------------------------
def logistic_function(t, K, P0, r):
    """
    逻辑斯蒂方程
    t: 时间数组（年份相对量）
    K: 环境容纳量（最大gdp）
    P0: 初始gdp
    r: 增长率
    """
    return K / (1 + (K / P0 - 1) * np.exp(-r * t))

# -----------------------------
# 分省gdp预测函数（基于拟合窗口）
# -----------------------------
def predict_province_gdp(province_data, base_year, predict_to):
    """
    对单个省份进行gdp预测
    province_data: 该省份的历史gdp数据（用于拟合的窗口数据，索引为年份）
    base_year: 拟合基准年份（窗口起点）
    predict_to: 预测截止年份
    """
    # 准备数据
    years = province_data.index.astype(int)
    gdp = province_data.values.astype(float)

    # 转换为相对时间（从0开始）
    t = years - base_year

    # 设置初始参数估计
    P0_est = gdp[0]  # 初始gdp取第一年数据
    K_est = max(gdp) * 1.5  # 环境容量设为历史最大值的1.5倍
    r_est = 0.01  # 初始增长率

    # 参数边界约束（防止不合理值）
    bounds = ([gdp.min(), gdp.min() * 0.8, -0.1],  # 下界
              [gdp.max() * 10, gdp.max() * 1.2, 0.2])  # 上界

    try:
        # 拟合逻辑斯蒂方程
        params, cov = curve_fit(
            logistic_function, t, gdp,
            p0=[K_est, P0_est, r_est],
            bounds=bounds,
            maxfev=5000
        )
        K, P0, r = params

        # 计算拟合优度
        fitted = logistic_function(t, K, P0, r)
        r_squared = 1 - np.sum((gdp - fitted) ** 2) / np.sum((gdp - np.mean(gdp)) ** 2)

        # 预测未来gdp（含历史窗口的起点到预测末年）
        future_years = np.arange(base_year, predict_to + 1)
        t_future = future_years - base_year
        predicted = logistic_function(t_future, K, P0, r)

        # 计算gdp峰值年份（拐点）
        if r > 0 and K > 0 and P0 > 0:
            inflection_point = np.log((K / P0) - 1) / r
            peak_year = base_year + inflection_point
            peak_gdp = K / 2  # 拐点gdp是K的一半
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

# -----------------------------
# 全国GDP增速约束函数（保留以备需要）
# -----------------------------
def apply_national_growth_constraint(province_predictions, base_year, target_growth_rates):
    """
    应用全国GDP增速约束调整各省份预测值
    province_predictions - 各省份预测GDP的DataFrame (年份为索引，省份为列)
    base_year - 基准年份 (例如 2025)
    target_growth_rates - 目标增速字典，格式: {2026: 0.05, 2027: 0.05, ...}
    返回: 调整后的各省份预测DataFrame, 调整因子, 目标全国GDP序列
    """
    national_gdp = province_predictions.sum(axis=1)

    target_national = pd.Series(index=province_predictions.index, dtype=float)
    target_national[base_year] = national_gdp[base_year]

    for year in sorted(target_growth_rates.keys()):
        growth_rate = target_growth_rates[year]
        target_national[year] = target_national[year - 1] * (1 + growth_rate)

    adjustment_factors = target_national / national_gdp

    adjusted_predictions = province_predictions.copy()
    for province in adjusted_predictions.columns:
        adjusted_predictions[province] = adjusted_predictions[province] * adjustment_factors

    return adjusted_predictions, adjustment_factors, target_national

# =========================================================
# 主程序
# =========================================================

# 1. 加载数据
df = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史GDP数据.csv', encoding='gbk')

# 1.1. 数据清洗和预处理
# 删除备注列（如果存在）
if '备注' in df.columns:
    df = df.drop(columns=['备注'])

# 1.2. 转置数据框
df = df.set_index('地区')          # 省份为索引
df_transposed = df.T               # 省份变列，年份为索引

# 1.3. 处理年份索引
df_transposed.index = df_transposed.index.str.replace('年', '', regex=False).astype(int)
df_transposed = df_transposed.sort_index(ascending=True)

# 1.4. 创建目标格式的DataFrame（保持所有历史年份）
years = df_transposed.index
provinces = df_transposed.columns
data = pd.DataFrame(index=years, columns=provinces)

for year in years:
    for province in provinces:
        data.loc[year, province] = df_transposed.loc[year, province]

# 数值化
data = data.apply(pd.to_numeric, errors='coerce')

# 另存一个转置历史数据（可选）
data.to_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史GDP数据转置.csv', encoding='utf-8-sig')
print("数据格式转换完成！")
print(f"数据形状: {data.shape}")
print(f"历史年份范围: {data.index.min()} - {data.index.max()}")
print(f"省份数量: {len(data.columns)}")

# 2. 定义宏观约束情景
# 假设从外部输入三组不同的全国GDP增长率约束 (2025-2035年)
df_rates = pd.read_excel('/lustre/home/litong/intern/京津冀货运发展/data/constraints/yx/GDP_Projection_CWJ.xlsx', sheet_name='年均增长率')
print("df_rates 的列名:", df_rates.columns.tolist())

scenario_rates = {
    '基准': None,
    '低增长': df_rates.loc[df_rates['年均增长率'] == 'SSP3', np.arange(2025, 2036).tolist()].values.tolist()[0],
    '中增长': df_rates.loc[df_rates['年均增长率'] == 'SSP1', np.arange(2025, 2036).tolist()].values.tolist()[0],
    '高增长': df_rates.loc[df_rates['年均增长率'] == 'SSP5', np.arange(2025, 2036).tolist()].values.tolist()[0]
}

# 3. 核心年份设置
syear = int(data.index.min())     # 历史最早年份
eyear = int(data.index.max())     # 历史最晚年份
history_syear = 2015              # 拟合窗口起年
history_eyear = 2025              # 拟合窗口止年（不含）
predict_syear = 2025
predict_eyear = 2035

# 3.1 保留全部历史年份（data_all），拟合仅使用窗口数据（data_hist）
data_all = data.copy()
fit_years = np.arange(history_syear, history_eyear, 1)
data_hist = data_all.loc[fit_years]  # 用于拟合

# 4. 基准情景预测
base_predictions = pd.DataFrame(index=np.arange(history_syear, predict_eyear + 1))
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
        # 保存预测结果（含 2015-2035 的模型序列）
        base_predictions[province] = result['predicted']

        # 保存参数
        params = result['params']
        params['r_squared'] = result['r_squared']
        params['peak_year'] = result['peak_year']
        params['peak_gdp'] = result['peak_gdp']
        base_params.loc[province] = params

        # ====== 绘图部分修改（优化 2024-2025 连接） ======
        plt.subplot(n_rows, n_cols, i + 1)

        # 1. 历史窗口原始点数据（蓝点，2015-2024）
        plt.plot(province_data.index, province_data, 'bo', markersize=4, label='历史数据点')

        # 2. 拟合曲线（绿线，2015-2024，无缝）
        # 这里用 arange(..., history_eyear + 1) 来确保包含 2024 这一年
        history_years_full = np.arange(history_syear, history_eyear + 1)
        t_history_full = history_years_full - history_syear
        fitted_full = logistic_function(t_history_full, result['params']['K'],
                                       result['params']['P0'],
                                       result['params']['r'])
        plt.plot(history_years_full, fitted_full, 'g-', alpha=0.7, label='历史拟合')

        # 3. 桥梁与预测数据（红线，2024-2035）
        # ====== 关键修改位置 ======
        # 我们创建一个包含 2024 和 2025-2035 的年份序列
        # 注意：np.arange 是不包含终点的，所以用 history_eyear (2024) 到 predict_eyear + 1 (2036)
        future_years_with_bridge = np.arange(history_eyear, predict_eyear + 1)
        
        # 从模型结果中提取对应的预测值
        # 注意：这里需要从 result['predicted'] 提取数值，它里面其实包含了 2024 的值。
        # 如果 result['predicted'] 没有 2024，我们需要用 logistic_function 计算一下
        if history_eyear in result['predicted'].index:
            predicted_with_bridge = result['predicted'].loc[future_years_with_bridge]
        else:
            # 兼容性处理：如果模型输出没有包含2024，现场计算一个模型值作为起点
            bridge_val = logistic_function(np.array([history_eyear - history_syear]), 
                                           result['params']['K'],
                                           result['params']['P0'],
                                           result['params']['r'])
            predicted_with_bridge = pd.concat([pd.Series(bridge_val, index=[history_eyear]), 
                                              result['predicted'].loc[predict_syear:predict_eyear]])
        
        # 绘制红线，起点在 2024
        plt.plot(future_years_with_bridge, predicted_with_bridge.values, 'r-', linewidth=1.5, label='未来预测')

        # ====== 标记拐点（gdp峰值）保持不变 ======
        if (result['peak_year'] is not None) and (history_syear < result['peak_year'] < predict_eyear):
            plt.axvline(x=result['peak_year'], color='gray', linestyle=':', alpha=0.7)
            plt.text(result['peak_year'], result['peak_gdp'] * 0.8,
                     f'峰值: {result["peak_year"]:.0f}\n{result["peak_gdp"]/10000:.1f}亿',
                     fontsize=8, ha='center')

        plt.title(province)
        plt.xlabel('年份')
        plt.ylabel('gdp(亿元)')
        plt.grid(alpha=0.3)

        # 显示R²
        plt.annotate(f'R²={result["r_squared"]:.3f}',
                     xy=(0.05, 0.85), xycoords='axes fraction',
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

plt.tight_layout()
plt.suptitle(f'中国各省gdp预测({history_syear}-{predict_eyear})', fontsize=16, y=1.02)
os.makedirs('/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/1.1GDP总量预测', exist_ok=True)
plt.savefig('/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/1.1GDP总量预测/province_gdp_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# 添加全国与区域汇总（基于预测期列）
bthsa_8prov = ['北京市', '天津市', '河北省', '河南省', '山东省', '山西省', '内蒙古自治区', '陕西省']
base_predictions['全国'] = base_predictions[data_all.columns].sum(axis=1)
base_predictions['京津冀及周边8省市'] = base_predictions[bthsa_8prov].sum(axis=1)

# 5. 生成“基准_完整年份”并保存（全部历史 + 预测）
provinces = list(data_all.columns)
full_index = sorted(set(data_all.index.tolist()) | set(range(predict_syear, predict_eyear + 1)))
full_predictions_base = pd.DataFrame(index=full_index, columns=provinces)

# 填入全部历史年份
full_predictions_base.loc[data_all.index, provinces] = data_all[provinces]

# 覆盖/追加 2025–2035 的预测
full_predictions_base.loc[predict_syear:predict_eyear, provinces] = base_predictions.loc[predict_syear:predict_eyear, provinces]

# 计算汇总列（对可得年份求和）
full_predictions_base['全国'] = full_predictions_base[provinces].sum(axis=1, min_count=1)
full_predictions_base['京津冀及周边8省市'] = full_predictions_base[bthsa_8prov].sum(axis=1, min_count=1)

# 保存基准情景结果（原基准）与参数
all_scenario_predictions = {'基准': base_predictions}
all_scenario_params = {'基准': base_params}

base_dir = '/lustre/home/litong/intern/京津冀货运发展/output/1.宏观经济指标/1.1GDP总量预测'
os.makedirs(base_dir, exist_ok=True)

# 保存“基准_完整年份”
full_path_base = f'{base_dir}/gdp_predictions_基准_完整年份.csv'
full_predictions_base.to_csv(full_path_base, encoding='utf-8-sig')
print(f"已保存 基准 情景完整年份至: {full_path_base}")

# 同时保存你的原基准（仅2015–2035模型行）与参数，保持原有输出
base_predictions.to_csv(f'{base_dir}/gdp_predictions_基准.csv', encoding='utf-8-sig')
base_params.to_csv(f'{base_dir}/gdp_params_基准.csv', encoding='utf-8-sig')

# 6. 约束情景预测（低/中/高）+ “完整年份”输出
for scenario_name, growth_rates in scenario_rates.items():
    if scenario_name == '基准':
        continue

    print("=" * 50)
    print(f"正在进行 {scenario_name} 情景预测...")
    print(f"增长率约束: {growth_rates}")
    print("=" * 50)

    # 创建当前情景的预测容器（从基准拷贝）
    scenario_predictions = base_predictions.copy()
    scenario_params = base_params.copy()

    # 基准情景全国GDP
    base_national_gdp = base_predictions['全国'].copy()

    # 以 2024 年为基准
    base_2024 = base_national_gdp.loc[2024]

    # 计算约束情景全国GDP路径（2025–2035）
    constrained_national = pd.Series(index=np.arange(2025, 2036), dtype=float)
    current_gdp = base_2024
    for year, growth_rate in zip(range(2025, 2036), growth_rates):
        current_gdp *= (1 + growth_rate)
        constrained_national.loc[year] = current_gdp

    # 计算调整因子 = 约束全国GDP / 基准全国GDP
    adjustment_factors = constrained_national / base_national_gdp.loc[2025:2035]

    # 应用调整因子到各省预测（仅 2025–2035）
    for year in range(2025, 2036):
        factor = adjustment_factors.loc[year]
        for province in data_all.columns:
            scenario_predictions.loc[year, province] = base_predictions.loc[year, province] * factor

    # 重新计算全国与区域汇总
    scenario_predictions['全国'] = scenario_predictions[data_all.columns].sum(axis=1)
    scenario_predictions['京津冀及周边8省市'] = scenario_predictions[bthsa_8prov].sum(axis=1)

    # 标记参数
    scenario_params['scenario'] = scenario_name
    scenario_params['growth_constraint'] = str(growth_rates)

    # 存储
    all_scenario_predictions[scenario_name] = scenario_predictions
    all_scenario_params[scenario_name] = scenario_params

    # 绘制情景对比图（全国）
    plt.figure(figsize=(12, 8))
    plt.plot(base_predictions.index, base_predictions['全国'] / 1e4, 'b-', label='基准情景')
    plt.plot(scenario_predictions.index, scenario_predictions['全国'] / 1e4, 'r--', label=f'{scenario_name}情景')
    plt.title(f'全国GDP预测对比: 基准 vs {scenario_name}')
    plt.xlabel('年份')
    plt.ylabel('全国GDP(万亿元)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{base_dir}/national_gdp_comparison_{scenario_name}.png')
    plt.show()

    # ===== 生成“完整年份”并保存 =====
    full_predictions_scn = pd.DataFrame(index=full_index, columns=provinces)
    # 历史全部年份
    full_predictions_scn.loc[data_all.index, provinces] = data_all[provinces]
    # 预测 2025–2035（使用情景调整后的）
    full_predictions_scn.loc[predict_syear:predict_eyear, provinces] = scenario_predictions.loc[predict_syear:predict_eyear, provinces]
    # 汇总列
    full_predictions_scn['全国'] = full_predictions_scn[provinces].sum(axis=1, min_count=1)
    full_predictions_scn['京津冀及周边8省市'] = full_predictions_scn[bthsa_8prov].sum(axis=1, min_count=1)

    # 保存情景“完整年份”
    scn_full_path = f'{base_dir}/gdp_predictions_{scenario_name}_完整年份.csv'
    full_predictions_scn.to_csv(scn_full_path, encoding='utf-8-sig')
    print(f"已保存 {scenario_name} 情景完整年份至: {scn_full_path}")

    # 也保存情景（模型期）与参数
    scenario_predictions.to_csv(f'{base_dir}/gdp_predictions_{scenario_name}.csv', encoding='utf-8-sig')
    scenario_params.to_csv(f'{base_dir}/gdp_params_{scenario_name}.csv', encoding='utf-8-sig')

# 7. 多情景对比分析（全国与2035分省）
# 多情景全国对比（仅 2025–2035）
plt.figure(figsize=(14, 8))
for scenario_name, predictions in all_scenario_predictions.items():
    gdp_data = predictions.loc[2025:2035, '全国'] / 1e4  # 万亿元
    plt.plot(gdp_data.index, gdp_data.values, label=scenario_name)
plt.title('多情景全国GDP预测对比 (2025-2035)', fontsize=16)
plt.xlabel('年份', fontsize=12)
plt.ylabel('GDP (万亿元)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{base_dir}/national_gdp_scenarios_comparison.png', dpi=300)
plt.show()

# 2035年各省GDP多情景对比
province_gdp_2035 = pd.DataFrame()
for scenario_name, predictions in all_scenario_predictions.items():
    province_gdp_2035[scenario_name] = predictions.loc[2035, data_all.columns] 

province_gdp_2035 = province_gdp_2035.sort_values(by='基准', ascending=False)

plt.figure(figsize=(16, 10))
province_gdp_2035.plot(kind='bar', width=0.8)
plt.title('2035年各省GDP预测多情景对比', fontsize=16)
plt.xlabel('省份', fontsize=12)
plt.ylabel('GDP (亿元)', fontsize=12)
plt.xticks(rotation=80, fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{base_dir}/province_gdp_2035_comparison.png', dpi=300)
plt.show()

print("所有情景预测完成！")
