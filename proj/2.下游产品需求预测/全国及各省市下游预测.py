# -*- coding: utf-8 -*-
"""
1. 各省份及区域（全国、京津冀及周边8省市）各指标的需求变化图
   - 历史数据：均来源于分省统计数据文件，确保包含2025年前完整数据
   - 平滑后曲线：单独曲线
   - 预测趋势：分别基于原始数据和平滑数据，用不同颜色曲线
2. 每个指标的区域总览图
3. 保存所有处理后的数据到CSV
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


# 1. σ原则去除离散值
def remove_outliers_by_sigma(data, factor_name, n_sigma=1):
    mean = data[factor_name].mean()
    std = data[factor_name].std()
    lower_bound = mean - n_sigma * std
    upper_bound = mean + n_sigma * std
    filtered_data = data[(data[factor_name] >= lower_bound) & 
                         (data[factor_name] <= upper_bound)]
    outlier_count = len(data) - len(filtered_data)
    if outlier_count > 0:
        print(f"使用{n_sigma}σ原则去除了{outlier_count}个异常值（{data['地区'].iloc[0]} - {factor_name}）")
    return filtered_data


# 2. 移动平均法平滑数据
def smooth_data_by_moving_average(data, factor_name, window_size=2):
    data_sorted = data.sort_values('年份').copy()
    data_sorted[f'smoothed_{factor_name}'] = data_sorted[factor_name].rolling(
        window=window_size, min_periods=1, center=False
    ).mean()
    return data_sorted


# 3. 数据加载与预处理（分省）
def load_and_preprocess_data(fi_predict_factor, factor_name):
    # 加载历史数据
    gdp = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史GDP数据.csv', encoding='gbk')
    gdp = gdp.iloc[:, :-1].melt(id_vars='地区', var_name='年份', value_name='GDP')
    gdp['年份'] = gdp['年份'].str.replace('年', '').astype(int)

    pop = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史人口数据.csv', encoding='gbk')
    pop = pop.iloc[:, :-1].melt(id_vars='地区', var_name='年份', value_name='人口')
    pop['年份'] = pop['年份'].str.replace('年', '').astype(int)

    third_gdp = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/第三产业增加值分省年度数据.csv', encoding='gbk')
    third_gdp = third_gdp.melt(id_vars='地区', var_name='年份', value_name='第三产业GDP')
    third_gdp['年份'] = third_gdp['年份'].str.replace('年', '').astype(int)

    urban_pop = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/城镇人口分省年度数据.csv', encoding='gbk')
    urban_pop = urban_pop.melt(id_vars='地区', var_name='年份', value_name='城镇人口')
    urban_pop['年份'] = urban_pop['年份'].str.replace('年', '').astype(int)

    prod_pre = pd.read_excel(fi_predict_factor, sheet_name='分省年度数据', skiprows=3)
    prod_pre = prod_pre.melt(id_vars='地区', var_name='年份', value_name=factor_name)
    prod_pre['年份'] = prod_pre['年份'].str.replace('年', '').astype(int)

    # 合并数据
    merged = pd.merge(gdp, pop, on=['地区', '年份'])
    merged = pd.merge(merged, third_gdp, on=['地区', '年份'])
    merged = pd.merge(merged, urban_pop, on=['地区', '年份'])
    merged = pd.merge(merged, prod_pre, on=['地区', '年份'])
    merged = merged.dropna(subset=['GDP', '人口', '第三产业GDP', '城镇人口', factor_name])
    
    # 按省份处理
    processed_data = []
    for province, group in merged.groupby('地区'):
        group_cleaned = remove_outliers_by_sigma(group, factor_name)
        group_smoothed = smooth_data_by_moving_average(group_cleaned, factor_name)
        processed_data.append(group_smoothed)
    
    final_data = pd.concat(processed_data).reset_index(drop=True)
    return final_data


# 4. 从分省数据汇总生成区域历史数据
def aggregate_regional_historical_data(provincial_data, region, factor_name):
    """
    从分省历史数据汇总生成区域历史数据
    与省份数据来源一致，确保2025年前数据完整
    """
    # 定义区域包含的省份
    region_provinces = {
        '全国': provincial_data['地区'].unique(),  # 全国包含所有省份
        '京津冀及周边8省市': ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', 
                             '山东省', '河南省', '辽宁省', '陕西省']
    }
    
    if region not in region_provinces:
        raise ValueError(f"不支持的区域: {region}")
    
    # 筛选区域内的省份数据
    region_province_list = region_provinces[region]
    region_data = provincial_data[provincial_data['地区'].isin(region_province_list)].copy()
    
    # 按年份汇总区域数据
    aggregated = region_data.groupby('年份').agg({
        'GDP': 'sum',
        '人口': 'sum',
        '第三产业GDP': 'sum',
        '城镇人口': 'sum',
        factor_name: 'sum',
        f'smoothed_{factor_name}': 'sum'
    }).reset_index()
    
    # 添加地区列标识
    aggregated['地区'] = region
    
    # 打印汇总信息
    pre_2025_data = aggregated[aggregated['年份'] < 2025]
    print(f"生成{region}历史数据: 共{len(aggregated)}条记录，其中2025年前{len(pre_2025_data)}条，年份范围: {aggregated['年份'].min()} - {aggregated['年份'].max()}")
    
    return aggregated


# 5. 加载未来特征数据 - 按省份加载
def load_future_province_data(gdp_future_file, pop_future_file, urban_pop_future_file, third_gdp_future_file):
    """加载各省未来特征数据"""
    # 处理GDP未来数据
    gdp_future = pd.read_csv(gdp_future_file)
    gdp_future = gdp_future.rename(columns={'Unnamed: 0': '年份'})
    # 转换为长格式
    gdp_long = gdp_future.melt(id_vars='年份', var_name='地区', value_name='GDP')
    
    # 处理人口未来数据
    pop_future = pd.read_csv(pop_future_file)
    pop_future = pop_future.rename(columns={'Unnamed: 0': '年份'})
    pop_long = pop_future.melt(id_vars='年份', var_name='地区', value_name='人口')
    
    # 处理第三产业GDP未来数据
    third_gdp_future = pd.read_csv(third_gdp_future_file)
    third_gdp_future = third_gdp_future.rename(columns={'Unnamed: 0': '年份'})
    third_gdp_long = third_gdp_future.melt(id_vars='年份', var_name='地区', value_name='第三产业GDP')
    
    # 处理城镇人口未来数据
    urban_pop_future = pd.read_csv(urban_pop_future_file)
    urban_pop_future = urban_pop_future.rename(columns={'Unnamed: 0': '年份'})
    urban_pop_long = urban_pop_future.melt(id_vars='年份', var_name='地区', value_name='城镇人口')
    
    # 合并所有未来数据
    future_data = pd.merge(gdp_long, pop_long, on=['地区', '年份'])
    future_data = pd.merge(future_data, third_gdp_long, on=['地区', '年份'])
    future_data = pd.merge(future_data, urban_pop_long, on=['地区', '年份'])
    
    # 只保留2025年及以后的数据
    future_data = future_data[future_data['年份'] >= 2025]
    
    return future_data


# 6. 模型拟合函数
def fit_model(X, y, max_degree=3, min_samples=3):
    """调整模型拟合函数，允许更少的数据点"""
    best_score = -np.inf
    best_degree = 1
    best_model = None
    
    # 根据样本量调整最大阶数，避免过拟合
    max_possible_degree = min(max_degree, len(X) - 1) if len(X) > 1 else 1
    
    for degree in range(1, max_possible_degree + 1):
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
        # 根据样本量调整交叉验证折数
        cv_folds = min(5, len(X)) if len(X) >= 2 else 1
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        avg_score = np.mean(scores)

        if avg_score > best_score:
            best_score = avg_score
            best_degree = degree
            best_model = model

    best_model.fit(X, y)
    return best_model, best_degree, best_score


# 7. 省份预测函数
def predict_future_province(historical_data, future_data, factor_name):
    """对每个省份进行预测，降低数据量要求"""
    results = []
    all_predictions = []
    
    # 获取所有省份列表
    provinces = historical_data['地区'].unique()
    print(f"发现{len(provinces)}个省份数据，开始处理...")
    
    for province in provinces:
        # 筛选当前省份数据
        province_data = historical_data[historical_data['地区'] == province].copy()
        
        # 降低数据量要求，从5个降到3个
        if len(province_data) < 3:
            print(f"{province}数据点不足（{len(province_data)}个），无法预测，将跳过")
            continue
        
        # 准备特征数据
        X = province_data[['GDP', '人口', '第三产业GDP', '城镇人口']]
        
        # 1. 基于原始数据的模型
        y_original = province_data[factor_name]
        model_original, degree_original, score_original = fit_model(X, y_original)
        
        # 2. 基于平滑数据的模型
        y_smoothed = province_data[f'smoothed_{factor_name}']
        model_smoothed, degree_smoothed, score_smoothed = fit_model(X, y_smoothed)

        # 获取该省份的未来数据
        province_future = future_data[future_data['地区'] == province].copy()
        if province_future.empty:
            print(f"{province}没有未来数据，尝试创建基础未来数据...")
            # 尝试为没有未来数据的省份创建基础数据
            if not province_data.empty:
                last_year = province_data['年份'].max()
                future_years = pd.DataFrame({'年份': range(2025, 2046)})  # 假设预测到2045年
                future_years['地区'] = province
                # 使用最后一年的数据作为基础
                last_data = province_data[province_data['年份'] == last_year].iloc[0]
                for col in ['GDP', '人口', '第三产业GDP', '城镇人口']:
                    future_years[col] = last_data[col] * (1.02 **(future_years['年份'] - last_year))  # 2%增长
                province_future = future_years
            
        if province_future.empty:
            print(f"{province}无法创建未来数据，跳过")
            continue
            
        X_future = province_future[['GDP', '人口', '第三产业GDP', '城镇人口']]
        
        # 生成两种预测
        province_future['预测_原始数据_' + factor_name] = model_original.predict(X_future)
        province_future['预测_平滑数据_' + factor_name] = model_smoothed.predict(X_future)
        
        all_predictions.append(province_future)

        # 记录模型结果
        results.append({
            '省份': province,
            '数据点数': len(province_data),
            '原始数据_R²': model_original.score(X, y_original),
            '原始数据_最佳阶数': degree_original,
            '平滑数据_R²': model_smoothed.score(X, y_smoothed),
            '平滑数据_最佳阶数': degree_smoothed
        })

    if not all_predictions:
        return pd.DataFrame(), pd.DataFrame()
        
    return pd.concat(all_predictions), pd.DataFrame(results)


# 8. 区域预测汇总函数（从省份预测结果汇总）
def aggregate_regional_predictions(provincial_predictions, factor_name):
    """从省份预测结果汇总得到区域预测数据"""
    # 京津冀及周边8省市的省份列表
    jingjinji_regions = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', 
                         '山东省', '河南省', '辽宁省', '陕西省']
    
    # 全国预测数据：所有省份预测结果汇总
    national_pred = provincial_predictions.groupby('年份').agg({
        'GDP': 'sum',
        '人口': 'sum',
        '第三产业GDP': 'sum',
        '城镇人口': 'sum',
        f'预测_原始数据_{factor_name}': 'sum',
        f'预测_平滑数据_{factor_name}': 'sum'
    }).reset_index()
    national_pred['地区'] = '全国'
    
    # 京津冀及周边8省市预测数据
    jjj_pred = provincial_predictions[provincial_predictions['地区'].isin(jingjinji_regions)].groupby('年份').agg({
        'GDP': 'sum',
        '人口': 'sum',
        '第三产业GDP': 'sum',
        '城镇人口': 'sum',
        f'预测_原始数据_{factor_name}': 'sum',
        f'预测_平滑数据_{factor_name}': 'sum'
    }).reset_index()
    jjj_pred['地区'] = '京津冀及周边8省市'
    
    return {
        '全国': national_pred,
        '京津冀及周边8省市': jjj_pred
    }


# 9. 确保预测值非负
def ensure_non_negative(predictions, factor_name):
    """将所有负数预测值修正为0，保证指标预测值非负"""
    predictions = predictions.copy()
    # 处理基于原始数据的预测列
    original_col = f'预测_原始数据_{factor_name}'
    if original_col in predictions.columns:
        predictions[original_col] = predictions[original_col].apply(lambda x: max(x, 0))
    
    # 处理基于平滑数据的预测列
    smoothed_col = f'预测_平滑数据_{factor_name}'
    if smoothed_col in predictions.columns:
        predictions[smoothed_col] = predictions[smoothed_col].apply(lambda x: max(x, 0))
    
    return predictions


# 10. 绘制区域/省份指标的需求变化图
def plot_region_indicator(historical, predictions, factor_name, region, save_path, is_province=False):
    """绘制指定区域或省份指标趋势图，确保显示2025年前的历史数据"""
    if historical.empty:
        print(f"警告: {region}没有历史数据，无法绘制趋势图")
        return
        
    # 确保历史数据按年份排序
    hist_region = historical.sort_values('年份').copy()
    pred_region = predictions.sort_values('年份').copy() if not predictions.empty else pd.DataFrame()

    # 筛选2025年前的历史数据并打印信息，确认数据存在
    pre_2025_history = hist_region[hist_region['年份'] < 2025]
    if len(pre_2025_history) > 0:
        print(f"绘制{region}趋势图，包含{len(pre_2025_history)}条2025年前的历史数据，年份范围: {pre_2025_history['年份'].min()} - {pre_2025_history['年份'].max()}")
    else:
        print(f"警告: {region}历史数据中没有2025年前的记录")

    # 定义单位
    if factor_name == '新增建筑面积':
        unit = '（万平方米）'
    elif factor_name == '汽车产量':
        unit = '（万辆）'
    else:
        unit = ''

    # 过渡点处理（确保历史数据与预测数据平滑连接）
    original_pred_col = f'预测_原始数据_{factor_name}'
    smoothed_pred_col = f'预测_平滑数据_{factor_name}'
    if not pred_region.empty and 2025 in pred_region['年份'].values:
        pred_2025 = pred_region[pred_region['年份'] == 2025].iloc[0]
        transition_data = {
            '地区': [region],
            '年份': [2025],
            factor_name: [pred_2025[original_pred_col] if original_pred_col in pred_2025 else np.nan],
            f'smoothed_{factor_name}': [pred_2025[smoothed_pred_col] if smoothed_pred_col in pred_2025 else np.nan]
        }
        transition_point = pd.DataFrame(transition_data)
        hist_region = pd.concat([hist_region, transition_point], ignore_index=True).sort_values('年份')

    plt.figure(figsize=(14, 8))

    # 绘制原始历史数据
    plt.scatter(hist_region['年份'], hist_region[factor_name], color='blue', alpha=0.7, label='原始历史数据（散点）')
    plt.plot(hist_region['年份'], hist_region[factor_name], 'b-', alpha=0.5, label='原始历史数据（折线）')
    
    # 绘制平滑后的数据曲线
    if f'smoothed_{factor_name}' in hist_region.columns:
        plt.plot(hist_region['年份'], hist_region[f'smoothed_{factor_name}'], 'g-', linewidth=2, label='平滑后历史数据')
    
    # 绘制两种预测曲线
    if not pred_region.empty:
        pred_region = pred_region[pred_region['年份'] >= 2025]
        plt.plot(pred_region['年份'], pred_region[original_pred_col], 'r--', linewidth=2, label='基于原始数据的预测')
        plt.plot(pred_region['年份'], pred_region[smoothed_pred_col], 'purple', linestyle='-.', linewidth=2, label='基于平滑数据的预测')

    # 图表设置
    title_prefix = "省份" if is_province else "区域"
    plt.title(f'{title_prefix} {region}{factor_name}历史趋势与预测', fontsize=16, pad=15)
    plt.xlabel('年份', fontsize=14)
    plt.ylabel(f'{factor_name}{unit}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # 强制包含2010-2025年的区间，确保历史数据可见
    all_years = hist_region['年份']
    if not pred_region.empty:
        all_years = pd.concat([all_years, pred_region['年份']])
    
    # 计算合适的x轴范围，确保包含2025年前的历史数据
    min_year = min(all_years.min(), 2010)
    max_year = all_years.max()
    plt.xlim(min_year - 1, max_year + 1)
    
    # 添加垂直线标记2025年（历史与预测的分界点）
    plt.axvline(x=2025, color='gray', linestyle=':', linewidth=1.5, label='2025年（预测起始点）')
    
    plt.tight_layout()
    # 创建省份和区域的不同保存目录
    if is_province:
        save_dir = os.path.join(save_path, '省份指标趋势图')
    else:
        save_dir = os.path.join(save_path, '区域指标趋势图')
    os.makedirs(save_dir, exist_ok=True)
    
    save_file = f"{save_dir}/{region}_{factor_name}_趋势与预测.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存{region}趋势图: {save_file}")


# 11. 保存完整数据到CSV（修改版）
def save_complete_data(historical, predictions, factor_name, region, save_path, is_province=False):
    # 处理历史数据
    historical_output = historical[['地区', '年份', 'GDP', '人口', 
                                  '第三产业GDP', '城镇人口', 
                                  factor_name, f'smoothed_{factor_name}']].copy() if not historical.empty else pd.DataFrame()
    if not historical_output.empty:
        # 重命名历史数据列，与预测数据对应
        historical_output.rename(columns={
            factor_name: f'原始_{factor_name}',
            f'smoothed_{factor_name}': f'平滑_{factor_name}'
        }, inplace=True)
        # 添加预测列（历史数据无预测值，用NaN填充）
        historical_output[f'预测_原始数据_{factor_name}'] = np.nan
        historical_output[f'预测_平滑数据_{factor_name}'] = np.nan
    
    # 处理预测数据
    predictions_output = predictions[['地区', '年份', 'GDP', '人口', 
                                    '第三产业GDP', '城镇人口', 
                                    f'预测_原始数据_{factor_name}', 
                                    f'预测_平滑数据_{factor_name}']].copy() if not predictions.empty else pd.DataFrame()
    if not predictions_output.empty:
        # 添加历史数据列（预测数据无原始历史值，用NaN填充）
        predictions_output[f'原始_{factor_name}'] = np.nan
        predictions_output[f'平滑_{factor_name}'] = np.nan
    
    # 合并历史数据和预测数据
    dfs_to_concat = []
    if not historical_output.empty:
        dfs_to_concat.append(historical_output)
    if not predictions_output.empty:
        dfs_to_concat.append(predictions_output)
        
    if dfs_to_concat:
        full_output = pd.concat(dfs_to_concat, ignore_index=True)
        # 按地区和年份排序，确保时间序列连续
        full_output.sort_values(by=['地区', '年份'], inplace=True)
        
        # 创建保存目录
        if is_province:
            save_dir = os.path.join(save_path, '省份完整数据')
        else:
            save_dir = os.path.join(save_path, '区域完整数据')
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存合并后的数据（包含历史+预测）
        save_file = f"{save_dir}/{region}{factor_name}完整数据.csv"
        full_output.to_csv(save_file, index=False)
        print(f"已保存完整数据（含历史+预测）: {save_file}")


# 12. 其他辅助函数
def calculate_feature_importance(model, feature_names):
    if hasattr(model, 'named_steps'):
        coefficients = model.named_steps['linearregression'].coef_
        poly_features = model.named_steps['polynomialfeatures']
        feature_names = poly_features.get_feature_names_out(feature_names)
    else:
        coefficients = model.coef_
    
    importance = np.abs(coefficients)
    importance = importance / np.sum(importance)
    
    return pd.DataFrame({
        '特征': feature_names,
        '重要性': importance
    }).sort_values('重要性', ascending=False)


def plot_feature_importance(importance_df, region, factor_name, save_path, is_province=False):
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['特征'], importance_df['重要性'], color='skyblue')
    plt.xlabel('相对重要性（归一化）')
    title_prefix = "省份" if is_province else "区域"
    plt.title(f'{title_prefix} {region} {factor_name} 预测因素重要性', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if is_province:
        save_dir = os.path.join(save_path, '省份特征重要性图')
    else:
        save_dir = os.path.join(save_path, '区域特征重要性图')
    os.makedirs(save_dir, exist_ok=True)
    
    save_file = f"{save_dir}/{region}_{factor_name}_特征重要性.png"
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"已保存特征重要性图: {save_file}")


def plot_factor_trends(historical_data, factor_name, region, save_path, is_province=False):
    if historical_data.empty:
        print(f"警告: {region}没有历史数据，无法绘制因素趋势图")
        return
        
    features = ['GDP', '人口', '第三产业GDP', '城镇人口']
    target = factor_name
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if feature not in historical_data.columns:
            continue
            
        ax1 = axes[i]
        ax1.plot(historical_data['年份'], historical_data[feature], 'b-', linewidth=2, label=feature)
        ax1.set_xlabel('年份')
        ax1.set_ylabel(feature, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(historical_data['年份'], historical_data[target], 'r--', linewidth=2, label=target)
        ax2.set_ylabel(target, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title(f'{feature} 与 {target} 趋势对比')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    
    if is_province:
        save_dir = os.path.join(save_path, '省份因素趋势图')
    else:
        save_dir = os.path.join(save_path, '区域因素趋势图')
    os.makedirs(save_dir, exist_ok=True)
    
    save_file = f"{save_dir}/{region}_{factor_name}_因素趋势对比.png"
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"已保存因素趋势对比图: {save_file}")


def calculate_marginal_contributions(model, X_mean, feature_names, factor_name):
    if hasattr(model, 'named_steps'):
        coefficients = model.named_steps['linearregression'].coef_
        poly_features = model.named_steps['polynomialfeatures']
        feature_names = poly_features.get_feature_names_out(feature_names)
    else:
        coefficients = model.coef_
    
    contributions = pd.DataFrame({
        '特征': feature_names,
        '系数': coefficients,
        '均值': X_mean,
        '边际贡献': coefficients * X_mean
    })
    
    total_contribution = np.sum(np.abs(contributions['边际贡献']))
    contributions['相对贡献'] = np.abs(contributions['边际贡献']) / total_contribution
    
    return contributions.sort_values('相对贡献', ascending=False)


# 主程序
if __name__ == "__main__":
    # 路径设置
    fipath = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/'
    fopath = '/lustre/home/litong/intern/京津冀货运发展/output/predict_results/Sce_Pop_BAU_GDP_BAU/5/'
    
    # 创建保存目录
    os.makedirs(fopath, exist_ok=True)

    # ====================== 新增：历史年份范围设置 ======================
    # 调整这里的年份值来控制用于预测的历史数据范围
    HISTORY_START_YEAR = 2015  # 历史数据起始年份
    HISTORY_END_YEAR = 2024    # 历史数据结束年份（需早于2025）
    # 检查年份有效性
    if HISTORY_END_YEAR >= 2025:
        raise ValueError("历史数据结束年份必须早于2025年")
    if HISTORY_START_YEAR >= HISTORY_END_YEAR:
        raise ValueError("起始年份必须早于结束年份")
    # ==================================================================


    # 定义要处理的区域
    regions = ['全国', '京津冀及周边8省市']
    
    # 指标配置
    fi_predict_factor_list = [
        '建筑业房屋竣工面积', '汽车产量', '新增铁路营业里程', 
        '家用洗衣机产量', '家用电冰箱产量', '房间空调器',
        '新增高速等级公路里程', '新增除高速外等级公路里程', 
        '农村水电建设本年完成投资额'
    ]
    predict_factor_list = [
        '新增建筑面积', '汽车产量',
        '新增铁路里程', 
        '洗衣机产量', '电冰箱产量', '空调产量',
        '新增高速公路里程', '新增其他等级公路里程', 
        '农村水电基础设施投资'
    ]
    
    # 未来数据文件
    fipath_future = '/lustre/home/litong/intern/京津冀货运发展/output/scenarios'
    gdp_future_file = f"{fipath_future}/gdp_predictions_基准.csv"
    pop_future_file = f"{fipath_future}/population_predictions_基准.csv"
    urban_pop_future_file = f"{fipath_future}/urban_population_predictions_基准.csv"
    third_gdp_future_file = f"{fipath_future}/third_gdp_predictions_基准.csv"
    
    # 处理每个指标
    for ip_, factor_name in enumerate(predict_factor_list):
        print(f'\n{"="*30} 处理指标: {factor_name} {"="*30}')
        
        # 输入文件路径
        fi_predict_factor = f"{fipath}{fi_predict_factor_list[ip_]}分省年度数据.xls"
        
        # 1. 加载并预处理历史数据（分省）- 省份和区域数据的共同来源
        print("加载历史数据并进行预处理...")
        historical_provincial = load_and_preprocess_data(fi_predict_factor, factor_name)
        
        #年份：＞=2010年
        historical_provincial = historical_provincial[historical_provincial['年份'] >= 2010].reset_index(drop=True)

        # ====================== 修改：按新年份范围筛选 ======================
        # 保留指定年份范围内的数据（替代原有的仅保留>=2010年的逻辑）
        # historical_provincial = historical_provincial[
        #     (historical_provincial['年份'] >= HISTORY_START_YEAR) & 
        #     (historical_provincial['年份'] <= HISTORY_END_YEAR)
        # ].reset_index(drop=True)
        
        # 检查筛选后的数据量
        total_years = historical_provincial['年份'].nunique()
        print(f"筛选后历史数据年份范围: {HISTORY_START_YEAR}-{HISTORY_END_YEAR}，共{total_years}个年份")
        
        # 检查是否有足够数据（至少3个年份，否则模型无法拟合）
        if len(historical_provincial) < 3:
            print(f"警告：筛选后的数据量不足3条，无法进行预测，跳过该指标")
            continue
        # ==================================================================

        
        # 2. 加载未来特征数据（分省）
        print("加载未来特征数据...")
        future_provincial_data = load_future_province_data(
            gdp_future_file, pop_future_file, 
            urban_pop_future_file, third_gdp_future_file
        )
        
        # 3. 对每个省份进行预测
        print("开始对各省进行预测...")
        provincial_predictions, model_results = predict_future_province(
            historical_provincial, future_provincial_data, factor_name
        )
        
        if provincial_predictions.empty:
            print(f"没有有效省份预测结果，跳过该指标: {factor_name}")
            continue
        
        # 4. 预测后校准
        def calibrate_predictions(predictions, factor_name, region_type='省份', max_growth=0.05):
            predictions = predictions.copy()
            # 按省份校准
            for region in predictions['地区'].unique():
                region_preds = predictions[predictions['地区'] == region].copy()
                hist_region = historical_provincial[historical_provincial['地区'] == region]
                
                if not hist_region.empty:
                    base_year = hist_region['年份'].max()  # 历史最后一年
                    if base_year in hist_region['年份'].values:
                        base_value = hist_region[hist_region['年份'] == base_year][f'smoothed_{factor_name}'].values[0]
                        
                        # 遍历预测年份，计算合理上限并校准
                        for idx, row in region_preds.iterrows():
                            year = row['年份']
                            years_since_base = year - base_year
                            upper_bound = base_value * (1 + max_growth) ** years_since_base
                            pred_idx = predictions[(predictions['地区'] == region) & (predictions['年份'] == year)].index
                            if not pred_idx.empty:
                                predictions.at[pred_idx[0], f'预测_平滑数据_{factor_name}'] = min(
                                    predictions.at[pred_idx[0], f'预测_平滑数据_{factor_name}'], 
                                    upper_bound
                                )
            return predictions
        
        # 调用校准函数
        provincial_predictions = calibrate_predictions(provincial_predictions, factor_name)
        
        # 5. 确保预测值非负
        provincial_predictions = ensure_non_negative(provincial_predictions, factor_name)
        
        # 6. 保存省份模型结果
        model_results.to_csv(f"{fopath}/各省{factor_name}模型结果.csv", index=False)
        
        # 7. 处理每个省份的可视化和数据保存
        print("处理各省预测结果...")
        # 获取所有有历史数据的省份
        all_provinces_with_data = historical_provincial['地区'].unique()
        for province in all_provinces_with_data:
            # 获取该省份的历史数据
            hist_prov = historical_provincial[historical_provincial['地区'] == province].copy()
            
            # 检查是否有预测数据
            if province in provincial_predictions['地区'].unique():
                pred_prov = provincial_predictions[provincial_predictions['地区'] == province].copy()
            else:
                print(f"{province}没有预测数据，仅绘制历史数据图")
                pred_prov = pd.DataFrame(columns=['地区', '年份', 'GDP', '人口', '第三产业GDP', 
                                                 '城镇人口', f'预测_原始数据_{factor_name}', 
                                                 f'预测_平滑数据_{factor_name}'])
            
            # 绘制省份需求变化图
            plot_region_indicator(hist_prov, pred_prov, factor_name, province, fopath, is_province=True)
            
            # 保存省份完整数据
            save_complete_data(hist_prov, pred_prov, factor_name, province, fopath, is_province=True)
        
        # 8. 从省份数据汇总得到区域历史数据和预测数据
        print("从省份数据汇总区域历史数据和预测数据...")
        regional_predictions = aggregate_regional_predictions(provincial_predictions, factor_name)
        
        # 9. 处理每个区域的可视化和数据保存
        for region in regions:
            print(f"\n----- 处理区域: {region} -----")
            # 从分省历史数据汇总生成区域历史数据（与省份数据来源一致）
            region_hist = aggregate_regional_historical_data(historical_provincial, region, factor_name)
            
            # 获取该区域的预测数据
            region_pred = regional_predictions.get(region, pd.DataFrame())
            
            # 绘制区域需求变化图（历史数据和预测数据均来自分省数据汇总）
            plot_region_indicator(region_hist, region_pred, factor_name, region, fopath, is_province=False)
            
            # 保存区域完整数据
            save_complete_data(region_hist, region_pred, factor_name, region, fopath, is_province=False)
                
    print("\n所有指标、省份和区域处理完成！")
