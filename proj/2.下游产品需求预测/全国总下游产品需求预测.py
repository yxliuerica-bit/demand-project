# -*- coding: utf-8 -*-
"""
1. 各区域（全国、京津冀及周边8省市）各指标的需求变化图：
   - 历史数据：散点+折线
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


# 3. 数据加载与预处理
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


# 4. 加载未来特征数据 - 修改为分别提取全国和京津冀及周边8省市数据
def load_future_data(gdp_future_file, pop_future_file, urban_pop_future_file, third_gdp_future_file):
    """
    加载未来特征数据，分别提取全国和京津冀及周边8省市两列数据
    返回一个字典，包含两个区域的数据
    """
    # 定义要提取的区域
    regions = ['全国', '京津冀及周边8省市']
    future_data = {}
    
    # 处理GDP未来数据
    gdp_future = pd.read_csv(gdp_future_file)
    gdp_future = gdp_future.rename(columns={'Unnamed: 0': '年份'})
    # 只保留需要的列和年份
    for region in regions:
        future_data[region] = gdp_future[['年份', region]].rename(columns={region: 'GDP'})
        future_data[region] = future_data[region][future_data[region]['年份'] >= 2025]
    
    # 处理人口未来数据
    pop_future = pd.read_csv(pop_future_file)
    pop_future = pop_future.rename(columns={'Unnamed: 0': '年份'})
    for region in regions:
        pop_df = pop_future[['年份', region]].rename(columns={region: '人口'})
        future_data[region] = pd.merge(future_data[region], pop_df, on='年份')
    
    # 处理第三产业GDP未来数据
    third_gdp_future = pd.read_csv(third_gdp_future_file)
    third_gdp_future = third_gdp_future.rename(columns={'Unnamed: 0': '年份'})
    for region in regions:
        third_gdp_df = third_gdp_future[['年份', region]].rename(columns={region: '第三产业GDP'})
        future_data[region] = pd.merge(future_data[region], third_gdp_df, on='年份')
    
    # 处理城镇人口未来数据
    urban_pop_future = pd.read_csv(urban_pop_future_file)
    urban_pop_future = urban_pop_future.rename(columns={'Unnamed: 0': '年份'})
    for region in regions:
        urban_pop_df = urban_pop_future[['年份', region]].rename(columns={region: '城镇人口'})
        future_data[region] = pd.merge(future_data[region], urban_pop_df, on='年份')
        # 添加地区列标识
        future_data[region]['地区'] = region
    
    return future_data


# 5. 模型拟合函数
def fit_model(X, y, max_degree=3):
    best_score = -np.inf
    best_degree = 1
    best_model = None

    for degree in range(1, max_degree + 1):
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        avg_score = np.mean(scores)

        if avg_score > best_score:
            best_score = avg_score
            best_degree = degree
            best_model = model

    best_model.fit(X, y)
    return best_model, best_degree, best_score


# 6. 区域预测函数
def predict_future_region(historical_data, future_data, factor_name, region):
    """对指定区域进行预测"""
    results = []
    all_predictions = []

    # 使用指定区域的数据
    data = historical_data.copy()
    if len(data) < 5:
        print(f"{region}数据点不足，无法预测")
        return pd.DataFrame(), pd.DataFrame()

    # 准备特征数据
    X = data[['GDP', '人口', '第三产业GDP', '城镇人口']]
    
    # 1. 基于原始数据的模型
    y_original = data[factor_name]
    model_original, degree_original, score_original = fit_model(X, y_original)
    
    # 2. 基于平滑数据的模型
    y_smoothed = data[f'smoothed_{factor_name}']
    model_smoothed, degree_smoothed, score_smoothed = fit_model(X, y_smoothed)

    # 获取未来数据
    future = future_data.copy()
    X_future = future[['GDP', '人口', '第三产业GDP', '城镇人口']]
    
    # 生成两种预测
    future['预测_原始数据_' + factor_name] = model_original.predict(X_future)
    future['预测_平滑数据_' + factor_name] = model_smoothed.predict(X_future)
    
    all_predictions.append(future)

    # 记录模型结果
    results.append({
        '区域': region,
        '数据点数': len(data),
        '原始数据_R²': model_original.score(X, y_original),
        '原始数据_最佳阶数': degree_original,
        '平滑数据_R²': model_smoothed.score(X, y_smoothed),
        '平滑数据_最佳阶数': degree_smoothed
    })

    return pd.concat(all_predictions), pd.DataFrame(results), model_original, model_smoothed


# 7. 确保预测值非负
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


# 8. 绘制区域指标的需求变化图
def plot_region_indicator(historical, predictions, factor_name, region, save_path):
    """绘制指定区域指标趋势图"""
    hist_region = historical.sort_values('年份').copy()
    pred_region = predictions.sort_values('年份').copy()

    # 定义单位
    if factor_name == '新增建筑面积':
        unit = '（万平方米）'
    elif factor_name == '汽车产量':
        unit = '（万辆）'
    else:
        unit = ''

    # 过渡点处理
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
    plt.plot(hist_region['年份'], hist_region[f'smoothed_{factor_name}'], 'g-', linewidth=2, label='平滑后历史数据')
    
    # 绘制两种预测曲线
    if not pred_region.empty:
        pred_region = pred_region[pred_region['年份'] >= 2025]
        plt.plot(pred_region['年份'], pred_region[original_pred_col], 'r--', linewidth=2, label='基于原始数据的预测')
        plt.plot(pred_region['年份'], pred_region[smoothed_pred_col], 'purple', linestyle='-.', linewidth=2, label='基于平滑数据的预测')

    # 图表设置
    plt.title(f'{region}{factor_name}历史趋势与预测', fontsize=16, pad=15)
    plt.xlabel('年份', fontsize=14)
    plt.ylabel(f'{factor_name}{unit}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # 调整x轴范围
    all_years = pd.concat([hist_region['年份'], pred_region['年份']]) if not pred_region.empty else hist_region['年份']
    plt.xlim(all_years.min() - 1, all_years.max() + 1)
    
    plt.tight_layout()
    save_file = f"{save_path}/{region}_{factor_name}_趋势与预测.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存{region}趋势图: {save_file}")


# 9. 保存完整数据到CSV
def save_complete_data(historical, predictions, factor_name, region, save_path):
    # 处理历史数据
    historical_output = historical[['地区', '年份', 'GDP', '人口', 
                                  '第三产业GDP', '城镇人口', 
                                  factor_name, f'smoothed_{factor_name}']].copy()
    historical_output.rename(columns={factor_name: f'原始_{factor_name}'}, inplace=True)
    historical_output[f'预测_原始数据_{factor_name}'] = np.nan
    historical_output[f'预测_平滑数据_{factor_name}'] = np.nan
    
    # 处理预测数据
    predictions_output = predictions[['地区', '年份', 'GDP', '人口', 
                                    '第三产业GDP', '城镇人口', 
                                    f'预测_原始数据_{factor_name}', 
                                    f'预测_平滑数据_{factor_name}']].copy()
    predictions_output[f'原始_{factor_name}'] = np.nan
    predictions_output[f'smoothed_{factor_name}'] = np.nan
    
    # 合并并保存
    full_output = pd.concat([historical_output, predictions_output], ignore_index=True)
    full_output.sort_values(by=['地区', '年份'], inplace=True)
    full_output.to_csv(f"{save_path}/{region}{factor_name}完整数据.csv", index=False)
    print(f"已保存完整数据: {save_path}/{region}{factor_name}完整数据.csv")


# 10. 区域数据汇总函数
def aggregate_to_region(data, factor_name, region):
    """将分省数据按年份汇总为指定区域总数据"""
    # 京津冀及周边8省市的省份列表（根据实际情况调整）
    jingjinji_regions = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', 
                         '山东省', '河南省', '辽宁省', '陕西省']  # 假设这9个是京津冀及周边8省市
    
    if region == '全国':
        # 全国数据：所有省份汇总
        region_data = data.groupby('年份').agg({
            'GDP': 'sum',
            '人口': 'sum',
            '第三产业GDP': 'sum',
            '城镇人口': 'sum',
            factor_name: 'sum'
        }).reset_index()
    elif region == '京津冀及周边8省市':
        # 京津冀及周边8省市：仅汇总指定省份
        filtered_data = data[data['地区'].isin(jingjinji_regions)]
        region_data = filtered_data.groupby('年份').agg({
            'GDP': 'sum',
            '人口': 'sum',
            '第三产业GDP': 'sum',
            '城镇人口': 'sum',
            factor_name: 'sum'
        }).reset_index()
    else:
        raise ValueError(f"不支持的区域: {region}")
    
    # 添加地区列标记
    region_data['地区'] = region
    return region_data


# 新增：计算特征重要性（适用于线性模型）
def calculate_feature_importance(model, feature_names):
    """从线性模型中提取特征重要性（系数绝对值）"""
    # 对于多项式特征管道，需要获取线性回归部分的系数
    if hasattr(model, 'named_steps'):
        # 提取线性回归系数
        coefficients = model.named_steps['linearregression'].coef_
        # 获取多项式特征名称
        poly_features = model.named_steps['polynomialfeatures']
        feature_names = poly_features.get_feature_names_out(feature_names)
    else:
        coefficients = model.coef_
    
    # 计算系数绝对值作为重要性
    importance = np.abs(coefficients)
    # 归一化
    importance = importance / np.sum(importance)
    
    # 生成结果DataFrame
    importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': importance
    }).sort_values('重要性', ascending=False)
    
    return importance_df


# 新增：绘制特征重要性条形图
def plot_feature_importance(importance_df, region, factor_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['特征'], importance_df['重要性'], color='skyblue')
    plt.xlabel('相对重要性（归一化）')
    plt.title(f'{region} {factor_name} 预测因素重要性', fontsize=14)
    plt.gca().invert_yaxis()  # 重要性高的在上方
    plt.tight_layout()
    save_file = f"{save_path}/{region}_{factor_name}_特征重要性.png"
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"已保存特征重要性图: {save_file}")


# 新增：绘制预测因素与目标指标的趋势对比图
def plot_factor_trends(historical_data, factor_name, region, save_path):
    """绘制各预测因素与目标指标的趋势对比"""
    # 选择需要对比的因素
    features = ['GDP', '人口', '第三产业GDP', '城镇人口']
    target = factor_name
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        # 左轴：特征值
        ax1 = axes[i]
        ax1.plot(historical_data['年份'], historical_data[feature], 'b-', linewidth=2, label=feature)
        ax1.set_xlabel('年份')
        ax1.set_ylabel(feature, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # 右轴：目标指标
        ax2 = ax1.twinx()
        ax2.plot(historical_data['年份'], historical_data[target], 'r--', linewidth=2, label=target)
        ax2.set_ylabel(target, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 添加标题和图例
        ax1.set_title(f'{feature} 与 {target} 趋势对比')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    save_file = f"{save_path}/{region}_{factor_name}_因素趋势对比.png"
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"已保存因素趋势对比图: {save_file}")


# 新增：计算各因素对预测结果的边际贡献
def calculate_marginal_contributions(model, X_mean, feature_names, factor_name):
    """计算各因素在均值点的边际贡献（单位变化引起的预测值变化）"""
    # 获取模型系数
    if hasattr(model, 'named_steps'):
        coefficients = model.named_steps['linearregression'].coef_
        poly_features = model.named_steps['polynomialfeatures']
        feature_names = poly_features.get_feature_names_out(feature_names)
    else:
        coefficients = model.coef_
    
    # 计算边际贡献（系数 × 特征均值）
    contributions = pd.DataFrame({
        '特征': feature_names,
        '系数': coefficients,
        '均值': X_mean,
        '边际贡献': coefficients * X_mean
    })
    
    # 计算相对贡献
    total_contribution = np.sum(np.abs(contributions['边际贡献']))
    contributions['相对贡献'] = np.abs(contributions['边际贡献']) / total_contribution
    
    return contributions.sort_values('相对贡献', ascending=False)



# 主程序
if __name__ == "__main__":
    # 路径设置
    fipath = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/'
    fopath = '/lustre/home/litong/intern/京津冀货运发展/output/predict_results/Sce_Pop_BAU_GDP_BAU/3'
    
    # 创建保存目录
    image_dir = os.path.join(fopath, '区域指标趋势图')
    os.makedirs(image_dir, exist_ok=True)
    
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
        '新增建筑面积', '汽车产量'
        #'新增铁路里程', 
        #'洗衣机产量', '电冰箱产量', '空调产量',
        #'新增高速公路里程', '新增其他等级公路里程', 
        #'农村水电基础设施投资'
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
        
        # 1. 加载并预处理历史数据
        print("加载历史数据并进行预处理...")
        historical_provincial = load_and_preprocess_data(fi_predict_factor, factor_name)
        historical_provincial = historical_provincial[historical_provincial['年份'] >= 2010].reset_index(drop=True)
        
        # 2. 加载未来特征数据（已按区域划分）
        print("加载未来特征数据...")
        future_data_regions = load_future_data(gdp_future_file, pop_future_file, 
                                              urban_pop_future_file, third_gdp_future_file)
        
        # 3. 对每个区域分别处理
        for region in regions:
            print(f"\n----- 处理区域: {region} -----")
            
            # 汇总为区域数据
            historical_data = aggregate_to_region(historical_provincial, factor_name, region)
            # 对区域数据重新进行平滑处理
            historical_data = smooth_data_by_moving_average(historical_data, factor_name, window_size=2)
            
            # 获取该区域的未来数据
            future_data = future_data_regions[region]
            
            # 生成区域预测
            print(f"生成{region}基于原始数据和平滑数据的预测...")
            predictions, model_results, model_original, model_smoothed = predict_future_region(
                historical_data, future_data, factor_name, region
            )
            
            if predictions.empty:
                print(f"{region}的{factor_name}无有效预测结果，跳过后续步骤")
                continue

            # 4. 预测后校准
            def calibrate_predictions(predictions, factor_name, region, max_growth=0.05):
                predictions = predictions.copy()
                # 区域数据校准
                hist_region = historical_data
                if not hist_region.empty:
                    base_year = hist_region['年份'].max()  # 历史最后一年
                    base_value = hist_region[hist_region['年份'] == base_year][f'smoothed_{factor_name}'].values[0]
                    
                    # 遍历预测年份，计算合理上限并校准
                    for idx, row in predictions.iterrows():
                        year = row['年份']
                        years_since_base = year - base_year
                        upper_bound = base_value * (1 + max_growth) ** years_since_base
                        predictions.at[idx, f'预测_平滑数据_{factor_name}'] = min(
                            predictions.at[idx, f'预测_平滑数据_{factor_name}'], 
                            upper_bound
                        )
                return predictions
            
            # 调用校准函数
            predictions = calibrate_predictions(predictions, factor_name, region, max_growth=0.05)
            
            # 5. 确保预测值非负
            predictions = ensure_non_negative(predictions, factor_name)


            # 6. 保存模型结果
            model_results.to_csv(f"{fopath}/{region}{factor_name}模型结果.csv", index=False)
            
            # 7. 绘制区域需求变化图
            print(f"绘制{region}需求变化图...")
            plot_region_indicator(historical_data, predictions, factor_name, region, image_dir)
            
            # 8. 保存完整数据到CSV
            print(f"保存{region}完整数据到CSV...")
            save_complete_data(historical_data, predictions, factor_name, region, fopath)

            # 新增分析代码：
            # 1. 计算特征重要性（使用平滑数据模型，更稳定）
            feature_names = ['GDP', '人口', '第三产业GDP', '城镇人口']
            importance_df = calculate_feature_importance(model_smoothed, feature_names)
            plot_feature_importance(importance_df, region, factor_name, image_dir)

            # 2. 绘制因素趋势对比图
            plot_factor_trends(historical_data, factor_name, region, image_dir)

            # 3. 计算边际贡献
            X_mean = historical_data[feature_names].mean().values
            contributions = calculate_marginal_contributions(model_smoothed, X_mean, feature_names, factor_name)
            contributions.to_csv(f"{fopath}/{region}_{factor_name}_因素贡献度.csv", index=False)
            print(f"已保存因素贡献度数据: {fopath}/{region}_{factor_name}_因素贡献度.csv")
                
    print("\n所有指标和区域处理完成！")
