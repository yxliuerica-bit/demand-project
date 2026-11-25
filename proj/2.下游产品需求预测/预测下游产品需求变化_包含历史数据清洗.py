# -*- coding: utf-8 -*-
"""
1. 各省市各指标的需求变化图：
   - 历史数据：散点+折线
   - 平滑后曲线：单独曲线
   - 预测趋势：分别基于原始数据和平滑数据，用不同颜色曲线
2. 每个指标的总览图：
   - 所有省市的平滑数据及基于平滑数据的预测趋势
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


# 4. 加载未来特征数据
def load_future_data(gdp_future_file, pop_future_file, urban_pop_future_file, third_gdp_future_file):
    gdp_future = pd.read_csv(gdp_future_file)
    gdp_future = gdp_future.melt(id_vars='Unnamed: 0', var_name='地区', value_name='GDP').rename(columns={'Unnamed: 0': '年份'})
    gdp_future = gdp_future[gdp_future['年份'] >= 2025]

    pop_future = pd.read_csv(pop_future_file)
    pop_future = pop_future.melt(id_vars='Unnamed: 0', var_name='地区', value_name='人口').rename(columns={'Unnamed: 0': '年份'})
    pop_future = pop_future[pop_future['年份'] >= 2025]

    third_gdp_future = pd.read_csv(third_gdp_future_file)
    third_gdp_future = third_gdp_future.melt(id_vars='Unnamed: 0', var_name='地区', value_name='第三产业GDP').rename(columns={'Unnamed: 0': '年份'})
    third_gdp_future = third_gdp_future[third_gdp_future['年份'] >= 2025]

    urban_pop_future = pd.read_csv(urban_pop_future_file)
    urban_pop_future = urban_pop_future.melt(id_vars='Unnamed: 0', var_name='地区', value_name='城镇人口').rename(columns={'Unnamed: 0': '年份'})
    urban_pop_future = urban_pop_future[urban_pop_future['年份'] >= 2025]

    future_data = pd.merge(gdp_future, pop_future, on=['地区', '年份'])
    future_data = pd.merge(future_data, third_gdp_future, on=['地区', '年份'])
    future_data = pd.merge(future_data, urban_pop_future, on=['地区', '年份'])
    
    return future_data


# 5. 模型拟合函数
def fit_model(X, y, max_degree=1):
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


# 6. 双重预测（基于原始数据和平滑数据）
def predict_future(historical_data, future_data, factor_name):
    results = []
    all_predictions = []

    for province, data in historical_data.groupby('地区'):
        if len(data) < 5:
            print(f"跳过 {province} - 数据点不足")
            continue

        # 准备特征数据
        X = data[['GDP', '人口', '第三产业GDP', '城镇人口']]
        
        # 1. 基于原始数据的模型
        y_original = data[factor_name]
        model_original, degree_original, score_original = fit_model(X, y_original)
        
        # 2. 基于平滑数据的模型
        y_smoothed = data[f'smoothed_{factor_name}']
        model_smoothed, degree_smoothed, score_smoothed = fit_model(X, y_smoothed)

        # 获取未来数据
        future = future_data[future_data['地区'] == province].copy()
        if future.empty:
            print(f"跳过 {province} - 无未来数据")
            continue

        X_future = future[['GDP', '人口', '第三产业GDP', '城镇人口']]
        
        # 生成两种预测
        future['预测_原始数据_' + factor_name] = model_original.predict(X_future)
        future['预测_平滑数据_' + factor_name] = model_smoothed.predict(X_future)
        
        all_predictions.append(future)

        # 记录模型结果
        results.append({
            '省份': province,
            '数据点数': len(data),
            '原始数据_R²': model_original.score(X, y_original),
            '原始数据_最佳阶数': degree_original,
            '平滑数据_R²': model_smoothed.score(X, y_smoothed),
            '平滑数据_最佳阶数': degree_smoothed
        })

    if all_predictions:
        return pd.concat(all_predictions), pd.DataFrame(results)
    else:
        return pd.DataFrame(), pd.DataFrame()


# 7. 确保预测值非负（新增函数）
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


# 8. 绘制各省市各指标的需求变化图
def plot_province_indicator(historical, predictions, province, factor_name, save_path):
    hist_prov = historical[historical['地区'] == province].sort_values('年份').copy()
    pred_prov = predictions[predictions['地区'] == province].sort_values('年份').copy()

    if hist_prov.empty:
        print(f"没有{province}的历史数据，无法绘图")
        return

    # --------------------------新增代码：定义单位--------------------------
    # 根据指标名称设置对应的单位
    if factor_name == '新增建筑面积':
        unit = '（万平方米）'
    elif factor_name == '汽车产量':
        unit = '（万辆）'
    else:
        unit = ''  # 其他指标默认无单位（可根据需要补充）
    # ----------------------------------------------------------------------

    # 提取2025年预测值作为过渡点（保持原有逻辑）
    original_pred_col = f'预测_原始数据_{factor_name}'
    smoothed_pred_col = f'预测_平滑数据_{factor_name}'
    if not pred_prov.empty and 2025 in pred_prov['年份'].values:
        pred_2025 = pred_prov[pred_prov['年份'] == 2025].iloc[0]
        transition_data = {
            '地区': [province],
            '年份': [2025],
            factor_name: [pred_2025[original_pred_col] if original_pred_col in pred_2025 else np.nan],
            f'smoothed_{factor_name}': [pred_2025[smoothed_pred_col] if smoothed_pred_col in pred_2025 else np.nan]
        }
        transition_point = pd.DataFrame(transition_data)
        hist_prov = pd.concat([hist_prov, transition_point], ignore_index=True).sort_values('年份')

    plt.figure(figsize=(14, 8))

    # 绘制原始历史数据
    plt.scatter(hist_prov['年份'], hist_prov[factor_name], color='blue', alpha=0.7, label='原始历史数据（散点）')
    plt.plot(hist_prov['年份'], hist_prov[factor_name], 'b-', alpha=0.5, label='原始历史数据（折线）')
    
    # 绘制平滑后的数据曲线
    plt.plot(hist_prov['年份'], hist_prov[f'smoothed_{factor_name}'], 'g-', linewidth=2, label='平滑后历史数据')
    
    # 绘制两种预测曲线
    if not pred_prov.empty:
        pred_prov = pred_prov[pred_prov['年份'] >= 2025]
        if original_pred_col in pred_prov.columns:
            plt.plot(pred_prov['年份'], pred_prov[original_pred_col], 'r--', linewidth=2, 
                     label='基于原始数据的预测')
        if smoothed_pred_col in pred_prov.columns:
            plt.plot(pred_prov['年份'], pred_prov[smoothed_pred_col], 'purple', linestyle='-.', 
                     linewidth=2, label='基于平滑数据的预测')

    # 图表设置（纵轴添加单位）
    plt.title(f'{province}{factor_name}历史趋势与预测', fontsize=16, pad=15)
    plt.xlabel('年份', fontsize=14)
    # --------------------------修改此处：纵轴标签加单位--------------------------
    plt.ylabel(f'{factor_name}{unit}', fontsize=14)  # 拼接指标名和单位
    # ----------------------------------------------------------------------
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # 调整x轴范围
    all_years = pd.concat([hist_prov['年份'], pred_prov['年份']]) if not pred_prov.empty else hist_prov['年份']
    plt.xlim(all_years.min() - 1, all_years.max() + 1)
    
    plt.tight_layout()
    save_file = f"{save_path}/{province}_{factor_name}_趋势与预测.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_file}")


# 9. 绘制每个指标的总览图
def plot_indicator_overview(historical, predictions, factor_name, save_path):
    plt.figure(figsize=(16, 10))
    
    # --------------------------新增代码：定义单位--------------------------
    if factor_name == '新增建筑面积':
        unit = '（万平方米）'
    elif factor_name == '汽车产量':
        unit = '（万辆）'
    else:
        unit = ''
    # ----------------------------------------------------------------------

    smoothed_pred_col = f'预测_平滑数据_{factor_name}'
    provinces = historical['地区'].unique()
    color_map = {prov: plt.cm.tab20(i % 20) for i, prov in enumerate(provinces)}
    
    for province in provinces:
        hist_prov = historical[historical['地区'] == province].sort_values('年份').copy()
        pred_prov = predictions[predictions['地区'] == province].sort_values('年份').copy()
        
        if hist_prov.empty:
            continue

        # 提取2025年预测值作为过渡点（保持原有逻辑）
        if not pred_prov.empty and 2025 in pred_prov['年份'].values:
            pred_2025 = pred_prov[pred_prov['年份'] == 2025].iloc[0]
            transition_data = {
                '地区': [province],
                '年份': [2025],
                f'smoothed_{factor_name}': [pred_2025[smoothed_pred_col] if smoothed_pred_col in pred_2025 else np.nan]
            }
            transition_point = pd.DataFrame(transition_data)
            hist_prov = pd.concat([hist_prov, transition_point], ignore_index=True).sort_values('年份')

        # 绘制平滑历史数据
        plt.plot(hist_prov['年份'], hist_prov[f'smoothed_{factor_name}'], 
                 color=color_map[province], linestyle='-', marker='o', 
                 markersize=4, label=province if province not in plt.gca().get_legend_handles_labels()[1] else "")
        
        # 绘制平滑数据的预测
        if not pred_prov.empty and smoothed_pred_col in pred_prov.columns:
            pred_prov = pred_prov[pred_prov['年份'] >= 2025]
            plt.plot(pred_prov['年份'], pred_prov[smoothed_pred_col], 
                     color=color_map[province], linestyle='--', marker='*', 
                     markersize=4)
    
    # 图表设置（纵轴添加单位）
    plt.title(f'{factor_name} 各省平滑历史与预测总览', fontsize=16, pad=15)
    plt.xlabel('年份', fontsize=14)
    # --------------------------修改此处：纵轴标签加单位--------------------------
    plt.ylabel(f'{factor_name}{unit}', fontsize=14)  # 拼接指标名和单位
    # ----------------------------------------------------------------------
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    save_file = f"{save_path}/{factor_name}_各省总览图.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存总览图: {save_file}")


# 10. 保存完整数据到CSV
def save_complete_data(historical, predictions, factor_name, save_path):
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
    full_output.to_csv(f"{save_path}/分省{factor_name}完整数据.csv", index=False)
    print(f"已保存完整数据: {save_path}/分省{factor_name}完整数据.csv")


# 主程序
if __name__ == "__main__":
    # 路径设置
    fipath = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/'
    fopath = '/lustre/home/litong/intern/京津冀货运发展/output/predict_results/Sce_Pop_BAU_GDP_BAU/2'
    
    # 创建保存目录
    image_dir = os.path.join(fopath, '指标趋势图')
    overview_dir = os.path.join(fopath, '指标总览图')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(overview_dir, exist_ok=True)
    
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
        historical_data = load_and_preprocess_data(fi_predict_factor, factor_name)
        historical_data = historical_data[historical_data['年份'] >= 2010].reset_index(drop=True)
        
        # 2. 加载未来特征数据
        print("加载未来特征数据...")
        future_data = load_future_data(gdp_future_file, pop_future_file, 
                                      urban_pop_future_file, third_gdp_future_file)
        
        # 3. 生成双重预测
        print("生成基于原始数据和平滑数据的预测...")
        predictions, model_results = predict_future(historical_data, future_data, factor_name)
        
        if predictions.empty:
            print(f"{factor_name} 无有效预测结果，跳过后续步骤")
            continue

        # 4. 预测后校准
        def calibrate_predictions(predictions, factor_name, max_growth=0.05):
            predictions = predictions.copy()
            # 按省份分组校准
            for province, group in predictions.groupby('地区'):
                # 取历史最后一年（预测起始年份前一年）的平滑后数据作为基准
                hist_prov = historical_data[historical_data['地区'] == province]
                if not hist_prov.empty:
                    base_year = hist_prov['年份'].max()  # 历史最后一年
                    base_value = hist_prov[hist_prov['年份'] == base_year][f'smoothed_{factor_name}'].values[0]
                    
                    # 遍历预测年份，计算合理上限并校准
                    for idx, row in group.iterrows():
                        year = row['年份']
                        years_since_base = year - base_year
                        upper_bound = base_value * (1 + max_growth) ** years_since_base
                        predictions.at[idx, f'预测_平滑数据_{factor_name}'] = min(
                            predictions.at[idx, f'预测_平滑数据_{factor_name}'], 
                            upper_bound
                        )
            return predictions
        
        # 调用校准函数
        predictions = calibrate_predictions(predictions, factor_name, max_growth=0.05)  # 可调整最大增长率
        
        # 5. 确保预测值非负（关键步骤：新增非负性约束）
        predictions = ensure_non_negative(predictions, factor_name)


        # 6. 保存模型结果
        model_results.to_csv(f"{fopath}/分省{factor_name}模型结果.csv", index=False)
        
        # 7. 为每个省份绘制需求变化图
        print("绘制各省市需求变化图...")
        for province in historical_data['地区'].unique():
            plot_province_indicator(historical_data, predictions, province, factor_name, image_dir)
        
        # 8. 绘制指标总览图
        print("绘制指标总览图...")
        plot_indicator_overview(historical_data, predictions, factor_name, overview_dir)
        
        # 9. 保存完整数据到CSV
        print("保存完整数据到CSV...")
        save_complete_data(historical_data, predictions, factor_name, fopath)
    
    print("\n所有指标处理完成！")
    