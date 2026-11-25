# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 17:23:12 2025

@author: Yuxi Liu
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


# 1. 数据加载与预处理
def load_and_preprocess_data(fi_predict_factor, factor_name):
    # 加载历史GDP数据
    gdp = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史GDP数据.csv', encoding='gbk')
    gdp = gdp.iloc[:, :-1]  # 移除备注列
    gdp = gdp.melt(id_vars='地区', var_name='年份', value_name='GDP')
    gdp['年份'] = gdp['年份'].str.replace('年', '').astype(int)

    # 加载历史人口数据
    pop = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史人口数据.csv', encoding='gbk')
    pop = pop.iloc[:, :-1]  # 移除备注列
    pop = pop.melt(id_vars='地区', var_name='年份', value_name='人口')
    pop['年份'] = pop['年份'].str.replace('年', '').astype(int)

    # 加载历史第三产业GDP数据
    third_gdp = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/第三产业增加值分省年度数据.csv',
                            encoding='gbk')
    third_gdp = third_gdp.melt(id_vars='地区', var_name='年份', value_name='第三产业GDP')
    third_gdp['年份'] = third_gdp['年份'].str.replace('年', '').astype(int)

    # 加载历史城镇人口数据
    urban_pop = pd.read_csv('/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/城镇人口分省年度数据.csv',
                            encoding='gbk')
    urban_pop = urban_pop.melt(id_vars='地区', var_name='年份', value_name='城镇人口')
    urban_pop['年份'] = urban_pop['年份'].str.replace('年', '').astype(int)

    # 加载相关变量数据
    prod_pre = pd.read_excel(fi_predict_factor, sheet_name='分省年度数据', skiprows=3)
    prod_pre = prod_pre.melt(id_vars='地区', var_name='年份', value_name=factor_name)
    prod_pre['年份'] = prod_pre['年份'].str.replace('年', '').astype(int)

    # 合并数据
    merged = pd.merge(gdp, pop, on=['地区', '年份'])
    merged = pd.merge(merged, third_gdp, on=['地区', '年份'])
    merged = pd.merge(merged, urban_pop, on=['地区', '年份'])
    merged = pd.merge(merged, prod_pre, on=['地区', '年份'])

    # 移除缺失值
    merged = merged.dropna(subset=['GDP', '人口', '第三产业GDP', '城镇人口', factor_name])
    
    # 保存当前指标的历史数据用于后续汇总可视化
    return merged


# 2. 加载未来预测数据
def load_future_data(gdp_future_file, pop_future_file, urban_pop_future_file, third_gdp_future_file):
    # 加载GDP预测数据
    gdp_future = pd.read_csv(gdp_future_file)
    gdp_future = gdp_future.melt(id_vars='Unnamed: 0', var_name='地区', value_name='GDP')
    gdp_future = gdp_future.rename(columns={'Unnamed: 0': '年份'})
    gdp_future = gdp_future[gdp_future['年份'] >= 2025]

    # 加载人口预测数据
    pop_future = pd.read_csv(pop_future_file)
    pop_future = pop_future.melt(id_vars='Unnamed: 0', var_name='地区', value_name='人口')
    pop_future = pop_future.rename(columns={'Unnamed: 0': '年份'})
    pop_future = pop_future[pop_future['年份'] >= 2025]

    # 加载第三产业GDP预测数据
    third_gdp_future = pd.read_csv(third_gdp_future_file)
    third_gdp_future = third_gdp_future.melt(id_vars='Unnamed: 0', var_name='地区', value_name='第三产业GDP')
    third_gdp_future = third_gdp_future.rename(columns={'Unnamed: 0': '年份'})
    third_gdp_future = third_gdp_future[third_gdp_future['年份'] >= 2025]

    # 加载城镇人口预测数据
    urban_pop_future = pd.read_csv(urban_pop_future_file)
    urban_pop_future = urban_pop_future.melt(id_vars='Unnamed: 0', var_name='地区', value_name='城镇人口')
    urban_pop_future = urban_pop_future.rename(columns={'Unnamed: 0': '年份'})
    urban_pop_future = urban_pop_future[urban_pop_future['年份'] >= 2025]

    # 合并数据
    future_data = pd.merge(gdp_future, pop_future, on=['地区', '年份'])
    future_data = pd.merge(future_data, third_gdp_future, on=['地区', '年份'])
    future_data = pd.merge(future_data, urban_pop_future, on=['地区', '年份'])
    
    return future_data


# 3. 模型拟合函数
def fit_model(X, y, max_degree=3):
    best_score = -np.inf
    best_degree = 1
    best_model = None

    for degree in range(1, max_degree + 1):
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            LinearRegression()
        )
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        avg_score = np.mean(scores)

        if avg_score > best_score:
            best_score = avg_score
            best_degree = degree
            best_model = model

    best_model.fit(X, y)
    print(f"最佳阶数: {best_degree}, 验证分数: {best_score:.4f}")
    return best_model


# 4. 预测未来数据
def predict_future(merged, future_data, factor_name):
    results = []
    all_predictions = []

    for province, data in merged.groupby('地区'):
        if len(data) < 5:
            print(f"跳过 {province} - 数据点不足 ({len(data)} < 5)")
            continue

        X = data[['GDP', '人口', '第三产业GDP', '城镇人口']]
        y = data[factor_name]

        model = fit_model(X, y)

        future = future_data[future_data['地区'] == province]
        if future.empty:
            print(f"跳过 {province} - 无未来数据")
            continue

        X_future = future[['GDP', '人口', '第三产业GDP', '城镇人口']]
        future_pred = model.predict(X_future)
        future['预测' + factor_name] = future_pred

        all_predictions.append(future)

        r2 = model.score(X, y)
        results.append({
            '省份': province,
            '数据点数': len(data),
            'R²': r2,
            '模型系数': model.steps[1][1].coef_,
            '模型截距': model.steps[1][1].intercept_
        })

    if all_predictions:
        return pd.concat(all_predictions), pd.DataFrame(results)
    else:
        return pd.DataFrame(), pd.DataFrame()


# 5. 可视化单个省份函数
def visualize_single_province(historical, predictions, province, factor_name, save_path):
    hist_prov = historical[historical['地区'] == province]
    pred_prov = predictions[predictions['地区'] == province]

    if hist_prov.empty or pred_prov.empty:
        print(f"没有{province}的数据可用于可视化")
        return

    plt.figure(figsize=(14, 8))

    # 历史与预测趋势
    plt.subplot(2, 1, 1)
    plt.plot(hist_prov['年份'], hist_prov[factor_name], 'bo-', label=f'历史{factor_name}')
    plt.plot(pred_prov['年份'], pred_prov['预测' + factor_name], 'r--', label=f'预测{factor_name}')
    plt.title(f'{province}{factor_name}历史与预测')
    plt.xlabel('年份')
    plt.ylabel(f'{factor_name}')
    plt.legend()
    plt.grid(True)

    # 相关性分析
    plt.subplot(2, 1, 2)
    sns.scatterplot(data=hist_prov, x='GDP', y=factor_name, hue='人口', size='人口', sizes=(20, 200))
    plt.title(f'{province}GDP与{factor_name}关系')
    plt.xlabel('GDP(亿元)')
    plt.ylabel(f'{factor_name}')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{save_path}/{province}_{factor_name}_预测结果.png", dpi=300, bbox_inches='tight')
    plt.close()


# 6. 可视化所有指标（每个指标一张折线图）
def visualize_all_factors(historical_dict, predictions_dict, factor_list, save_path):
    """
    为每个指标绘制一张折线图，展示所有省份的历史与预测趋势（同一省份使用相同颜色）
    """
    # 创建指标总览图像目录
    overview_dir = os.path.join(save_path, '指标总览')
    os.makedirs(overview_dir, exist_ok=True)
    
    # 获取所有省份列表（用于统一分配颜色）
    all_provinces = []
    for factor_name in factor_list:
        all_provinces.extend(historical_dict[factor_name]['地区'].unique())
    all_provinces = list(set(all_provinces))  # 去重
    
    # 为每个省份分配唯一颜色（使用matplotlib的颜色循环）
    color_map = {
        province: plt.cm.tab20(i % 20)  # 使用tab20色系，支持最多20个省份，超出自动循环
        for i, province in enumerate(all_provinces)
    }
    
    for factor_name in factor_list:
        plt.figure(figsize=(18, 10))
        
        # 获取当前指标的历史和预测数据
        historical_data = historical_dict[factor_name]
        predictions = predictions_dict[factor_name]
        
        # 遍历所有省份绘制曲线
        for province in historical_data['地区'].unique():
            # 历史数据
            hist_prov = historical_data[
                (historical_data['地区'] == province) & 
                (historical_data['年份'] >= 2010)
            ]
            # 预测数据
            pred_prov = predictions[predictions['地区'] == province]
            
            if not hist_prov.empty and not pred_prov.empty:
                # 获取当前省份的专属颜色
                province_color = color_map[province]
                
                # 绘制历史数据（实线，同省份颜色）
                plt.plot(hist_prov['年份'], hist_prov[factor_name], 
                         linestyle='-', marker='o', markersize=4, 
                         color=province_color,  # 指定颜色
                         label=f'{province} (历史)')
                
                # 绘制预测数据（虚线，同省份颜色）
                plt.plot(pred_prov['年份'], pred_prov['预测' + factor_name], 
                         linestyle='--', marker='*', markersize=4,
                         color=province_color,  # 与历史数据同色
                         label=f'{province} (预测)')
        
        plt.title(f'{factor_name} 各省历史与预测趋势对比', fontsize=16)
        plt.xlabel('年份', fontsize=14)
        plt.ylabel(factor_name, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 处理图例（放置在右侧）
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{overview_dir}/{factor_name}_各省趋势对比.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已生成 {factor_name} 指标总览图")


# 主程序
if __name__ == "__main__":
    # 设置路径
    fipath = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/'
    fopath = '/lustre/home/litong/intern/京津冀货运发展/output/predict_results/Sce_Pop_BAU_GDP_BAU'
    
    # 创建图像保存目录
    image_dir = os.path.join(fopath, 'images')
    os.makedirs(image_dir, exist_ok=True)
    
    # 指标列表
    fi_predict_factor_list = [
        '建筑业房屋竣工面积', '汽车产量', '新增铁路营业里程', 
        '家用洗衣机产量', '家用电冰箱产量', '房间空调器',
        '新增高速等级公路里程', '新增除高速外等级公路里程', 
        '农村水电建设本年完成投资额'
    ]
    predict_factor_list = [
        '新增建筑面积', '汽车产量', '新增铁路里程', 
        '洗衣机产量', '电冰箱产量', '空调产量',
        '新增高速公路里程', '新增其他等级公路里程', 
        '农村水电基础设施投资'
    ]
    
    # 存储所有指标的历史数据和预测结果
    historical_dict = {}  # {指标名: 历史数据}
    predictions_dict = {}  # {指标名: 预测数据}
    
    # 循环处理每个指标
    for ip_, factor_name in enumerate(predict_factor_list):
        print(f'\n{"="*20} 处理指标: {factor_name} {"="*20}')
        
        # 输入文件路径
        fi_predict_factor = fipath + fi_predict_factor_list[ip_] + '分省年度数据.xls'
        
        # 未来数据文件路径
        fipath_future = '/lustre/home/litong/intern/京津冀货运发展/output'
        gdp_future_file = fipath_future + '/scenarios/gdp_predictions_基准.csv'
        pop_future_file = fipath_future + '/scenarios/population_predictions_基准.csv'
        urban_pop_future_file = fipath_future + '/scenarios/urban_population_predictions_基准.csv'
        third_gdp_future_file = fipath_future + '/scenarios/third_gdp_predictions_基准.csv'
        
        # 加载历史数据
        print("加载历史数据...")
        historical_data = load_and_preprocess_data(fi_predict_factor, factor_name)
        historical_data = historical_data.loc[historical_data['年份'] >= 2010].reset_index(drop=True)
        historical_dict[factor_name] = historical_data  # 保存到字典
        
        # 加载未来数据
        print("加载未来预测数据...")
        future_data = load_future_data(gdp_future_file, pop_future_file, 
                                      urban_pop_future_file, third_gdp_future_file)
        
        # 模型预测
        print("开始预测...")
        predictions, model_results = predict_future(historical_data, future_data, factor_name)
        
        if predictions.empty:
            print(f"{factor_name} 无预测结果，跳过后续处理")
            continue
        
        # 保存预测结果到字典
        predictions_dict[factor_name] = predictions
        
        # 保存结果文件
        historical_output = historical_data[['地区', '年份', 'GDP', '人口', 
                                           '第三产业GDP', '城镇人口', factor_name]].copy()
        historical_output.rename(columns={factor_name: f'真实{factor_name}'}, inplace=True)
        historical_output[f'预测{factor_name}'] = np.nan
        
        predictions_output = predictions[['地区', '年份', 'GDP', '人口', 
                                        '第三产业GDP', '城镇人口', f'预测{factor_name}']].copy()
        predictions_output[f'真实{factor_name}'] = np.nan
        
        full_output = pd.concat([historical_output, predictions_output], ignore_index=True)
        full_output.sort_values(by=['地区', '年份'], inplace=True)
        
        # 保存CSV文件
        full_output.to_csv(f"{fopath}/分省{factor_name}完整结果.csv", index=False)
        predictions.to_csv(f"{fopath}/分省{factor_name}预测结果.csv", index=False)
        model_results.to_csv(f"{fopath}/分省{factor_name}模型拟合结果.csv", index=False)
        
        # 可视化单个省份
        if not model_results.empty:
            top_provinces = model_results.sort_values('R²', ascending=False)['省份'].head(5)
            print(f"可视化R²最高的5个省份: {', '.join(top_provinces)}")
            
            for province in top_provinces:
                visualize_single_province(historical_data, predictions, 
                                        province, factor_name, image_dir)
    
    # 生成所有指标的总览图（每个指标一张折线图）
    print("\n生成所有指标的总览可视化图...")
    visualize_all_factors(historical_dict, predictions_dict, predict_factor_list, image_dir)
    
    print("\n所有处理完成！")
