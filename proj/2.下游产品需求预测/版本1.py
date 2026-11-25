# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# -------------------------- 1. 基础配置 --------------------------
# 1.1 路径设置（替换为你的省份完整数据目录路径）
PROVINCE_DATA_DIR = "/lustre/home/litong/intern/京津冀货运发展/output/predict_results/Sce_Pop_BAU_GDP_BAU/5/省份完整数据"

# 1.2 目标省份与指标
TARGET_PROVINCES = ["浙江省", "江苏省", "北京市"]  # 待对比省份
FACTOR_NAME = "新增建筑面积"  # 目标指标

# 1.3 绘图配置（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150


# -------------------------- 2. 数据读取与预处理 --------------------------
def load_province_data(data_dir, target_provinces, factor_name):
    """
    读取各省单独的数据文件，筛选目标省份的新增建筑面积相关数据
    :param data_dir: 省份数据目录
    :param target_provinces: 目标省份列表
    :param factor_name: 目标指标（新增建筑面积）
    :return: 合并后的DataFrame（含历史+预测数据）
    """
    all_data = []
    
    # 遍历目标省份
    for province in target_provinces:
        # 构建文件名（如"浙江省新增建筑面积完整数据.csv"）
        file_pattern = f"{province}{factor_name}完整数据.csv"
        file_path = os.path.join(data_dir, file_pattern)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            # 尝试用通配符查找，防止文件名有微小差异
            search_pattern = os.path.join(data_dir, f"{province}*{factor_name}*.csv")
            matching_files = glob.glob(search_pattern)
            if not matching_files:
                raise FileNotFoundError(f"未找到{province}的数据文件：{file_path}")
            else:
                file_path = matching_files[0]
                print(f"找到匹配文件：{file_path}")
        
        # 读取单个省份数据
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 确保地区列存在并正确设置
        if '地区' not in df.columns:
            df['地区'] = province
        else:
            df['地区'] = province  # 确保地区名称一致
        
        all_data.append(df)
    
    # 合并所有省份数据
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # 重命名关键列
    col_mapping = {
        f"原始_{factor_name}": "历史_原始值",
        f"平滑_{factor_name}": "历史_平滑值",
        f"预测_原始数据_{factor_name}": "预测_基于原始",
        f"预测_平滑数据_{factor_name}": "预测_基于平滑"
    }
    # 只重命名存在的列
    existing_cols = {k: v for k, v in col_mapping.items() if k in df_combined.columns}
    df_combined.rename(columns=existing_cols, inplace=True)
    
    # 处理年份列
    df_combined['年份'] = df_combined['年份'].astype(int)
    
    # 合并历史与预测数据
    if '历史_平滑值' in df_combined.columns and '历史_原始值' in df_combined.columns:
        df_combined['历史值'] = df_combined['历史_平滑值'].fillna(df_combined['历史_原始值'])
    elif '历史_原始值' in df_combined.columns:
        df_combined['历史值'] = df_combined['历史_原始值']
    elif '历史_平滑值' in df_combined.columns:
        df_combined['历史值'] = df_combined['历史_平滑值']
    
    if '预测_基于平滑' in df_combined.columns and '预测_基于原始' in df_combined.columns:
        df_combined['预测值'] = df_combined['预测_基于平滑'].fillna(df_combined['预测_基于原始'])
    elif '预测_基于原始' in df_combined.columns:
        df_combined['预测值'] = df_combined['预测_基于原始']
    elif '预测_基于平滑' in df_combined.columns:
        df_combined['预测值'] = df_combined['预测_基于平滑']
    
    # 只保留关键列
    key_cols = ['地区', '年份', 'GDP', '人口', '第三产业GDP', '城镇人口', '历史值', '预测值']
    # 过滤掉不存在的列
    key_cols = [col for col in key_cols if col in df_combined.columns]
    df_clean = df_combined[key_cols].dropna(subset=['年份', '地区'])
    
    return df_clean


# 执行数据读取
try:
    df_prov = load_province_data(
        data_dir=PROVINCE_DATA_DIR,
        target_provinces=TARGET_PROVINCES,
        factor_name=FACTOR_NAME
    )

    # 查看数据结构
    print("数据预览（前5行）：")
    print(df_prov.head())
    print(f"\n数据时间范围：{df_prov['年份'].min()} - {df_prov['年份'].max()}")
    print(f"目标省份数据量：{df_prov.groupby('地区').size()}")
except Exception as e:
    print(f"数据读取错误：{str(e)}")
    exit()


# -------------------------- 3. 三省预测结果对比 --------------------------
def compare_pred_results(df, factor_name):
    """对比目标省份的预测结果（表格+折线图）"""
    # 3.1 输出关键年份的预测对比表
    key_years = [2025, 2035, 2045]
    # 找到数据中实际存在的关键年份
    available_years = [year for year in key_years if year in df['年份'].unique()]
    if not available_years:
        # 如果没有关键年份，使用最新的3个年份
        available_years = sorted(df['年份'].unique())[-3:]
    
    df_key_years = df[df['年份'].isin(available_years)].copy()
    
    # 整理表格
    pred_table = df_key_years.pivot_table(
        index='地区', 
        columns='年份', 
        values='预测值', 
        fill_value=0
    ).round(2)
    
    print(f"\n=== {factor_name} 预测结果对比表（单位：万平方米）===")
    print(pred_table)
    
    # 3.2 绘制历史+预测趋势图
    plt.figure(figsize=(12, 6))
    
    # 定义省份样式
    province_style = {
        "浙江省": ("#1f77b4", "o-"),    # 蓝色圆点折线
        "江苏省": ("#ff7f0e", "s-"),    # 橙色方形折线
        "北京市": ("#2ca02c", "^-")     # 绿色三角形折线
    }
    
    # 绘制每个省份的趋势
    for province in TARGET_PROVINCES:
        df_province = df[df['地区'] == province].sort_values('年份')
        if df_province.empty:
            print(f"警告：{province}没有数据，无法绘图")
            continue
            
        color, style = province_style[province]
        
        # 绘制历史数据（2025年前）
        df_hist = df_province[df_province['年份'] < 2025]
        if not df_hist.empty and '历史值' in df_hist.columns:
            plt.plot(df_hist['年份'], df_hist['历史值'], style, color=color, 
                     linewidth=2, markersize=5, label=f'{province}（历史）')
        
        # 绘制预测数据（2025年及以后）
        df_pred = df_province[df_province['年份'] >= 2025]
        if not df_pred.empty and '预测值' in df_pred.columns:
            plt.plot(df_pred['年份'], df_pred['预测值'], style, color=color, 
                     linewidth=2, markersize=5, alpha=0.8, label=f'{province}（预测）')
    
    # 图表配置
    plt.axvline(x=2025, color='gray', linestyle='--', alpha=0.7, label='2025年（预测起点）')
    plt.xlabel('年份', fontsize=12)
    plt.ylabel(f'{factor_name}（万平方米）', fontsize=12)
    plt.title(f'浙江省、江苏省、北京市 {factor_name} 历史趋势与预测对比', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 图例放在右侧
    plt.tight_layout()
    
    # 保存图片
    save_dir = os.path.join(PROVINCE_DATA_DIR, "对比分析图")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"三省{factor_name}趋势对比.png"), bbox_inches='tight')
    plt.close()
    print(f"\n趋势对比图已保存至：{save_dir}")


# 执行预测结果对比
compare_pred_results(df_prov, FACTOR_NAME)


# -------------------------- 4. 影响因素分析 --------------------------
def analyze_influencing_factors(df, factor_name):
    """分析影响新增建筑面积预测的关键因素"""
    # 4.1 计算特征与新增建筑面积的相关性
    # 筛选2025年前的历史数据
    df_hist = df[df['年份'] < 2025].copy()
    if '历史值' in df_hist.columns:
        df_hist = df_hist.dropna(subset=['历史值'])
    else:
        print("警告：没有历史值数据，无法进行相关性分析")
        return
    
    # 定义特征列
    feature_cols = []
    for col in ['GDP', '人口', '第三产业GDP', '城镇人口']:
        if col in df_hist.columns:
            feature_cols.append(col)
    
    if not feature_cols:
        print("警告：没有找到特征数据，无法进行相关性分析")
        return
    
    # 计算每个省份的相关性
    print(f"\n=== {factor_name} 影响因素相关性分析（历史数据）===")
    corr_results = {}
    for province in TARGET_PROVINCES:
        df_p = df_hist[df_hist['地区'] == province]
        if len(df_p) < 3:
            print(f"{province}历史数据不足，无法计算相关性")
            continue
        
        # 计算相关性
        try:
            corr_matrix = df_p[feature_cols + ['历史值']].corr()
            corr_with_target = corr_matrix['历史值'].drop('历史值').sort_values(ascending=False)
            corr_results[province] = corr_with_target
            
            # 输出相关性结果
            print(f"\n{province}特征与{factor_name}的相关性：")
            for feat, corr in corr_with_target.items():
                strength = "强正相关" if corr > 0.7 else "中等正相关" if corr > 0.3 else "弱相关或负相关"
                print(f"  {feat}: {corr:.3f}（{strength}）")
        except Exception as e:
            print(f"{province}相关性计算出错：{str(e)}")
    
    # 绘制相关性热力图（合并三省数据）
    try:
        plt.figure(figsize=(10, 6))
        corr_all = df_hist[feature_cols + ['历史值']].corr()
        im = plt.imshow(corr_all, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # 添加数值标签
        for i in range(len(corr_all.columns)):
            for j in range(len(corr_all.columns)):
                text = plt.text(j, i, f'{corr_all.iloc[i, j]:.2f}', 
                               ha="center", va="center", color="black", fontsize=10)
        
        # 图表配置
        plt.colorbar(im, label='Pearson相关系数')
        plt.xticks(range(len(corr_all.columns)), corr_all.columns, rotation=45)
        plt.yticks(range(len(corr_all.columns)), corr_all.columns)
        plt.title(f'特征与{factor_name}的相关性热力图（三省历史数据合并）', fontsize=12, pad=15)
        plt.tight_layout()
        
        save_dir = os.path.join(PROVINCE_DATA_DIR, "对比分析图")
        plt.savefig(os.path.join(save_dir, "特征相关性热力图.png"), bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"相关性热力图绘制出错：{str(e)}")

    # 4.2 对比三省关键特征的数值差异
    print(f"\n=== 三省关键特征数值对比 ===")
    # 找到预测起始年附近的数据
    target_year = 2025
    if target_year not in df['年份'].unique():
        target_year = min([year for year in df['年份'].unique() if year >= 2025], default=None)
    
    if target_year:
        df_target_year = df[df['年份'] == target_year].copy()
        print(f"使用{target_year}年数据进行特征对比")
    else:
        # 若没有2025年及以后数据，使用最新年份
        target_year = df['年份'].max()
        df_target_year = df[df['年份'] == target_year].copy()
        print(f"未找到2025年及以后数据，使用最新年份{target_year}的数据进行对比")
    
    # 整理特征对比表
    if not df_target_year.empty:
        value_cols = feature_cols.copy()
        if '预测值' in df_target_year.columns:
            value_cols.append('预测值')
        
        feature_compare = df_target_year.pivot_table(
            index='地区', 
            values=value_cols, 
            fill_value=0
        ).round(2)
        
        # 添加单位
        new_columns = []
        for col in feature_compare.columns:
            if 'GDP' in col:
                new_columns.append(f'{col}（亿元）')
            elif '人口' in col:
                new_columns.append(f'{col}（万人）')
            elif col == '预测值':
                new_columns.append(f'{factor_name}预测值（万平方米）')
            else:
                new_columns.append(col)
        feature_compare.columns = new_columns
        
        print(feature_compare)
        
        # 绘制关键特征对比柱状图
        try:
            # 选择两个最相关的特征进行可视化
            if corr_results:
                # 找到平均相关性最高的两个特征
                avg_corr = {}
                for feat in feature_cols:
                    avg = []
                    for prov_corr in corr_results.values():
                        if feat in prov_corr.index:
                            avg.append(abs(prov_corr[feat]))
                    if avg:
                        avg_corr[feat] = sum(avg)/len(avg)
                
                top_features = sorted(avg_corr.items(), key=lambda x: x[1], reverse=True)[:2]
                top_features = [f[0] for f in top_features]
            else:
                top_features = feature_cols[:2]  # 默认取前两个
            
            if len(top_features) >= 2:
                plt.figure(figsize=(14, 5))
                
                # 子图1：第一个关键特征
                plt.subplot(1, 2, 1)
                provinces = df_target_year['地区'].tolist()
                values = df_target_year[top_features[0]].tolist()
                colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                
                plt.bar(provinces, values, color=colors, alpha=0.8)
                plt.xlabel('省份')
                plt.ylabel(f'{top_features[0]}（亿元）' if 'GDP' in top_features[0] else f'{top_features[0]}（万人）')
                plt.title(f'{target_year}年三省{top_features[0]}对比')
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                
                # 添加数值标签
                for i, v in enumerate(values):
                    plt.text(i, v + max(values)*0.01, f'{int(v)}', ha='center', va='bottom')
                
                # 子图2：第二个关键特征
                plt.subplot(1, 2, 2)
                values2 = df_target_year[top_features[1]].tolist()
                
                plt.bar(provinces, values2, color=colors, alpha=0.8)
                plt.xlabel('省份')
                plt.ylabel(f'{top_features[1]}（亿元）' if 'GDP' in top_features[1] else f'{top_features[1]}（万人）')
                plt.title(f'{target_year}年三省{top_features[1]}对比')
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                
                # 添加数值标签
                for i, v in enumerate(values2):
                    plt.text(i, v + max(values2)*0.01, f'{int(v)}', ha='center', va='bottom')
                
                plt.tight_layout()
                save_dir = os.path.join(PROVINCE_DATA_DIR, "对比分析图")
                plt.savefig(os.path.join(save_dir, "三省关键特征对比.png"), bbox_inches='tight')
                plt.close()
                print(f"\n关键特征对比图已保存至：{os.path.join(PROVINCE_DATA_DIR, '对比分析图')}")
            else:
                print("特征不足，无法绘制特征对比图")
        except Exception as e:
            print(f"特征对比图绘制出错：{str(e)}")
    else:
        print("没有找到可用于对比的特征数据")

    # 4.3 输出影响因素结论
    print(f"\n=== 浙江省、北京市预测值偏大的可能原因分析 ===")
    print("1. 经济发展水平：")
    print("   - 若GDP与新增建筑面积呈强相关，浙江省和北京市较高的GDP可能是预测值偏大的主要原因。")
    print("2. 城镇化进程：")
    print("   - 城镇人口规模和增长速度是影响建筑需求的关键因素，北京市作为首都，城镇化水平高，可能推动建筑需求增长。")
    print("3. 产业结构影响：")
    print("   - 第三产业占比高的地区通常对商业建筑、办公楼需求更大，可能导致新增建筑面积预测值较高。")
    print("4. 政策与规划：")
    print("   - 北京市作为首都，可能有更多的基础设施建设规划；浙江省作为经济活跃地区，可能有更多的商业和住宅建设需求。")
    print("5. 模型特性：")
    print("   - 预测模型基于历史数据中的特征关系，如果历史上GDP等指标与建筑面积相关性强，这些指标的高水平会直接导致高预测值。")
