# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

def calculate_thermal_power_ratio():
    # ================= 配置路径 =================
    # 基础输出目录
    base_fopath = '/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_基准/'
    # 源数据目录（存放各省CSV的文件夹）
    prov_data_dir = os.path.join(base_fopath, '省份完整数据')
    # 结果输出目录
    output_dir = os.path.join(base_fopath, '火力发电量占比分析')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # ================= 逻辑处理 =================
    print("开始扫描省份数据并计算火力占比...")
    
    all_province_results = []

    # 1. 获取目录下所有文件
    if not os.path.exists(prov_data_dir):
        print(f"错误: 找不到目录 {prov_data_dir}，请确认路径是否正确。")
        return

    files = os.listdir(prov_data_dir)
    
    # 2. 筛选出总发电量文件，并寻找对应的火力发电量文件
    total_gen_files = [f for f in files if '发电量完整数据.csv' in f and '火力' not in f]
    
    for file_name in total_gen_files:
        # 提取省份名称（例如：从 "北京市发电量完整数据.csv" 提取 "北京市"）
        province = file_name.replace('发电量完整数据.csv', '')
        thermal_file_name = f"{province}火力发电量完整数据.csv"
        
        if thermal_file_name in files:
            try:
                # 读取数据
                df_total = pd.read_csv(os.path.join(prov_data_dir, file_name))
                df_thermal = pd.read_csv(os.path.join(prov_data_dir, thermal_file_name))
                
                # 合并两个指标
                # 关键：我们需要把历史平滑和预测平滑合并成一个连续的序列
                df = pd.merge(df_total[['地区', '年份', '历史_平滑_发电量', '预测_平滑优先_发电量']],
                              df_thermal[['年份', '历史_平滑_火力发电量', '预测_平滑优先_火力发电量']],
                              on='年份', how='inner')
                
                # 融合数值：优先取历史平滑，如果没有则取预测值
                df['总发电量_融合'] = df['历史_平滑_发电量'].fillna(df['预测_平滑优先_发电量'])
                df['火力发电量_融合'] = df['历史_平滑_火力发电量'].fillna(df['预测_平滑优先_火力发电量'])
                
                # 计算占比 (处理除以0的情况)
                df['火力占比'] = df['火力发电量_融合'] / df['总发电量_融合'].replace(0, np.nan)
                
                # 保留核心列
                res = df[['地区', '年份', '总发电量_融合', '火力发电量_融合', '火力占比']].copy()
                all_province_results.append(res)
                print(f"已处理: {province}")
                
            except Exception as e:
                print(f"处理 {province} 时出错: {e}")
        else:
            print(f"跳过: {province} (未找到对应的火力数据文件)")

    # 3. 汇总与导出
    if all_province_results:
        # 合并所有省份数据
        final_df = pd.concat(all_province_results, ignore_index=True)
        final_df = final_df.sort_values(['地区', '年份'])
        
        # 导出分省详细占比
        detail_file = os.path.join(output_dir, '分省火力发电占比汇总.csv')
        final_df.to_csv(detail_file, index=False, encoding='utf-8-sig')
        
        # 计算全国维度的平均占比（加权平均：sum(火力)/sum(总量)）
        national_summary = final_df.groupby('年份').agg({
            '总发电量_融合': 'sum',
            '火力发电量_融合': 'sum'
        }).reset_index()
        national_summary['全国平均火力占比'] = national_summary['火力发电量_融合'] / national_summary['总发电量_融合'].replace(0, np.nan)
        
        # 导出全国汇总表
        summary_file = os.path.join(output_dir, '全国火力发电占比趋势.csv')
        national_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*30)
        print(f"分析完成！结果已保存至:\n1. {detail_file}\n2. {summary_file}")
        print("="*30)
    else:
        print("未匹配到任何有效的发电量数据。")

if __name__ == '__main__':
    calculate_thermal_power_ratio()