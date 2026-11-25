# -*- coding: utf-8 -*-
import pandas as pd
import plotly.graph_objects as go
import os

# ---------------------------
# 1. 路径配置
# ---------------------------
# 火电消费预测数据
coal_pred_path = '/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_基准/全国完整数据/分省火电消费量预测.xlsx'
# 2024年运输流量基准数据
flow_2024_path = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/transfer_matrix_province_2024annual.csv'
# 输出目录
save_dir = '/lustre/home/litong/intern/京津冀货运发展/output/4.运量'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ---------------------------
# 2. 数据加载与比例计算
# ---------------------------
# 读取火电预测数据
df_coal = pd.read_excel(coal_pred_path)
# 将列名统一处理，去除可能存在的空格
df_coal.columns = [c.strip() for c in df_coal.columns]

def get_growth_ratios(target_year):
    """计算各省 target_year 相对 2024 年的火电消费比例"""
    row_2024 = df_coal[df_coal['year'] == 2024].iloc[0]
    row_target = df_coal[df_coal['year'] == target_year].iloc[0]
    
    # 计算比例 (目标年份 / 2024)
    # 注意：排除 'year', '全国', '京津冀及周边8省市' 等非省份列
    exclude_cols = ['year', '全国', '京津冀及周边8省市']
    provinces = [c for c in df_coal.columns if c not in exclude_cols]
    
    ratios = {}
    for prov in provinces:
        val_2024 = row_2024[prov]
        if val_2024 == 0 or pd.isna(val_2024):
            ratios[prov] = 1.0 # 默认不增长
        else:
            ratios[prov] = row_target[prov] / val_2024
    return ratios

# 获取比例字典
ratios_2030 = get_growth_ratios(2030)
ratios_2035 = get_growth_ratios(2035)

# 读取2024流量数据
df_flow_2024 = pd.read_csv(flow_2024_path)
df_flow_2024 = df_flow_2024.dropna(subset=['src_prov_name', 'dst_prov_name']).copy()

# ---------------------------
# 3. 流量预测与系数调整函数
# ---------------------------
def predict_and_adjust_flow(target_year, ratios, factor_op):
    """
    根据流入省(dst)的消费比例调整流量
    target_year: 目标年份
    ratios: 火电增长比例字典
    factor_op: 特定的数学调整逻辑函数
    """
    df = df_flow_2024.copy()
    
    # 步骤1: 基于流入省的火电需求比例进行缩放
    # 假设流量与流入地的火电消费需求成正比
    def apply_ratio(row):
        dst = row['dst_prov_name']
        # 匹配省份名称（处理简称或全称差异，如需要可加 fuzzy match）
        ratio = ratios.get(dst, 1.0)
        return row['nox'] * ratio

    df['predicted_nox'] = df.apply(apply_ratio, axis=1)
    
    # 步骤2: 执行题目要求的特定数学调整
    # 2030: / 0.35 * 0.8
    # 2035: / 0.35
    df['adjusted_nox'] = factor_op(df['predicted_nox'])
    
    return df

# 定义调整系数逻辑
op_2030 = lambda x: (x / 0.35) * 0.8
op_2035 = lambda x: (x / 0.35)

df_2030 = predict_and_adjust_flow(2030, ratios_2030, op_2030)
df_2035 = predict_and_adjust_flow(2035, ratios_2035, op_2035)

# ---------------------------
# 4. 桑基图绘制函数
# ---------------------------
def plot_future_sankey(df, year, top_n=50):
    # 过滤自循环
    df_plot = df[df['src_prov_name'] != df['dst_prov_name']].nlargest(top_n, 'adjusted_nox').copy()
    
    # 强制左右布局逻辑
    df_plot['src_label'] = df_plot['src_prov_name'] + " (出)"
    df_plot['dst_label'] = df_plot['dst_prov_name'] + " (进)"
    
    all_src = sorted(df_plot['src_label'].unique())
    all_dst = sorted(df_plot['dst_label'].unique())
    all_labels = all_src + all_dst
    label_map = {label: i for i, label in enumerate(all_labels)}
    
    fig = go.Figure(data=[go.Sankey(
        arrangement = "perpendicular",
        node = dict(
            pad = 20, thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = [l.replace(" (出)", "").replace(" (进)", "") for l in all_labels],
            color = ["#3498db"] * len(all_src) + ["#e74c3c"] * len(all_dst)
        ),
        link = dict(
            source = df_plot['src_label'].map(label_map),
            target = df_plot['dst_label'].map(label_map),
            value = df_plot['adjusted_nox'],
            color = 'rgba(46, 204, 113, 0.2)'
        )
    )])

    fig.update_layout(
        title_text=f"{year}年省际预测运输流量桑基图 (调整后 | Top {top_n})",
        font_size=12, width=1200, height=1000
    )

    save_path = os.path.join(save_dir, f'sankey_{year}_adjusted.html')
    fig.write_html(save_path)
    print(f"✅ {year}年桑基图已保存至: {save_path}")

# ---------------------------
# 执行
# ---------------------------
plot_future_sankey(df_2030, 2030)
plot_future_sankey(df_2035, 2035)