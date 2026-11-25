# -*- coding: utf-8 -*-
import pandas as pd
import plotly.graph_objects as go
import os

# ---------------------------
# 1. 数据加载与预处理
# ---------------------------
file_path = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/transfer_matrix_province_2024annual.csv'
df = pd.read_csv(file_path)

# 过滤掉自循环（dst_prov_name == src_prov_name），只看省际传输
df_inter = df[df['src_prov_name'] != df['dst_prov_name']].copy()

# ---------------------------
# 2. 绘制桑基图
# ---------------------------
def plot_sankey(df, top_n=50):
    # 设定保存路径
    save_dir = '/lustre/home/litong/intern/京津冀货运发展/output/4.运量'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # --- 数据清洗 ---
    # 1. 删除省份名称为空的行，避免排序和绘图报错
    df_clean = df.dropna(subset=['src_prov_name', 'dst_prov_name']).copy()
    
    # 2. 确保省份列为字符串类型
    df_clean['src_prov_name'] = df_clean['src_prov_name'].astype(str)
    df_clean['dst_prov_name'] = df_clean['dst_prov_name'].astype(str)
    
    # 3. 筛选流量最大的前 N 条数据
    df_top = df_clean.nlargest(top_n, 'nox').copy()
    
    # --- 布局优化逻辑 ---
    # 给起点和终点加上不同的后缀，强制将它们分为“流出组”和“流入组”，实现左右对开布局
    df_top['src_label'] = df_top['src_prov_name'] + " (出)"
    df_top['dst_label'] = df_top['dst_prov_name'] + " (进)"
    
    # 构建节点列表并排序
    all_src = sorted(df_top['src_label'].unique())
    all_dst = sorted(df_top['dst_label'].unique())
    all_labels = all_src + all_dst
    
    # 建立标签到索引的映射
    label_map = {label: i for i, label in enumerate(all_labels)}
    
    # 配置节点颜色：左侧流出组用蓝色，右侧流入组用红色
    node_colors = ["#3498db"] * len(all_src) + ["#e74c3c"] * len(all_dst)

    # 绘制桑基图
    fig = go.Figure(data=[go.Sankey(
        arrangement = "perpendicular", # 强制垂直排列，避免节点重叠或交叉混乱
        node = dict(
            pad = 20,                  # 节点垂直间距
            thickness = 20,            # 节点宽度
            line = dict(color = "black", width = 0.5),
            # 显示时去掉辅助后缀，保持图表美观
            label = [label.replace(" (出)", "").replace(" (进)", "") for label in all_labels],
            color = node_colors
        ),
        link = dict(
            source = df_top['src_label'].map(label_map),
            target = df_top['dst_label'].map(label_map),
            value = df_top['nox'],
            color = 'rgba(46, 204, 113, 0.2)' # 连线使用淡绿色半透明
        )
    )])

    fig.update_layout(
        title_text=f"省际运输流量桑基图 (左侧：流出 | 右侧：流入 | Top {top_n})", 
        font_size=12,
        width=1200,   # 增加宽度让长距离连线更清晰
        height=1000   # 增加高度确保所有省份标签不重叠
    )

    # 保存为 HTML 格式（交互式）
    save_path_html = os.path.join(save_dir, 'sankey_fixed.html')
    fig.write_html(save_path_html)
    print(f"✅ 桑基图已保存至: {save_path_html}")

# ---------------------------
# 执行绘图
# ---------------------------
if __name__ == "__main__":
    # 展示流量最大的前 50 条路径
    plot_sankey(df_inter, top_n=50)