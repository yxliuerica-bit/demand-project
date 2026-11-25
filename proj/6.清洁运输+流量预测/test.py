# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =================== 1. 环境与路径配置 ===================
# 基础数据路径（请确保这些文件在您的目录下）
PROD_PATH = "/lustre/home/litong/intern/京津冀货运发展/output/产量分配/预测表/分省粗钢产量_历史+预测_2010-2035_逐年衰减版.xlsx"

# 输出路径
OUT_DIR = "/lustre/home/litong/intern/京津冀货运发展/output/钢铁物流预测_看板展示/"
os.makedirs(OUT_DIR, exist_ok=True)

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 时间范围
YEARS = list(range(2024, 2036))
# 目标 8 省市定义
TARGET_8_PROVINCES = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '山东省', '河南省', '陕西省']

# =================== 2. 模型参数设定 ===================
# 大宗原燃料运量系数 (吨/吨钢) - 对应生产环节进项
# 包含：矿石(1.6)、燃料煤(0.22)、焦炭(0.46)、石灰石(0.20)、固废(0.68)
UF_INPUTS = 1.60 + 0.22 + 0.46 + 0.20 + 0.68 

# 短流程占比 (EAF Share) 线性插值设定
EAF_POINTS = {2020: 0.10, 2025: 0.12, 2030: 0.17, 2035: 0.22}

# 用于看板展示的指定配色
BAR_COLORS = ['#e1f0f9', '#b3cde3', '#e4f4d1', '#b2df8a', '#fbb4ae', '#e78ac3', '#fee0b6', '#fdbf6f']

# =================== 3. 核心计算函数 ===================

def get_eaf_series():
    """生成逐年短流程占比序列"""
    s = pd.Series(EAF_POINTS).sort_index()
    return s.reindex(range(2020, 2036)).interpolate().reindex(YEARS).ffill().bfill().values

def predict_policy_scenario_share():
    """根据'政策情景'定义清洁运输比例序列 (2025年后达80%)"""
    df_share = pd.DataFrame(index=YEARS)
    # 假设 2024 年现状为 35%，2025年及以后达到 80%
    shares = np.full(len(YEARS), 0.80)
    shares[0] = 0.35 # 2024
    df_share['政策情景'] = shares
    return df_share

def run_logistics_model():
    # --- A. 数据加载 ---
    df_prod = pd.read_excel(PROD_PATH, sheet_name="wide_tons", index_col=0)
    
    # 自动获取 Excel 中所有合法的省份列（用于计算全国汇总）
    all_valid_columns = [c for c in df_prod.columns if c not in ['year', '全国', '京津冀及周边8省市']]
    
    # 提取所需年份数据并转换单位（吨）
    prod_data_all = df_prod.loc[YEARS, all_valid_columns].astype(float) # 全部省份数据

    # --- B. 计算原燃料运输需求与清洁运输量 ---
    eaf_v = get_eaf_series()
    # 只有长流程需要消耗大量矿石和焦炭
    total_input_logistics = prod_data_all.mul((1 - eaf_v), axis=0) * UF_INPUTS
    
    # 加载指定的政策情景比例 (例如：政策情景 80%)
    policy_share = predict_policy_scenario_share()['政策情景']
    
    # 计算清洁运输量 (吨)
    clean_vol_all_prov = total_input_logistics.mul(policy_share, axis=0)
            
    return total_input_logistics, clean_vol_all_prov

# =================== 4. 绘图与展示函数 ===================

def draw_combined_dashboard(total_vol_df, clean_vol_df):
    """
    绘制复合看板：全国政策总需求(折线) + 区域构成(柱状)
    """
    # 1. 准备绘图数据 (全部换算为万吨)
    total_vol_10k = total_vol_df / 10000
    clean_vol_10k = clean_vol_df / 10000
    
    # 全国总原燃料清洁运输需求量 (折线图数据对象)
    national_clean_demand = clean_vol_10k.sum(axis=1)
    
    # 提取 8 省市的清洁运输量 (柱状图堆叠对象)
    # 注意：这里需要检查原始数据中是否包含所有定义的 8 省市
    valid_target_provinces = [p for p in TARGET_8_PROVINCES if p in clean_vol_10k.columns]
    data_8prov_clean = clean_vol_10k[valid_target_provinces]
    
    # 绘图逻辑
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
    
    # --- B. 绘制右侧堆叠柱状图 (京津冀及周边8省市构成) ---
    # 我们将柱状图作为底层背景
    bottom = np.zeros(len(YEARS))
    for i, prov in enumerate(valid_target_provinces):
        values = data_8prov_clean[prov].values
        ax1.bar(YEARS, values, bottom=bottom, color=BAR_COLORS[i % len(BAR_COLORS)], 
                label=prov, width=0.7, alpha=0.9, zorder=2)
        bottom += values
        
    ax1.set_ylabel("原燃料清洁运输量 (万吨)", fontsize=11, fontweight='bold')
    ax1.set_xticks(YEARS)
    ax1.tick_params(axis='both', labelsize=10)
    ax1.grid(axis='y', linestyle=':', alpha=0.3, zorder=1)

    # --- A. 绘制全国总清洁运输需求折线 (叠加在柱状图上方) ---
    # 这里我们直接叠加在 ax1 上，因为单位一致（均为万吨）。如果单位不一致则需要使用 ax1.twinx()
    ax1.plot(YEARS, national_clean_demand.values, color='#d62728', 
             marker='o', markersize=6, linewidth=2.5, label='全国总清洁运输需求 (政策情景)', zorder=10)
    
    # 在折线顶部标注数值 (Top Label)
    for x, y in zip(YEARS, national_clean_demand.values):
        ax1.text(x, y + (national_clean_demand.max()*0.02), f'{y:.0f}', 
                 ha='center', va='bottom', fontsize=9, color='#d62728', fontweight='bold', zorder=11)

    # 整体标题与修饰
    plt.title("全国钢铁原燃料清洁运输总需求与京津冀及周边区域构成预测 (2024-2035)", fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlim(YEARS[0]-0.5, YEARS[-1]+0.5) # 调整x轴范围使柱子不贴边

    # 合并两个来源的图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    # 调整图例顺序，将折线图放在第一位
    order = [labels1.index('全国总清洁运输需求 (政策情景)')] + [i for i, l in enumerate(labels1) if l != '全国总清洁运输需求 (政策情景)']
    ax1.legend([handles1[idx] for idx in order], [labels1[idx] for idx in order],
               loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, fontsize=10, shadow=True)

    # 保存图片
    plt.tight_layout()
    save_path = os.path.join(OUT_DIR, "BAU情景_清洁运输预测看板.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f">>> 复合展示看板已生成并保存至: {save_path}")
    plt.show()

# =================== 5. 执行主程序 ===================
if __name__ == "__main__":
    total_v, clean_v_all = run_logistics_model()
    draw_combined_dashboard(total_v, clean_v_all)
    print(">>> 任务完成！")