# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- 1. 路径与配置 ----------------
SCENARIO = "基准"   # 基准情景

GCAM_CSV = "/lustre/home/litong/intern/京津冀货运发展/data/GCAM_results/GCAM_results_china_for_plot.csv"
BASE_INPUT_DIR = f"/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_{SCENARIO}/全国完整数据"
OUT_DIR = "/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配"
os.makedirs(OUT_DIR, exist_ok=True)

SHARE_YEAR = 2020      # 使用 2020 年份额
OUTPUT_FACTOR = 1  # 放大倍数

# 消费量预测输入文件
INPUT_FILES = {
    "Steel": os.path.join(BASE_INPUT_DIR, "分省钢铁消费量预测.xlsx"),
    "Cement": os.path.join(BASE_INPUT_DIR, "分省水泥消费量预测.xlsx")
}

# 输出列名定义（严格对应要求）
COLUMN_ORDER = [
    'year', '北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑龙江省', 
    '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省', 
    '湖南省', '广东省', '广西壮族自治区', '海南省', '重庆市', '四川省', '贵州省', '云南省', 
    '西藏自治区', '陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区', '全国', '京津冀及周边8省市'
]

# GCAM 区域映射（使用全称以匹配消费量文件列名）
REGION_MAP = {
    "AH":"安徽省","BJ":"北京市","CQ":"重庆市","FJ":"福建省","GD":"广东省","GS":"甘肃省","GX":"广西壮族自治区","GZ":"贵州省",
    "HA":"河南省","HB":"湖北省","HE":"河北省","HI":"海南省","HL":"黑龙江省","HN":"湖南省","JL":"吉林省","JS":"江苏省",
    "JX":"江西省","LN":"辽宁省","NM":"内蒙古自治区","NX":"宁夏回族自治区","QH":"青海省","SC":"四川省","SD":"山东省","SH":"上海市",
    "SN":"陕西省","SX":"山西省","TJ":"天津市","XJ":"新疆维吾尔自治区","XZ":"西藏自治区","YN":"云南省","ZJ":"浙江省"
}

# ---------------- 2. 工具函数 ----------------

def load_gcam_share(var_prefix):
    """从 GCAM CSV 获取指定变量的省份份额"""
    gcam = pd.read_csv(GCAM_CSV)
    mask = (gcam["year"] == SHARE_YEAR) & (gcam["scenario"].str.lower() == SCENARIO)
    t = gcam[mask & gcam["var"].str.startswith(var_prefix, na=False)].copy()
    
    t = t.groupby("region")["value"].sum()
    shares = t / t.sum()
    
    # 转换为全称映射
    share_dict = {REGION_MAP[r]: v for r, v in shares.items() if r in REGION_MAP}
    return share_dict

def process_and_save(category, input_path, gcam_prefix):
    """读取消费量总量并按份额分配"""
    print(f"正在处理 {category} 数据...")
    
    # A. 获取份额
    share_dict = load_gcam_share(gcam_prefix)
    
    # B. 读取全国总量（从消费预测表中提取『全国』列）
    df_input = pd.read_excel(input_path)
    if '全国' not in df_input.columns:
        raise ValueError(f"文件 {input_path} 中缺少 '全国' 列")
    
    nat_series = df_input.set_index('year')['全国']
    years = nat_series.index.tolist()
    
    # C. 构建输出结果
    out_df = pd.DataFrame(index=years, columns=COLUMN_ORDER)
    out_df['year'] = years
    
    # 填入分配后的省份值
    for prov in COLUMN_ORDER:
        if prov in ['year', '全国', '京津冀及周边8省市']:
            continue
        share = share_dict.get(prov, 0.0)
        out_df[prov] = (nat_series * share * OUTPUT_FACTOR).values
    
    # 填入总量列（原始全国总量 * 10000）
    out_df['全国'] = (nat_series * OUTPUT_FACTOR).values
    
    # 填入京津冀及周边8省市（北京、天津、河北、山西、内蒙古、山东、河南、陕西）
    jjj_8_list = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '山东省', '河南省', '陕西省']
    out_df['京津冀及周边8省市'] = out_df[jjj_8_list].sum(axis=1)
    
    # D. 保存
    file_label = "粗钢" if category == "Steel" else "水泥"
    out_path = os.path.join(OUT_DIR, f"分省历史{file_label}产量_按GCAM{SHARE_YEAR}份额分配.xlsx")
    out_df.to_excel(out_path, index=False)
    print(f"[OK] 已保存至: {out_path}")

# ---------------- 3. 执行主程序 ----------------

if __name__ == "__main__":
    # 处理钢铁
    process_and_save("Steel", INPUT_FILES["Steel"], "Production|Steel")
    # 处理水泥
    process_and_save("Cement", INPUT_FILES["Cement"], "Production|Cement")