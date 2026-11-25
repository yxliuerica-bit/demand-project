# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np

# ---------------- 1. 路径与配置 ----------------
scenario_name = '基准'
BASE_DIR = "/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/"
EXP_FILE = os.path.join(BASE_DIR, "上游产品出口年度数据.xlsx")
IMP_FILE = os.path.join(BASE_DIR, "上游产品进口年度数据.xlsx")

# 历史产量文件路径
PRODUCTION_FILES = {
    "Steel": os.path.join(BASE_DIR, "分省历史粗钢产量数据.xls"),
    "Cement": os.path.join(BASE_DIR, "分省历史水泥产量数据.xls")
}

# 手动补充 2025 年产量数据 (单位：万吨)
MANUAL_2025_PROD = {
    "Steel": 96081.0,
    "Cement": 169300.0
}

# 输出路径
OUT_DIR = f"/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/历史消费需求核算结果"
os.makedirs(OUT_DIR, exist_ok=True)

# 进出口列名映射
TRADE_COLS = {
    "Steel": {"exp": "钢材出口数量 (万吨)", "imp": "钢材进口数量 (万吨)"},
    "Cement": {"exp": "水泥及水泥熟料出口数量 (万吨)", "imp": "水泥及水泥熟料进口数量 (万吨)"}
}

# 修改年份范围至 2025
TARGET_YEARS = list(range(2010, 2026)) 
CONVERSION_FACTOR = 0.95  

# ---------------- 2. 核心工具函数 ----------------

def load_national_production_history(path):
    """读取产量 Excel 并返回全国总量序列（吨）"""
    if not os.path.exists(path):
        print(f"❌ 找不到产量文件: {path}")
        return None
    
    df0 = pd.read_excel(path, header=None)
    header_row = 0
    for i in range(min(len(df0), 50)):
        if str(df0.iat[i, 0]).strip() == '地区':
            header_row = i
            break
    
    df = pd.read_excel(path, header=header_row)
    year_cols = [c for c in df.columns if re.fullmatch(r'\d{4}年', str(c).strip())]
    
    # 万吨 -> 吨
    df_years = df[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    nat_series = df_years.sum(axis=0) * 10000.0
    
    nat_series.index = nat_series.index.map(lambda x: int(re.search(r'\d{4}', str(x)).group()))
    return nat_series.sort_index()

def load_trade_data(file_path, col_name, target_years):
    """读取贸易数据并进行系数折算"""
    if col_name is None or not os.path.exists(file_path):
        return pd.Series(0.0, index=target_years)
    
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    
    time_col = next((c for c in df.columns if '时间' in c), None)
    if time_col is None: return pd.Series(0.0, index=target_years)
    
    df['year'] = df[time_col].astype(str).str.extract(r'(\d{4})').astype(int)
    
    search_pattern = col_name.strip().replace('(', r'[(（]').replace(')', r'[)）]')
    found_col = next((c for c in df.columns if re.search(search_pattern, c)), None)
    
    if not found_col:
        return pd.Series(0.0, index=target_years)
    
    # 万吨 -> 吨，并除以系数
    raw_series = (df.set_index('year')[found_col] * 10000.0) / CONVERSION_FACTOR
    
    # 对齐年份，如果 2025 年在贸易文件中不存在，ffill 会沿用 2024 的值
    return raw_series.reindex(target_years).ffill().fillna(0.0)

# ---------------- 3. 主计算程序 ----------------

def run_reverse_consumption_v2():
    for category, prod_path in PRODUCTION_FILES.items():
        print(f"正在核算 {category} 消费量 (含 2025 手动补全)...")
        
        # 1. 获取历史产量
        nat_prod = load_national_production_history(prod_path)
        if nat_prod is None: continue
        
        # --- 新增：手动添加 2025 年数据 ---
        # 将万吨转换为吨并填入序列
        nat_prod[2025] = MANUAL_2025_PROD[category] * 10000.0
        nat_prod = nat_prod.sort_index()
        
        common_years = [y for y in TARGET_YEARS if y in nat_prod.index]
        
        # 2. 获取进出口
        exp_col = TRADE_COLS[category]["exp"]
        imp_col = TRADE_COLS[category]["imp"]
        
        ser_exp_adj = load_trade_data(EXP_FILE, exp_col, common_years)
        ser_imp_adj = load_trade_data(IMP_FILE, imp_col, common_years)
        
        # 3. 计算消费量: 消费 = 产量 + (进口/0.95) - (出口/0.95)
        nat_consumption = nat_prod.reindex(common_years) + ser_imp_adj - ser_exp_adj
        
        # 4. 整理结果并换算为万吨输出
        res_df = pd.DataFrame({
            '年份': common_years,
            f'历史产量(万吨)': nat_prod.reindex(common_years).values / 10000.0,
            f'进口(万吨)': ser_imp_adj.values / 10000.0,
            f'出口(万吨)': ser_exp_adj.values / 10000.0,
            f'历史消费量(万吨)': nat_consumption.values / 10000.0
        })
        
        # 打印 2025 验证信息
        if 2025 in common_years:
            row_25 = res_df[res_df['年份'] == 2025].iloc[0]
            print(f"2025年核算结果 (万吨):")
            print(f"  - 手动录入产量: {row_25['历史产量(万吨)']:,.2f}")
            print(f"  - 核算消费量: {row_25['历史消费量(万吨)']:,.2f}")

        save_name = "粗钢" if category == "Steel" else "水泥"
        out_path = os.path.join(OUT_DIR, f"历史{save_name}表观消费量核算结果.xlsx")
        res_df.to_excel(out_path, index=False)
        print(f"✅ 保存完成: {out_path}\n")

if __name__ == "__main__":
    run_reverse_consumption_v2()