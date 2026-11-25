# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import numpy as np

# ---------------- 1. 路径与配置 ----------------
GCAM_SCENARIO = "Current"
scenario_name = '基准'

GCAM_CSV = "/lustre/home/litong/intern/京津冀货运发展/data/GCAM_results/GCAM_results_china_for_plot.csv"
BASE_INPUT_DIR = f"/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_{scenario_name}/全国完整数据"
HIST_DATA_DIR = "/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据"
EXP_FILE = os.path.join(HIST_DATA_DIR, "上游产品出口年度数据.xlsx")
IMP_FILE = os.path.join(HIST_DATA_DIR, "上游产品进口年度数据.xlsx")

OUT_DIR = f"/lustre/home/litong/intern/京津冀货运发展/output/3.产量分配/Sce_{scenario_name}"
os.makedirs(OUT_DIR, exist_ok=True)

SHARE_YEAR = 2020
OUTPUT_FACTOR = 1
TRADE_CONVERSION = 0.95  # 新增：贸易量折算系数 (如钢材转粗钢)
HIST_LIMIT_YEAR = 2024

# 2025年手动总量配置 (单位：万吨)
MANUAL_2025_TOTAL = {
    "Steel": 96081.0,
    # NBS 2025 communique rounds cement output to 16.9 hundred-million tonnes.
    "Cement": 169000.0
}

# 消费量预测输入 (单位：吨)
INPUT_FILES = {
    "Steel": os.path.join(BASE_INPUT_DIR, "分省钢铁消费量预测.xlsx"),
    "Cement": os.path.join(BASE_INPUT_DIR, "分省水泥消费量预测.xlsx"),
    "Power": os.path.join(BASE_INPUT_DIR, "分省火电消费量预测.xlsx")
}

# 历史产量文件 (单位：万吨)
HIST_PROD_FILES = {
    "Steel": os.path.join(HIST_DATA_DIR, "分省历史粗钢产量数据.xls"),
    "Cement": os.path.join(HIST_DATA_DIR, "分省历史水泥产量数据.xls")
}

TRADE_COLS = {
    "Steel": {"exp": "钢材出口数量 (万吨)", "imp": "钢材进口数量 (万吨)"},
    "Cement": {"exp": "水泥及水泥熟料出口数量 (万吨)", "imp": None},
    "Power": {"exp": "煤出口数量 (万吨)", "imp": "煤进口数量 (万吨)"}
}

COLUMN_ORDER = [
    'year', '北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑龙江省', 
    '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省', 
    '湖南省', '广东省', '广西壮族自治区', '海南省', '重庆市', '四川省', '贵州省', '云南省', 
    '西藏自治区', '陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区', '全国', '京津冀及周边8省市'
]

REGION_MAP = {
    "AH":"安徽省","BJ":"北京市","CQ":"重庆市","FJ":"福建省","GD":"广东省","GS":"甘肃省","GX":"广西壮族自治区","GZ":"贵州省",
    "HA":"河南省","HB":"湖北省","HE":"河北省","HI":"海南省","HL":"黑龙江省","HN":"湖南省","JL":"吉林省","JS":"江苏省",
    "JX":"江西省","LN":"辽宁省","NM":"内蒙古自治区","NX":"宁夏回族自治区","QH":"青海省","SC":"四川省","SD":"山东省","SH":"上海市",
    "SN":"陕西省","SX":"山西省","TJ":"天津市","XJ":"新疆维吾尔自治区","XZ":"西藏自治区","YN":"云南省","ZJ":"浙江省"
}

# ---------------- 2. 工具函数 ----------------

def load_national_hist_total(file_path):
    """提取历史Excel全国产量总和 (吨)"""
    df_raw = pd.read_excel(file_path, header=None)
    header_idx = 0
    for i, row in df_raw.iterrows():
        if "地区" in str(row.values):
            header_idx = i
            break
    df = pd.read_excel(file_path, header=header_idx)
    year_cols = [c for c in df.columns if re.search(r'\d{4}', str(c))]
    # 万吨转吨
    df_years = df[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0) * 10000.0
    nat_total = df_years.sum(axis=0)
    nat_total.index = nat_total.index.map(lambda x: int(re.search(r'\d{4}', str(x)).group()))
    return nat_total

def load_provincial_hist(file_path):
    """Load NBS provincial product output in tonnes, indexed by year."""
    raw = pd.read_excel(file_path, header=None)
    header_idx = next(
        (i for i, row in raw.iterrows() if "地区" in str(row.values)),
        None,
    )
    if header_idx is None:
        raise ValueError(f"历史数据表中未找到地区表头：{file_path}")
    data = pd.read_excel(file_path, header=header_idx)
    area_col = next((c for c in data.columns if "地区" in str(c)), None)
    if area_col is None:
        raise ValueError(f"历史数据表中未找到地区列：{file_path}")
    data[area_col] = data[area_col].astype(str).str.strip()
    year_cols = {
        int(match.group()): col
        for col in data.columns
        if (match := re.search(r"\d{4}", str(col)))
    }
    provinces = [c for c in COLUMN_ORDER if c not in {"year", "全国", "京津冀及周边8省市"}]
    missing = set(provinces) - set(data[area_col])
    if missing:
        raise ValueError(f"历史数据缺少省级记录：{sorted(missing)}")
    values = data.set_index(area_col).reindex(provinces)
    history = values[[year_cols[y] for y in sorted(year_cols)]].T
    history.index = sorted(year_cols)
    history.columns.name = "省份"
    history.index.name = "year"
    history = history.apply(pd.to_numeric, errors="coerce")
    if history.isna().any().any():
        raise ValueError(f"历史省级产量存在缺失值：{file_path}")
    return history * 10000.0

def forecast_shares_from_history(history, years):
    """Extend recent official provincial shares with bounded relative trends."""
    shares = history.div(history.sum(axis=1), axis=0)
    base_year = int(shares.index.max())
    trend_start = max(int(shares.index.min()), base_year - 5)
    start = shares.loc[trend_start].clip(lower=1e-8)
    end = shares.loc[base_year].clip(lower=1e-8)
    annual_change = (np.log(end / start) / (base_year - trend_start)).clip(-0.03, 0.03)
    projected = {}
    for year in years:
        raw = end * np.exp(annual_change * (year - base_year))
        projected[year] = raw / raw.sum()
    return pd.DataFrame(projected).T.reindex(columns=history.columns)

def load_trade_data_adjusted(file_path, col_name, target_years):
    """读取贸易数据并除以 0.95 折算系数 (吨)"""
    if col_name is None or not os.path.exists(file_path):
        return pd.Series(0.0, index=target_years)
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    time_col = next((c for c in df.columns if '时间' in c), None)
    df['year'] = df[time_col].astype(str).str.extract(r'(\d{4})').astype(int)
    
    target_col = col_name.strip()
    if target_col not in df.columns:
        search_pattern = target_col.replace('(', r'[(（]').replace(')', r'[)）]')
        found_col = next((c for c in df.columns if re.search(search_pattern, c)), None)
        target_col = found_col
        
    # 万吨转吨，并除以 0.95
    raw_series = (df.set_index('year')[target_col] * 10000.0) / TRADE_CONVERSION
    return raw_series.reindex(target_years).ffill().fillna(0.0)

def load_gcam_share(var_prefix):
    """归一化分配份额"""
    gcam = pd.read_csv(GCAM_CSV)
    mask = (gcam["year"] == SHARE_YEAR) & (gcam["scenario"].str.lower() == GCAM_SCENARIO.lower())
    t = gcam[mask & (gcam["var"] == var_prefix)].copy()
    if t.empty: return {}
    t_grouped = t.groupby("region")["value"].sum()
    mapped = {REGION_MAP[r]: v for r, v in t_grouped.items() if r in REGION_MAP}
    total_val = sum(mapped.values())
    if total_val > 0:
        return {k: v / total_val for k, v in mapped.items()}
    return {}

def process_and_save(category, input_path, gcam_prefix):
    print(f"================ 正在处理 {category} 行业 (贸易量除以{TRADE_CONVERSION}) ================")
    
    # A. 基础数据加载 (吨)
    df_input = pd.read_excel(input_path)
    nat_consumption = df_input.set_index('year')['全国']
    years = sorted(nat_consumption.index.tolist())
    share_dict = load_gcam_share(gcam_prefix) if category != "Cement" else {}
    
    # B. 进出口处理 (吨，已含系数)
    ser_exp_adj = load_trade_data_adjusted(EXP_FILE, TRADE_COLS[category]["exp"], years)
    ser_imp_adj = load_trade_data_adjusted(IMP_FILE, TRADE_COLS[category]["imp"], years)
    
    # C. 2026+ 产量预测
    nat_prod_forecast = (nat_consumption - ser_imp_adj + ser_exp_adj) * OUTPUT_FACTOR

    # D. 2010-2024 历史总量 (吨)
    nat_hist_total = load_national_hist_total(HIST_PROD_FILES[category]) if category in HIST_PROD_FILES else None
    provincial_history = (
        load_provincial_hist(HIST_PROD_FILES[category])
        if category == "Cement"
        else None
    )
    cement_forecast_shares = (
        forecast_shares_from_history(provincial_history, years)
        if provincial_history is not None
        else None
    )

    # E. 全年份 GCAM 分配
    out_df = pd.DataFrame(index=years, columns=COLUMN_ORDER)
    out_df['year'] = years
    prov_cols = [c for c in COLUMN_ORDER if c not in ['year', '全国', '京津冀及周边8省市']]

    for yr in years:
        if category == "Cement" and yr in provincial_history.index:
            for prov in prov_cols:
                out_df.loc[out_df['year'] == yr, prov] = provincial_history.loc[yr, prov]
            continue

        # 确定年度总量锚点 (吨)
        if yr <= HIST_LIMIT_YEAR and nat_hist_total is not None:
            current_nat_total = nat_hist_total.get(yr, 0)
        elif yr == 2025 and category in MANUAL_2025_TOTAL:
            current_nat_total = MANUAL_2025_TOTAL[category] * 10000.0
        else:
            current_nat_total = nat_prod_forecast.loc[yr]

        # 空间分配
        for prov in prov_cols:
            if category == "Cement" and yr > HIST_LIMIT_YEAR:
                share = cement_forecast_shares.loc[yr, prov]
            else:
                share = share_dict.get(prov, 0.0)
            out_df.loc[out_df['year'] == yr, prov] = current_nat_total * share

    # F. 汇总
    out_df['全国'] = out_df[prov_cols].sum(axis=1)
    jjj_8_list = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '山东省', '河南省', '陕西省']
    out_df['京津冀及周边8省市'] = out_df[jjj_8_list].sum(axis=1)
    
    # G. 保存
    label_map = {"Steel": "粗钢", "Cement": "水泥", "Power": "煤炭"}
    file_label = label_map.get(category, category)
    out_path = os.path.join(OUT_DIR, f"分省{file_label}产量_历史加预测.xlsx")
    out_df.to_excel(out_path, index=False)
    
    actual_2025 = out_df.loc[out_df['year'] == 2025, '全国'].values[0] / 10000.0
    print(f"2025年核对: 目标总量 {MANUAL_2025_TOTAL.get(category, 'N/A')} 万吨 | 分配总量 {actual_2025:.2f} 万吨")
    print(f"✅ 文件已保存: {out_path}\n")

# ---------------- 3. 执行 ----------------
if __name__ == "__main__":
    try:
        process_and_save("Steel", INPUT_FILES["Steel"], "Production|Steel")
        process_and_save("Cement", INPUT_FILES["Cement"], "Production|Cement")
        process_and_save("Power", INPUT_FILES["Power"], "Capacity|Electricity|Coal")
        print("🎉 任务完成！")
    except Exception as e:
        print(f"🔥 错误: {e}")