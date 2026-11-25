import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# --- 1. 初始化设置 ---
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False

base_path = "/lustre/home/litong/intern/京津冀货运发展/"
data_path = base_path + "data/预测原始数据/电力部门相关数据/"
out_path = base_path + "output/7.电力部门/7.1电力需求预测/工业-高耗能行业"
if not os.path.exists(out_path): os.makedirs(out_path)

products = {
    '乙烯': '乙烯产量分省年度数据.csv', 
    '玻璃': '平板玻璃产量分省年度数据.csv', 
    '烧碱': '烧碱产量分省年度数据.csv', 
    '焦炭': '焦炭产量分省年度数据.csv'
}

cols = ['Year', '北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑龙江省', '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省', '湖南省', '广东省', '广西壮族自治区', '海南省', '重庆市', '四川省', '贵州省', '云南省', '西藏自治区', '陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区', '全国', '京津冀及周边8省市']

# --- 2. 基础数据加载函数 ---
def load_base_data(path):
    df = pd.read_csv(path, header=None)
    df.columns = cols
    for c in cols[1:]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    return df

print("正在加载 GDP 和人口数据...")
df_gdp = load_base_data(base_path + "output/scenarios/gdp_predictions_基准_完整年份.csv")
df_pop = load_base_data(base_path + "output/scenarios/population_predictions_基准_完整年份.csv")

# --- 3. 核心计算循环 ---
for prod_name, file_name in products.items():
    print(f"\n[{prod_name}] 正在处理中...")
    
    # A. 鲁棒性读取产量数据
    try:
        df_prod_raw = pd.read_csv(data_path + file_name, header=2)
    except Exception as e:
        print(f"读取文件 {file_name} 失败: {e}")
        continue

    # 彻底清洗：去除列名和数据中的制表符 \t 和空格
    df_prod_raw.columns = df_prod_raw.columns.str.replace(r'[\t\s]', '', regex=True)
    df_prod_raw['地区'] = df_prod_raw['地区'].astype(str).str.replace(r'[\t\s]', '', regex=True)
    actual_cols = df_prod_raw.columns.tolist()
    
    # B. 提取历史长表 (2010-2024)
    hist_years = range(2010, 2025)
    hist_list = []
    prov_only_list = [p for p in cols[1:] if p not in ['全国', '京津冀及周边8省市']]
    
    for y in hist_years:
        y_col = next((c for c in [f"{y}年", f"{y}"] if c in actual_cols), None)
        row_data = {'Year': y}
        for province in prov_only_list:
            if y_col:
                val_row = df_prod_raw[df_prod_raw['地区'] == province]
                if not val_row.empty:
                    val = val_row[y_col].values[0]
                    # 处理单元格内的制表符和逗号
                    clean_val = pd.to_numeric(str(val).replace('\t', '').replace(',', '').strip(), errors='coerce')
                    row_data[province] = clean_val if not np.isnan(clean_val) else 0.0
                else: row_data[province] = 0.0
            else: row_data[province] = 0.0
        
        # 处理全国数据：如果文件中没有“全国”，则汇总各省
        quanguo_col = next((c for c in [f"{y}年", f"{y}"] if c in actual_cols), None)
        quanguo_val = 0.0
        if quanguo_col:
            quanguo_row = df_prod_raw[df_prod_raw['地区'] == '全国']
            if not quanguo_row.empty:
                val = quanguo_row[quanguo_col].values[0]
                quanguo_val = pd.to_numeric(str(val).replace('\t', '').replace(',', '').strip(), errors='coerce')
        
        # 如果读取到的全国值为 0 或 NaN，则使用省份汇总
        if pd.isna(quanguo_val) or quanguo_val == 0:
            quanguo_val = sum(row_data[p] for p in prov_only_list)
        
        row_data['全国'] = quanguo_val
        hist_list.append(row_data)
    
    df_hist_prod = pd.DataFrame(hist_list)
    print(f"DEBUG: {prod_name} 2023年全国读取/计算值: {df_hist_prod.loc[df_hist_prod['Year']==2023, '全国'].values[0]}")

    # C. 分省独立回归拟合
    models = {}
    for p in prov_only_list:
        pop_hist = df_pop.loc[df_pop['Year'].isin(hist_years), p].values
        gdp_hist = df_gdp.loc[df_gdp['Year'].isin(hist_years), p].values
        prod_hist = df_hist_prod[p].values
        
        x_vals = (gdp_hist / pop_hist) * 10000
        y_vals = (prod_hist / pop_hist) * 10000
        
        valid_mask = ~np.isnan(x_vals) & ~np.isnan(y_vals) & (y_vals > 0)
        if valid_mask.sum() > 3:
            reg = LinearRegression().fit(x_vals[valid_mask].reshape(-1, 1), y_vals[valid_mask])
            models[p] = (reg.coef_[0], reg.intercept_)
        else: models[p] = (0.0, 0.0)

    # D. 计算 2024 年个性化系数 k_p_2024
    k_p_2024 = {}
    for p in prov_only_list:
        actual_24 = df_hist_prod.loc[df_hist_prod['Year'] == 2024, p].values[0]
        gdp_24 = df_gdp.loc[df_gdp['Year'] == 2024, p].values[0]
        pop_24 = df_pop.loc[df_pop['Year'] == 2024, p].values[0]
        a, b = models[p]
        theory_24 = max(0, (a * (gdp_24 / pop_24 * 10000) + b) * pop_24 / 10000)
        k_p_2024[p] = actual_24 / theory_24 if theory_24 > 0 else 1.0

    # E. 预测与校准 (2005-2035)
    predict_years = range(2005, 2036)
    final_output = []
    for y in predict_years:
        res_row = {'Year': y}
        # 只有在历史数据范围内（2010-2024）才校准到真实值
        if 2010 <= y <= 2024:
            for p in prov_only_list:
                res_row[p] = df_hist_prod.loc[df_hist_prod['Year'] == y, p].values[0]
            res_row['全国'] = df_hist_prod.loc[df_hist_prod['Year'] == y, '全国'].values[0]
        else:
            # 预测期或 2010 以前：使用回归值 * k_p_2024
            current_sum = 0
            for p in prov_only_list:
                gdp_y, pop_y = df_gdp.loc[df_gdp['Year']==y, p].values[0], df_pop.loc[df_pop['Year']==y, p].values[0]
                a, b = models[p]
                theory_val = max(0, (a * (gdp_y / pop_y * 10000) + b) * pop_y / 10000)
                res_row[p] = theory_val * k_p_2024[p]
                current_sum += res_row[p]
            res_row['全国'] = current_sum
        
        jjj_list = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '山东省', '河南省', '陕西省']
        res_row['京津冀及周边8省市'] = sum(res_row.get(p, 0) for p in jjj_list)
        final_output.append(res_row)

    df_final = pd.DataFrame(final_output)
    df_final.to_csv(out_path + f"{prod_name}_prov_specific_calibrated.csv", index=False, encoding='utf-8-sig')
    print(f"[{prod_name}] 处理完成")

print("\n>>> 任务结束。")