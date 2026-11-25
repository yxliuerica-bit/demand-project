import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# --- 1. 初始化路径与配置 ---
base_path = "/lustre/home/litong/intern/京津冀货运发展/"
data_path = base_path + "data/预测原始数据/电力部门相关数据/"
out_path = base_path + "output/7.电力部门/7.1电力需求预测/工业-高耗能行业/"
if not os.path.exists(out_path): os.makedirs(out_path)

cols_standard = ['Year', '北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑龙江省', '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省', '湖南省', '广东省', '广西壮族自治区', '海南省', '重庆市', '四川省', '贵州省', '云南省', '西藏自治区', '陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区', '全国', '京津冀及周边8省市']
prov_only_list = [p for p in cols_standard[1:] if p not in ['全国', '京津冀及周边8省市']]

# --- 2. 预处理函数：读取全国总量数据 ---
def get_national_total(target_name):
    # 读取总量数据 CSV
    df_total = pd.read_csv(data_path + "工业产品产量年度数据.csv", header=2)
    df_total.columns = df_total.columns.str.replace(r'[\t\s]', '', regex=True)
    df_total['指标'] = df_total['指标'].astype(str).str.replace(r'[\t\s]', '', regex=True)
    
    # 模糊匹配指标名称
    row = df_total[df_total['指标'].str.contains(target_name)]
    if row.empty: 
        print(f"警告：未能在总量文件中找到指标 {target_name}")
        return None
    
    # 提取 2010-2024 的数据
    totals = {}
    for y in range(2010, 2025):
        y_col = next((c for c in [f"{y}年", f"{y}"] if c in df_total.columns), None)
        if y_col:
            val = pd.to_numeric(str(row[y_col].values[0]).replace(',', '').strip(), errors='coerce')
            totals[y] = val if not np.isnan(val) else 0.0
    return totals

# --- 3. 第一阶段：历史产量分配 (2010-2024) ---

# A. 氧化铝分配 (固定比例)
print("正在进行氧化铝历史产量分配...")
alum_totals = get_national_total("氧化铝产量")
df_alum_ratio = pd.read_csv(data_path + "氧化铝比例分省年度数据.csv", header=None, names=['地区', '比例'])
df_alum_ratio['地区'] = df_alum_ratio['地区'].astype(str).str.replace(r'[\t\s]', '', regex=True)

total_ratio_sum = df_alum_ratio['比例'].sum()
alum_hist_list = []
for y in range(2010, 2025):
    total_val = alum_totals.get(y, 0)
    row = {'Year': y, '全国': total_val}
    for p in prov_only_list:
        ratio = df_alum_ratio.loc[df_alum_ratio['地区'] == p, '比例'].values
        p_ratio = ratio[0] if len(ratio) > 0 else 0
        row[p] = (total_val * p_ratio / total_ratio_sum) if total_ratio_sum > 0 else 0
    alum_hist_list.append(row)
df_alum_hist = pd.DataFrame(alum_hist_list)
df_alum_hist.to_csv(out_path + "氧化铝产量分省年度数据_按比例分配.csv", index=False, encoding='utf-8-sig')

# B. 合成氨分配 (动态化肥比例)
print("正在进行合成氨历史产量分配...")
ammonia_totals = get_national_total("合成氨产量")
df_fert = pd.read_csv(data_path + "农用氮磷钾化肥产量分省年度数据.csv", header=2)
df_fert.columns = df_fert.columns.str.replace(r'[\t\s]', '', regex=True)
df_fert['地区'] = df_fert['地区'].astype(str).str.replace(r'[\t\s]', '', regex=True)

ammonia_hist_list = []
for y in range(2010, 2025):
    total_val = ammonia_totals.get(y, 0)
    y_col = next((c for c in [f"{y}年", f"{y}"] if c in df_fert.columns), None)
    row = {'Year': y, '全国': total_val}
    
    prov_fert_temp = {}
    for p in prov_only_list:
        v_row = df_fert[df_fert['地区'] == p]
        if not v_row.empty and y_col:
            v = pd.to_numeric(str(v_row[y_col].values[0]).replace(',', '').strip(), errors='coerce')
            prov_fert_temp[p] = v if not np.isnan(v) else 0
        else: prov_fert_temp[p] = 0
    
    year_fert_sum = sum(prov_fert_temp.values())
    for p in prov_only_list:
        row[p] = (total_val * prov_fert_temp[p] / year_fert_sum) if year_fert_sum > 0 else 0
    ammonia_hist_list.append(row)
df_ammonia_hist = pd.DataFrame(ammonia_hist_list)
df_ammonia_hist.to_csv(out_path + "合成氨产量分省年度数据_按比例分配.csv", index=False, encoding='utf-8-sig')

# --- 4. 第二阶段：建立回归预测与校准 (2010-2035) ---

def load_standard_csv(path):
    df = pd.read_csv(path, header=None)
    df.columns = cols_standard
    for c in cols_standard[1:]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '').str.strip(), errors='coerce')
    return df

df_gdp = load_standard_csv(base_path + "output/scenarios/gdp_predictions_基准_完整年份.csv")
df_pop = load_standard_csv(base_path + "output/scenarios/population_predictions_基准_完整年份.csv")

for prod_name, df_hist in [("氧化铝", df_alum_hist), ("合成氨", df_ammonia_hist)]:
    print(f"正在对 {prod_name} 进行回归建模与 2025-2035 预测...")
    
    hist_years = range(2010, 2025)
    models = {}
    k_p = {}
    
    # A. 独立省份回归拟合
    for p in prov_only_list:
        pop_h = df_pop.loc[df_pop['Year'].isin(hist_years), p].values
        gdp_h = df_gdp.loc[df_gdp['Year'].isin(hist_years), p].values
        prod_h = df_hist.loc[df_hist['Year'].isin(hist_years), p].values
        
        x = (gdp_h / pop_h) * 10000
        y_val = (prod_h / pop_h) * 10000
        mask = ~np.isnan(x) & ~np.isnan(y_val) & (y_val > 0)
        
        if mask.sum() > 3:
            reg = LinearRegression().fit(x[mask].reshape(-1, 1), y_val[mask])
            models[p] = (reg.coef_[0], reg.intercept_)
        else:
            models[p] = (0.0, 0.0)

        # B. 计算 2024 校准系数 k_p
        actual_24 = df_hist.loc[df_hist['Year'] == 2024, p].values[0]
        gdp_24 = df_gdp.loc[df_gdp['Year'] == 2024, p].values[0]
        pop_24 = df_pop.loc[df_pop['Year'] == 2024, p].values[0]
        a, b = models[p]
        theory_24 = max(0, (a * (gdp_24 / pop_24 * 10000) + b) * pop_24 / 10000)
        k_p[p] = actual_24 / theory_24 if theory_24 > 0 else 1.0

    # C. 执行预测循环 (2010-2035)
    results = []
    for y in range(2010, 2036):
        res = {'Year': y}
        if y <= 2024:
            # 历史期：直接取分配后的真实值
            for p in prov_only_list:
                res[p] = df_hist.loc[df_hist['Year'] == y, p].values[0]
            res['全国'] = df_hist.loc[df_hist['Year'] == y, '全国'].values[0]
        else:
            # 预测期：回归预测值 * k_p_2024
            current_sum = 0
            for p in prov_only_list:
                g_y = df_gdp.loc[df_gdp['Year'] == y, p].values[0]
                p_y = df_pop.loc[df_pop['Year'] == y, p].values[0]
                a, b = models[p]
                theory = max(0, (a * (g_y / p_y * 10000) + b) * p_y / 10000)
                res[p] = theory * k_p[p]
                current_sum += res[p]
            res['全国'] = current_sum
        
        # 重新计算区域汇总
        jjj_list = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '山东省', '河南省', '陕西省']
        res['京津冀及周边8省市'] = sum(res.get(p, 0) for p in jjj_list)
        results.append(res)
    
    pd.DataFrame(results).to_csv(out_path + f"{prod_name}_final_prediction.csv", index=False, encoding='utf-8-sig')

print("\n>>> 处理完毕。已生成分配文件及 2010-2035 最终预测文件。")