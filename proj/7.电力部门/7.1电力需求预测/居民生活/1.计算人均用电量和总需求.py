import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 设置中文字体（根据你服务器环境调整，如无黑体可删此行）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径定义
path_gdp = "/lustre/home/litong/intern/京津冀货运发展/output/scenarios/gdp_predictions_基准_完整年份.csv"
path_pop = "/lustre/home/litong/intern/京津冀货运发展/output/scenarios/population_predictions_基准_完整年份.csv"
path_elec = "/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/电力部门相关数据/分行业电力消费总量年度数据.csv"

# 1. 读取 GDP 和人口数据 (第一列为年份，无列名)
df_gdp = pd.read_csv(path_gdp, header=None)
df_pop = pd.read_csv(path_pop, header=None)

cols = ['Year', '北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑龙江省', '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省', '湖南省', '广东省', '广西壮族自治区', '海南省', '重庆市', '四川省', '贵州省', '云南省', '西藏自治区', '陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区', '全国', '京津冀及周边8省市']
df_gdp.columns = cols
df_pop.columns = cols

# 2. 读取电力数据 (第三行为列名)
df_elec_raw = pd.read_csv(path_elec, header=2)

#检查列名，去掉可能存在的空格
df_elec_raw.columns = df_elec_raw.columns.str.strip()

# 提取“居民生活电力消费总量 (亿千瓦小时)”这一行
res_elec_row = df_elec_raw[df_elec_raw['指标'].str.contains('居民生活电力消费总量', na=False)]

# 转置电力数据，方便按年份合并
years_elec = [str(y) + '年' for y in range(2010, 2024)] # 2024年数据若尚未出炉，请确保csv中有对应列
# 注意：附图中2024和2025可能是空的，我们取2010-2023或2024的有效交集
valid_years = range(2010, 2024) 

hist_data = []
for y in valid_years:
    y_str = f"{y}年"
    if y_str in res_elec_row.columns:
        val = res_elec_row[y_str].values[0]
        hist_data.append({'Year': y, 'Total_Res_Elec_100M_kWh': val})

df_hist_elec = pd.DataFrame(hist_data)

# 合并全国历史数据进行回归
df_merge = pd.merge(df_hist_elec, df_gdp[['Year', '全国']], on='Year')
df_merge = pd.merge(df_merge, df_pop[['Year', '全国']], on='Year', suffixes=('_GDP', '_Pop'))

# 定义需要转换的列
cols_to_fix = ['全国_GDP', '全国_Pop', 'Total_Res_Elec_100M_kWh']

for col in cols_to_fix:
    # 这一步会：1. 去掉可能的逗号 2. 强制转换非数字为 NaN 3. 转换为 float
    df_merge[col] = (
        df_merge[col]
        .astype(str)
        .str.replace(',', '')  # 处理千分位 1,234.5
        .replace('nan', np.nan)
        .apply(pd.to_numeric, errors='coerce')
    )

# 检查是否有转换失败产生的空值
if df_merge[cols_to_fix].isnull().any().any():
    print("警告：部分数据包含非数字字符，已转换为 NaN。")
    df_merge = df_merge.dropna(subset=cols_to_fix) # 剔除无法计算的年份

# 计算人均指标
# 人均GDP = 全国GDP(亿元) / 全国人口(万人) * 10000 -> 元/人
# 人均用电 = 居民总用电(亿kWh) / 全国人口(万人) * 10000 -> kWh/人
df_merge['Per_GDP'] = (df_merge['全国_GDP'] / df_merge['全国_Pop']) * 10000
df_merge['Per_Elec'] = (df_merge['Total_Res_Elec_100M_kWh'] / df_merge['全国_Pop']) * 10000

# 3. 线性回归: y = ax + b
X = df_merge['Per_GDP'].values.reshape(-1, 1)
y = df_merge['Per_Elec'].values
model = LinearRegression()
model.fit(X, y)

a = model.coef_[0]
b = model.intercept_
r2 = r2_score(y, model.predict(X))

print(f"回归方程: y = {a:.5f}x + {b:.5f}")
print(f"R² = {r2:.4f}")

# 4. 绘图 (复刻附图1)
plt.figure(figsize=(10, 6))
plt.scatter(df_merge['Per_GDP'], df_merge['Per_Elec'], color='royalblue', s=100)
plt.plot(df_merge['Per_GDP'], model.predict(X), color='royalblue', linestyle=':', linewidth=2)
plt.xlabel('人均GDP (元)', fontsize=12)
plt.ylabel('人均居民生活用电量 (千瓦时/年)', fontsize=12)
plt.text(df_merge['Per_GDP'].max()*0.7, df_merge['Per_Elec'].max()*0.9, 
         f"y = {a:.4f}x + {b:.3f}\nR² = {r2:.4f}", fontsize=12)
plt.grid(False)
plt.savefig("per_capita_elec_regression.png", dpi=300)
plt.show()

# 5. 预测 2025-2035

# --- 在执行循环预测之前，添加以下清洗逻辑 ---

# 1. 定义需要预测的所有省份列（排除第一列年份）
target_cols = cols[1:] 

# 2. 强制转换 GDP 和人口表中的所有数据为数值型
# errors='coerce' 会把无法转换的字符串变成 NaN，然后再用 ffill/bfill 填补或处理
for col in target_cols:
    df_gdp[col] = pd.to_numeric(df_gdp[col].astype(str).str.replace(',', ''), errors='coerce')
    df_pop[col] = pd.to_numeric(df_pop[col].astype(str).str.replace(',', ''), errors='coerce')

# --- 5. 预测与总量校准逻辑 ---

# --- 修改后的预测循环部分 ---

predict_years = list(range(2005, 2036)) # 明确转为 list
df_output_per = pd.DataFrame({'Year': predict_years})
temp_total_dict = {}

# 关键修复：确保只处理对应年份的数据行
# 假设 df_gdp 的第一列是年份且已经通过前面的 load 处理过
for province in cols[1:]:
    # 筛选出 df_gdp 中年份在 2005-2035 范围内的行
    mask = df_gdp['Year'].isin(predict_years)
    p_gdp = df_gdp.loc[mask, province].reset_index(drop=True)
    p_pop = df_pop.loc[mask, province].reset_index(drop=True)
    
    # 如果数据行数还是对不上，强制截断或补齐
    p_gdp = p_gdp.iloc[:len(predict_years)]
    p_pop = p_pop.iloc[:len(predict_years)]
    
    per_gdp = (p_gdp / p_pop) * 10000
    per_elec_pred = a * per_gdp + b
    
    # 填充人均表
    df_output_per[province] = per_elec_pred
    # 暂存总量计算值
    temp_total_dict[province] = (per_elec_pred * p_pop) / 10000

# 此时生成的 DataFrame 长度必然是 31
df_temp_total = pd.DataFrame(temp_total_dict)
df_temp_total['Year'] = predict_years # 现在长度匹配了，不会报错

# --- 5. 执行总量放缩校准 ---
final_total_data = []

for idx, row in df_temp_total.iterrows():
    year = int(row['Year'])
    
    # 1. 确定该年份的“全国基准总量” (锚点)
    if year <= 2023:
        # 历史年份：优先使用统计局真实值
        actual_row = df_hist_elec[df_hist_elec['Year'] == year]
        if not actual_row.empty:
            national_target = float(actual_row['Total_Res_Elec_100M_kWh'].iloc[0])
        else:
            national_target = float(row['全国'])
    else:
        # 未来年份：使用基于全国平均GDP回归出的总量
        national_target = float(row['全国'])
    
    # 2. 计算各省初步预测值之和（用于计算分配系数）
    provinces_only = [c for c in cols[1:] if c not in ['全国', '京津冀及周边8省市']]
    # 确保转换为标量，避免 TypeError
    current_prov_values = pd.to_numeric(row[provinces_only], errors='coerce').fillna(0)
    sum_provinces = float(current_prov_values.sum())
    
    # 3. 计算放缩系数 k
    if sum_provinces != 0:
        k = national_target / sum_provinces
    else:
        k = 1.0
    
    # 4. 对各省进行比例放缩 (Top-Down 分配)
    calibrated_row = {'Year': year}
    for p in provinces_only:
        calibrated_row[p] = float(row[p]) * k
    
    # 5. 记录校准后的固定值
    calibrated_row['全国'] = national_target
    
    # 6. 重新计算区域汇总（基于校准后的省份数据）
    jjj_provinces = ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '山东省', '河南省', '陕西省']
    calibrated_row['京津冀及周边8省市'] = sum(calibrated_row[p] for p in jjj_provinces)
    
    # --- 重要：仅执行一次 append ---
    final_total_data.append(calibrated_row)

# 转换回 DataFrame
df_res_total_elec = pd.DataFrame(final_total_data)

# 6. 同步更新人均用电量表（根据校准后的总量逆算，确保总量=人均*人口）
for province in cols[1:]:
    p_pop = df_pop[province]
    # 人均 = (总量 * 10000) / 人口
    df_output_per[province] = (df_res_total_elec[province] * 10000) / p_pop

# 7. 保存结果
df_output_per.to_csv("per_capita_residential_electricity_calibrated.csv", index=False, encoding='utf-8-sig')
df_res_total_elec.to_csv("total_residential_electricity_calibrated.csv", index=False, encoding='utf-8-sig')

print("总量控制校准计算完成！")