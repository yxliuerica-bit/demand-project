import pandas as pd

# 读取文件
file_path = "/lustre/home/litong/intern/京津冀货运发展/data/GCAM_results/GCAM_results_china_for_plot.csv"
df = pd.read_csv(file_path)

# 去掉 NaN
vars_nonan = df['var'].dropna().unique()

# 筛选 steel 和 cement 相关的变量
# steel_cement_vars = [v for v in vars_nonan if ('steel' in v.lower()) or ('cement' in v.lower())]

# print("与钢铁和水泥相关的变量：")
# for v in steel_cement_vars:
#     print(v)

# 查看 region 列的所有唯一值
unique_vars = df['region'].unique()
print("region列的所有变量：")
for v in unique_vars:
    print(v)