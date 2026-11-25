import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# 创建保存结果的目录
output_dir = '/lustre/home/litong/intern/京津冀货运发展/output/约束插值'
os.makedirs(output_dir, exist_ok=True)

# 原始数据
# years = np.array([2019, 2020, 2021, 2022, 2023, 2024, 2025, 2030, 2035])
years = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2035])
data = np.array([10702.71, 10495.20, 10396.96, 9934.08, 9812.84, 9582.05, 8961.59, 8574.06, 7994.23, 8012.93, 7676.71, 7017.84, 6919.48, 6061.38])

# 转换为相对时间（以2019年为基准，t=0）
t = years - 2010

# 逻辑斯蒂方程
def logistic_function(t, K, P0, r):
    return K / (1 + (K/P0 - 1) * np.exp(-r * t))

# 初始参数估计
P0_est = data[0]  # 初始值
K_est = max(data) * 1.1  # 假设上限略高于已知最大值
r_est = 0.02  # 初始增长率猜测

# 拟合逻辑斯蒂方程
try:
    params, _ = curve_fit(
        logistic_function, 
        t, 
        data, 
        p0=[K_est, P0_est, r_est],
        bounds=([max(data), 0, 0], [np.inf, np.inf, np.inf])  # 设置参数约束
    )
    
    K, P0, r = params
    print(f"拟合参数: K={K:.2f}, P0={P0:.2f}, r={r:.6f}")
    
    # 生成插值数据（2019-2035年每年的数据）
    interp_years = np.arange(2010, 2036)
    interp_t = interp_years - 2010
    interpolated_data = logistic_function(interp_t, K, P0, r)
    
    # 计算R²评估拟合质量
    predicted = logistic_function(t, K, P0, r)
    r_squared = 1 - np.sum((data - predicted)**2) / np.sum((data - np.mean(data))**2)
    print(f"拟合优度 R²: {r_squared:.4f}")
    
    # 创建结果DataFrame并保存为CSV
    result_df = pd.DataFrame({
        '年份': interp_years,
        '逻辑斯蒂插值': interpolated_data
    })
    csv_path = os.path.join(output_dir, 'logistic_interpolation_results.csv')
    result_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"结果已保存至: {csv_path}")
    
    # 保存拟合参数
    params_df = pd.DataFrame({
        '参数': ['K', 'P0', 'r', 'R²'],
        '值': [K, P0, r, r_squared]
    })
    params_path = os.path.join(output_dir, 'logistic_fit_parameters.csv')
    params_df.to_csv(params_path, index=False, encoding='utf-8')
    print(f"拟合参数已保存至: {params_path}")
    
    # 可视化结果并保存图像
    plt.figure(figsize=(12, 6))
    plt.scatter(years, data, color='red', label='原始数据')
    plt.plot(interp_years, interpolated_data, 'b-', label='逻辑斯蒂插值')
    
    # 添加参数和R²信息到图像
    plt.figtext(0.15, 0.01, f'参数: K={K:.2f}, P0={P0:.2f}, r={r:.6f}, R²={r_squared:.4f}', ha="left", fontsize=10)
    
    plt.title('逻辑斯蒂方程插值结果')
    plt.xlabel('年份')
    plt.ylabel('GDP值')
    plt.grid(True)
    plt.legend()
    
    # 保存图像
    image_path = os.path.join(output_dir, 'logistic_interpolation.png')
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存至: {image_path}")
    plt.close()
    
except Exception as e:
    print(f"拟合过程出错: {str(e)}")