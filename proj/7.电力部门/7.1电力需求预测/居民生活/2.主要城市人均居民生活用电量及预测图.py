import pandas as pd
import matplotlib.pyplot as plt

# 1. 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False

# 2. 读取 CSV 文件
file_path = "per_capita_residential_electricity_calibrated.csv"
df = pd.read_csv(file_path)

# 3. 绘图函数定义
def plot_electricity_comparison(regions, file_name, title_suffix, figsize=(12, 7)):
    # 映射“全国平均”到数据列“全国”
    data_cols = [r if r != '全国平均' else '全国' for r in regions]
    
    # 获取年份数据
    val_2023 = df[df['Year'] == 2023][data_cols].values[0]
    val_2025_avg = df[df['Year'] == 2025]['全国'].values[0]
    val_2030_avg = df[df['Year'] == 2030]['全国'].values[0]
    val_2035_avg = df[df['Year'] == 2035]['全国'].values[0]

    plt.figure(figsize=figsize)
    
    # 绘制 2023 年柱状图
    plt.bar(regions, val_2023, color='#4A74B4', width=0.6, label='2023年人均用电量')

    # 绘制全国平均预测横线
    plt.hlines(y=val_2025_avg, xmin=-0.5, xmax=len(regions)-0.5, 
               colors='#E87E35', linestyles='-', linewidth=2.5)
    plt.hlines(y=val_2030_avg, xmin=-0.5, xmax=len(regions)-0.5, 
               colors='#A5A5A5', linestyles='-', linewidth=2.5)
    plt.hlines(y=val_2035_avg, xmin=-0.5, xmax=len(regions)-0.5, 
               colors='#FFC000', linestyles='-', linewidth=2.5)

    # 添加预测年份文字标注
    plt.text(len(regions)-0.5, val_2025_avg + 15, '2025年全国平均', ha='right', va='bottom', color='#E87E35', fontweight='bold', fontsize=18)
    plt.text(len(regions)-0.5, val_2030_avg + 15, '2030年全国平均', ha='right', va='bottom', color='#7F7F7F', fontweight='bold', fontsize=18)
    plt.text(len(regions)-0.5, val_2035_avg + 15, '2035年全国平均', ha='right', va='bottom', color='#BF9000', fontweight='bold', fontsize=18)

    # 图表装饰
    plt.ylabel('人均居民生活用电量 (千瓦时/年·人)', fontsize=18)
    plt.title(f'人均居民生活用电量对比 - {title_suffix}', fontsize=18, pad=20)
    
    # 动态调整 y 轴上限，留出标注空间
    plt.ylim(0, max(val_2023.max(), val_2035_avg) * 1.2)
    
    # 如果地区较多，旋转标签
    if len(regions) > 10:
        plt.xticks(rotation=45, ha='right', fontsize=16)
    else:
        plt.xticks(rotation=0, fontsize=18)

    # 移除多余边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    print(f"图像已保存至: {file_name}")
    plt.close()

# 4. 执行生成任务

# 第一张图：特定省份对比
#specific_regions = ['北京市', '天津市', '河北省', '河南省', '山东省', '山西省', '内蒙古自治区', '陕西省', '全国平均']
# specific_regions = ['北京市', '上海市', '福建省', '浙江省', '天津市', '全国平均']
# plot_electricity_comparison(specific_regions, "elec_specific_provinces_放缩.png", "重点省份")

# 第二张图：全国所有省市
# 排除'Year'和'京津冀及周边8省市'，并确保'全国'改名为'全国平均'放在最后
all_provinces = [c for c in df.columns if c not in ['Year', '全国', '京津冀及周边8省市']]
all_regions_with_avg = all_provinces + ['全国平均']

# 全国图表建议增加宽度
plot_electricity_comparison(all_regions_with_avg, "elec_all_provinces_放缩.png", "全国所有省市", figsize=(18, 8))