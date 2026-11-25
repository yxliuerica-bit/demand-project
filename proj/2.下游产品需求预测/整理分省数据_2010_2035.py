import pandas as pd
import os

# 定义指标列表
indicators = [
    '新增建筑面积', '汽车产量',
    '新增铁路里程', 
    '洗衣机产量', '电冰箱产量', '空调产量',
    '新增高速公路里程', '新增其他等级公路里程', 
    '农村水电基础设施投资',
    '火力发电量', "发电量", "电力消费量"
]

scenario_name = '高增长'

# 数据文件所在路径
data_path = f'/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_{scenario_name}/全国完整数据'

# 遍历每个指标，处理对应的数据文件
for indicator in indicators:
    file_name = f'分省{indicator}完整结果.csv'
    file_path = os.path.join(data_path, file_name)
    
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在，跳过该指标。")
        continue
    except Exception as e:
        print(f"读取文件 {file_path} 时出错：{e}，跳过该指标。")
        continue
    
    # 检查是否存在所需列
    required_cols = ['地区', '年份', indicator]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"文件 {file_path} 缺少列 {missing_cols}，跳过该指标。")
        continue
    
    # 筛选出年份在2010-2035之间的数据
    df = df[(df['年份'] >= 2010) & (df['年份'] <= 2035)]
    
    # 按年份和地区进行透视，行是年份，列是地区
    pivot_df = df.pivot(index='年份', columns='地区', values=indicator)
    
    # 计算全国列（各地区该指标数据之和）
    pivot_df['全国'] = pivot_df.sum(axis=1)
    
    # 保存整理后的数据到新的Excel文件
    output_file = os.path.join(data_path, f'分省{indicator}数据(2005-2035).xlsx')
    pivot_df.to_excel(output_file)
    print(f"已成功整理 {indicator} 数据，并保存到 {output_file}。")