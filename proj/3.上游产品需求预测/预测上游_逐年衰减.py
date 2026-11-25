# -*- coding: utf-8 -*-
"""
钢铁和水泥需求预测代码
同时支持钢铁和水泥下游行业需求计算及约束处理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False

def load_reformed_data(path, factor):
    """从Excel读取基础数据并进行有效性校验"""
    file_path = os.path.join(path, f'分省{factor}数据(2005-2035).xlsx')
    if not os.path.exists(file_path):
        print(f"警告: {file_path} 不存在")
        return None
    
    try:
        df = pd.read_excel(file_path, index_col=0)
        df = df.loc[2010:2035].copy()  # 截取所需年份范围
        df = df.fillna(0)
        df[df < 0] = 0
        
        # 检查数据是否全为0
        if (df == 0).all().all():
            print(f"警告: {factor} 基础数据全为0，请检查原始文件")
        
        # 计算非零值占比，用于数据有效性判断
        non_zero_ratio = round((df != 0).mean().mean()*100, 2)
        print(f"成功加载 {factor} 数据，形状: {df.shape}，非零值占比: {non_zero_ratio}%")
        return df
    except Exception as e:
        print(f"读取 {file_path} 失败: {str(e)}")
        return None

def apply_province_decay(df, prov2cfg):
    """
    对 df（index=年份, columns含各省列）按省份配置做逐年衰减。
    prov2cfg: { '省名': {'start': 2026, 'rate': 0.98}, ... }
      - start: 从该年起开始衰减
      - rate : 年衰减率，例如 0.98 表示每年再下降 2%
    """
    for prov, cfg in prov2cfg.items():
        start = int(cfg.get('start', 2026))
        rate  = float(cfg.get('rate', 0.98))
        if prov not in df.columns: 
            continue
        years_after = df.index[df.index >= start]
        for i, y in enumerate(years_after, start=1):
            # 逐年累计衰减：start年乘 rate^1，下一年乘 rate^2，……
            df.loc[y, prov] = df.loc[y, prov] * (rate ** i)
    return df


def load_national_series_from_wide_excel(path, sheet_name, year_limit_max=2024):
    """
    从“地区 + YYYY年列”为结构的宽表中，读取每年的全国真实值（单位：吨）。
    历史表单位是“万吨”，这里自动×10000 转为“吨”。
    """
    import re
    df0 = pd.read_excel(path, sheet_name=sheet_name, header=None)

    # 找到表头行（第一列为“地区”）
    header_row = None
    for i in range(min(len(df0), 50)):
        if str(df0.iat[i, 0]).strip() == '地区':
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"在 {path} 的 {sheet_name} 中找不到'地区'表头行")

    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    if '地区' not in df.columns:
        raise ValueError(f"{path}/{sheet_name} 未包含'地区'列")

    # 只保留“YYYY年”格式的列
    year_cols = []
    for c in df.columns:
        s = str(c)
        if re.fullmatch(r'\d{4}年', s):
            year_cols.append(s)

    if not year_cols:
        raise ValueError(f"{path}/{sheet_name} 未识别到形如'YYYY年'的列")

    # 这些列转数值，缺失按 0
    df_years = df[year_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)

    # 全国(万吨) = 各省求和（按列求和），再转为“吨”
    nat_wan_ton = df_years.sum(axis=0)  # index 是 'YYYY年'
    nat_ton = nat_wan_ton * 10000.0
    # 把索引改成 int 年份，并按升序排序
    nat_ton.index = nat_ton.index.map(lambda x: int(str(x).replace('年', '')))
    nat_ton = nat_ton.sort_index()

    # 只取到 year_limit_max（通常 2024）
    nat_ton = nat_ton[nat_ton.index <= year_limit_max]
    return nat_ton  # Series: index=年份(int), value=吨


# 配置省份和指标列表
provinces = [
    '北京市', '天津市', '河北省', '山西省', '内蒙古自治区', '辽宁省', '吉林省', '黑龙江省',
    '上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '河南省', '湖北省',
    '湖南省', '广东省', '广西壮族自治区', '海南省', '重庆市', '四川省', '贵州省', '云南省',
    '西藏自治区', '陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区'
]
provlist_bthsa = ['北京市','天津市','河北省','河南省','山东省','山西省','内蒙古自治区','陕西省']

# 钢铁和水泥下游行业指标
steel_stream = ['新增建筑面积','汽车产量','新增铁路里程','洗衣机产量','电冰箱产量','空调产量']
cement_stream = ['新增建筑面积','新增高速公路里程','新增其他等级公路里程','新增铁路里程','农村水电基础设施投资']

# 路径配置
sce = 'Sce_Pop_BAU_GDP_BAU'
path_down_stream = '/lustre/home/litong/intern/京津冀货运发展/output/predict_results/'+sce+'/12_跨省应用模型预测/分省完整数据/'

# 控制参数
reform_data_ = 0  # 1: 重新生成基础数据；0: 读取已有数据
do_steel = 1      # 1: 计算钢铁需求
do_cement = 1     # 1: 计算水泥需求
constraint_steel = 1  # 1: 应用钢铁约束
constraint_cement = 1 # 1: 应用水泥约束
plot_predict = 1      # 1: 绘制预测图表

# 存储基础数据
base_data = {}

# 重新生成基础数据（如需）
if reform_data_:
    def reform_data(path, factor):
        """从原始CSV生成Excel格式的基础数据"""
        try:
            df = pd.read_csv(path + f'分省{factor}完整结果.csv')
            if df.empty:
                print(f"错误: {factor}原始CSV文件为空")
                return
        except Exception as e:
            print(f"读取原始CSV失败 {factor}: {e}")
            return
        
        years = list(range(2010, 2036))
        provinces = df['地区'].unique()
        result_df = pd.DataFrame(index=years, columns=provinces)
        
        for province in provinces:
            province_data = df[df['地区'] == province].copy()
            # 确保年份列正确
            province_data['年份'] = pd.to_numeric(province_data['年份'], errors='coerce')
            province_data = province_data.dropna(subset=['年份'])
            
            for year in years:
                # 匹配年份（允许±1误差）
                year_mask = (province_data['年份'] >= year - 1) & (province_data['年份'] <= year + 1)
                year_data = province_data[year_mask]
                
                if not year_data.empty:
                    # 优先提取真实值，无则提取预测值
                    real_col = f'平滑_{factor}' if f'平滑_{factor}' in year_data.columns else None
                    pred_col = f'预测_平滑数据_{factor}' if f'预测_平滑数据_{factor}' in year_data.columns else None
                    
                    if real_col and not pd.isna(year_data[real_col].iloc[0]):
                        result_df.at[year, province] = year_data[real_col].iloc[0]
                    elif pred_col and not pd.isna(year_data[pred_col].iloc[0]):
                        result_df.at[year, province] = year_data[pred_col].iloc[0]
                    else:
                        # 尝试从数值列提取
                        numeric_cols = year_data.select_dtypes(include='number').columns.tolist()
                        if numeric_cols:
                            result_df.at[year, province] = year_data[numeric_cols[0]].iloc[0]
    
        # 数据清洗
        result_df = result_df.fillna(0)
        result_df[result_df < 0] = 0
        result_df['全国'] = result_df.sum(axis=1)
        result_df.to_excel(path + f'分省{factor}数据(2005-2035).xlsx')
        print(f"重新生成 {factor} 数据，非零值占比: {round((result_df != 0).mean().mean()*100, 2)}%")
    
    # 生成所有需要的基础数据
    all_factors = list(set(steel_stream + cement_stream))  # 去重
    for factor in all_factors:
        reform_data(path_down_stream, factor)

# 加载基础数据（钢铁和水泥共用）
all_factors = list(set(steel_stream + cement_stream))
for factor in all_factors:
    data = load_reformed_data(path_down_stream, factor)
    if data is not None:
        # 强制对齐省份列，缺失省份填充0
        data_aligned = data.reindex(columns=provinces, fill_value=0)
        base_data[factor] = data_aligned

def steel_intensity_series(product: str, years_index: pd.Index, base_intensity: float) -> pd.Series:
    """
    给定 2010 年的基准用钢强度 base_intensity，按规则生成每年的用钢强度（2010–2035）。
    规则：
      - 汽车产量：2010→2020 线性下降 20%（到 0.8倍）；
                 2020→2030 线性再降 10%（到 0.72倍）；
                 2030→2040 线性再降 10%（到 0.648倍）；
                 2040→2050 线性再降 10%（到 0.5832倍）。
                 （我们只用到 2035，按 2020→2030、2030→2040 两段线性插值）
      - 洗衣机产量/电冰箱产量/空调产量：每十年线性下降 10%（相对于上一段末值）。
                 2010→2020 到 0.9倍；2020→2030 到 0.81倍；2030→2040 到 0.729倍；……
      - 其它产品：不变（1.0 倍）
    返回：一个以 years_index 为索引的强度（不是“倍率”），已经乘上 base_intensity。
    """
    yrs = pd.Index(sorted(set(int(y) for y in years_index)))
    factors = pd.Series(1.0, index=yrs, dtype=float)

    def linear_segment(y0, f0, y1, f1):
        """把 [y0,y1] 两端的倍率 f0→f1 做逐年线性插值（含 y0、y1）。"""
        if y1 <= y0:
            return {y0: f0}
        seg_years = np.arange(y0, y1 + 1)
        seg_vals = np.linspace(f0, f1, len(seg_years))
        return dict(zip(seg_years, seg_vals))

    product = str(product).strip()

    if product == '汽车产量':
        # 2010->2020: 1.0 → 0.8
        mapping = linear_segment(2010, 1.0, 2020, 0.8)
        # 2020->2030: 0.8 → 0.8*0.9=0.72
        mapping.update(linear_segment(2020, 0.8, 2030, 0.72))
        # 2030->2040: 0.72 → 0.72*0.9=0.648
        mapping.update(linear_segment(2030, 0.72, 2040, 0.648))
        # 2040->2050: 0.648 → 0.648*0.9=0.5832（用不到 2050，只是保持规则完整）
        mapping.update(linear_segment(2040, 0.648, 2050, 0.5832))
        for y in yrs:
            # 超出定义区间的，使用邻近端点的倍率（<=2010 用1.0，>=2050 用 0.5832）
            if y in mapping:
                factors.loc[y] = mapping[y]
            elif y < 2010:
                factors.loc[y] = 1.0
            elif y > 2050:
                factors.loc[y] = 0.5832
            elif 2020 < y < 2030:
                # 上面的 mapping 已经覆盖整年；这个分支基本不会进
                factors.loc[y] = mapping[2020] + (mapping[2030] - mapping[2020]) * (y - 2020) / (2030 - 2020)

    elif product in ['洗衣机产量', '电冰箱产量', '空调产量']:
        # 每十年下降 10%，分段线性：2010→2020 1.0→0.9，2020→2030 0.9→0.81，2030→2040 0.81→0.729，2040→2050 0.729→0.6561
        mapping = {}
        mapping.update(linear_segment(2010, 1.0, 2020, 0.9))
        mapping.update(linear_segment(2020, 0.9, 2030, 0.81))
        mapping.update(linear_segment(2030, 0.81, 2040, 0.729))
        mapping.update(linear_segment(2040, 0.729, 2050, 0.6561))
        for y in yrs:
            if y in mapping:
                factors.loc[y] = mapping[y]
            elif y < 2010:
                factors.loc[y] = 1.0
            elif y > 2050:
                factors.loc[y] = 0.6561
    else:
        # 其它未指定产品：强度不变
        factors.loc[:] = 1.0

    return factors * float(base_intensity)

# 钢铁需求计算
if do_steel:
    years = pd.Index(range(2010, 2036), name='year')
    steel_result_df = pd.DataFrame(0.0, index=years, columns=provinces)  # 显式浮点型
    
    try:
        # 读取单位用钢强度数据
        steel_intensity_path = '/lustre/home/litong/intern/京津冀货运发展/data/单位用料强度/单位用钢强度.xlsx'
        df_steel_intensity = pd.read_excel(steel_intensity_path)
        
        # 统一列名
        if '标识' not in df_steel_intensity.columns:
            if '指标' in df_steel_intensity.columns:
                df_steel_intensity = df_steel_intensity.rename(columns={'指标': '标识'})
            else:
                print(f"错误: 单位用钢强度数据无'标识'列，列名: {df_steel_intensity.columns.tolist()}")
                do_steel = 0
    except Exception as e:
        print(f"读取单位用钢强度数据失败: {e}")
        do_steel = 0
    
    if do_steel:
        with pd.ExcelWriter(path_down_stream+'下游行业用钢需求预测.xlsx') as writer:
            has_valid_data = False
            intermediate_results = {}  # 保存中间结果用于调试
            
            for d_prod in steel_stream:
                if d_prod not in base_data:
                    print(f"跳过 {d_prod}：无基础数据")
                    continue
                
                df_d_prod = base_data[d_prod].copy()
                # 检查基础数据是否有效
                if (df_d_prod == 0).all().all():
                    print(f"警告: {d_prod} 基础数据全为0，跳过计算")
                    continue
                
                # 检查强度数据
                intensity_mask = df_steel_intensity['标识'] == d_prod
                if not intensity_mask.any():
                    print(f"警告: 单位用钢强度无'{d_prod}'记录，跳过")
                    continue
                
                # 提取强度和单位
                try:
                    intensity = float(df_steel_intensity.loc[intensity_mask, '用钢强度'].values[0])
                    unit = float(df_steel_intensity.loc[intensity_mask, 'unit'].values[0])
                    
                    # 校验强度和单位
                    if intensity <= 0 or unit <= 0:
                        print(f"警告: {d_prod} 强度或单位为非正值（强度={intensity}, unit={unit}），跳过")
                        continue
                except Exception as e:
                    print(f"{d_prod} 强度数据解析失败: {e}，跳过")
                    continue
                
                # 计算用钢需求（转换为万吨）
                # df_prod = df_d_prod * intensity * unit

                # —— 新：按年份生成“用钢强度”时间序列，再逐年相乘 ——
                years_idx = df_d_prod.index  # 2010–2035
                intensity_by_year = steel_intensity_series(d_prod, years_idx, intensity)  # Series(index=years)
                # 逐年相乘（按行对齐）：每一年的省份量 × 当年的用钢强度 × unit
                df_prod = df_d_prod.mul(intensity_by_year, axis=0) * unit


                intermediate_results[d_prod] = df_prod
                has_valid_data = True
                
                # 打印校验信息
                non_zero_ratio = round((df_prod != 0).mean().mean()*100, 2)
                print(f"{d_prod} 用钢需求计算完成，非零值占比: {non_zero_ratio}%，示例值: {df_prod.iloc[0, 0]:.2f}")
                
                # 累加结果
                steel_result_df = steel_result_df.add(df_prod.reindex_like(steel_result_df), fill_value=0)
                
                # —— 钢铁：对目标省份施加逐年衰减（示例：山东省 2026 起每年*0.98）——
                steel_decay_cfg = {
                    '山东省': {'start': 2026, 'rate': 0.98},
                    # 如需更多省份，按需添加：
                    # '河北省': {'start': 2026, 'rate': 0.99},
                }
                steel_result_df = apply_province_decay(steel_result_df, steel_decay_cfg)

            # 写入结果
            if has_valid_data:
                steel_result_df['全国'] = steel_result_df[provinces].sum(axis=1)
                steel_result_df['京津冀及周边8省市'] = steel_result_df[provlist_bthsa].sum(axis=1)
                steel_result_df.to_excel(writer, sheet_name='下游行业用钢需求')
                
                # 写入中间结果
                for prod, df in intermediate_results.items():
                    df.to_excel(writer, sheet_name=f'{prod}_用钢中间结果')
                
                # 最终校验
                if (steel_result_df['全国'] == 0).all():
                    print("警告: 钢铁需求汇总后全国列仍全为0，检查中间结果表")
                else:
                    print(f"钢铁需求汇总完成，全国列非零值年份: {sum(steel_result_df['全国'] != 0)} 个")
            else:
                pd.DataFrame().to_excel(writer, sheet_name='无有效数据')

# ======================== 应用钢铁约束（固定 2024 比例） ========================
if constraint_steel and do_steel:
    try:
        steel_forecast_path = path_down_stream + '下游行业用钢需求预测.xlsx'
        df_pred = pd.read_excel(steel_forecast_path, sheet_name='下游行业用钢需求', index_col=0)
        df_pred = df_pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 历史真实（单位：万吨）读取为“吨”的全国合计（按列求和）
        steel_history_path  = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史粗钢产量数据.xls'
        steel_history_sheet = '分省历史粗钢产量数据'
        real_nat_ton = load_national_series_from_wide_excel(
            steel_history_path, steel_history_sheet, year_limit_max=2024
        )  # Series: index=年份(int), value=吨

        years_all = pd.Index(range(2010, 2036), name='year')
        df_pred = df_pred.reindex(index=years_all, fill_value=0.0)

        # 预测的“全国”
        if '全国' in df_pred.columns:
            pred_nat = df_pred['全国'].astype(float)
        else:
            pred_nat = df_pred[provinces].sum(axis=1).astype(float)

        # 1) 只在“有真实值的年份(<=2024)”逐年算比例；2) 2025+ 全部使用“2024年的比例”
        scale_factors = {}
        last_known_scale = 1.0

        # 逐年计算 2010–2024 的比例（有真实就计算，没有就沿用上一年）
        for y in years_all:
            if y > 2024:
                break
            if (y in real_nat_ton.index) and (pred_nat.get(y, 0.0) > 0):
                s = float(real_nat_ton.loc[y]) / float(pred_nat.get(y, 0.0))
                s = float(np.clip(s, 0.1, 10.0))
                last_known_scale = s
            # 如果该年没有真实或预测为 0，就沿用上一年最后一个可用比例
            scale_factors[y] = last_known_scale
            print(f"[钢铁] {y}年: 比例={scale_factors[y]:.4f} （{'计算' if y in real_nat_ton.index and pred_nat.get(y,0.0)>0 else '沿用上一年'}）")

        # 锁定 2024 年的比例，作为 2025–2035 的固定比例
        ratio_2024 = scale_factors.get(2024, last_known_scale)
        for y in range(2025, 2036):
            scale_factors[y] = ratio_2024
            print(f"[钢铁] {y}年: 比例={scale_factors[y]:.4f} （沿用2024比例）")

        # 按比例缩放“各省份列”，再回算全国与区域
        df_out = df_pred.astype(float).copy()
        for y, s in scale_factors.items():
            if y in df_out.index:
                df_out.loc[y, provinces] = df_out.loc[y, provinces].values * s

        df_out['全国'] = df_out[provinces].sum(axis=1)
        if set(provlist_bthsa).issubset(df_out.columns):
            df_out['京津冀及周边8省市'] = df_out[provlist_bthsa].sum(axis=1)

        df_out.to_excel(path_down_stream + '分省钢铁消费量预测.xlsx')
        print("钢铁约束处理完成（2025+ 固定使用 2024 年比例）")

    except Exception as e:
        print(f"处理钢铁约束时出错: {e}")




# 水泥需求计算（保持与钢铁相同的逻辑框架）
if do_cement:
    years = pd.Index(range(2010, 2036), name='year')
    cement_result_df = pd.DataFrame(0.0, index=years, columns=provinces)  # 显式浮点型
    
    try:
        # 读取单位水泥强度数据
        cement_intensity_path = '/lustre/home/litong/intern/京津冀货运发展/data/单位用料强度/单位水泥强度.xlsx'
        df_cement_intensity = pd.read_excel(cement_intensity_path)
        
        # 统一列名
        if '标识' not in df_cement_intensity.columns:
            if '指标' in df_cement_intensity.columns:
                df_cement_intensity = df_cement_intensity.rename(columns={'指标': '标识'})
            elif '项目' in df_cement_intensity.columns:
                df_cement_intensity = df_cement_intensity.rename(columns={'项目': '标识'})
            else:
                print(f"错误: 单位水泥强度数据无'标识'列，列名: {df_cement_intensity.columns.tolist()}")
                do_cement = 0
    except Exception as e:
        print(f"读取单位水泥强度数据失败: {e}")
        do_cement = 0
    
    if do_cement:
        with pd.ExcelWriter(path_down_stream+'下游行业水泥需求预测.xlsx') as writer:
            has_valid_data = False
            intermediate_results = {}  # 保存中间结果用于调试
            
            for d_prod in cement_stream:
                if d_prod not in base_data:
                    print(f"跳过 {d_prod}：无基础数据")
                    continue
                
                df_d_prod = base_data[d_prod].copy()
                # 检查基础数据是否有效
                if (df_d_prod == 0).all().all():
                    print(f"警告: {d_prod} 基础数据全为0，跳过计算")
                    continue
                
                # 检查强度数据
                intensity_mask = df_cement_intensity['标识'] == d_prod
                if not intensity_mask.any():
                    print(f"警告: 单位水泥强度无'{d_prod}'记录，跳过")
                    continue
                
                # 提取强度和单位
                try:
                    intensity = float(df_cement_intensity.loc[intensity_mask, '水泥强度'].values[0])
                    unit = float(df_cement_intensity.loc[intensity_mask, 'unit'].values[0])
                    
                    # 校验强度和单位
                    if intensity <= 0 or unit <= 0:
                        print(f"警告: {d_prod} 强度或单位为非正值（强度={intensity}, unit={unit}），跳过")
                        continue
                except Exception as e:
                    print(f"{d_prod} 强度数据解析失败: {e}，跳过")
                    continue
                
                # 计算水泥需求（转换为万吨）
                #df_prod = df_d_prod * intensity * unit / 10000.0
                df_prod = df_d_prod * intensity * unit
                intermediate_results[d_prod] = df_prod
                has_valid_data = True
                
                # 打印校验信息
                non_zero_ratio = round((df_prod != 0).mean().mean()*100, 2)
                print(f"{d_prod} 水泥需求计算完成，非零值占比: {non_zero_ratio}%，示例值: {df_prod.iloc[0, 0]:.2f}")
                
                # 累加结果
                cement_result_df = cement_result_df.add(df_prod.reindex_like(cement_result_df), fill_value=0)

                # —— 水泥：对目标省份施加逐年衰减（示例：山东省 2026 起每年*0.98）——
                cement_decay_cfg = {
                    '山东省': {'start': 2026, 'rate': 0.98},
                    # 需要的话可给其他省也加：
                    # '河北省': {'start': 2026, 'rate': 0.99},
                }
                cement_result_df = apply_province_decay(cement_result_df, cement_decay_cfg)

            
            # 写入结果
            if has_valid_data:
                cement_result_df['全国'] = cement_result_df[provinces].sum(axis=1)
                cement_result_df['京津冀及周边8省市'] = cement_result_df[provlist_bthsa].sum(axis=1)
                cement_result_df.to_excel(writer, sheet_name='下游行业水泥需求')
                
                # 写入中间结果
                for prod, df in intermediate_results.items():
                    df.to_excel(writer, sheet_name=f'{prod}_水泥中间结果')
                
                # 最终校验
                if (cement_result_df['全国'] == 0).all():
                    print("警告: 水泥需求汇总后全国列仍全为0，检查中间结果表")
                else:
                    print(f"水泥需求汇总完成，全国列非零值年份: {sum(cement_result_df['全国'] != 0)} 个")
            else:
                pd.DataFrame().to_excel(writer, sheet_name='无有效数据')

# ======================== 应用水泥约束（固定 2024 比例） ========================
if constraint_cement and do_cement:
    try:
        cement_forecast_path = path_down_stream + '下游行业水泥需求预测.xlsx'
        df_pred = pd.read_excel(cement_forecast_path, sheet_name='下游行业水泥需求', index_col=0)
        df_pred = df_pred.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        cement_history_path  = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史水泥产量数据.xls'
        cement_history_sheet = '分省历史水泥产量数据'
        real_nat_ton = load_national_series_from_wide_excel(
            cement_history_path, cement_history_sheet, year_limit_max=2024
        )  # Series: index=年份, value=吨

        years_all = pd.Index(range(2010, 2036), name='year')
        df_pred = df_pred.reindex(index=years_all, fill_value=0.0)

        if '全国' in df_pred.columns:
            pred_nat = df_pred['全国'].astype(float)
        else:
            pred_nat = df_pred[provinces].sum(axis=1).astype(float)

        scale_factors = {}
        last_known_scale = 1.0

        # 逐年计算 2010–2024 的比例（有真实就计算，没有就沿用上一年）
        for y in years_all:
            if y > 2024:
                break
            if (y in real_nat_ton.index) and (pred_nat.get(y, 0.0) > 0):
                s = float(real_nat_ton.loc[y]) / float(pred_nat.get(y, 0.0))
                s = float(np.clip(s, 0.1, 10.0))
                last_known_scale = s
            scale_factors[y] = last_known_scale
            print(f"[水泥] {y}年: 比例={scale_factors[y]:.4f} （{'计算' if y in real_nat_ton.index and pred_nat.get(y,0.0)>0 else '沿用上一年'}）")

        # 锁定 2024 年的比例，作为 2025–2035 的固定比例
        ratio_2024 = scale_factors.get(2024, last_known_scale)
        for y in range(2025, 2036):
            scale_factors[y] = ratio_2024
            print(f"[水泥] {y}年: 比例={scale_factors[y]:.4f} （沿用2024比例）")

        # 按比例缩放省份列
        df_out = df_pred.astype(float).copy()
        for y, s in scale_factors.items():
            if y in df_out.index:
                df_out.loc[y, provinces] = df_out.loc[y, provinces].values * s

        df_out['全国'] = df_out[provinces].sum(axis=1)
        if set(provlist_bthsa).issubset(df_out.columns):
            df_out['京津冀及周边8省市'] = df_out[provlist_bthsa].sum(axis=1)

        df_out.to_excel(path_down_stream + '分省水泥消费量预测.xlsx')
        print("水泥约束处理完成（2025+ 固定使用 2024 年比例）")

    except Exception as e:
        print(f"处理水泥约束时出错: {e}")




# 绘制预测图表
if plot_predict:
    # 绘制钢铁消费量图表
    if do_steel and os.path.exists(path_down_stream+'分省钢铁消费量预测.xlsx'):
        try:
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 2.4))
            prod = '钢铁'
            df = pd.read_excel(path_down_stream+f'分省{prod}消费量预测.xlsx')
            df.set_index('year', inplace=True)
            #df = df.replace([np.inf, -np.inf], np.nan).fillna(0) / 10000  # 转换为亿吨
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0) / 100000000
            
            years = np.arange(2010, 2036, 1)
            df_subset = df.loc[years].fillna(0)
            df_provinces = df_subset[provlist_bthsa]
            
            colors = ['#DEECF6','#AFC8E2','#E2F2CD','#B6DAA7','#F9D5D5','#EF98A1','#FBE3C0','#FBC99A']
            bottom = np.zeros(len(years))
            
            for i, province in enumerate(provlist_bthsa):
                province_data = df_provinces[province].clip(lower=0)
                ax1.bar(years, province_data, bottom=bottom, color=colors[i], width=0.7, alpha=0.9, label=province)
                bottom += province_data
            
            ax1.set_ylabel(f'分省{prod}需求 (亿吨)')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
            
            ax2 = ax1.twinx()
            national_data = df_subset['全国'].clip(lower=0)
            line_plot, = ax2.plot(years, national_data, color='#d62728', marker='o', markersize=5, linewidth=1.5, label='全国消费量')
            ax2.set_ylabel(f'全国{prod}消费量 (亿吨)')
            ax2.set_ylim(0, max(national_data.max() * 1.2, 0.1))
            
            # 添加数据标签
            for i, value in enumerate(national_data):
                if value > 0:
                    ax2.text(years[i], value + max(national_data.max() * 0.05, 0.1), f'{value:.1f}', 
                             ha='center', va='bottom', fontsize=6)
            
            ax1.set_xticks(np.arange(2010, 2036, 5))
            ax1.set_xticklabels(np.arange(2010, 2036, 5))
            ax2.grid(visible=False)
            
            # 合并图例
            handles_bar, labels_bar = ax1.get_legend_handles_labels()
            ax1.legend(handles_bar + [line_plot], labels_bar + ['全国消费量'], 
                      loc='upper left', bbox_to_anchor=(1.13, 1))
            
            plt.tight_layout()
            plt.savefig(path_down_stream+f'京津冀及周边8省份{prod}消费量.png', dpi=500)
            plt.close()
            print(f"{prod}消费量图表生成完成")
        except Exception as e:
            print(f"绘制{prod}消费量图表时出错: {e}")
    
    # 绘制水泥消费量图表
    if do_cement and os.path.exists(path_down_stream+'分省水泥消费量预测.xlsx'):
        try:
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 2.4))
            prod = '水泥'
            df = pd.read_excel(path_down_stream+f'分省{prod}消费量预测.xlsx')
            df.set_index('year', inplace=True)
            #df = df.replace([np.inf, -np.inf], np.nan).fillna(0) / 10000  # 转换为亿吨
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0) / 100000000
            
            years = np.arange(2010, 2036, 1)
            df_subset = df.loc[years].fillna(0)
            df_provinces = df_subset[provlist_bthsa]
            
            colors = ['#DEECF6','#AFC8E2','#E2F2CD','#B6DAA7','#F9D5D5','#EF98A1','#FBE3C0','#FBC99A']
            bottom = np.zeros(len(years))
            
            for i, province in enumerate(provlist_bthsa):
                province_data = df_provinces[province].clip(lower=0)
                ax1.bar(years, province_data, bottom=bottom, color=colors[i], width=0.7, alpha=0.9, label=province)
                bottom += province_data
            
            ax1.set_ylabel(f'分省{prod}需求 (亿吨)')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
            
            ax2 = ax1.twinx()
            national_data = df_subset['全国'].clip(lower=0)
            line_plot, = ax2.plot(years, national_data, color='#d62728', marker='o', markersize=5, linewidth=1.5, label='全国消费量')
            ax2.set_ylabel(f'全国{prod}消费量 (亿吨)')
            ax2.set_ylim(0, max(national_data.max() * 1.2, 0.1))
            
            # 添加数据标签
            for i, value in enumerate(national_data):
                if value > 0:
                    ax2.text(years[i], value + max(national_data.max() * 0.05, 0.1), f'{value:.1f}', 
                             ha='center', va='bottom', fontsize=6)
            
            ax1.set_xticks(np.arange(2010, 2036, 5))
            ax1.set_xticklabels(np.arange(2010, 2036, 5))
            ax2.grid(visible=False)
            
            # 合并图例
            handles_bar, labels_bar = ax1.get_legend_handles_labels()
            ax1.legend(handles_bar + [line_plot], labels_bar + ['全国消费量'], 
                      loc='upper left', bbox_to_anchor=(1.13, 1))
            
            plt.tight_layout()
            plt.savefig(path_down_stream+f'京津冀及周边8省份{prod}消费量.png', dpi=500)
            plt.close()
            print(f"{prod}消费量图表生成完成")
        except Exception as e:
            print(f"绘制{prod}消费量图表时出错: {e}")

print("程序执行完毕")