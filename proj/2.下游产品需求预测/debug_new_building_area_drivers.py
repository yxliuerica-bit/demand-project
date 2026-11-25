# -*- coding: utf-8 -*-
"""
新增建筑面积：驱动诊断（山东/湖北/北京）
- 读取省份完整数据（历史原始/回填/平滑 + 预测）
- 读取未来特征（GDP/人口/三产GDP/城镇人口）
- 画两张图：A) 目标(历史/平滑/预测)；B) X 特征(2010–2035)
- 输出增长诊断表：2025–2030 & 2030–2035 的 CAGR / 区间增幅
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- 基本路径 ----------------
sce = 'Sce_Pop_BAU_GDP_BAU'
base_out = f'/lustre/home/litong/intern/京津冀货运发展/output/predict_results/{sce}/11_全国建模回填无法建模省份历史数据/'
prov_full_dir = os.path.join(base_out, '省份完整数据')  # 省份{指标}完整数据.csv
scenario_dir  = '/lustre/home/litong/intern/京津冀货运发展/output/scenarios'  # 未来特征CSV所在
save_dir      = os.path.join(base_out, '调试_新增建筑面积_驱动诊断')
os.makedirs(save_dir, exist_ok=True)

# ---------------- 配置 ----------------
TARGET_FACTOR = '新增建筑面积'
PROVINCES = ['山东省', '湖北省', '北京市']

YEARS_ALL   = list(range(2010, 2036))
YEARS_FUT   = list(range(2025, 2036))

plt.rcParams['font.sans-serif']  = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['font.size']        = 11
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 读取省份目标（历史/平滑/预测） ----------------
def load_province_target(province: str) -> pd.DataFrame:
    """
    从 '省份完整数据/省名新增建筑面积完整数据.csv' 读取
    返回列：
      年份, 原始, 模型回填, 历史_融合值, 历史_平滑, 预测
    """
    path = os.path.join(prov_full_dir, f'{province}{TARGET_FACTOR}完整数据.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'缺少省份完整数据：{path}')
    df = pd.read_csv(path)
    # 统一列名
    col_map = {
        f'原始_{TARGET_FACTOR}': '原始',
        f'模型回填_{TARGET_FACTOR}': '模型回填',
        '历史_融合值': '历史_融合值',
        f'历史_平滑_{TARGET_FACTOR}': '历史_平滑',
        f'预测_平滑优先_{TARGET_FACTOR}': '预测'  # 你当前绘图使用“平滑优先=原始预测”
    }
    # 构建输出框架
    out = pd.DataFrame({'年份': pd.to_numeric(df['年份'], errors='coerce')})
    for k, v in col_map.items():
        out[v] = df[k] if k in df.columns else np.nan
    out = out.dropna(subset=['年份'])
    out['年份'] = out['年份'].astype(int)
    out = out.sort_values('年份')
    # 对齐年份并清理
    out = out.set_index('年份').reindex(YEARS_ALL).reset_index()
    for c in ['原始','模型回填','历史_融合值','历史_平滑','预测']:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
            out[c] = out[c].clip(lower=0)  # 非负
    return out

# ---------------- 读取未来特征 ----------------
def melt_future(path, value_name):
    df = pd.read_csv(path)
    df = df.rename(columns={'Unnamed: 0': '年份'})
    df = df.melt(id_vars='年份', var_name='地区', value_name=value_name)
    df['年份'] = pd.to_numeric(df['年份'], errors='coerce').astype(int)
    return df

def load_future_features_for_province(province: str) -> pd.DataFrame:
    """
    返回列：年份, GDP, 人口, 第三产业GDP, 城镇人口（2010–2035；若历史期无则为 NaN/0）
    """
    gdp_f     = os.path.join(scenario_dir, 'gdp_predictions_基准_完整年份.csv')
    pop_f     = os.path.join(scenario_dir, 'population_predictions_基准_完整年份.csv')
    urban_f   = os.path.join(scenario_dir, 'urban_population_predictions_基准_完整年份.csv')
    third_f   = os.path.join(scenario_dir, 'third_gdp_predictions_基准_完整年份.csv')

    gdp   = melt_future(gdp_f, 'GDP')
    pop   = melt_future(pop_f, '人口')
    third = melt_future(third_f, '第三产业GDP')
    urban = melt_future(urban_f, '城镇人口')

    fut = (gdp.merge(pop,   on=['年份','地区'], how='inner')
               .merge(third, on=['年份','地区'], how='inner')
               .merge(urban, on=['年份','地区'], how='inner'))
    fut = fut[fut['地区'] == province].copy()
    fut = fut[['年份','GDP','人口','第三产业GDP','城镇人口']]
    # 对齐全年份
    fut = fut.set_index('年份').reindex(YEARS_ALL).reset_index()
    for c in ['GDP','人口','第三产业GDP','城镇人口']:
        fut[c] = pd.to_numeric(fut[c], errors='coerce')
    return fut

# ---------------- 诊断指标 ----------------
def cagr(v0, v1, years):
    if v0 is None or v1 is None or v0 <= 0 or years <= 0:
        return np.nan
    return (v1 / v0) ** (1.0/years) - 1.0

def summary_growth(series: pd.Series, a: int, b: int) -> dict:
    """给定年份段 [a,b] 的 CAGR 与区间增幅"""
    s = series.reindex(range(a, b+1)).astype(float)
    v0, v1 = s.iloc[0], s.iloc[-1]
    return {
        '起点': float(v0),
        '终点': float(v1),
        '区间增幅(%)': (float(v1 - v0) / v0 * 100.0) if v0 > 0 else np.nan,
        'CAGR(%)': cagr(v0, v1, b - a) * 100.0 if (v0 > 0 and (b - a) > 0) else np.nan
    }

def make_plots_and_stats(province: str):
    # 读目标
    tgt = load_province_target(province)  # 年份, 原始/模型回填/历史_融合值/历史_平滑/预测
    # 读未来特征
    fut = load_future_features_for_province(province)  # 年份, 4个X
    # ---------- 图 A：目标（历史→平滑→预测） ----------
    fig1 = plt.figure(figsize=(10.5, 4.2))
    ax = fig1.gca()
    ax.plot(tgt['年份'], tgt['历史_融合值'], label='历史（原始→回填融合）', linewidth=2)
    if '历史_平滑' in tgt:
        ax.plot(tgt['年份'], tgt['历史_平滑'], label='历史（平滑）', linewidth=2, linestyle='--')
    if '预测' in tgt:
        ax.plot(tgt['年份'], tgt['预测'], label='预测', linewidth=2, linestyle='-.')
    ax.axvline(x=2025, color='gray', linestyle=':', linewidth=1.5)
    ax.set_title(f'{province} 新增建筑面积：历史与预测')
    ax.set_xlabel('年份'); ax.set_ylabel('新增建筑面积（万平方米）')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    out_png1 = os.path.join(save_dir, f'{province}_A_目标_历史与预测.png')
    plt.savefig(out_png1, dpi=300, bbox_inches='tight'); plt.close()
    print(f'[OK] {out_png1}')

    # ---------- 图 B：四个驱动特征 ----------
    fig2 = plt.figure(figsize=(10.5, 4.2))
    ax2 = fig2.gca()
    for col in ['GDP','人口','第三产业GDP','城镇人口']:
        if col in fut.columns:
            ax2.plot(fut['年份'], fut[col], label=col, linewidth=2)
    ax2.axvline(x=2025, color='gray', linestyle=':', linewidth=1.5)
    ax2.set_title(f'{province} 四个特征（2010–2035）')
    ax2.set_xlabel('年份'); ax2.set_ylabel('原值（单位依输入文件）')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    plt.tight_layout()
    out_png2 = os.path.join(save_dir, f'{province}_B_特征_时间轨迹.png')
    plt.savefig(out_png2, dpi=300, bbox_inches='tight'); plt.close()
    print(f'[OK] {out_png2}')

    # ---------- 诊断表：增长率/区间增幅 ----------
    # 目标：优先用“预测”；历史期给个参考
    pred_series = tgt.set_index('年份')['预测'].astype(float)
    hist_series = tgt.set_index('年份')['历史_平滑'].astype(float)

    rows = []
    # 目标诊断
    rows.append({'省份': province, '变量': '目标:新增建筑面积(预测)', '区间': '2025-2030', **summary_growth(pred_series, 2025, 2030)})
    rows.append({'省份': province, '变量': '目标:新增建筑面积(预测)', '区间': '2030-2035', **summary_growth(pred_series, 2030, 2035)})
    # 历史参考
    rows.append({'省份': province, '变量': '目标:新增建筑面积(历史平滑)', '区间': '2019-2024', **summary_growth(hist_series, 2019, 2024)})

    # 特征诊断
    for col in ['GDP','人口','第三产业GDP','城镇人口']:
        s = fut.set_index('年份')[col].astype(float)
        rows.append({'省份': province, '变量': f'特征:{col}', '区间': '2025-2030', **summary_growth(s, 2025, 2030)})
        rows.append({'省份': province, '变量': f'特征:{col}', '区间': '2030-2035', **summary_growth(s, 2030, 2035)})

    diag = pd.DataFrame(rows)
    return diag

# ---------------- 主流程 ----------------
all_stats = []
for prov in PROVINCES:
    try:
        diag = make_plots_and_stats(prov)
        all_stats.append(diag)
    except Exception as e:
        print(f'[失败] {prov}: {e}')

if all_stats:
    out_csv = os.path.join(save_dir, '三省_新增建筑面积_增长诊断.csv')
    pd.concat(all_stats, ignore_index=True).to_csv(out_csv, index=False)
    print(f'[OK] 导出诊断：{out_csv}')
else:
    print('没有生成任何诊断结果，请检查输入文件路径。')
