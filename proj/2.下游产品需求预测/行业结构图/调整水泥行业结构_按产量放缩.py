# -*- coding: utf-8 -*-
"""
按目标行业份额调整“下游行业水泥需求预测.xlsx”的中间结果，
保持两类公路（高速/其他等级）不合并；
为 2010–2035 做逐年插值并对 2024→2025→2026 进行余弦缓冲平滑；
——新增——
读取历史真值（分省历史水泥产量数据.xls），
按年把所有行业等比例放缩，使“当年全国总量=历史真值”，
且 2025 年及以后使用 2024 年的放缩系数。
保存放缩后的结果到新文件，并（可选）绘图。
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Matplotlib 字体 ----------------
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Droid Sans Fallback', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 基本路径 ----------------
sce = 'Sce_Pop_BAU_GDP_BAU'
base_dir = f'/lustre/home/litong/intern/京津冀货运发展/output/predict_results/{sce}/12_跨省应用模型预测/分省完整数据/'
src_xlsx = os.path.join(base_dir, '下游行业水泥需求预测.xlsx')
dst_xlsx = os.path.join(base_dir, '下游行业水泥需求预测_调整行业结构_v2.xlsx')

# 历史真值（分省历史水泥产量）
hist_xls = '/lustre/home/litong/intern/京津冀货运发展/data/预测原始数据/分省历史水泥产量数据.xls'
# 历史表单位：万吨；若你的模型用“吨”，需要 ×1e4；若模型也是“万吨”，改为 1
HIST_UNIT_FACTOR = 1e4

# 放缩后另存的新文件
dst_scaled_xlsx = os.path.join(base_dir, '下游行业水泥需求预测_调整行业结构_v2_按历史总量放缩.xlsx')

years = list(range(2010, 2035 + 1))

# ---------------- 行业名称（与原文件一致） ----------------
SECT_BUILD   = '新增建筑面积'
SECT_RAIL    = '新增铁路里程'
SECT_RURAL   = '农村水电基础设施投资'
SECT_HWY     = '新增高速公路里程'
SECT_ROAD_OT = '新增其他等级公路里程'

# ---------------- 目标份额（四类之和=1） ----------------
target_shares = {
    'building': 0.35,
    'railway' : 0.03,
    'rural'   : 0.45,
    'road'    : 0.17,
}

# ---------------- 颜色 ----------------
cement_colors = ['#bdbdbd', '#1e88e5', '#90caf9', '#7e57c2', '#26a69a']

# ---------------- 工具函数 ----------------
def read_sheet(xlsx_path, sector, suffix='水泥中间结果'):
    """读取某行业 省份×年份 中间结果，做年份对齐 + 插值 + 清理。"""
    df = pd.read_excel(xlsx_path, sheet_name=f'{sector}_{suffix}', index_col=0)
    df = df.select_dtypes(include='number')
    df = df.reindex(index=years)
    df = df.sort_index().interpolate(method='index', axis=0, limit_direction='both')
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    return df

def nat(df):  # 全国合计：各省求和（Series，索引为年份）
    return df.sum(axis=1)

def safe_ratio(tgt, base):
    return (tgt / base.replace(0, np.nan)).fillna(0.0).clip(lower=0.0, upper=50.0)

def apply_row_scale(df, k):
    out = df.copy()
    for y in years:
        if y in k.index:
            out.loc[y, :] = out.loc[y, :].values * float(k.loc[y])
    return out

def smooth_2024_2025(df, year0=2024, w_2025=0.60, w_2026=0.30):
    if year0-1 not in df.index or year0+2 not in df.index:
        return df
    def cosmix(a, b, t):
        w = 0.5 - 0.5*np.cos(np.pi*t)
        return (1-w)*a + w*b
    out = df.copy()
    mid_2025 = cosmix(df.loc[year0], df.loc[year0+2], 0.5)
    out.loc[year0+1] = (1-w_2025)*df.loc[year0+1] + w_2025*mid_2025
    if (year0+3) in df.index:
        mid_2026 = cosmix(out.loc[year0+1], df.loc[year0+3], 1/3)
        out.loc[year0+2] = (1-w_2026)*df.loc[year0+2] + w_2026*mid_2026
    return out

def plot_structure(df_plot, title, out_png):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.stackplot(
        df_plot.index,
        df_plot[SECT_RURAL], df_plot[SECT_RAIL],
        df_plot[SECT_ROAD_OT], df_plot[SECT_HWY],
        df_plot[SECT_BUILD],
        labels=[SECT_RURAL, SECT_RAIL, SECT_ROAD_OT, SECT_HWY, SECT_BUILD],
        colors=cement_colors, alpha=0.95, linewidth=0.2
    )
    ax.set_title(title)
    ax.set_xlabel('Year'); ax.set_ylabel('Cement demand')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ 已保存图片：{out_png}')

# ========== 1) 读取原始五个行业表 ==========
build_df   = read_sheet(src_xlsx, SECT_BUILD)
rail_df    = read_sheet(src_xlsx, SECT_RAIL)
rural_df   = read_sheet(src_xlsx, SECT_RURAL)
hwy_df     = read_sheet(src_xlsx, SECT_HWY)
road_ot_df = read_sheet(src_xlsx, SECT_ROAD_OT)

# ========== 2) 计算全国原始总量与目标份额 ==========
nat_build = nat(build_df)
nat_rail  = nat(rail_df)
nat_rural = nat(rural_df)
nat_hwy   = nat(hwy_df)
nat_road_ot = nat(road_ot_df)
nat_road_total = nat_hwy + nat_road_ot

total_four = (nat_build + nat_rail + nat_rural + nat_road_total).replace(0, np.nan)

tgt_build      = total_four * target_shares['building']
tgt_rail       = total_four * target_shares['railway']
tgt_rural      = total_four * target_shares['rural']
tgt_road_total = total_four * target_shares['road']

share_hwy   = (nat_hwy / nat_road_total.replace(0, np.nan)).fillna(0.5).clip(0, 1)
tgt_hwy     = tgt_road_total * share_hwy
tgt_road_ot = tgt_road_total * (1.0 - share_hwy)

# ========== 3) 逐年缩放到目标份额 ==========
k_build   = safe_ratio(tgt_build,   nat_build)
k_rail    = safe_ratio(tgt_rail,    nat_rail)
k_rural   = safe_ratio(tgt_rural,   nat_rural)
k_hwy     = safe_ratio(tgt_hwy,     nat_hwy)
k_road_ot = safe_ratio(tgt_road_ot, nat_road_ot)

build_adj   = apply_row_scale(build_df,   k_build)
rail_adj    = apply_row_scale(rail_df,    k_rail)
rural_adj   = apply_row_scale(rural_df,   k_rural)
hwy_adj     = apply_row_scale(hwy_df,     k_hwy)
road_ot_adj = apply_row_scale(road_ot_df, k_road_ot)

# ========== 4) 2024→2025→2026 余弦缓冲 ==========
build_adj   = smooth_2024_2025(build_adj,   w_2025=0.60, w_2026=0.30)
rail_adj    = smooth_2024_2025(rail_adj,    w_2025=0.60, w_2026=0.30)
rural_adj   = smooth_2024_2025(rural_adj,   w_2025=0.60, w_2026=0.30)
hwy_adj     = smooth_2024_2025(hwy_adj,     w_2025=0.60, w_2026=0.30)
road_ot_adj = smooth_2024_2025(road_ot_adj, w_2025=0.60, w_2026=0.30)

# ========== 5) 另存“调整后 v2”（未放缩） ==========
with pd.ExcelWriter(dst_xlsx) as writer:
    build_adj.to_excel(writer,   sheet_name=f'{SECT_BUILD}_水泥中间结果_调整后v2')
    rail_adj.to_excel(writer,    sheet_name=f'{SECT_RAIL}_水泥中间结果_调整后v2')
    rural_adj.to_excel(writer,   sheet_name=f'{SECT_RURAL}_水泥中间结果_调整后v2')
    hwy_adj.to_excel(writer,     sheet_name=f'{SECT_HWY}_水泥中间结果_调整后v2')
    road_ot_adj.to_excel(writer, sheet_name=f'{SECT_ROAD_OT}_水泥中间结果_调整后v2')
print(f'✅ 已生成（未放缩）：{dst_xlsx}')

# ========== 6) 读取历史真值（专门适配你截图的 .xls） ==========
def read_hist_nat_series_from_bureau_xls(xls_path, unit_factor=1.0):
    """
    适配类似截图的结构：
      第1~3行是说明；第4行为表头：'地区','2024年','2023年',..., '2010年' ...
      行为各省（含直辖市/自治区）。
    返回 Series: index=年份(int), value=全国总量(已按 unit_factor 换算)
    """
    # 用第 4 行作为列名
    df = pd.read_excel(xls_path, header=3)
    if df is None or df.dropna(how='all').empty:
        raise ValueError("历史表为空")

    # 统一列名，提取年份列
    df.columns = [str(c).strip() if not re.search(r'\d{4}', str(c)) else str(c) for c in df.columns]
    year_cols = [c for c in df.columns if re.search(r'\d{4}', c)]
    if not year_cols:
        raise ValueError("未识别到年份列（如 2020年）")

    # 仅保留年份列，转换为数值
    df_years = df[year_cols].apply(pd.to_numeric, errors='coerce')

    # 全国总量 = 各省相加（跨行求和），得到按“列=年份”的 Series
    s = df_years.sum(axis=0)

    # 列名里提取 4 位年份并转为 int 索引
    years_int = s.index.to_series().str.extract(r'(\d{4})')[0].astype(int).values
    s.index = years_int
    s = s.sort_index()

    # 单位换算：历史表是“万吨”，若模型用“吨”，乘以 1e4
    s = s * float(unit_factor)
    return s

hist_nat = read_hist_nat_series_from_bureau_xls(hist_xls, unit_factor=HIST_UNIT_FACTOR)

# ========== 7) 计算当前全国总量 & 构造放缩系数 ==========
nat_build_new   = nat(build_adj)
nat_rail_new    = nat(rail_adj)
nat_rural_new   = nat(rural_adj)
nat_hwy_new     = nat(hwy_adj)
nat_road_ot_new = nat(road_ot_adj)
nat_total_new   = (nat_build_new + nat_rail_new + nat_rural_new + nat_hwy_new + nat_road_ot_new).reindex(years).fillna(0.0)

# k_hist[y] = 历史全国真值 / 当前（五行业合计）全国总量
k_hist = pd.Series(index=years, dtype=float)
for y in years:
    if y in hist_nat.index:
        base = nat_total_new.loc[y] if y in nat_total_new.index else np.nan
        k_hist.loc[y] = (hist_nat.loc[y] / base) if base not in (None, 0) else np.nan

# 2025+ 使用 2024 的系数；若没有 2024，则用历史里的最后一年
if len(hist_nat.index) == 0:
    raise ValueError("历史真值为空，无法放缩。")
ref_year = 2024 if 2024 in hist_nat.index else int(hist_nat.index.max())
k_ref = k_hist.get(ref_year, np.nan)
if not np.isfinite(k_ref):
    base = nat_total_new.get(ref_year, np.nan)
    k_ref = (hist_nat.loc[ref_year] / base) if base not in (None, 0) else 1.0

for y in years:
    if pd.isna(k_hist.loc[y]):
        k_hist.loc[y] = k_ref if y >= 2025 else (k_hist.dropna().iloc[0] if len(k_hist.dropna())>0 else 1.0)

k_hist = k_hist.fillna(1.0).clip(lower=0.0, upper=50.0)

# ========== 8) 对五个行业做“按年等比例放缩” ==========
build_scaled   = apply_row_scale(build_adj,   k_hist)
rail_scaled    = apply_row_scale(rail_adj,    k_hist)
rural_scaled   = apply_row_scale(rural_adj,   k_hist)
hwy_scaled     = apply_row_scale(hwy_adj,     k_hist)
road_ot_scaled = apply_row_scale(road_ot_adj, k_hist)

# 审计：历史年份全国总量的匹配程度
nat_scaled_total = (nat(build_scaled) + nat(rail_scaled) + nat(rural_scaled) + nat(hwy_scaled) + nat(road_ot_scaled)).reindex(hist_nat.index)
audit_compare = pd.DataFrame({'hist_total': hist_nat, 'scaled_total': nat_scaled_total})
print("\n=== 放缩校验（历史年份） ===")
print(audit_compare.head(20))
if (audit_compare['hist_total'] != 0).any():
    rel_err = ((audit_compare['scaled_total'] - audit_compare['hist_total']) / audit_compare['hist_total']).abs().mean()
    print("相对误差（均值）:", float(rel_err))

# ========== 9) 保存放缩后的结果 ==========
with pd.ExcelWriter(dst_scaled_xlsx) as writer:
    build_scaled.to_excel(writer,   sheet_name=f'{SECT_BUILD}_水泥中间结果_调整后v2_放缩')
    rail_scaled.to_excel(writer,    sheet_name=f'{SECT_RAIL}_水泥中间结果_调整后v2_放缩')
    rural_scaled.to_excel(writer,   sheet_name=f'{SECT_RURAL}_水泥中间结果_调整后v2_放缩')
    hwy_scaled.to_excel(writer,     sheet_name=f'{SECT_HWY}_水泥中间结果_调整后v2_放缩')
    road_ot_scaled.to_excel(writer, sheet_name=f'{SECT_ROAD_OT}_水泥中间结果_调整后v2_放缩')

    audit = pd.DataFrame({
        'current_total_before_scale': nat_total_new,
        f'history_total({os.path.basename(hist_xls)})': hist_nat.reindex(years),
        'scale_factor_used': k_hist,
        'total_after_scale': (nat(build_scaled) + nat(rail_scaled) + nat(rural_scaled) + nat(hwy_scaled) + nat(road_ot_scaled)).reindex(years)
    })
    audit.to_excel(writer, sheet_name='放缩系数_与全国总量对比')

print(f'\n✅ 已生成（放缩后）：{dst_scaled_xlsx}')

# ========== 10) （可选）放缩后再绘图 ==========
def region_sum(df, provs):
    cols = [c for c in provs if c in df.columns]
    return df[cols].sum(axis=1)

df_nat_scaled = pd.DataFrame({
    SECT_RURAL:   nat(rural_scaled),
    SECT_RAIL:    nat(rail_scaled),
    SECT_ROAD_OT: nat(road_ot_scaled),
    SECT_HWY:     nat(hwy_scaled),
    SECT_BUILD:   nat(build_scaled),
}).reindex(years).fillna(0.0)

plot_structure(df_nat_scaled, 'Cement demand structure（全国，按下游行业）',
               os.path.join(base_dir, '全国_水泥消费结构_堆叠面积_放缩后.png'))

bthsa = ['北京市','天津市','河北省','河南省','山东省','山西省','内蒙古自治区','陕西省']
df_region_scaled = pd.DataFrame({
    SECT_RURAL:   region_sum(rural_scaled,   bthsa),
    SECT_RAIL:    region_sum(rail_scaled,    bthsa),
    SECT_ROAD_OT: region_sum(road_ot_scaled, bthsa),
    SECT_HWY:     region_sum(hwy_scaled,     bthsa),
    SECT_BUILD:   region_sum(build_scaled,   bthsa),
}).reindex(years).fillna(0.0)

plot_structure(df_region_scaled, 'Cement demand structure（京津冀及周边8省份，按下游行业）',
               os.path.join(base_dir, '京津冀及周边8省份_水泥消费结构_堆叠面积_放缩后.png'))

print('🎉 All done.')

