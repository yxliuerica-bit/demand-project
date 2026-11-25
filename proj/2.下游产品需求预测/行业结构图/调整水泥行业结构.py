# -*- coding: utf-8 -*-
"""
按目标行业份额调整“下游行业水泥需求预测.xlsx”的中间结果，
保持两类公路（高速/其他等级）不合并；
为 2010–2035 做逐年插值并对 2024→2025→2026 进行余弦缓冲平滑；
另存为新文件，并绘制 全国 与 京津冀及周边8省份 的堆叠结构图。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- 基本路径 ----------------
sce = 'Sce_Pop_BAU_GDP_BAU'
base_dir = f'/lustre/home/litong/intern/京津冀货运发展/output/predict_results/{sce}/12_跨省应用模型预测/分省完整数据/'
src_xlsx = os.path.join(base_dir, '下游行业水泥需求预测.xlsx')
dst_xlsx = os.path.join(base_dir, '下游行业水泥需求预测_调整行业结构_v2.xlsx')

years = list(range(2010, 2035 + 1))

# ---------------- 行业名称（与原文件一致） ----------------
SECT_BUILD   = '新增建筑面积'
SECT_RAIL    = '新增铁路里程'
SECT_RURAL   = '农村水电基础设施投资'
SECT_HWY     = '新增高速公路里程'
SECT_ROAD_OT = '新增其他等级公路里程'

# ---------------- 目标份额（可按需微调；四类之和=1） ----------------
# 参照你的“目标图2”（忽略 Others），设定四类可控行业目标份额：
#   Building / Railway / Rural / Road(total)
target_shares = {
    'building': 0.35,
    'railway' : 0.03,
    'rural'   : 0.45,
    'road'    : 0.17,
}

# ---------------- 颜色（沿用你之前的 cement_colors 顺序） ----------------
# 对应（从下到上/传参顺序）：
# 农村水电(灰) / 铁路(深蓝) / 其他等级公路(浅蓝) / 高速公路(紫) / 建筑(青绿)
cement_colors = ['#bdbdbd', '#1e88e5', '#90caf9', '#7e57c2', '#26a69a']

# ---------------- 工具函数 ----------------
def read_sheet(xlsx_path, sector, suffix='水泥中间结果'):
    """
    读取某行业“省份×年份”的中间结果表，做年份对齐与缺口插值（修补 2024 等断点）。
    """
    df = pd.read_excel(xlsx_path, sheet_name=f'{sector}_{suffix}', index_col=0)
    df = df.select_dtypes(include='number')           # 仅数值列（省份）
    df = df.reindex(index=years)                      # 年份对齐
    # 按年份做线性插值，向两端延拓，清理无穷/缺失并裁零
    df = df.sort_index().interpolate(method='index', axis=0, limit_direction='both')
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    return df

def nat(series_df):
    """全国合计：各省求和（Series，索引为年份）"""
    return series_df.sum(axis=1)

def safe_ratio(tgt, base):
    """逐年缩放系数：tgt/base，遇到 0 及缺失做保护，限制到合理区间"""
    return (tgt / base.replace(0, np.nan)).fillna(0.0).clip(lower=0.0, upper=50.0)

def apply_row_scale(df, k):
    """按年份把系数 k 应用到省份×年份矩阵"""
    out = df.copy()
    for y in years:
        out.loc[y, :] = out.loc[y, :].values * float(k.loc[y])
    return out

def smooth_2024_2025(df, year0=2024, w_2025=0.60, w_2026=0.30):
    """
    让 2024→2025→2026 更顺滑：对每个省份做余弦缓冲。
      - w_2025: 2025 的缓冲强度（0~1，越大越接近光滑过渡）
      - w_2026: 2026 的缓冲强度（一般取 w_2025 的一半左右）
    """
    if year0-1 not in df.index or year0 not in df.index or year0+1 not in df.index or year0+2 not in df.index:
        return df

    def cosmix(a, b, t):
        # t∈[0,1] 的余弦插值：更柔和
        w = 0.5 - 0.5*np.cos(np.pi*t)
        return (1-w)*a + w*b

    out = df.copy()
    # 2025 拉向 2024~2026 的中间值
    mid_2025 = cosmix(df.loc[year0], df.loc[year0+2], 0.5)
    out.loc[year0+1] = (1-w_2025)*df.loc[year0+1] + w_2025*mid_2025

    # 2026 轻微拉向 2025(已缓冲)~2027 的 1/3 位置
    if (year0+3) in df.index:
        mid_2026 = cosmix(out.loc[year0+1], df.loc[year0+3], 1/3)
        out.loc[year0+2] = (1-w_2026)*df.loc[year0+2] + w_2026*mid_2026

    return out

def plot_structure(df_plot, title, out_png):
    """
    绘制堆叠结构图（df_plot 列顺序：Rural / Rail / Road-Other / Highway / Building）
    """
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC','Droid Sans Fallback','sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

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
    ax.set_xlabel('Year'); ax.set_ylabel('Cement demand (tons)')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ 已保存图片：{out_png}')

# ---------------- 读取原始五个行业表（带插值与清理） ----------------
build_df   = read_sheet(src_xlsx, SECT_BUILD)
rail_df    = read_sheet(src_xlsx, SECT_RAIL)
rural_df   = read_sheet(src_xlsx, SECT_RURAL)
hwy_df     = read_sheet(src_xlsx, SECT_HWY)
road_ot_df = read_sheet(src_xlsx, SECT_ROAD_OT)

# ---------------- 计算全国原始总量与目标份额 ----------------
nat_build = nat(build_df)
nat_rail  = nat(rail_df)
nat_rural = nat(rural_df)
nat_hwy   = nat(hwy_df)
nat_road_ot = nat(road_ot_df)
nat_road_total = nat_hwy + nat_road_ot

# 以四类之和作为“可分配总量”
total_four = (nat_build + nat_rail + nat_rural + nat_road_total).replace(0, np.nan)

# 目标全国序列
tgt_build      = total_four * target_shares['building']
tgt_rail       = total_four * target_shares['railway']
tgt_rural      = total_four * target_shares['rural']
tgt_road_total = total_four * target_shares['road']

# Road 的目标总量按“当年原始高速/其他等级占比”拆回两表
share_hwy = (nat_hwy / nat_road_total.replace(0, np.nan)).fillna(0.5).clip(0, 1)
tgt_hwy     = tgt_road_total * share_hwy
tgt_road_ot = tgt_road_total * (1.0 - share_hwy)

# ---------------- 逐年缩放到目标份额 ----------------
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

# ---------------- 对 2024→2025→2026 做余弦缓冲平滑 ----------------
build_adj   = smooth_2024_2025(build_adj,   w_2025=0.60, w_2026=0.30)
rail_adj    = smooth_2024_2025(rail_adj,    w_2025=0.60, w_2026=0.30)
rural_adj   = smooth_2024_2025(rural_adj,   w_2025=0.60, w_2026=0.30)
hwy_adj     = smooth_2024_2025(hwy_adj,     w_2025=0.60, w_2026=0.30)
road_ot_adj = smooth_2024_2025(road_ot_adj, w_2025=0.60, w_2026=0.30)

# ---------------- 另存为新 Excel（不覆盖原文件） ----------------
with pd.ExcelWriter(dst_xlsx) as writer:
    build_adj.to_excel(writer,   sheet_name=f'{SECT_BUILD}_水泥中间结果_调整后v2')
    rail_adj.to_excel(writer,    sheet_name=f'{SECT_RAIL}_水泥中间结果_调整后v2')
    rural_adj.to_excel(writer,   sheet_name=f'{SECT_RURAL}_水泥中间结果_调整后v2')
    hwy_adj.to_excel(writer,     sheet_name=f'{SECT_HWY}_水泥中间结果_调整后v2')
    road_ot_adj.to_excel(writer, sheet_name=f'{SECT_ROAD_OT}_水泥中间结果_调整后v2')
print(f'✅ 已生成：{dst_xlsx}')

# ---------------- 计算“全国”并作图 ----------------
nat_build_new   = nat(build_adj)
nat_rail_new    = nat(rail_adj)
nat_rural_new   = nat(rural_adj)
nat_hwy_new     = nat(hwy_adj)
nat_road_ot_new = nat(road_ot_adj)

df_nat = pd.DataFrame({
    SECT_RURAL:   nat_rural_new,
    SECT_RAIL:    nat_rail_new,
    SECT_ROAD_OT: nat_road_ot_new,
    SECT_HWY:     nat_hwy_new,
    SECT_BUILD:   nat_build_new,
}).reindex(years).fillna(0.0)

plot_structure(df_nat, 'Cement demand structure（全国，调整后）',
               os.path.join(base_dir, '全国_水泥消费结构_堆叠面积_调整后v2.png'))

# ---------------- 计算“京津冀及周边8省份”并作图 ----------------
bthsa = ['北京市','天津市','河北省','河南省','山东省','山西省','内蒙古自治区','陕西省']

def region_sum(df, provs):
    cols = [c for c in provs if c in df.columns]
    return df[cols].sum(axis=1)

reg_build   = region_sum(build_adj,   bthsa)
reg_rail    = region_sum(rail_adj,    bthsa)
reg_rural   = region_sum(rural_adj,   bthsa)
reg_hwy     = region_sum(hwy_adj,     bthsa)
reg_road_ot = region_sum(road_ot_adj, bthsa)

df_region = pd.DataFrame({
    SECT_RURAL:   reg_rural,
    SECT_RAIL:    reg_rail,
    SECT_ROAD_OT: reg_road_ot,
    SECT_HWY:     reg_hwy,
    SECT_BUILD:   reg_build,
}).reindex(years).fillna(0.0)

plot_structure(df_region, 'Cement demand structure（京津冀及周边8省份，调整后）',
               os.path.join(base_dir, '京津冀及周边8省份_水泥消费结构_堆叠面积_调整后v2.png'))

print('🎉 All done.')
    