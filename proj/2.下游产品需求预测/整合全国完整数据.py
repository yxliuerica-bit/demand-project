import os
import pandas as pd

# ================== 路径与指标 ==================
scenario_name = '高增长'
base = f"/lustre/home/litong/intern/京津冀货运发展/output/2.消费需求预测/Sce_{scenario_name}"
input_dir = os.path.join(base, "省份完整数据")
output_dir = os.path.join(base, "全国完整数据")
os.makedirs(output_dir, exist_ok=True)

# 与主流程一致的指标清单
predict_factor_list = [
    "新增建筑面积", "汽车产量",
    "新增铁路里程",
    "洗衣机产量", "电冰箱产量", "空调产量",
    "新增高速公路里程", "新增其他等级公路里程",
    "农村水电基础设施投资",
    "火力发电量", "发电量", "电力消费量"
]

YEAR_MIN, YEAR_MAX = 2005, 2035

# ================== 工具函数 ==================
def parse_filename(filename: str):
    """从 '<省份><指标>完整数据.csv' 提取省份与指标"""
    for factor in predict_factor_list:
        if factor in filename and filename.endswith("完整数据.csv"):
            prov = filename.replace(factor, "").replace("完整数据.csv", "").strip()
            return prov, factor
    return None, None

def choose_hist_value(row, factor):
    """年份<=2024时的取值优先级"""
    for col in [
        f"历史_平滑_{factor}", 
        f"历史_融合值",
        f"模型回填_{factor}",
        f"原始_{factor}",
    ]:
        if col in row and pd.notna(row[col]):
            return row[col]
    return pd.NA

def choose_future_value(row, factor):
    """年份>=2025时的取值优先级"""
    for col in [
        f"预测_原始数据_{factor}",
        f"预测_平滑数据_{factor}",
        f"预测_平滑优先_{factor}",     # 兼容旧导出（如果还在）
    ]:
        if col in row and pd.notna(row[col]):
            return row[col]
    return pd.NA

# ================== 主合并流程 ==================
# 指标 -> 各省DataFrame列表
factor_buckets = {f: [] for f in predict_factor_list}

files = [f for f in os.listdir(input_dir) if f.endswith("完整数据.csv")]
if not files:
    raise SystemExit(f"未在目录中找到CSV：{input_dir}")

for fname in files:
    prov, factor = parse_filename(fname)
    if not prov or not factor:
        print(f"跳过无法解析的文件名：{fname}")
        continue

    path = os.path.join(input_dir, fname)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"读取失败：{fname} -> {e}")
        continue

    # 年份处理
    if "年份" not in df.columns:
        print(f"跳过：{fname} 缺少 '年份' 列")
        continue
    df["年份"] = pd.to_numeric(df["年份"], errors="coerce")

    # 地区兜底
    if "地区" not in df.columns:
        df.insert(0, "地区", prov)
    else:
        df["地区"] = df["地区"].fillna(prov)

    # 只保留 2005-2035
    df = df[(df["年份"] >= YEAR_MIN) & (df["年份"] <= YEAR_MAX)].copy()

    # 生成统一指标列
    df[factor] = df.apply(
        lambda r: choose_hist_value(r, factor) if r["年份"] <= 2024 else choose_future_value(r, factor),
        axis=1
    )

    # 选择输出列（存在则保留）
    keep_cols = ["地区", "年份", "GDP", "人口", "第三产业GDP", "城镇人口", factor]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    factor_buckets[factor].append(out)

# 按指标写出CSV
for factor, frames in factor_buckets.items():
    if not frames:
        print(f"指标 {factor} 无有效数据，跳过")
        continue

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["地区", "年份"]).reset_index(drop=True)

    out_name = f"分省{factor}完整结果.csv"
    out_path = os.path.join(output_dir, out_name)
    merged.to_csv(out_path, index=False)
    print(f"已输出：{out_path} （{len(merged)} 行）")

print("所有指标数据整合完成！")

    