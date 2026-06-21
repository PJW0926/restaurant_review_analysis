import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =========================================================
# 1. 경로 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "all_크롤링_익명화.csv"

OUTPUT_OVERALL = BASE_DIR / "missing_summary_overall.csv"
OUTPUT_PLATFORM = BASE_DIR / "missing_summary_by_platform.csv"
OUTPUT_ROW = BASE_DIR / "missing_summary_by_row.csv"
OUTPUT_STRUCTURAL = BASE_DIR / "missing_structural_candidates.csv"

BAR_PLOT = BASE_DIR / "missing_rate_bar.png"
HEATMAP_PLOT = BASE_DIR / "missing_matrix_heatmap.png"


# =========================================================
# 2. 파일 불러오기
# =========================================================

def load_file(path):
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


df = load_file(INPUT_FILE)

print("파일 불러오기 완료")
print(f"데이터 크기: {df.shape[0]}행 × {df.shape[1]}열")


# =========================================================
# 3. 전체 결측치 요약
# =========================================================

total_rows = len(df)

overall_summary = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_rate": (df.isna().sum().values / total_rows).round(4),
    "non_missing_count": df.notna().sum().values,
    "dtype": [df[col].dtype for col in df.columns]
})

overall_summary = overall_summary.sort_values(
    by="missing_rate",
    ascending=False
)

overall_summary.to_csv(OUTPUT_OVERALL, index=False, encoding="utf-8-sig")

print("\n[전체 결측치 요약]")
print(overall_summary.head(20))


# =========================================================
# 4. 플랫폼별 결측치 요약
# =========================================================

platform_summary_rows = []

if "platform" in df.columns:
    for platform_name, group in df.groupby("platform"):
        group_rows = len(group)

        for col in df.columns:
            missing_count = group[col].isna().sum()
            missing_rate = missing_count / group_rows

            platform_summary_rows.append({
                "platform": platform_name,
                "column": col,
                "total_rows": group_rows,
                "missing_count": missing_count,
                "missing_rate": round(missing_rate, 4),
                "non_missing_count": group[col].notna().sum(),
                "dtype": group[col].dtype
            })

    platform_summary = pd.DataFrame(platform_summary_rows)
    platform_summary = platform_summary.sort_values(
        by=["column", "platform"]
    )

    platform_summary.to_csv(OUTPUT_PLATFORM, index=False, encoding="utf-8-sig")

    print("\n[플랫폼별 결측치 요약 저장 완료]")
    print(OUTPUT_PLATFORM)

else:
    print("\nplatform 컬럼이 없어서 플랫폼별 결측치 요약은 생략됨")


# =========================================================
# 5. 구조적 결측 후보 탐지
# =========================================================

structural_rows = []

if "platform" in df.columns:
    platforms = df["platform"].dropna().unique()

    for col in df.columns:
        if col == "platform":
            continue

        platform_rates = {}

        for platform_name in platforms:
            group = df[df["platform"] == platform_name]
            rate = group[col].isna().mean()
            platform_rates[platform_name] = rate

        max_rate = max(platform_rates.values())
        min_rate = min(platform_rates.values())

        # 한 플랫폼에서는 거의 전부 결측이고, 다른 플랫폼에서는 결측이 낮으면 구조적 결측 후보
        if max_rate >= 0.95 and min_rate <= 0.20:
            note = "플랫폼 구조 차이로 인한 구조적 결측 후보"

            # 대표적인 경우 자동 설명
            if col in ["rating", "account_avg_rating", "reviewer_level"]:
                note = "카카오에는 존재하지만 네이버에는 없는 플랫폼 고유 변수일 가능성"
            elif col in ["visit_count"]:
                note = "네이버에는 존재하지만 카카오에는 없는 플랫폼 고유 변수일 가능성"

            row = {
                "column": col,
                "mechanism_suggestion": "MNAR / structural missing",
                "note": note
            }

            for platform_name, rate in platform_rates.items():
                row[f"{platform_name}_missing_rate"] = round(rate, 4)

            structural_rows.append(row)

    structural_df = pd.DataFrame(structural_rows)
    structural_df.to_csv(OUTPUT_STRUCTURAL, index=False, encoding="utf-8-sig")

    print("\n[구조적 결측 후보]")
    if len(structural_df) > 0:
        print(structural_df)
    else:
        print("구조적 결측 후보가 뚜렷하게 탐지되지 않음")

else:
    print("\nplatform 컬럼이 없어서 구조적 결측 후보 탐지는 생략됨")


# =========================================================
# 6. 행별 결측치 개수 확인
# =========================================================

row_summary = df.copy()
row_summary["missing_count_by_row"] = df.isna().sum(axis=1)
row_summary["missing_rate_by_row"] = (df.isna().sum(axis=1) / df.shape[1]).round(4)

row_summary_small = row_summary[[
    col for col in [
        "platform",
        "store_name",
        "review_text",
        "missing_count_by_row",
        "missing_rate_by_row"
    ] if col in row_summary.columns
]]

row_summary_small = row_summary_small.sort_values(
    by="missing_count_by_row",
    ascending=False
)

row_summary_small.to_csv(OUTPUT_ROW, index=False, encoding="utf-8-sig")

print("\n[행별 결측치 상위 10개]")
print(row_summary_small.head(10))


# =========================================================
# 7. 결측률 막대그래프 저장
# =========================================================

plot_df = overall_summary[overall_summary["missing_rate"] > 0].copy()

if len(plot_df) > 0:
    plt.figure(figsize=(10, max(5, len(plot_df) * 0.35)))
    plt.barh(plot_df["column"], plot_df["missing_rate"])
    plt.xlabel("Missing Rate")
    plt.ylabel("Column")
    plt.title("Missing Rate by Column")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(BAR_PLOT, dpi=300)
    plt.close()

    print(f"\n결측률 막대그래프 저장 완료: {BAR_PLOT}")
else:
    print("\n결측치가 있는 컬럼이 없어 막대그래프 생략")


# =========================================================
# 8. 결측 매트릭스 heatmap 저장
# =========================================================

# 행이 너무 많으면 보기 좋게 최대 1000개만 샘플링
if len(df) > 1000:
    heatmap_df = df.sample(n=1000, random_state=42)
else:
    heatmap_df = df.copy()

missing_matrix = heatmap_df.isna().astype(int)

plt.figure(figsize=(12, 6))
plt.imshow(missing_matrix, aspect="auto", interpolation="nearest")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.title("Missing Value Matrix")
plt.xticks(
    ticks=np.arange(len(df.columns)),
    labels=df.columns,
    rotation=90,
    fontsize=7
)
plt.tight_layout()
plt.savefig(HEATMAP_PLOT, dpi=300)
plt.close()

print(f"결측 매트릭스 heatmap 저장 완료: {HEATMAP_PLOT}")


# =========================================================
# 9. 발표용 요약 출력
# =========================================================

print("\n==============================")
print("결측치 분석 완료")
print("==============================")
print(f"전체 결측치 요약: {OUTPUT_OVERALL}")
print(f"플랫폼별 결측치 요약: {OUTPUT_PLATFORM}")
print(f"행별 결측치 요약: {OUTPUT_ROW}")
print(f"구조적 결측 후보: {OUTPUT_STRUCTURAL}")
print(f"결측률 그래프: {BAR_PLOT}")
print(f"결측 매트릭스: {HEATMAP_PLOT}")
