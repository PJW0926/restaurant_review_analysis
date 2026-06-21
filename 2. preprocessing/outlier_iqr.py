import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# 1. 경로 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# 여기에 네 파일명 넣기
INPUT_FILE = BASE_DIR / "all_크롤링_익명화.csv"

OUTPUT_FILE = BASE_DIR / "outlier_iqr_result.csv"
SUMMARY_FILE = BASE_DIR / "outlier_iqr_summary.csv"


# =========================================================
# 2. 파일 불러오기 함수
# =========================================================

def load_file(path):
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


df = load_file(INPUT_FILE)


# =========================================================
# 3. 기본 전처리
# =========================================================

# has_photo가 문자형이면 0/1로 변환
if "has_photo" in df.columns:
    df["has_photo"] = df["has_photo"].replace({
        True: 1,
        False: 0,
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        "Y": 1,
        "N": 0,
        "yes": 1,
        "no": 0,
        "있음": 1,
        "없음": 0,
        "O": 1,
        "X": 0
    })

# 숫자형 변환 대상
numeric_candidates = [
    "review_length",
    "account_review_count",
    "has_photo",
    "rating",
    "account_avg_rating",
    "visit_count",
    "reviewer_level"
]

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 카카오 별점 편향 변수 생성
# 해당 리뷰 별점과 계정 평균 별점의 차이
if "rating" in df.columns and "account_avg_rating" in df.columns:
    df["rating_bias"] = (df["rating"] - df["account_avg_rating"]).abs()


# =========================================================
# 4. IQR 이상치 탐지 함수
# =========================================================

def add_iqr_outlier_flag(data, col, group_col=None):
    """
    col 하나에 대해 IQR 이상치 여부를 계산한다.
    platform 컬럼이 있으면 네이버/카카오를 나눠서 계산한다.
    """

    outlier_col = f"{col}_iqr_outlier"
    lower_col = f"{col}_iqr_lower"
    upper_col = f"{col}_iqr_upper"

    data[outlier_col] = 0
    data[lower_col] = np.nan
    data[upper_col] = np.nan

    summary_rows = []

    if group_col and group_col in data.columns:
        groups = data.groupby(group_col).groups.items()
    else:
        groups = [("ALL", data.index)]

    for group_name, idx in groups:
        values = data.loc[idx, col].dropna()

        # 값이 너무 적으면 IQR 계산 생략
        if len(values) < 4:
            summary_rows.append({
                "column": col,
                "group": group_name,
                "q1": np.nan,
                "q3": np.nan,
                "iqr": np.nan,
                "lower": np.nan,
                "upper": np.nan,
                "outlier_count": 0,
                "note": "값이 너무 적어 계산 생략"
            })
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (data.loc[idx, col] < lower) | (data.loc[idx, col] > upper)

        data.loc[idx, outlier_col] = mask.astype(int)
        data.loc[idx, lower_col] = lower
        data.loc[idx, upper_col] = upper

        summary_rows.append({
            "column": col,
            "group": group_name,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
            "outlier_count": int(mask.sum()),
            "total_count": int(len(idx)),
            "outlier_ratio": round(mask.sum() / len(idx), 4),
            "note": ""
        })

    return data, summary_rows


# =========================================================
# 5. IQR 적용
# =========================================================

target_cols = [
    "review_length",
    "account_review_count",
    "visit_count",
    "reviewer_level",
    "rating",
    "account_avg_rating",
    "rating_bias"
]

target_cols = [col for col in target_cols if col in df.columns]

group_col = "platform" if "platform" in df.columns else None

all_summary = []

for col in target_cols:
    df, summary = add_iqr_outlier_flag(df, col, group_col=group_col)
    all_summary.extend(summary)

summary_df = pd.DataFrame(all_summary)

# 전체 IQR 이상치 개수
iqr_flag_cols = [col for col in df.columns if col.endswith("_iqr_outlier")]
df["iqr_outlier_count"] = df[iqr_flag_cols].sum(axis=1)

# 하나라도 이상치면 1
df["has_iqr_outlier"] = (df["iqr_outlier_count"] > 0).astype(int)


# =========================================================
# 6. 저장
# =========================================================

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
summary_df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

print("IQR 이상치 탐지 완료")
print(f"결과 파일: {OUTPUT_FILE}")
print(f"요약 파일: {SUMMARY_FILE}")
print()
print("IQR 이상치 후보 리뷰 수:")
print(df["has_iqr_outlier"].sum())
print()
print("변수별 요약:")
print(summary_df)
