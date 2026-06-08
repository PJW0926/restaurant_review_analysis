import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# 1. 파일 불러오기
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

input_file = BASE_DIR / "final_high_trust_reviews_category_sentiment_v9_token_only.csv"
output_file = BASE_DIR / "final_high_trust_reviews_with_sentiment_star_v9_token_only_star_정규화수정.csv"

df = pd.read_csv(input_file, encoding="utf-8-sig")

# 숫자형 변환
df["category_total_score"] = pd.to_numeric(df["category_total_score"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# =========================================================
# 2. 카카오 감성점수 기준값 설정
# =========================================================
# 카카오는 실제 별점이 있으므로, 카카오 감성점수 분포를 기준으로 사용

kakao_scores = df.loc[
    df["platform"] == "kakao",
    "category_total_score"
].dropna()

# 극단값 영향을 줄이기 위해 1%~85% 분위수 사용
score_min = kakao_scores.quantile(0.01)
score_max = kakao_scores.quantile(0.85)
 
print("카카오 기준 감성점수 1%:", score_min)
print("카카오 기준 감성점수 85%:", score_max)

# =========================================================
# 3. 감성점수 → 1~5점 별점 변환
# =========================================================

def sentiment_to_star(score):
    if pd.isna(score):
        return np.nan

    # min-max 정규화
    normalized = (score - score_min) / (score_max - score_min)

    # 0~1 범위로 제한
    normalized = max(0, min(1, normalized))

    # 1~5점 변환
    star = 1 + normalized * 4

    return round(star, 2)

df["sentiment_star"] = df["category_total_score"].apply(sentiment_to_star)

# =========================================================
# 4. 식당별 감성 별점 요약
# =========================================================

store_star_summary = df.groupby(["store_name", "platform"]).agg(
    review_count=("review_text", "count"),
    avg_sentiment_score=("category_total_score", "mean"),
    avg_sentiment_star=("sentiment_star", "mean"),
    avg_original_rating=("rating", "mean")
).reset_index()

store_star_summary["avg_sentiment_star"] = store_star_summary["avg_sentiment_star"].round(2)
store_star_summary["avg_sentiment_score"] = store_star_summary["avg_sentiment_score"].round(2)
store_star_summary["avg_original_rating"] = store_star_summary["avg_original_rating"].round(2)

# =========================================================
# 5. 저장
# =========================================================

df.to_csv(output_file, index=False, encoding="utf-8-sig")

store_output = BASE_DIR / "store_sentiment_star_summary.csv"
store_star_summary.to_csv(store_output, index=False, encoding="utf-8-sig")

print("완료:", output_file)
print("식당별 감성 별점 요약:", store_output)

print(store_star_summary.head())