import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

input_file = BASE_DIR / "final_high_trust_reviews_category_sentiment_v9_token_only.csv"

df = pd.read_csv(input_file, encoding="utf-8-sig")

df["category_total_score"] = pd.to_numeric(df["category_total_score"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

kakao_df = df[df["platform"] == "kakao"].copy()
kakao_df = kakao_df.dropna(subset=["category_total_score", "rating"])

kakao_scores = kakao_df["category_total_score"].dropna()

def sentiment_to_star(score, score_min, score_max):
    if pd.isna(score):
        return np.nan

    normalized = (score - score_min) / (score_max - score_min)
    normalized = max(0, min(1, normalized))

    return round(1 + normalized * 4, 2)

results = []

# 비교할 분위수 후보
min_quantiles = [0.01, 0.03, 0.05, 0.07, 0.10]
max_quantiles = [0.85, 0.88, 0.90, 0.92, 0.95, 0.97, 0.99]

for q_min in min_quantiles:
    for q_max in max_quantiles:
        if q_min >= q_max:
            continue

        score_min = kakao_scores.quantile(q_min)
        score_max = kakao_scores.quantile(q_max)

        temp = kakao_df.copy()

        temp["test_sentiment_star"] = temp["category_total_score"].apply(
            lambda x: sentiment_to_star(x, score_min, score_max)
        )

        temp["diff"] = temp["rating"] - temp["test_sentiment_star"]
        temp["diff_abs"] = temp["diff"].abs()

        mae = temp["diff_abs"].mean()
        rmse = np.sqrt((temp["diff"] ** 2).mean())
        median_abs_error = temp["diff_abs"].median()
        within_1 = (temp["diff_abs"] <= 1).mean()
        within_2 = (temp["diff_abs"] <= 2).mean()

        results.append({
            "q_min": q_min,
            "q_max": q_max,
            "score_min": round(score_min, 3),
            "score_max": round(score_max, 3),
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "median_abs_error": round(median_abs_error, 3),
            "within_1_ratio": round(within_1, 3),
            "within_2_ratio": round(within_2, 3)
        })

result_df = pd.DataFrame(results).sort_values("MAE")

print(result_df.to_string(index=False))

output_file = BASE_DIR / "normalization_quantile_comparison.csv"
result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("저장 완료:", output_file)