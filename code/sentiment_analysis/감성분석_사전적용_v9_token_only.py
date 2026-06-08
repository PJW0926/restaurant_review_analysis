import pandas as pd
from pathlib import Path
from collections import defaultdict

# =========================================================
# 1. 경로 설정
# =========================================================
# 이 파이썬 파일과 입력 CSV 파일들을 같은 폴더에 넣고 실행하세요.

BASE_DIR = Path(__file__).resolve().parent

review_file = BASE_DIR / "final_high_trust_reviews_pos.csv"
dict_file = BASE_DIR / "통합_감성사전_v9.csv"

output_file = BASE_DIR / "final_high_trust_reviews_category_sentiment_v9_token_only.csv"
platform_output = BASE_DIR / "platform_category_sentiment_summary_v9_token_only.csv"
store_output = BASE_DIR / "store_category_sentiment_summary_v9_token_only.csv"
review_score_output = BASE_DIR / "review_sentiment_scores_v9_token_only.csv"

# =========================================================
# 2. CSV 안전하게 불러오기
# =========================================================

def read_csv_safely(path):
    """utf-8-sig, utf-8, cp949, euc-kr 순서로 CSV를 읽습니다."""
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc), enc
        except UnicodeDecodeError as e:
            last_error = e

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"CSV 인코딩을 읽지 못했습니다: {path} / 마지막 오류: {last_error}",
    )


df, review_enc = read_csv_safely(review_file)
sent_dict, dict_enc = read_csv_safely(dict_file)

print(f"[입력 리뷰 파일 인코딩] {review_enc}")
print(f"[입력 사전 파일 인코딩] {dict_enc}")

# =========================================================
# 3. 감성사전 정리
# =========================================================

required_cols = ["word", "category", "polarity", "score"]
missing_cols = [col for col in required_cols if col not in sent_dict.columns]

if missing_cols:
    raise ValueError(f"감성사전에 필요한 열이 없습니다: {missing_cols}")

# 필요한 열만 사용
sent_dict = sent_dict[required_cols].copy()

# 결측 제거
sent_dict = sent_dict.dropna(subset=required_cols)

# 자료형 정리
sent_dict["word"] = sent_dict["word"].astype(str).str.strip()
sent_dict["category"] = sent_dict["category"].astype(str).str.strip()
sent_dict["polarity"] = sent_dict["polarity"].astype(str).str.strip()

# polarity 오타 보정
sent_dict["polarity"] = sent_dict["polarity"].replace({"negetive": "negative"})

# 빈 단어 제거
sent_dict = sent_dict[sent_dict["word"] != ""].copy()

# score 숫자 변환
sent_dict["score"] = pd.to_numeric(sent_dict["score"], errors="coerce")
sent_dict = sent_dict.dropna(subset=["score"])
sent_dict["score"] = sent_dict["score"].astype(int)

# 완전 중복 제거
sent_dict = sent_dict.drop_duplicates(subset=["word", "category", "polarity", "score"])

# =========================================================
# 4. 리뷰 데이터 기본 확인
# =========================================================

if "review_text" not in df.columns:
    raise ValueError("리뷰 파일에 'review_text' 열이 없습니다.")

if "tokens" not in df.columns:
    raise ValueError("리뷰 파일에 'tokens' 열이 없습니다. 먼저 Okt 토큰 생성 코드를 실행해야 합니다.")

# 결측 처리
df["review_text"] = df["review_text"].fillna("").astype(str)
df["tokens"] = df["tokens"].fillna("").astype(str)

# =========================================================
# 5. 카테고리 설정
# =========================================================

categories = ["food", "price", "service", "atmosphere", "general"]

dict_categories = sorted(sent_dict["category"].unique())
unknown_categories = [cat for cat in dict_categories if cat not in categories]

if unknown_categories:
    print("[주의] categories 목록에 없는 카테고리가 사전에 있습니다:")
    print(unknown_categories)
    print("이 카테고리들은 점수 계산에서 제외됩니다.")

# =========================================================
# 6. 사전 구조 만들기
# =========================================================
# 같은 단어가 여러 카테고리에 들어갈 수 있으므로 list 형태로 저장

sentiment_map = defaultdict(list)

for _, row in sent_dict.iterrows():
    word = row["word"]
    category = row["category"]
    score = row["score"]

    sentiment_map[word].append((category, score))

# 사전에 있는 최대 어절 수
max_ngram = max(len(str(word).split()) for word in sentiment_map.keys())

print(f"[사전 유효 행 수] {len(sent_dict):,}")
print(f"[사전 고유 표현 수] {len(sentiment_map):,}")
print(f"[최대 n-gram 어절 수] {max_ngram}")

# =========================================================
# 7. 유틸 함수
# =========================================================

def clean_token(token):
    """
    혹시 tokens에 품사 태그가 남아 있을 경우 제거.
    예: 맛있다/Adjective -> 맛있다
    이미 품사 태그가 없다면 그대로 반환.
    """
    token = str(token).strip()
    if "/" in token:
        return token.rsplit("/", 1)[0]
    return token

# =========================================================
# 8. 리뷰별 감성점수 계산
# =========================================================
# 방식:
# 1) review_text 원문 매칭은 하지 않음
#    - 원문 substring 매칭이 과하게 잡히는 문제 방지
# 2) tokens에서만 n-gram 표현을 찾음
#    - 맛있다, 비싸다, 고성 방가, 별로 사람 모르다 같은 토큰 표현 탐지용
# 3) 긴 표현을 먼저 매칭함
#    - 3어절 표현이 있으면 내부 1어절/2어절은 중복 계산하지 않음


def analyze_review(tokens_text):
    raw_tokens = str(tokens_text).split()
    tokens = [clean_token(t) for t in raw_tokens if clean_token(t) != ""]

    scores = {cat: 0 for cat in categories}
    matched = {cat: [] for cat in categories}

    # 이미 점수화한 표현 기록
    matched_terms = set()

    # 토큰에서 긴 표현과 짧은 표현이 중복 매칭되는 것 방지
    used_token_idx = set()

    # =====================================================
    # tokens 기반 n-gram 매칭만 수행
    # =====================================================
    # 긴 표현부터 먼저 매칭
    for n in range(max_ngram, 0, -1):
        for i in range(len(tokens) - n + 1):
            idx_range = set(range(i, i + n))

            # 이미 긴 표현으로 잡힌 토큰이면 스킵
            if used_token_idx & idx_range:
                continue

            term = " ".join(tokens[i:i + n])

            if term not in sentiment_map:
                continue

            # 같은 표현 반복 점수화를 막음
            # 예: 맛있다 맛있다 맛있다 -> 맛있다 1회만 반영
            if term in matched_terms:
                continue

            for category, score in sentiment_map[term]:
                if category in scores:
                    scores[category] += score
                    matched[category].append(f"{term}:{score}[token]")

            matched_terms.add(term)
            used_token_idx.update(idx_range)

    # =====================================================
    # 결과 정리
    # =====================================================
    result = {}

    for cat in categories:
        result[f"{cat}_score"] = scores[cat]
        result[f"{cat}_matched_words"] = " ".join(matched[cat])

        if scores[cat] > 0:
            result[f"{cat}_label"] = "positive"
        elif scores[cat] < 0:
            result[f"{cat}_label"] = "negative"
        else:
            result[f"{cat}_label"] = "neutral"

    result["category_total_score"] = sum(scores.values())

    if result["category_total_score"] > 0:
        result["category_total_label"] = "positive"
    elif result["category_total_score"] < 0:
        result["category_total_label"] = "negative"
    else:
        result["category_total_label"] = "neutral"

    return pd.Series(result)


result_df = df["tokens"].apply(analyze_review)

df = pd.concat([df, result_df], axis=1)

# =========================================================
# 9. 플랫폼별 요약
# =========================================================

if "platform" in df.columns:
    agg_dict = {
        "review_count": ("review_text", "count"),
        "avg_total_score": ("category_total_score", "mean"),
        "positive_ratio": ("category_total_label", lambda x: (x == "positive").mean()),
        "neutral_ratio": ("category_total_label", lambda x: (x == "neutral").mean()),
        "negative_ratio": ("category_total_label", lambda x: (x == "negative").mean()),
    }

    for cat in categories:
        agg_dict[f"avg_{cat}_score"] = (f"{cat}_score", "mean")

    platform_summary = df.groupby("platform").agg(**agg_dict).reset_index()

    platform_cols = (
        ["platform", "review_count"]
        + [f"avg_{cat}_score" for cat in categories]
        + ["avg_total_score", "positive_ratio", "neutral_ratio", "negative_ratio"]
    )

    platform_summary = platform_summary[platform_cols]
    platform_summary.to_csv(platform_output, index=False, encoding="utf-8-sig")

# =========================================================
# 10. 식당별 요약
# =========================================================

if "store_name" in df.columns and "platform" in df.columns:
    agg_dict = {
        "review_count": ("review_text", "count"),
        "avg_total_score": ("category_total_score", "mean"),
        "positive_ratio": ("category_total_label", lambda x: (x == "positive").mean()),
        "neutral_ratio": ("category_total_label", lambda x: (x == "neutral").mean()),
        "negative_ratio": ("category_total_label", lambda x: (x == "negative").mean()),
    }

    for cat in categories:
        agg_dict[f"avg_{cat}_score"] = (f"{cat}_score", "mean")

    store_summary = df.groupby(["store_name", "platform"]).agg(**agg_dict).reset_index()

    store_cols = (
        ["store_name", "platform", "review_count"]
        + [f"avg_{cat}_score" for cat in categories]
        + ["avg_total_score", "positive_ratio", "neutral_ratio", "negative_ratio"]
    )

    store_summary = store_summary[store_cols]
    store_summary.to_csv(store_output, index=False, encoding="utf-8-sig")

# =========================================================
# 11. 리뷰별 감성점수 확인용 파일 저장
# =========================================================

review_score_cols = [
    "platform",
    "store_name",
    "review_text",
    "tokens",
]

# 카테고리별 score
review_score_cols += [f"{cat}_score" for cat in categories]

# 전체 score
review_score_cols += [
    "category_total_score",
    "category_total_label",
]

# 카테고리별 label
review_score_cols += [f"{cat}_label" for cat in categories]

# 카테고리별 matched words
review_score_cols += [f"{cat}_matched_words" for cat in categories]

# 실제 존재하는 열만 저장
review_score_cols = [col for col in review_score_cols if col in df.columns]

df[review_score_cols].to_csv(
    review_score_output,
    index=False,
    encoding="utf-8-sig",
)

# 전체 결과 저장
df.to_csv(output_file, index=False, encoding="utf-8-sig")

# =========================================================
# 12. 실행 결과 출력
# =========================================================

print("\n완료:", output_file)
print("리뷰별 감성점수 확인 파일:", review_score_output)

if "platform" in df.columns:
    print("플랫폼 요약:", platform_output)

if "store_name" in df.columns and "platform" in df.columns:
    print("식당 요약:", store_output)

print("\n[사용 카테고리]")
print(categories)

print("\n[사전에 들어 있는 카테고리]")
print(dict_categories)

print("\n[감성분석 결과 미리보기]")

preview_cols = [
    "review_text",
    "tokens",
]

preview_cols += [f"{cat}_score" for cat in categories]
preview_cols += ["category_total_score", "category_total_label"]
preview_cols += [f"{cat}_matched_words" for cat in categories]

preview_cols = [col for col in preview_cols if col in df.columns]

print(df[preview_cols].head(20).to_string())
