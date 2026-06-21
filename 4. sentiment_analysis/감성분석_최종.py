import pandas as pd
import numpy as np
import re
import sys
import unicodedata
import json
from pathlib import Path
from collections import defaultdict

# =========================================================
# 1. 경로 설정
# =========================================================
# 입력 CSV와 감성사전이 들어 있는 폴더를 직접 지정합니다.
# 폴더를 이동한 경우 아래 경로만 수정하면 됩니다.

BASE_DIR = Path(__file__).resolve().parent

review_file = BASE_DIR / "final_high_trust_reviews_pos.csv"
dict_candidates = [
    BASE_DIR / "통합_감성사전_v10.csv",
    BASE_DIR / "통합_감성사전_v10_이상토큰정제.csv",
    BASE_DIR / "통합_감성사전_v9.csv",
]
dict_file = next((path for path in dict_candidates if path.exists()), dict_candidates[-1])

missing_input_files = [
    path
    for path in [review_file, dict_file]
    if not path.exists()
]

if missing_input_files:
    missing_text = "\n".join(f"- {path}" for path in missing_input_files)
    raise FileNotFoundError(
        "다음 입력 파일을 찾을 수 없습니다. BASE_DIR 경로와 파일명을 확인하세요.\n"
        f"{missing_text}"
    )

# positive evidence 안전 회복 출력 파일
output_file = BASE_DIR / "all_final_high_trust_reviews_with_sentiment_star_positive_recovery.csv"
platform_output = BASE_DIR / "all_platform_aspect_sentiment_summary_positive_recovery.csv"
store_output = BASE_DIR / "all_store_aspect_sentiment_summary_positive_recovery.csv"
review_score_output = BASE_DIR / "all_review_sentiment_scores_positive_recovery.csv"
compact_output = BASE_DIR / "all_final_high_trust_reviews_with_sentiment_star_positive_recovery_compact.csv"
review_view_output = BASE_DIR / "all_final_high_trust_reviews_sentiment_star_positive_recovery_review_view.csv"
error_cases_output = BASE_DIR / "all_positive_recovery_remaining_diff_abs_over_2_cases.csv"
low_score_high_rating_output = BASE_DIR / "all_positive_recovery_remaining_high_rating_low_sentiment_cases.csv"
high_score_low_rating_output = BASE_DIR / "all_positive_recovery_remaining_low_rating_high_sentiment_cases.csv"
overestimation_cases_output = BASE_DIR / "all_positive_recovery_remaining_low_rating_high_sentiment_cases.csv"
overestimation_diagnosis_output = BASE_DIR / "all_positive_recovery_remaining_error_diagnosis.csv"
possible_cap_misfire_output = BASE_DIR / "all_positive_recovery_possible_cap_misfire_high_rating_cases.csv"
possible_floor_misfire_output = BASE_DIR / "all_positive_recovery_possible_floor_misfire_low_rating_cases.csv"
unmatched_candidate_output = BASE_DIR / "all_positive_recovery_unmatched_positive_candidate_terms.csv"
error_summary_output = BASE_DIR / "all_positive_recovery_sentiment_error_summary.csv"
test_results_output = BASE_DIR / "all_positive_regression_test_results_positive_recovery.csv"
overcap_test_results_output = BASE_DIR / "all_overestimation_prevention_test_results_positive_recovery.csv"
final_tuning_test_results_output = BASE_DIR / "all_final_tuning_additional_test_results_positive_recovery.csv"
minimal_tuning_test_results_output = BASE_DIR / "all_minimal_error_tuning_test_results_positive_recovery.csv"
positive_recovery_test_results_output = BASE_DIR / "all_positive_evidence_recovery_test_results.csv"
refactor_report_output = BASE_DIR / "all_sentiment_analysis_positive_evidence_recovery_report.md"

# 현재 요청은 규칙 수정 전 구조 진단이 목적이므로 기본값을 True로 둡니다.
# 전체 결과 CSV까지 다시 저장하려면 False로 변경하세요.
DEBUG_ONLY = False

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
original_input_columns = list(df.columns)

print(f"[입력 리뷰 파일 인코딩] {review_enc}")
print(f"[입력 사전 파일 인코딩] {dict_enc}")

# =========================================================
# 3. 감성사전 정리
# =========================================================

required_cols = ["word", "category", "polarity", "score"]
missing_cols = [col for col in required_cols if col not in sent_dict.columns]

if missing_cols:
    raise ValueError(f"감성사전에 필요한 열이 없습니다: {missing_cols}")

sent_dict = sent_dict[required_cols].copy()
sent_dict = sent_dict.dropna(subset=required_cols)

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
# 3-1. 같은 단어가 같은 카테고리에 여러 점수로 들어간 경우 정리
# =========================================================
# 예:
# 훨씬 맛있다 / food / positive / 1
# 훨씬 맛있다 / food / positive / 2
# 이런 경우 둘 다 점수화되지 않도록 절댓값이 큰 점수 하나만 남김.

sent_dict["abs_score"] = sent_dict["score"].abs()

sent_dict = (
    sent_dict
    .sort_values(
        by=["word", "category", "polarity", "abs_score"],
        ascending=[True, True, True, False]
    )
    .drop_duplicates(subset=["word", "category", "polarity"], keep="first")
    .drop(columns=["abs_score"])
    .reset_index(drop=True)
)

# =========================================================
# 4. 리뷰 데이터 기본 확인
# =========================================================

if "review_text" not in df.columns:
    raise ValueError("리뷰 파일에 'review_text' 열이 없습니다.")

if "tokens" not in df.columns:
    raise ValueError("리뷰 파일에 'tokens' 열이 없습니다. 먼저 Okt 토큰 생성 코드를 실행해야 합니다.")

df["review_text"] = df["review_text"].fillna("").astype(str)
df["tokens"] = df["tokens"].fillna("").astype(str)

if "tokens_pos" in df.columns:
    df["tokens_pos"] = df["tokens_pos"].fillna("").astype(str)
else:
    # tokens_pos가 없으면 '별로/Josa' 같은 품사 구분은 생략하되 분석은 계속합니다.
    print("[주의] tokens_pos 열이 없습니다. 품사 기반 부정어 구분은 제한됩니다.")
    df["tokens_pos"] = ""

if "rating" in df.columns:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
else:
    print("[주의] rating 열이 없습니다. diff, diff_abs, diff_squared 계산은 생략됩니다.")

# =========================================================
# 5. 카테고리 설정
# =========================================================

categories = ["food", "price", "service", "atmosphere", "general"]

# 이전 결과 열이 이미 있으면 제거
generated_cols = []
for cat in categories:
    generated_cols += [
        f"{cat}_score",
        f"{cat}_matched_words",
        f"{cat}_label",
    ]

generated_cols += [
    "tokens_fixed",
    "category_total_score",
    "category_total_label",
    "strong_positive_count",
    "weak_positive_count",
    "strong_negative_count",
    "weak_negative_count",
    "revisit_positive_count",
    "recommendation_count",
    "waiting_positive_count",
    "waiting_negative_count",
    "positive_phrase_count",
    "phrase_score_total",
    "dictionary_score_total",
    "context_guard_score_total",
    "evidence_items",
    "evidence_floor_items",
    "overall_sentiment_score",
    "overall_context_reason",
    "sentiment_star_raw",
    "sentiment_star",
    "sentiment_calibration_reason",
    "diff",
    "diff_abs",
    "diff_squared",
]

generated_cols = [col for col in generated_cols if col in df.columns]
if generated_cols:
    df = df.drop(columns=generated_cols)

dict_categories = sorted(sent_dict["category"].unique())
unknown_categories = [cat for cat in dict_categories if cat not in categories]

if unknown_categories:
    print("[주의] categories 목록에 없는 카테고리가 사전에 있습니다:")
    print(unknown_categories)
    print("이 카테고리들은 점수 계산에서 제외됩니다.")

# =========================================================
# 6. 사전 구조 만들기
# =========================================================

sentiment_map = defaultdict(list)

for _, row in sent_dict.iterrows():
    word = row["word"]
    category = row["category"]
    score = row["score"]

    sentiment_map[word].append((category, score))

# 같은 표현이 general과 구체 카테고리에 동시에 있으면 구체 카테고리만 사용.
# 예: 불편하다 / atmosphere:-1, general:-1 -> atmosphere:-1
general_duplicate_terms = 0
normalized_category_family_terms = 0

for word, entries in list(sentiment_map.items()):
    # '불편하다' 표현 가족은 사전에서 general/atmosphere가 섞여 있으므로
    # atmosphere로 통일해 같은 의미가 카테고리별로 중복 점수화되는 것을 막음.
    if word == "불편하다" or word.endswith(" 불편하다"):
        normalized_entries = []

        for _, score in entries:
            entry = ("atmosphere", score)
            if entry not in normalized_entries:
                normalized_entries.append(entry)

        sentiment_map[word] = normalized_entries
        entries = normalized_entries
        normalized_category_family_terms += 1

    has_specific_category = any(category != "general" for category, _ in entries)

    if has_specific_category and any(category == "general" for category, _ in entries):
        sentiment_map[word] = [
            (category, score)
            for category, score in entries
            if category != "general"
        ]
        general_duplicate_terms += 1

if not sentiment_map:
    raise ValueError("감성사전에 사용할 수 있는 표현이 없습니다.")

max_ngram = max(len(str(word).split()) for word in sentiment_map.keys())

print(f"[사전 유효 행 수] {len(sent_dict):,}")
print(f"[사전 고유 표현 수] {len(sentiment_map):,}")
print(f"[최대 n-gram 어절 수] {max_ngram}")
print(f"[general 중복 제거 표현 수] {general_duplicate_terms:,}")
print(f"[카테고리 통일 표현 수] {normalized_category_family_terms:,}")

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


def print_console_safely(value):
    """Windows 콘솔 인코딩에서 표시할 수 없는 문자는 대체해 안전하게 출력."""
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_text)


def parse_pos_tokens(tokens_pos_text):
    """
    '맛있다/Adjective 별로/Josa' 형식의 tokens_pos를
    [(표현, 품사), ...] 형태로 변환.
    """
    parsed = []

    for raw in str(tokens_pos_text).split():
        raw = str(raw).strip()
        if not raw:
            continue

        if "/" in raw:
            word, pos = raw.rsplit("/", 1)
        else:
            word, pos = raw, ""

        parsed.append((word, pos))

    return parsed


def normalize_review_text(review_text):
    """
    원문 phrase 탐지용 최소 정규화.

    의미를 추정하거나 토큰화하지 않고 Unicode/공백/소수점 표기만 통일합니다.
    예: '총점 4 5'와 '총점 4.5'를 모두 명시적 점수 phrase로 탐지 가능하게 함.
    """
    text = unicodedata.normalize("NFKC", str(review_text or ""))
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = re.sub(r"(?<=\d)\s*[.,]\s*(?=\d)", ".", text)
    text = re.sub(r"(?<=\d)\s+(?=\d(?:\s|$))", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_analysis_tokens(tokens, review_text):
    """
    감성분석 직전에 명백한 형태소 분석 오류만 최소한으로 보정.

    예:
    - 질기지 않고 -> 기지 않다
      위 결과를 질기다 않다로 복원해 '질기다'의 직접 부정을 인식.
    - 달달 허다 -> 달달하다
    - 누리다 나다 -> 누린내
    - 매워서/매웠다 -> 맵다

    보정 후에도 tokens 기반 n-gram 매칭, 긴 표현 우선, used_token_idx 구조는
    그대로 유지됩니다.
    """
    normalized = []
    source = list(tokens)
    compact_text = str(review_text).replace(" ", "")

    i = 0
    while i < len(source):
        current = source[i]
        next_token = source[i + 1] if i + 1 < len(source) else None
        third_token = source[i + 2] if i + 2 < len(source) else None

        if current == "달달" and next_token == "허다":
            normalized.append("달달하다")
            i += 2
            continue

        if current == "누리다" and next_token == "나다":
            normalized.append("누린내")
            i += 2
            continue

        if (
            current == "줄"
            and next_token in {"설", "서다"}
            and third_token in {"만하다", "만한", "만함"}
        ):
            normalized.append("줄설만하다")
            i += 3
            continue

        if (
            current == "재방문"
            and next_token
            and str(next_token).lower() in {"ok", "오케이"}
        ):
            normalized.append("재방문OK")
            i += 2
            continue

        if (
            current in {"먹어", "먹다"}
            and next_token in {"볼", "보다"}
            and third_token in {"만하다", "만한", "만함"}
        ):
            normalized.append("먹어볼만하다")
            i += 3
            continue

        # '누린내가/누린내도'가 단독으로 누리다로 분석되는 실제 사례 보정.
        # 주변 음식/냄새 문맥이 있을 때만 바꾸어, 동사 '누리다'는 보존합니다.
        if current == "누리다":
            local = source[max(0, i - 3):min(len(source), i + 4)]
            smell_context = {
                "고기", "돼지", "돼지고기", "순대", "내장", "냄새", "특유",
                "식다", "포장", "곰탕", "국밥", "해장국", "잡내",
                "나다", "없이", "없다", "심하다", "예민하다",
            }
            if any(token in smell_context for token in local):
                normalized.append("누린내")
                i += 1
                continue

        spicy_variants = {
            "매워서", "매워요", "매웠다", "매웠음", "매운", "매움",
        }
        if current in spicy_variants:
            normalized.append("맵다")
            i += 1
            continue

        normalized.append(current)
        i += 1

    for i in range(len(normalized) - 1):
        if (
            normalized[i] == "기지"
            and normalized[i + 1] == "않다"
            and "질기지" in compact_text
        ):
            normalized[i] = "질기다"

    return normalized


def fix_tokens_for_sentiment(tokens_text, review_text):
    """
    기존 tokens는 보존하고 감성분석에 필요한 명백한 오류만 복원합니다.

    특히 형태소 분석기가 원문 '맛나다/맛나게'를 '나다/달다'로 바꾸는 경우,
    주변에 음식 대상이 있을 때만 사전 표제어 '맛나다'로 복원합니다.
    """
    raw_tokens = str(tokens_text).split()
    tokens = [clean_token(token) for token in raw_tokens if clean_token(token)]
    fixed = normalize_analysis_tokens(tokens, review_text)
    compact_text = re.sub(r"\s+", "", normalize_review_text(review_text))
    food_context_tokens = {
        "김치", "고기", "음식", "맛", "국물", "찌개", "면", "칼국수",
        "족발", "보쌈", "곰탕", "냉면", "반찬", "요리", "메뉴",
    }

    if re.search(r"맛나(?:다|는|게|고|서|네요|요|더라|며|ㄴ)", compact_text):
        for i, token in enumerate(fixed):
            if token not in {"나다", "달다"}:
                continue

            local = fixed[max(0, i - 5):min(len(fixed), i + 3)]
            if any(item in food_context_tokens for item in local):
                fixed[i] = "맛나다"
                break

    return " ".join(fixed)


def make_tokens_fixed(tokens_text, review_text):
    """이전 함수명과의 호환용 별칭."""
    return fix_tokens_for_sentiment(tokens_text, review_text)


# 단독으로는 너무 일반적이거나 토큰화 오류 가능성이 큰 표현.
# n-gram 길이가 1일 때만 차단하므로 '냄새 나다', '안 주다' 같은
# 의미 있는 복합 표현은 정상적으로 긴 표현 우선 매칭됩니다.
RISKY_SINGLE_TOKENS = {
    "불다", "나다", "들다", "주다", "하다", "있다",
    "없다", "먹다", "되다", "보다", "가다", "오다", "사라지다",
}

INTENSITY_MODIFIERS = {
    "너무", "매우", "정말", "진짜", "엄청", "완전",
    "되게", "무척", "아주", "굉장히", "지나치게",
}


def is_suppressed_by_stronger_phrase(term, start_idx, stronger_phrase_spans, window=8):
    """
    이미 '너무 맵다' 같은 강한 표현을 점수화했다면 가까운 범위의 '맵다'
    단독은 추가 점수화하지 않음. 리뷰 전체를 억제하지 않아 다른 문장의
    독립적인 평가는 보존합니다.
    """
    for base_term, span_start, span_end in stronger_phrase_spans:
        distance = min(abs(start_idx - span_start), abs(start_idx - (span_end - 1)))

        if base_term == term and distance <= window:
            return True

    return False


DIRECT_LEFT_NEGATORS = {"안", "못"}
DIRECT_RIGHT_NEGATORS = {"않다", "아니다", "못하다"}
DIRECT_ABSENCE_TOKENS = DIRECT_RIGHT_NEGATORS | {"없다", "없이"}
NEGATION_MODIFIERS = {"전혀", "하나", "하나도", "거의"}
NEGATION_TOKENS = DIRECT_LEFT_NEGATORS | DIRECT_ABSENCE_TOKENS
STANDALONE_NEGATIVE_TERMS = {"별로", "별루", "그닥", "그다지"}


def term_contains_negation(term):
    """사전 표현 자체에 부정 토큰이 포함되어 있는지 확인."""
    return any(part in NEGATION_TOKENS for part in str(term).split())


def has_left_boundary_occurrence(text, phrase):
    """
    phrase 앞이 다른 한글/영문/숫자에 붙지 않은 경우만 True.

    예:
    - '별로 안 좋다'의 '별로' -> True
    - '메뉴별로 맛있다'의 '별로' -> False
    """
    start_idx = 0

    while True:
        idx = text.find(phrase, start_idx)
        if idx == -1:
            return False

        if idx == 0 or not text[idx - 1].isalnum():
            return True

        start_idx = idx + len(phrase)


def is_standalone_negative_term(review_text, term):
    """'메뉴별로'처럼 다른 단어에 붙은 표현은 독립 부정어로 점수화하지 않음."""
    if term not in STANDALONE_NEGATIVE_TERMS:
        return True

    return has_left_boundary_occurrence(str(review_text), term)


def is_contextually_valid_ambiguous_term(review_text, term):
    """
    대상에 따라 극성이 달라지는 모호한 표현은 평가 대상과 함께 있을 때만 인정.

    예:
    - 웨이팅이 길다 -> 부정
    - 고기가 길게 들어있다 -> 감성 점수 제외
    """
    if term != "길다":
        return True

    compact_text = str(review_text).replace(" ", "")
    waiting_length_patterns = [
        "웨이팅이길", "웨이팅은길", "웨이팅너무길",
        "대기가길", "대기는길", "대기너무길",
        "줄이길", "줄은길", "줄너무길",
        "시간이길", "시간은길", "시간너무길",
        "오래기다리", "한참기다리",
    ]

    return any(pattern in compact_text for pattern in waiting_length_patterns)


def has_concessive_preference_context(review_text, term, window=45):
    """
    단독 부정어가 현재 음식 평가가 아니라 원래 취향을 설명하는 양보절인지 확인.

    예:
    - 보쌈 별로 안 좋아하는데도 안 느끼하고 맛있음
    - 고기 별로 좋아하지 않지만 잡내 없고 맛있음

    이때 '별로'는 현재 식당의 품질 평가가 아니므로 0점 처리.
    """
    if term not in STANDALONE_NEGATIVE_TERMS:
        return False

    text = str(review_text)
    preference_patterns = ["좋아하", "선호하", "즐기", "찾아먹", "사먹"]
    concessive_patterns = [
        "는데도", "은데도", "인데도", "지만", "더라도",
        "임에도", "인데", "는데", "않지만", "않는데도",
    ]

    start_idx = 0

    while True:
        idx = text.find(term, start_idx)
        if idx == -1:
            return False

        if idx == 0 or not text[idx - 1].isalnum():
            context = text[idx:min(len(text), idx + window)].replace(" ", "")

            has_preference = any(pattern in context for pattern in preference_patterns)
            has_concessive = any(pattern in context for pattern in concessive_patterns)

            if has_preference and has_concessive:
                return True

        start_idx = idx + len(term)


def has_direct_negative_context(review_text, term):
    """
    원문에서 긍정 감성어에 실제로 붙어 있는 부정만 탐지.

    인정:
    - 안 맛있다 / 못 먹다 / 별로 맛있다
    - 맛있지 않다 / 좋지는 않다 / 친절하진 않다
    - 맛집이 아니다

    불인정:
    - 감칠맛 ... 질기지 않고
    - 쫄깃 ... 질기지 않고
    - 메뉴별로 맛있다
    - 딱입니다. 특은 아닌데처럼 문장 경계를 넘은 토큰 인접
    """
    if term_contains_negation(term):
        return False

    text = str(review_text)
    compact_text = text.replace(" ", "")

    term_parts = str(term).split()
    last_part = term_parts[-1]
    targets = {str(term).replace(" ", "")}

    if last_part == "이다" and len(term_parts) > 1:
        # '별미 이다'에서 의미 대상은 일반 동사 '이다'가 아니라 '별미'입니다.
        targets.add(term_parts[-2].replace(" ", ""))
    else:
        targets.add(last_part.replace(" ", ""))

        if last_part.endswith("하다"):
            targets.add(last_part[:-1])  # 친절하
            targets.add(last_part[:-2])  # 친절
        elif last_part.endswith("다"):
            targets.add(last_part[:-1])

    targets = {target for target in targets if target}

    # 부정어 앞이 다른 한글/영문/숫자이면 독립 부정어가 아닙니다.
    # 예: '메뉴별로 맛있다'의 '별로'는 제외.
    spaced_left_negators = ["안", "못", "별로", "그닥", "그다지", "별루"]

    for target in targets:
        for negator in spaced_left_negators:
            if has_left_boundary_occurrence(text, f"{negator} {target}"):
                return True

        for negator in DIRECT_LEFT_NEGATORS:
            if has_left_boundary_occurrence(text, f"{negator}{target}"):
                return True

    direct_negative_suffixes = [
        "지않", "진않", "지는않", "지도않", "지못", "지는못",
        "아니다", "아니고", "아닌", "아니라", "아님",
        "이아니다", "이아니고", "이아닌", "이아니라",
        "은아니다", "은아니고", "은아닌", "은아니라",
        "는아니다", "는아니고", "는아닌", "는아니라",
        "은건아니", "는건아니", "것은아니", "건아니",
        "도없", "가없", "이없", "은없", "는없",
    ]

    for target in targets:
        if any(target + suffix in compact_text for suffix in direct_negative_suffixes):
            return True

    return False


def has_direct_comparative_reference(review_text, term):
    """
    감성어가 실제 속성 단정이 아니라 비교 기준으로만 쓰였는지 확인.

    예:
    - 바삭보다는 조금 딱딱하다
    - 부드럽다기보다는 쫄깃하다

    비교 기준 감성어는 안전하게 0점 처리.
    """
    text = str(review_text).replace(" ", "")
    term_parts = str(term).split()
    last_part = term_parts[-1]
    targets = {str(term).replace(" ", ""), last_part.replace(" ", "")}

    if last_part.endswith("하다"):
        targets.add(last_part[:-1])
        targets.add(last_part[:-2])
    elif last_part.endswith("다"):
        targets.add(last_part[:-1])

    suffixes = [
        "보다는", "보단",
        "기보다는", "기보단",
        "라기보다는", "라기보단",
        "이라기보다는", "이라기보단",
        "하다기보다는", "하다기보단",
    ]

    return any(
        target and target + suffix in text
        for target in targets
        for suffix in suffixes
    )


POSITIVE_RESOLUTION_PATTERNS = [
    "가치가있", "가치있",
    "맛있", "좋", "추천", "만족", "훌륭", "최고",
    "먹어볼만", "먹을만", "재방문", "또오", "다시오",
]

CONCESSIVE_BOUNDARIES = [
    "지만", "하지만", "그러나", "그래도", "근데",
    "는데도", "은데도", "임에도", "인데도",
]


def has_resolved_positive_concession(review_text, term):
    """
    general 부정어가 양보절 앞에 있고, 뒤에서 긍정 결론으로 해소되는지 확인.

    예:
    - 웨이팅은 힘들지만 가치가 있다
    """
    compact_text = str(review_text).replace(" ", "")
    last_part = str(term).split()[-1]
    targets = {str(term).replace(" ", ""), last_part.replace(" ", "")}

    if last_part.endswith("다"):
        targets.add(last_part[:-1])

    negative_positions = [
        compact_text.find(target)
        for target in targets
        if target and compact_text.find(target) != -1
    ]

    if not negative_positions:
        return False

    first_negative_idx = min(negative_positions)

    for boundary in CONCESSIVE_BOUNDARIES:
        boundary_idx = compact_text.find(boundary, first_negative_idx)

        if boundary_idx == -1:
            continue

        tail = compact_text[boundary_idx + len(boundary):]

        if any(pattern in tail for pattern in POSITIVE_RESOLUTION_PATTERNS):
            return True

    return False


def should_decompose_negated_mixed_term(review_text, term):
    """
    음수 복합표현 안의 부정 특성이 원문에서 직접 부정되면 긴 표현을 분해.

    예:
    - 사전: 느끼하다 맛있다 / -1
    - 원문: 안느끼하고 맛있음
    -> 긴 -1 표현을 사용하지 않고, 느끼하다 0 + 맛있다 +1로 각각 분석.
    """
    parts = str(term).split()

    if len(parts) < 2:
        return False

    entries = sentiment_map.get(term, [])

    if not any(score < 0 for _, score in entries):
        return False

    negated_negative_parts = {
        part
        for part in parts
        if (
            has_direct_negative_context(review_text, part)
            and any(score < 0 for _, score in sentiment_map.get(part, []))
        )
    }

    has_separate_positive_part = any(
        part not in negated_negative_parts
        and any(score > 0 for _, score in sentiment_map.get(part, []))
        for part in parts
    )

    return bool(negated_negative_parts) and has_separate_positive_part


def has_direct_absence_of_negative_token_context(tokens, start_i, end_i):
    """
    부정 감성어가 바로 뒤의 부정/부재 표현으로 상쇄되는지 검사.

    예:
    - 질기다 않다
    - 잡내 없다
    - 비린내 전혀 없다
    """
    if start_i > 0 and tokens[start_i - 1] in DIRECT_LEFT_NEGATORS:
        return True

    right_tokens = tokens[end_i:min(len(tokens), end_i + 2)]

    if not right_tokens:
        return False

    if right_tokens[0] in DIRECT_ABSENCE_TOKENS:
        return True

    if (
        len(right_tokens) >= 2
        and right_tokens[0] in NEGATION_MODIFIERS
        and right_tokens[1] in DIRECT_ABSENCE_TOKENS
    ):
        return True

    return False


def is_cold_dish_temperature_context(review_text, term):
    """
    냉면류에서 '차갑다'가 정상적인 제공 온도를 설명하면 중립 처리.
    명시적인 온도 불만이 함께 있으면 음수 점수를 유지.
    """
    if term != "차갑다":
        return False

    compact_text = str(review_text).replace(" ", "")

    cold_dish_patterns = [
        "냉면", "평냉", "평양냉면", "막국수", "거냉",
        "냉모밀", "냉메밀", "메밀소바", "소바", "콩국수", "냉국수",
    ]

    cold_complaint_patterns = [
        "너무차갑", "지나치게차갑", "과하게차갑",
        "차갑기만", "차가워서별로", "차가워별로",
        "차가워서싫", "차가워싫", "차가워서아쉽",
        "차가워아쉽", "차가워서맛없", "차갑고맛없",
    ]

    has_cold_dish = any(pattern in compact_text for pattern in cold_dish_patterns)
    has_temperature_complaint = any(
        pattern in compact_text for pattern in cold_complaint_patterns
    )

    return has_cold_dish and not has_temperature_complaint


def make_keyword_variants(keyword):
    """
    감성어의 원문 변형을 생성.
    예:
    맛있다 -> 맛있, 맛있는, 맛있고, 맛있습니다
    친절하다 -> 친절, 친절한, 친절하고, 친절함
    약하다 -> 약, 약한, 약하고, 약함
    맵다 -> 매운, 매운맛, 매콤
    비리다 -> 비린, 비린내, 비리지
    """
    keyword = str(keyword).strip()
    compact_keyword = keyword.replace(" ", "")

    variants = set()
    variants.add(keyword)
    variants.add(compact_keyword)

    for part in keyword.split():
        variants.add(part)
        variants.add(part.replace(" ", ""))

    base_variants = set(variants)

    for kw in base_variants:
        if not kw:
            continue

        if kw.endswith("하다"):
            stem = kw[:-2]
            variants.add(stem)
            variants.add(stem + "하")
            variants.add(stem + "한")
            variants.add(stem + "하고")
            variants.add(stem + "하지만")
            variants.add(stem + "하니")
            variants.add(stem + "하지")
            variants.add(stem + "하지는")
            variants.add(stem + "하지도")
            variants.add(stem + "함")
            variants.add(stem + "했다")
            variants.add(stem + "했")
            variants.add(stem + "해")

        if kw.endswith("다"):
            stem = kw[:-1]
            variants.add(stem)
            variants.add(stem + "는")
            variants.add(stem + "고")
            variants.add(stem + "게")
            variants.add(stem + "지만")
            variants.add(stem + "지는")
            variants.add(stem + "지도")
            variants.add(stem + "은")
            variants.add(stem + "한")
            variants.add(stem + "합니다")
            variants.add(stem + "했습니다")
            variants.add(stem + "습니다")
            variants.add(stem + "스러울")
            variants.add(stem + "스럽")
            variants.add(stem + "었다")
            variants.add(stem + "었음")
            variants.add(stem + "웠다")
            variants.add(stem + "웠음")

        # 불규칙/음식 리뷰용 보완
        if kw.startswith("맵") or kw == "맵다":
            variants.update([
                "맵", "맵고", "맵지만", "매운", "매운맛", "매운 맛",
                "매워", "매콤", "매콤한", "매콤하고"
            ])

        if kw.startswith("비리") or kw == "비리다":
            variants.update([
                "비리", "비린", "비린내", "비리지", "비리지않", "비리지 않",
                "비린내는", "비린내가"
            ])

        if kw.startswith("아쉽") or kw == "아쉽다":
            variants.update([
                "아쉽", "아쉬움", "아쉬웠", "아쉬웠음", "아쉬웠다", "아쉬운"
            ])

        if kw.startswith("좋아") or kw == "좋아하다":
            variants.update([
                "좋아", "좋아하", "좋아한", "좋아하고", "좋아했", "좋아했구",
                "좋아했구요", "좋아했다", "좋아함"
            ])

        if kw.startswith("바삭") or kw == "바삭하다":
            variants.update([
                "바삭", "바삭하", "바삭한", "바삭하고", "바삭하니", "바삭함"
            ])

    return [v for v in variants if v]


def has_blocking_boundary(context, keyword_text, negative_patterns, boundaries):
    """
    [이전 버전 참고용]
    v14의 analyze_review에서는 이 함수와 아래의 넓은 부정 문맥 함수를 호출하지 않음.

    '다만/하지만/근데/그런데' 같은 경계 뒤의 부정 표현이
    앞쪽 긍정어를 무효화하지 않도록 막는 함수.

    예:
    '바삭하니 좋아했구요 다만 육수가 불편했어요'
    -> 바삭/좋아하다 주변에 불편이 있어도, 다만 뒤의 부정이므로 무효화하지 않음.
    """
    kw_idx = context.find(keyword_text)
    if kw_idx == -1:
        return False

    neg_positions = [
        context.find(pattern)
        for pattern in negative_patterns
        if context.find(pattern) != -1
    ]

    if not neg_positions:
        return False

    first_neg_idx = min(neg_positions)

    for boundary in boundaries:
        boundary_idx = context.find(boundary)
        if boundary_idx != -1 and kw_idx < boundary_idx < first_neg_idx:
            return True

    return False


def has_negative_context(review_text, keyword, window=35):
    """
    긍정 감성어 주변에 부정/대조 표현이 있으면 True 반환.
    단, '다만/하지만/근데/그런데' 뒤의 부정이 앞 긍정어를 무효화하지 않도록 처리.

    중요:
    - '없다/없고/없음'은 일반 부정 패턴에서 제외.
      이유: '비린내는 전혀 없고 불맛...' 같은 긍정 문맥을 잘못 무효화하기 때문.
    """
    if pd.isna(review_text):
        return False

    text = str(review_text)
    compact_text = text.replace(" ", "")

    keyword_variants = make_keyword_variants(keyword)

    negative_patterns = [
        # 직접 부정
        "아닌", "아니", "아니라", "아니고", "아니다", "아님",
        "않", "못",
        "안함", "안 함", "안하", "안 하",
        "하지 않", "지는 않", "진 않", "지도 않", "지도 안",

        # 별로/부족/실망 계열
        "별로", "그닥", "그다지", "별루",
        "기대 이하", "잘 모르", "모르겠",
        "실망", "아쉽", "아쉬", "비추",

        # '~것도 아니고' 계열
        "것도 아니고", "것도 아니", "것도 아니다", "것도 아닌", "것도 아님",
        "것은 아니", "건 아니", "게 아니",
        "것 같지는 않", "같지는 않", "같진 않",

        # 대조 표현
        "기보다는", "라기보다는", "이라기보다는",
        "하다기보다는", "하기보다는", "보다는",

        # 서비스/분위기 부정 맥락
        "거리감", "불편", "편하게 식사하기에는", "편하지",
    ]

    compact_negative_patterns = [p.replace(" ", "") for p in negative_patterns]

    boundaries = ["다만", "하지만", "그런데", "근데"]
    compact_boundaries = [b.replace(" ", "") for b in boundaries]

    direct_negative_suffixes = [
        "아닌", "아니", "아니고", "아니다", "아님",
        "않", "못", "없", "별로",
        "기보다는", "라기보다는", "이라기보다는",
        "하다기보다는", "하기보다는", "보다는",
        "것도아니고", "것도아니", "것도아니다", "것도아닌",
        "건아니", "게아니",
        "지는않", "진않", "지도않",
    ]

    # 붙어 있는 표현 직접 탐지
    for kw in keyword_variants:
        compact_kw = kw.replace(" ", "")
        for suffix in direct_negative_suffixes:
            if compact_kw + suffix in compact_text:
                return True

    # 원문 기준 주변 문맥 검사
    for kw in keyword_variants:
        start_idx = 0

        while True:
            idx = text.find(kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)

            context = text[start:end]

            if any(pattern in context for pattern in negative_patterns):
                if not has_blocking_boundary(context, kw, negative_patterns, boundaries):
                    return True

            start_idx = idx + len(kw)

    # 공백 제거 문장 기준 주변 문맥 검사
    for kw in keyword_variants:
        compact_kw = kw.replace(" ", "")
        start_idx = 0

        while True:
            idx = compact_text.find(compact_kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(compact_text), idx + len(compact_kw) + window)

            context = compact_text[start:end]

            if any(pattern in context for pattern in compact_negative_patterns):
                if not has_blocking_boundary(context, compact_kw, compact_negative_patterns, compact_boundaries):
                    return True

            start_idx = idx + len(compact_kw)

    return False


def has_negative_token_context(tokens, start_i, end_i, window=5):
    """
    토큰 기준 부정 문맥 검사.
    기존보다 보수적으로 처리함.

    핵심:
    - '없다/없고/없음'이 감성어 앞에 있다고 해서 감성어를 무효화하지 않음.
      예: 비린내 없다 + 불맛 좋다
    - '별로 맛있다', '맛있지 않다', '맛있는 것도 아니다' 같은 경우는 잡음.
    - '다만/하지만/근데/그런데' 뒤의 부정이 앞 감성어를 무효화하지 않도록 처리.
    """
    left_start = max(0, start_i - 3)
    right_end = min(len(tokens), end_i + window)

    left_tokens = tokens[left_start:start_i]
    right_tokens = tokens[end_i:right_end]

    left_context = " ".join(left_tokens)
    right_context = " ".join(right_tokens)
    full_context = " ".join(tokens[left_start:right_end])

    compact_left = left_context.replace(" ", "")
    compact_right = right_context.replace(" ", "")
    compact_full = full_context.replace(" ", "")

    # 감성어 앞에서 작동해도 되는 부정 표현
    left_negative_patterns = [
        "별로",
        "그닥",
        "그다지",
        "별루",
    ]

    # 감성어 뒤에서 작동하는 부정 표현
    right_negative_patterns = [
        "아니다", "아니고", "아닌", "아니", "아님",
        "않다", "않음", "않고", "않은", "않",
        "못하다", "못",

        "것 아니다", "것 아니고", "것 아니",
        "것도 아니다", "것도 아니고", "것도 아니",
        "건 아니다", "건 아니고", "건 아니",
        "게 아니다", "게 아니고", "게 아니",

        "기 보다는", "기보다는",
        "라기 보다는", "라기보다는",
        "하다기 보다는", "하다기보다는",
        "하기 보다는", "하기보다는",
        "보다는",

        "불편", "불편하다", "불편했",
    ]

    compact_left_negative_patterns = [p.replace(" ", "") for p in left_negative_patterns]
    compact_right_negative_patterns = [p.replace(" ", "") for p in right_negative_patterns]

    # 앞쪽 부정 검사
    if any(p in left_context for p in left_negative_patterns):
        return True

    if any(p in compact_left for p in compact_left_negative_patterns):
        return True

    # 뒤쪽 부정 검사
    if any(p in right_context for p in right_negative_patterns):
        return True

    if any(p in compact_right for p in compact_right_negative_patterns):
        return True

    # 붙어 있는 표현 보조 검사
    compact_full_negative_patterns = [
        "것도아니다", "것도아니고", "것도아니",
        "건아니다", "건아니고",
        "게아니다", "게아니고",
        "지는않", "진않", "지도않",
        "기보다는", "라기보다는",
        "하다기보다는", "하기보다는",
    ]

    # 다만/하지만 경계 처리
    compact_boundaries = ["다만", "하지만", "그런데", "근데"]

    if any(p in compact_full for p in compact_full_negative_patterns):
        neg_positions = [
            compact_full.find(p)
            for p in compact_full_negative_patterns
            if compact_full.find(p) != -1
        ]

        if neg_positions:
            first_neg_idx = min(neg_positions)

            for boundary in compact_boundaries:
                boundary_idx = compact_full.find(boundary)
                if boundary_idx != -1 and boundary_idx < first_neg_idx:
                    return False

        return True

    return False


def has_comparative_or_ironic_context(review_text, keyword, tokens=None, start_i=None, end_i=None, window=55):
    """
    긍정어가 현재 식당이 아니라 비교 대상/대체재/반어 문맥에 쓰였는지 검사.
    과잉 보정을 줄이기 위해 비교 대상 + 비교 결론이 같이 있을 때만 비교 문맥으로 처리.
    '옆집' 단독 패턴은 제외.

    예:
    - 일반 고깃집 된찌가 훨씬 맛있다
    - 컵라면에 삼김만 먹어도 이거보단 만족스럽다
    - 이걸 먹고 맛있다고 하는 사람들은 이해불가
    """
    if pd.isna(review_text):
        return False

    text = str(review_text)
    compact_text = text.replace(" ", "")

    keyword_variants = make_keyword_variants(keyword)

    alt_target_patterns = [
        "이거보단", "이것보단", "이거보다", "이것보다",
        "여기보다", "여기보단",
        "이집보다", "이 집보다",
        "이 식당보다", "이곳보다",

        "차라리",
        "컵라면", "삼김", "편의점", "씨유", "CU",
        "라면에삼김", "컵라면에삼김",

        "일반고깃집", "일반 고깃집",
        "서비스로나오는", "서비스로 나오는",
        "다른곳", "다른 곳",
        "다른집", "다른 집",
        "다른식당", "다른 식당",

        "건너편에있는", "건너편에 있는",
        "맞은편에있는", "맞은편에 있는",
    ]

    comparison_result_patterns = [
        "훨씬낫", "훨씬 낫",
        "더낫", "더 낫",
        "훨씬맛있", "훨씬 맛있",
        "더맛있", "더 맛있",
        "만족스러울것", "만족스러울 것",
        "낫다", "낫습니다",
    ]

    ironic_patterns = [
        "맛있다고하는사람", "맛있다고 하는 사람",
        "맛있다고하는사람들", "맛있다고 하는 사람들",
        "맛있다고하는분", "맛있다고 하는 분",
        "맛있다고느끼", "맛있다고 느끼",
        "맛있다고생각", "맛있다고 생각",
    ]

    ironic_negative_patterns = [
        "이해불가", "이해 불가",
        "대체뭘", "대체 뭘",
        "뭘드시고", "뭘 드시고",
        "살아계시는",
        "아무리생각해도", "아무리 생각해도",
        "이해안", "이해 안",
    ]

    improvement_patterns = [
        "제발", "바꾸세요", "고치세요", "개선",
        "문제", "심각",
        "맛이없", "맛 없", "맛없",
    ]

    compact_alt_target_patterns = [p.replace(" ", "") for p in alt_target_patterns]
    compact_result_patterns = [p.replace(" ", "") for p in comparison_result_patterns]
    compact_ironic_patterns = [p.replace(" ", "") for p in ironic_patterns]
    compact_ironic_negative_patterns = [p.replace(" ", "") for p in ironic_negative_patterns]
    compact_improvement_patterns = [p.replace(" ", "") for p in improvement_patterns]

    def check_context(context):
        compact_context = context.replace(" ", "")

        has_alt_target = (
            any(p in context for p in alt_target_patterns)
            or any(p in compact_context for p in compact_alt_target_patterns)
        )

        has_comparison_result = (
            any(p in context for p in comparison_result_patterns)
            or any(p in compact_context for p in compact_result_patterns)
        )

        if has_alt_target and has_comparison_result:
            return True

        has_ironic = (
            any(p in context for p in ironic_patterns)
            or any(p in compact_context for p in compact_ironic_patterns)
        )

        has_ironic_negative = (
            any(p in context for p in ironic_negative_patterns)
            or any(p in compact_context for p in compact_ironic_negative_patterns)
        )

        if has_ironic and has_ironic_negative:
            return True

        has_improvement = (
            any(p in context for p in improvement_patterns)
            or any(p in compact_context for p in compact_improvement_patterns)
        )

        if has_improvement and ("제발" in context or "제발" in compact_context):
            return True

        return False

    # 원문 기준 검사
    for kw in keyword_variants:
        start_idx = 0

        while True:
            idx = text.find(kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)
            context = text[start:end]

            if check_context(context):
                return True

            start_idx = idx + len(kw)

    # 공백 제거 기준 검사
    for kw in keyword_variants:
        compact_kw = kw.replace(" ", "")
        start_idx = 0

        while True:
            idx = compact_text.find(compact_kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(compact_text), idx + len(compact_kw) + window)
            context = compact_text[start:end]

            if check_context(context):
                return True

            start_idx = idx + len(compact_kw)

    # 토큰 기준 검사
    if tokens is not None and start_i is not None and end_i is not None:
        left = max(0, start_i - 8)
        right = min(len(tokens), end_i + 8)

        token_context = " ".join(tokens[left:right])

        if check_context(token_context):
            return True

    return False


def has_positive_preference_context(review_text, keyword, window=45):
    """
    [이전 버전 참고용]
    v14에서는 주변 긍정어만으로 음수 점수를 지우지 않으므로 호출하지 않음.

    부정어가 취향/선호/만족 문맥에서 긍정적으로 쓰였는지 검사.

    예:
    - 탄력이 약한 편인데 이 점이 마음에 든다
    - 슴슴한 맛이 개인적으로 좋았다
    - 자극적이지 않아서 좋다
    """
    if pd.isna(review_text):
        return False

    text = str(review_text)
    compact_text = text.replace(" ", "")

    keyword_variants = make_keyword_variants(keyword)

    positive_preference_patterns = [
        "마음에 든다",
        "마음에 들",
        "마움에 든다",
        "마움에 들",
        "맘에 든다",
        "맘에 들",

        "개인적으로 좋",
        "개인적으로 가장",
        "개인적으로 선호",
        "가장 선호",
        "선호하는",
        "선호",

        "취향",
        "만족감",
        "만족",
        "좋았다",
        "좋을",
        "좋은",
        "좋습니다",
        "도전해봐도 좋",

        "이 점이",
        "이점이",
        "이 부분이",
        "이부분이",
    ]

    compact_positive_patterns = [p.replace(" ", "") for p in positive_preference_patterns]

    for kw in keyword_variants:
        start_idx = 0

        while True:
            idx = text.find(kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)

            context = text[start:end]
            compact_context = context.replace(" ", "")

            if any(p in context for p in positive_preference_patterns):
                return True

            if any(p in compact_context for p in compact_positive_patterns):
                return True

            start_idx = idx + len(kw)

    return False


def has_other_target_negative_context(review_text, keyword, window=60):
    """
    부정어가 현재 리뷰 대상 식당이 아니라 다른 가게/다른 대상에 향하는지 검사.

    예:
    - 다른 가게들은 탄성이 높아서 아쉬웠음
    - 다른 평냉집은 만족감을 느끼지 못했다
    - 다른 집은 별로였다
    """
    if pd.isna(review_text):
        return False

    text = str(review_text)
    compact_text = text.replace(" ", "")

    keyword_variants = make_keyword_variants(keyword)

    other_target_patterns = [
        "다른 가게", "다른가게",
        "다른 집", "다른집",
        "다른 곳", "다른곳",
        "다른 식당", "다른식당",
        "다른 평냉집", "다른평냉집",
        "다른 평양냉면집", "다른평양냉면집",
        "타 가게", "타가게",
        "타 식당", "타식당",
        "남의 가게",
        "근처 가게",
        "주변 가게",
    ]

    other_negative_patterns = [
        "아쉬웠",
        "아쉬움",
        "아쉽",
        "별로",
        "실망",
        "만족감을 느끼진 못",
        "만족감을 느끼지 못",
        "만족감 못",
        "못했다",
        "못했",
        "부족",
        "별루",
        "그닥",
        "그다지",
    ]

    compact_other_target_patterns = [p.replace(" ", "") for p in other_target_patterns]
    compact_other_negative_patterns = [p.replace(" ", "") for p in other_negative_patterns]

    for kw in keyword_variants:
        start_idx = 0

        while True:
            idx = text.find(kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)

            context = text[start:end]
            compact_context = context.replace(" ", "")

            has_other_target = (
                any(p in context for p in other_target_patterns)
                or any(p in compact_context for p in compact_other_target_patterns)
            )

            has_other_negative = (
                any(p in context for p in other_negative_patterns)
                or any(p in compact_context for p in compact_other_negative_patterns)
            )

            if has_other_target and has_other_negative:
                return True

            start_idx = idx + len(kw)

    return False


def has_absence_of_negative_context(review_text, keyword, window=45):
    """
    부정어가 실제 부정이 아니라 '부정 요소가 없다'는 의미인지 검사.

    예:
    - 비린내는 전혀 없고
    - 잡내가 없다
    - 냄새 없이 깔끔하다
    - 비리지 않다
    """
    if pd.isna(review_text):
        return False

    text = str(review_text)
    compact_text = text.replace(" ", "")

    keyword_variants = make_keyword_variants(keyword)

    absence_patterns = [
        "전혀 없",
        "하나도 없",
        "거의 없",
        "없고",
        "없다",
        "없어요",
        "없습니다",
        "없음",
        "없는",
        "없이",
        "안 나",
        "나지 않",
        "나지않",
        "않다",
        "않음",
        "비리지 않",
        "비리지않",
    ]

    compact_absence_patterns = [p.replace(" ", "") for p in absence_patterns]

    for kw in keyword_variants:
        start_idx = 0

        while True:
            idx = text.find(kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)

            context = text[start:end]
            compact_context = context.replace(" ", "")

            if any(p in context for p in absence_patterns):
                return True

            if any(p in compact_context for p in compact_absence_patterns):
                return True

            start_idx = idx + len(kw)

    return False


def has_positive_flavor_context(review_text, keyword, window=50):
    """
    부정 또는 중립으로 잡힌 맛 표현이 긍정적인 풍미 문맥에서 쓰였는지 검사.

    예:
    - 약간의 매운 맛이 감칠맛으로 매우 조화롭다
    - 매콤해서 좋다
    - 느끼하지 않고 깔끔하다
    """
    if pd.isna(review_text):
        return False

    text = str(review_text)
    compact_text = text.replace(" ", "")

    keyword_variants = make_keyword_variants(keyword)

    positive_flavor_patterns = [
        "감칠맛",
        "불맛",
        "조화롭",
        "잘 어울",
        "잘어울",
        "맛있",
        "좋",
        "매력",
        "깔끔",
        "개운",
        "중독",
        "입 안에 가득",
        "입안에가득",
        "약간의",
        "적당",
        "기분 좋",
        "기분좋",
        "반해",
        "단골",
    ]

    compact_positive_flavor_patterns = [p.replace(" ", "") for p in positive_flavor_patterns]

    for kw in keyword_variants:
        start_idx = 0

        while True:
            idx = text.find(kw, start_idx)
            if idx == -1:
                break

            start = max(0, idx - window)
            end = min(len(text), idx + len(kw) + window)

            context = text[start:end]
            compact_context = context.replace(" ", "")

            if any(p in context for p in positive_flavor_patterns):
                return True

            if any(p in compact_context for p in compact_positive_flavor_patterns):
                return True

            start_idx = idx + len(kw)

    return False


# =========================================================
# 7-1. 토큰 주변 문맥 기반 조건부 감성어 보정
# =========================================================
# 이 규칙들은 review_text 전체를 넓게 검색하지 않습니다.
# 감성어 앞 5토큰, 뒤 8토큰만 사용해 다른 절의 문맥이 전파되는 것을 막습니다.
# 확실한 반대 극성이 보여도 점수를 바로 반전하지 않고 0점으로 중립화합니다.
#
# 검증 예시와 기대 matched_words:
# - 고기만 빠르게 입에 욱여넣고 나왔습니다
#   -> 빠르다:1->0[speed_not_service_context]
# - 주문하고 음식이 빠르게 나왔어요
#   -> 빠르다:1[speed_service_context]
# - 탄력이 약한 편인데 이 점이 마음에 든다
#   -> 약하다:-1->0[positive_preference_context]
# - 간이 약해서 아쉬웠다
#   -> 약하다:-1[token]
# - 비싸지만 만족스러웠다
#   -> 비싸다:-1->0[expensive_but_satisfied_context]
# - 웨이팅할 만한 맛이다
#   -> 사전에 '웨이팅' 단독 표현이 없으면 기존 구조상 미매칭되어 자동 0점
#   -> 사전의 웨이팅 음수 표현이 매칭되면 -1->0[worth_waiting_context]
# - 특별할 건 없었다
#   -> 특별하다:1->0[negated_special_context]
# - 목소리가 지나치게 크고 불편하다
#   -> 크다:1->0[noise_context]
# - 너무 매워서 혼남
#   -> 너무 맵다:-2[token], 맵다:-1->0[suppressed_by_stronger_phrase]
# - 음식이 식으면서 누린내가 났다
#   -> 누리다 나다를 누린내로 보정 후 누린내:-1[token]
# - 아구힘 센 분이라면
#   -> 불다:-1->0[blocked_risky_single_token]
# - 너무 바쁜 매장이라 먹을 수 있을지 걱정
#   -> 바쁘다:-1->0[busy_indirect_context]
#   -> 걱정:-1->0[indirect_concern_context]

CONTEXT_DEPENDENT_ROOTS = {
    "빠르게", "빠르다", "빨리", "금방", "바로",
    "약하다", "강하다", "세다", "크다",
    "많다", "적다",
    "작다", "얇다", "두껍다",
    "오래",
    "차갑다", "뜨겁다",
    "기름지다", "자극적이다", "슴슴하다", "심심하다",
    "느끼하다", "맵다", "매콤하다",
    "무난하다", "깔끔하다",
    "사람 많다", "웨이팅", "웨이", "시끄럽다", "혼잡", "붐비다",
    "비싸다", "싸다", "저렴하다",
    "평범하다", "특별하다",
    "왠만하다", "웬만하다",
    "맛집", "설레다",
    "바쁘다", "걱정",
    "싫다", "이상하다", "짜다", "힘들다", "사라지다",
    "별로", "불친절", "불친절하다", "투박하다", "길다",
    "맛없다", "나쁘다", "굳이", "필요없다", "안되다", "없다", "멈추다",
}


def compact_token_text(token_list):
    """토큰 window 안의 표현만 공백 없이 연결."""
    return "".join(str(token) for token in token_list)


def token_window_has(token_list, patterns):
    """토큰 window 내부에서만 단어/구문 패턴을 검사."""
    compact = compact_token_text(token_list)
    return any(str(pattern).replace(" ", "") in compact for pattern in patterns)


def is_context_dependent_term(term):
    """term 자체에 문맥 의존 감성어가 포함되어 있는지 확인."""
    compact_term = str(term).replace(" ", "")
    return any(root.replace(" ", "") in compact_term for root in CONTEXT_DEPENDENT_ROOTS)


def adjusted_match_text(term, original_score, adjusted_score, tag=None):
    """matched_words에 원점수, 조정점수, 조정 이유를 일관되게 기록."""
    if not tag:
        return f"{term}:{original_score}[token]"

    if adjusted_score == original_score:
        return f"{term}:{original_score}[{tag}]"

    return f"{term}:{original_score}->{adjusted_score}[{tag}]"


PHRASE_RULES = [
    # v22: 명확한 최종 부정 결론은 strong negative evidence로 기록
    {
        "pattern": r"(?:다시는안(?:올|갈)|다시가지(?:는)?않|재방문(?:의사)?없)",
        "label": "다시는 안 갈 것",
        "category": "general",
        "score": -2,
        "tag": "strong_rejection_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"맛이존재하지않",
        "label": "맛이 존재하지 않다",
        "category": "food",
        "score": -2,
        "tag": "strong_food_negative_phrase",
        "suppress_terms": set(),
        "exclude_patterns": [r"맛이존재하지않는게아니"],
    },

    # v22: 재방문/재이용 의사
    {
        "pattern": r"(?:또|다시)(?:가고|방문하고)싶",
        "label": "또 가고 싶다",
        "category": "general",
        "score": 2,
        "tag": "revisit_positive_phrase",
        "suppress_terms": set(),
        "exclude_patterns": [r"(?:또|다시)(?:가고|방문하고)싶지않"],
    },
    {
        "pattern": r"집가는길에생각(?:이)?나",
        "label": "집 가는 길에 생각이 난다",
        "category": "general",
        "score": 1,
        "tag": "craving_memory_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"다음에또",
        "label": "다음에 또",
        "category": "general",
        "score": 1,
        "tag": "revisit_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"다음에(?:남비|냄비)들고가야겠",
        "label": "다음에 냄비 들고 가야겠다",
        "category": "general",
        "score": 1,
        "tag": "takeout_intention_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"다음에포장(?:해야겠|할게|해야지)",
        "label": "다음에 포장해야겠다",
        "category": "general",
        "score": 1,
        "tag": "takeout_intention_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"웨이팅(?:을)?했는데도(?:또|다시)가고싶",
        "label": "웨이팅 했는데도 또 가고 싶다",
        "category": "general",
        "score": 2,
        "tag": "waiting_revisit_positive",
        "suppress_terms": set(),
    },

    # v22: 추천/가치/대기 긍정
    {
        "pattern": r"괜히줄길게설까",
        "label": "괜히 줄 길게 설까",
        "category": "general",
        "score": 1,
        "tag": "popularity_positive_context",
        "suppress_terms": {"길다"},
    },
    {
        "pattern": r"줄(?:이|은|이렇게)?긴이유(?:가)?있",
        "label": "줄 긴 이유가 있다",
        "category": "general",
        "score": 1,
        "tag": "popularity_positive_context",
        "suppress_terms": {"길다"},
    },
    {
        "pattern": r"(?:테이블)?회전율(?:이|은)?빠른?",
        "label": "회전율이 빠르다",
        "category": "general",
        "score": 1,
        "tag": "wait_turnover_positive",
        "suppress_terms": {"빠르다"},
    },
    {
        "pattern": r"금방들어(?:가|갈|갔)",
        "label": "금방 들어가다",
        "category": "general",
        "score": 1,
        "tag": "short_wait_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"거부감없이먹",
        "label": "거부감 없이 먹다",
        "category": "food",
        "score": 1,
        "tag": "approachable_taste_positive",
        "suppress_terms": set(),
    },

    # v22: 맛/품질/경험 긍정
    {
        "pattern": r"맛있음",
        "label": "맛있음",
        "category": "food",
        "score": 1,
        "tag": "food_positive_phrase",
        "suppress_terms": {"맛있다"},
    },
    {
        "pattern": r"맛은투박하지만있을맛은다있",
        "label": "투박하지만 있을 맛은 다 있다",
        "category": "food",
        "score": 2,
        "tag": "complete_flavor_positive",
        "suppress_terms": {"투박하다"},
    },
    {
        "pattern": r"있을맛은다있",
        "label": "있을 맛은 다 있다",
        "category": "food",
        "score": 1,
        "tag": "complete_flavor_positive",
        "suppress_terms": {"투박하다"},
        "exclude_patterns": [r"맛은투박하지만있을맛은다있"],
    },
    {
        "pattern": r"치즈폭포",
        "label": "치즈 폭포",
        "category": "food",
        "score": 1,
        "tag": "food_experience_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"믿고먹",
        "label": "믿고 먹다",
        "category": "food",
        "score": 2,
        "tag": "trust_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"장인볶음밥",
        "label": "장인볶음밥",
        "category": "food",
        "score": 1,
        "tag": "menu_must_try_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"마지막볶음밥까지",
        "label": "마지막 볶음밥까지",
        "category": "food",
        "score": 1,
        "tag": "complete_meal_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"끝나쥬",
        "label": "끝나쥬",
        "category": "general",
        "score": 1,
        "tag": "satisfying_finish_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"여기100(?:국내산)?",
        "label": "여기 100",
        "category": "general",
        "score": 1,
        "tag": "trust_positive_phrase",
        "suppress_terms": set(),
    },

    # v22: 가격/양/인기
    {
        "pattern": r"비싸도맛있",
        "label": "비싸도 맛있음",
        "category": "food",
        "score": 1,
        "tag": "expensive_but_tasty_context",
        "suppress_terms": {"비싸다", "맛있다"},
    },
    {
        "pattern": r"이정도면가성비좋",
        "label": "이 정도면 가성비 좋다",
        "category": "price",
        "score": 1,
        "tag": "price_positive_phrase",
        "suppress_terms": {"좋다", "가성비 좋다"},
    },
    {
        "pattern": r"쌈채소무한리필",
        "label": "쌈채소 무한리필",
        "category": "price",
        "score": 1,
        "tag": "value_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"(?<!쌈채소)무한리필",
        "label": "무한리필",
        "category": "price",
        "score": 1,
        "tag": "value_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"(?:(?:냄비|남비)\d+개.{0,50}(?:김치통\d+개|테이크아웃).{0,50}(?:포장나가|못셈|못세)|(?:테이크아웃|포장)(?:이|도)?(?:많이|계속)나가)",
        "label": "포장 많이 나감",
        "category": "general",
        "score": 1,
        "tag": "popularity_positive_context",
        "suppress_terms": set(),
    },

    # v21: 형태소 분석에서 자주 사라지는 명확한 문장형 긍정
    {
        "pattern": r"곰탕1티어",
        "label": "곰탕 1티어",
        "category": "food",
        "score": 2,
        "tag": "rank_tier_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"(?<!곰탕)1티어",
        "label": "1티어",
        "category": "general",
        "score": 2,
        "tag": "rank_tier_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"무조건특으로먹을것",
        "label": "무조건 특으로 먹을 것",
        "category": "food",
        "score": 1,
        "tag": "menu_recommendation_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"(?:[2-9]|[1-9][0-9]+)(?:번째|회차)방문",
        "label": "2번째 방문",
        "category": "general",
        "score": 1,
        "tag": "repeat_visit_context",
        "suppress_terms": set(),
    },
    {
        "pattern": r"김치가맛나(?:다|는|게|고|서|네요|요|더라|며)",
        "label": "김치가 맛나다",
        "category": "food",
        "score": 1,
        "tag": "food_positive_phrase",
        "suppress_terms": {"맛나다"},
    },
    {
        "pattern": r"구수함(?:의)?조화",
        "label": "구수함의 조화",
        "category": "food",
        "score": 1,
        "tag": "flavor_harmony_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"구수함",
        "label": "구수함",
        "category": "food",
        "score": 1,
        "tag": "food_depth_positive",
        "suppress_terms": {"구수하다"},
    },
    {
        "pattern": r"(?:맛|향|구수함).{0,12}조화|조화(?:가좋|롭)",
        "label": "조화",
        "category": "food",
        "score": 1,
        "tag": "flavor_harmony_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"고추향(?:을)?살리",
        "label": "고추향을 살리다",
        "category": "food",
        "score": 1,
        "tag": "flavor_harmony_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"매운맛(?:은|을)?덜어(?:내|낸)",
        "label": "매운맛은 덜어내다",
        "category": "food",
        "score": 1,
        "tag": "balanced_spicy_context",
        "suppress_terms": {"맵다"},
    },
    {
        "pattern": r"독특한양념",
        "label": "독특한 양념",
        "category": "food",
        "score": 1,
        "tag": "flavor_character_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"날씨좋은",
        "label": "날씨 좋은",
        "category": "atmosphere",
        "score": 1,
        "tag": "outdoor_atmosphere_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"야장느낌",
        "label": "야장 느낌",
        "category": "atmosphere",
        "score": 1,
        "tag": "outdoor_atmosphere_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"고기먹을수있는게너무좋",
        "label": "고기 먹을 수 있는 게 너무 좋다",
        "category": "general",
        "score": 1,
        "tag": "experience_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"멀리갈필요없",
        "label": "멀리 갈 필요 없다",
        "category": "general",
        "score": 1,
        "tag": "convenience_positive_context",
        "suppress_terms": {"굳이", "필요없다", "없다"},
    },
    {
        "pattern": r"이게바로캠핑",
        "label": "이게 바로 캠핑",
        "category": "general",
        "score": 2,
        "tag": "experience_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"넘예",
        "label": "넘예",
        "category": "atmosphere",
        "score": 1,
        "tag": "atmosphere_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"낭만넘치",
        "label": "낭만 넘치다",
        "category": "atmosphere",
        "score": 1,
        "tag": "atmosphere_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"분위기굿",
        "label": "분위기 굿",
        "category": "atmosphere",
        "score": 1,
        "tag": "atmosphere_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"맛있게잘먹",
        "label": "맛있게 잘 먹다",
        "category": "food",
        "score": 1,
        "tag": "food_positive_phrase",
        "suppress_terms": {"맛있다"},
    },
    {
        "pattern": r"총점(?:4\.[0-9]|5(?:\.0)?)",
        "label": "총점 4.5",
        "category": "general",
        "score": 3,
        "tag": "explicit_rating_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"맛(?:은|이)?(?:4\.[0-9]|5(?:\.0)?)",
        "label": "맛 4.5",
        "category": "food",
        "score": 2,
        "tag": "explicit_taste_rating_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"맛(?:을)?알아버",
        "label": "맛을 알아버림",
        "category": "food",
        "score": 1,
        "tag": "taste_discovery_positive",
        "suppress_terms": set(),
    },
    {
        "pattern": r"또다른맛",
        "label": "또다른 맛",
        "category": "food",
        "score": 1,
        "tag": "taste_discovery_positive",
        "suppress_terms": set(),
    },
    # 약한 긍정과 웨이팅 가치 표현
    {
        "pattern": r"먹어볼만(?:하|합니다|해|한|함)?",
        "label": "먹어볼만하다",
        "category": "food",
        "score": 1,
        "tag": "worth_trying_phrase",
        "suppress_terms": set(),
        "exclude_patterns": [r"먹어볼만하지않"],
    },
    {
        "pattern": r"(?:줄(?:을)?설만|줄서도아깝지않|기다릴만|기다릴가치(?:가)?있|웨이팅할만)",
        "label": "줄 설만하다",
        "category": "general",
        "score": 1,
        "tag": "worth_waiting_phrase",
        "suppress_terms": set(),
        "exclude_patterns": [r"(?:줄설만|기다릴만|웨이팅할만)하지않"],
    },

    # 재방문, 반복 방문, 추천
    {
        "pattern": r"재방문(?:은|도)?(?:ok|오케이)",
        "label": "재방문 OK",
        "category": "general",
        "score": 2,
        "tag": "revisit_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"재방문(?:의사(?:가)?있|예정|하고싶|할게|할듯)",
        "label": "재방문 의사 있음",
        "category": "general",
        "score": 2,
        "tag": "revisit_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"(?:또갈듯|다시갈집|종종갈듯|(?:나중에)?다시가(?:는)?걸로|다시갈(?:게|듯|예정))",
        "label": "또 갈 듯",
        "category": "general",
        "score": 2,
        "tag": "revisit_intention_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"(?:[5-9]|[1-9][0-9]+)회이상방문",
        "label": "5회 이상 방문",
        "category": "general",
        "score": 1,
        "tag": "repeat_visit_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"(?:여러번|수차례|몇번이나)방문",
        "label": "여러 번 방문",
        "category": "general",
        "score": 1,
        "tag": "repeat_visit_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"단골집|단골(?!손님|은아니|이아니|도아니)",
        "label": "단골집",
        "category": "general",
        "score": 2,
        "tag": "loyal_customer_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"(?:강추|꼭가보세요|꼭가보시)",
        "label": "강추",
        "category": "general",
        "score": 2,
        "tag": "recommendation_positive_phrase",
        "suppress_terms": {"추천"},
    },
    {
        "pattern": r"추천(?:합니다|드립니다|드려요|해요|하고싶|할게|할만)",
        "label": "추천",
        "category": "general",
        "score": 1,
        "tag": "recommendation_positive_phrase",
        "suppress_terms": {"추천"},
        "exclude_patterns": [r"추천(?:하지않|안함|못하)"],
    },
    {
        "pattern": r"(?:감동받|맛있|좋아|마음에들).{0,30}다시갔",
        "label": "다시 갔다",
        "category": "general",
        "score": 1,
        "tag": "revisit_positive_phrase",
        "suppress_terms": set(),
        "exclude_patterns": [r"(?:맛있지않|좋지않|마음에들지않).{0,30}다시갔"],
    },

    # 최종 강한 긍정과 점수형 표현
    {
        "pattern": r"서울최고",
        "label": "서울 최고",
        "category": "general",
        "score": 3,
        "tag": "strong_final_positive",
        "suppress_terms": {"최고"},
        "exclude_patterns": [r"서울최고(?:는|가|라고)?아니"],
    },
    {
        "pattern": r"최고(?:의)?.{0,12}(?:김치찌개집|보쌈집|냉면집|맛집|식당)",
        "label": "최고 김치찌개집",
        "category": "general",
        "score": 3,
        "tag": "strong_final_positive",
        "suppress_terms": {"최고"},
        "exclude_patterns": [r"최고.{0,12}(?:김치찌개집|보쌈집|냉면집|맛집|식당).{0,8}아니"],
    },
    {
        "pattern": r"인생맛집",
        "label": "인생 맛집",
        "category": "general",
        "score": 3,
        "tag": "strong_final_positive",
        "suppress_terms": {"맛집"},
        "exclude_patterns": [r"인생맛집.{0,8}아니"],
    },
    {
        "pattern": r"극락",
        "label": "극락",
        "category": "general",
        "score": 2,
        "tag": "strong_positive_phrase",
        "suppress_terms": {"극락"},
        "exclude_patterns": [r"극락.{0,8}아니"],
    },
    {
        "pattern": r"신세계(?:를)?경험",
        "label": "신세계 경험",
        "category": "general",
        "score": 2,
        "tag": "strong_positive_phrase",
        "suppress_terms": {"신세계"},
        "exclude_patterns": [r"신세계.{0,8}아니"],
    },
    {
        "pattern": r"신세계(?!백화점|경험)",
        "label": "신세계",
        "category": "general",
        "score": 2,
        "tag": "strong_positive_phrase",
        "suppress_terms": {"신세계"},
        "exclude_patterns": [r"신세계(?:를)?경험", r"신세계.{0,8}아니"],
    },
    {
        "pattern": r"감동받",
        "label": "감동 받고",
        "category": "general",
        "score": 2,
        "tag": "strong_positive_phrase",
        "suppress_terms": {"감동"},
        "exclude_patterns": [r"감동받(?:지못|지않)"],
    },
    {
        "pattern": r"감동(?!받)",
        "label": "감동",
        "category": "general",
        "score": 2,
        "tag": "strong_positive_phrase",
        "suppress_terms": {"감동"},
        "exclude_patterns": [r"감동(?:은|이|도)?없|감동.{0,8}아니"],
    },
    {
        "pattern": r"후회(?:가)?없(?!지않|진않)(?:는선택|을선택|다|음|었|겠)?",
        "label": "후회없는 선택",
        "category": "general",
        "score": 2,
        "tag": "strong_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"백점만점(?!에?(?:[0-8]?[0-9])점)(?:에?(?:9[0-9]|100)점(?:정도)?(?:주고싶)?)?|(?:9[0-9]|100)점(?:정도)?주고싶|100점(?!만점에(?:[0-8]?[0-9])점)|만점(?!에(?:[0-8]?[0-9])점)",
        "label": "백점 만점 고득점",
        "category": "general",
        "score": 3,
        "tag": "strong_rating_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"가격과맛퀄리티모두.{0,20}(?:백점만점|9[0-9]점|100점)",
        "label": "가격과 맛 퀄리티 모두 고득점",
        "category": "food",
        "score": 1,
        "tag": "quality_positive_context",
        "suppress_terms": set(),
    },

    # 맛, 품질, 가격, 양 조합형 긍정
    {
        "pattern": r"깊은맛",
        "label": "깊은 맛",
        "category": "food",
        "score": 1,
        "tag": "food_depth_positive",
        "suppress_terms": set(),
        "exclude_patterns": [r"깊은맛.{0,8}아니"],
    },
    {
        "pattern": r"진한맛",
        "label": "진한 맛",
        "category": "food",
        "score": 1,
        "tag": "food_depth_positive",
        "suppress_terms": set(),
        "exclude_patterns": [r"진한맛.{0,8}아니"],
    },
    {
        "pattern": r"맛(?:이|은|는|도)?(?:정말|아주|너무)?좋(?!지않|지는않)",
        "label": "맛 좋다",
        "category": "food",
        "score": 1,
        "tag": "food_positive_phrase",
        "suppress_terms": {"좋다", "맛 좋다"},
    },
    {
        "pattern": r"맛(?:이|은|는|도)?괜찮(?!지않|지는않)",
        "label": "맛 괜찮다",
        "category": "food",
        "score": 1,
        "tag": "food_positive_phrase",
        "suppress_terms": {"괜찮다", "맛 괜찮다"},
    },
    {
        "pattern": r"맛(?:이|은|는|도)?(?:훌륭(?!하지않)|미쳤)",
        "label": "맛이 훌륭하다",
        "category": "food",
        "score": 2,
        "tag": "strong_food_positive_phrase",
        "suppress_terms": {"훌륭하다"},
    },
    {
        "pattern": r"퀄리티(?:도|가|는)?(?:좋|높|괜찮|미쳤)(?!지않)",
        "label": "퀄리티 좋다",
        "category": "food",
        "score": 1,
        "tag": "quality_positive_phrase",
        "suppress_terms": {"좋다", "퀄리티 좋다"},
        "exclude_patterns": [
            r"퀄리티.{0,12}좋을수(?:가)?없",
            r"퀄리티.{0,12}좋지않",
        ],
    },
    {
        "pattern": r"가격(?:이|은|는|도)?(?:괜찮|좋|착하)(?!지않)",
        "label": "가격 괜찮다",
        "category": "price",
        "score": 1,
        "tag": "price_positive_phrase",
        "suppress_terms": {"가격 괜찮다"},
    },
    {
        "pattern": r"가성비(?:가|는|도)?(?:매우|정말|진짜|아주)?굿",
        "label": "가성비 굿",
        "category": "price",
        "score": 2,
        "tag": "strong_positive_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"가성비(?:가|는|도)?(?:정말|진짜|아주|매우)?좋(?!지않)",
        "label": "가성비 좋다",
        "category": "price",
        "score": 1,
        "tag": "price_positive_phrase",
        "suppress_terms": {"좋다", "가성비 좋다"},
    },
    {
        "pattern": r"(?:다|전부|둘이서|2인이서)못먹",
        "label": "다 못먹음",
        "category": "food",
        "score": 1,
        "tag": "large_portion_context",
        "suppress_terms": set(),
        "exclude_patterns": [r"(?:맛없|상했|상해|비려|느끼해서|매워서).{0,15}(?:다|전부|둘이서|2인이서)못먹"],
    },
    {
        "pattern": r"양(?:이|은|는|도)?(?:정말|진짜|아주|매우)?많(?!지않)",
        "label": "양 많다",
        "category": "food",
        "score": 1,
        "tag": "large_portion_context",
        "suppress_terms": {"많다", "양 많다"},
    },
    {
        "pattern": r"푸짐(?!하지않)",
        "label": "푸짐하다",
        "category": "food",
        "score": 1,
        "tag": "large_portion_context",
        "suppress_terms": {"푸짐하다"},
    },

    # 긍정 관용구와 명확한 메뉴 칭찬
    {
        "pattern": r"게눈감추듯사라",
        "label": "게눈 감추듯 사라짐",
        "category": "food",
        "score": 1,
        "tag": "positive_idiom",
        "suppress_terms": {"사라지다"},
    },
    {
        "pattern": r"순삭",
        "label": "순삭",
        "category": "food",
        "score": 1,
        "tag": "positive_idiom",
        "suppress_terms": set(),
        "exclude_patterns": [r"(?:손님|사람).{0,12}계속들어"],
    },
    {
        "pattern": r"계속들어(?:가|감)",
        "label": "계속 들어감",
        "category": "food",
        "score": 1,
        "tag": "positive_idiom",
        "suppress_terms": set(),
    },
    {
        "pattern": r"젓가락(?:이)?멈추지않",
        "label": "젓가락이 멈추지 않음",
        "category": "food",
        "score": 1,
        "tag": "positive_idiom",
        "suppress_terms": set(),
    },
    {
        "pattern": r"볶음밥(?:은|는)?필수",
        "label": "볶음밥은 필수",
        "category": "food",
        "score": 1,
        "tag": "menu_must_try_phrase",
        "suppress_terms": set(),
    },
    {
        "pattern": r"사람들데려가기좋",
        "label": "사람들 데려가기 좋다",
        "category": "general",
        "score": 1,
        "tag": "recommendation_positive_phrase",
        "suppress_terms": set(),
    },

    # 접근/대기 불편은 약한 general 음수로만 반영
    {
        "pattern": r"(?:웨이팅|대기)장난아니",
        "label": "웨이팅 장난 아니다",
        "category": "general",
        "score": -1,
        "tag": "waiting_negative_phrase",
        "suppress_terms": set(),
    },
]


# =========================================================
# 7-1. 일반화 phrase pattern layer
# =========================================================
# 기존 PHRASE_RULES는 고확신 관용구와 회귀 호환을 위해 유지합니다.
# 아래 규칙은 특정 문장을 하나씩 추가하지 않고, 의미 역할을 가진 slot을
# 조합합니다. 예를 들어 revisit_time + revisit_action + future_intent 조합은
# "또 먹으러 갈게", "다음에 포장해야겠다" 같은 미등록 변형도 포착합니다.
PATTERN_SLOTS = {
    "revisit_time": [r"또", r"다시", r"다음에", r"나중에", r"종종", r"자주"],
    "revisit_action": [r"가", r"방문하", r"먹으러가", r"먹", r"포장하", r"시키"],
    "future_intent": [r"고싶", r"아야겠", r"어야겠", r"야겠", r"할게", r"할듯", r"갈듯", r"예정", r"확정"],
    "positive_predicate": [
        r"좋", r"괜찮", r"훌륭", r"미쳤", r"맛있", r"만족", r"가치있",
        r"착하", r"높", r"굿",
    ],
    "food_target": [
        r"맛", r"음식", r"메뉴", r"국물", r"고기", r"김치", r"면", r"반찬",
        r"퀄리티", r"풍미", r"향",
    ],
    "value_target": [r"가성비", r"가격", r"가격대비", r"가격에"],
    "value_positive_predicate": [r"좋", r"괜찮", r"착하", r"합리적", r"저렴", r"굿", r"갑", r"짱"],
    "atmosphere_target": [r"분위기", r"공간", r"인테리어", r"야장", r"뷰", r"플레이트"],
    "wait_target": [r"줄", r"웨이팅", r"대기", r"기다림"],
    "worth_predicate": [
        r"설만하", r"할만하", r"기다릴만하", r"기다려볼만하",
        r"가치(?:가)?있", r"아깝지않",
    ],
    "concession": [r"도", r"지만", r"인데도", r"그래도", r"하지만", r"해도"],
    "quantity_target": [r"양", r"고기", r"반찬", r"토핑", r"건더기"],
    "quantity_positive": [r"많", r"푸짐", r"넉넉", r"배부르"],
    "recommendation": [
        r"추천(?:합니다|드립|드려|해요|하고싶|할만|할게)",
        r"강추", r"꼭가보", r"꼭먹어보",
    ],
    "positive_idiom": [r"게눈감추듯사라", r"순삭", r"계속들어가", r"젓가락이멈추지않"],
}


def slot_pattern(slot_name):
    """의미 slot의 여러 표면형을 하나의 정규식 그룹으로 만듭니다."""
    return "(?:" + "|".join(PATTERN_SLOTS[slot_name]) + ")"


def build_generalized_rule(
    family,
    pattern,
    label,
    category,
    score,
    tag,
    suppress_terms=None,
    exclude_patterns=None,
):
    """일반화 규칙 메타데이터를 기존 phrase rule 형식으로 변환합니다."""
    return {
        "family": family,
        "pattern": pattern,
        "label": label,
        "category": category,
        "score": score,
        "tag": tag,
        "suppress_terms": set(suppress_terms or set()),
        "exclude_patterns": list(exclude_patterns or []),
        "generalized": True,
    }


GENERALIZED_PATTERN_RULES = [
    # 재방문: 시점/반복 부사 + 이용 행동 + 미래 의사
    build_generalized_rule(
        "strong_revisit_explicit",
        r"재방문(?:의사)?(?:은|는|도|가)?100",
        "재방문 의사 100",
        "general",
        3,
        "strong_revisit_positive",
        exclude_patterns=[r"재방문(?:의사)?(?:은|는|도|가)?(?:없|안|않)"],
    ),
    build_generalized_rule(
        "revisit_intent",
        rf"{slot_pattern('revisit_time')}.{{0,10}}{slot_pattern('revisit_action')}.{{0,8}}{slot_pattern('future_intent')}",
        "재방문 의사",
        "general",
        2,
        "revisit_positive_pattern",
        exclude_patterns=[
            r"(?:안|못|않).{0,5}(?:가|방문|먹|포장)",
            r"(?:냄비|남비|용기|통).{0,8}(?:들고|가져)",
        ],
    ),
    build_generalized_rule(
        "revisit_explicit",
        r"재방문(?:할)?(?:의사)?(?:은|는|도|가)?(?:ok|오케이|굿|o|있|예정|확정|100|할듯|하고싶)",
        "재방문 의사",
        "general",
        2,
        "revisit_positive_pattern",
        exclude_patterns=[r"재방문(?:할)?(?:의사)?(?:은|는|도|가)?(?:없|안|않|x)"],
    ),
    build_generalized_rule(
        "revisit_explicit_good",
        r"재방문의사.{0,10}(?:굿|좋|있|o)",
        "재방문 의사",
        "general",
        2,
        "revisit_positive_pattern",
        exclude_patterns=[r"재방문의사.{0,10}(?:없|안|않|x)"],
    ),
    build_generalized_rule(
        "worth_traveling",
        r"(?:굳이)?찾아서(?:라도)?올만(?:하|한|해)|찾아올만(?:하|한|해)",
        "찾아서 올 만하다",
        "general",
        1,
        "revisit_positive_pattern",
        exclude_patterns=[
            r"(?:찾아서(?:라도)?올만(?:하|한|해)|찾아올만(?:하|한|해)).{0,4}(?:않|아니|없|못)",
            r"(?:않|아니|없|못).{0,4}(?:찾아서(?:라도)?올만(?:하|한|해)|찾아올만(?:하|한|해))",
        ],
    ),
    build_generalized_rule(
        "repeat_visit",
        r"(?:[2-9]|[1-9][0-9]+)(?:번째|회차|회이상)방문",
        "반복 방문",
        "general",
        1,
        "repeat_visit_pattern",
    ),
    build_generalized_rule(
        "repeat_visit_nth",
        r"n번째방문",
        "반복 방문",
        "general",
        1,
        "repeat_visit_pattern",
    ),
    build_generalized_rule(
        "repeat_visit_count_action",
        r"(?:[2-9]|[1-9][0-9]+)(?:번|회)(?:정도는?)?.{0,6}(?:갔|가봤|가보|가고|가게|가야|왔|와봤|오지|오고|방문|먹으러|먹었|먹어)",
        "여러 번 이용하다",
        "general",
        1,
        "repeat_visit_pattern",
        exclude_patterns=[
            r"(?:[2-9]|[1-9][0-9]+)(?:번|회).{0,8}(?:안|못|않).{0,5}(?:가|오|방문|먹)",
            r"(?:[2-9]|[1-9][0-9]+)(?:번|회)(?:정도는?)?.{0,6}(?:갔|가봤|가보|가고|가게|가야|왔|와봤|오지|오고|방문|먹으러|먹었|먹어).{0,35}(?:별로|실망|후회|다신|이제안|안갑|않을|속상|아쉽|계속그러|연속불|문제)",
        ],
    ),
    build_generalized_rule(
        "habitual_visit",
        r"(?:(?:갈일있을때마다|올때마다|갈때마다).{0,10}(?:가는맛집|찾아|방문|먹으러|포장)|(?:자주|종종).{0,8}(?:갔|가봤|가보|가고|왔|와봤|방문|먹으러|포장))",
        "꾸준히 이용하다",
        "general",
        1,
        "repeat_visit_pattern",
        exclude_patterns=[
            r"(?:자주|종종).{0,8}(?:리뷰|후기|검색|구경)",
            r"(?:(?:갈일있을때마다|올때마다|갈때마다)|(?:자주|종종).{0,8}(?:갔|가봤|가보|가고|왔|와봤|방문|먹으러|포장)).{0,45}(?:이제안|다신|안갑|못가|실망|속상|아쉽|별로|이해안|후회|잃|나빠|떨어|예전같지|길어지)",
        ],
    ),
    build_generalized_rule(
        "craving_memory",
        r"(?:집가는길|먹고나서|돌아가는길)?.{0,12}생각(?:이)?(?:나|났|날)",
        "먹고 나서 생각이 나다",
        "general",
        1,
        "craving_memory_pattern",
        exclude_patterns=[r"생각(?:이)?나지않"],
    ),
    build_generalized_rule(
        "takeout_intent",
        r"(?:다음에|나중에).{0,10}(?:냄비|남비|용기|통).{0,8}(?:들고|가져).{0,6}(?:가|오|야겠)",
        "다음에 용기를 들고 가다",
        "general",
        1,
        "takeout_intention_pattern",
    ),

    # 추천/기다릴 가치: 대기 대상 + 가치 판단, 대기 양보 뒤 재방문
    build_generalized_rule(
        "recommendation",
        rf"{slot_pattern('recommendation')}",
        "추천",
        "general",
        1,
        "recommendation_positive_pattern",
        suppress_terms={"추천"},
        exclude_patterns=[r"(?:비|안|못)추천", r"추천(?:하지않|안하|못하)"],
    ),
    build_generalized_rule(
        "menu_recommendation",
        r"무조건.{0,14}(?:먹|시키|주문|가보|ㄱㄱ)",
        "무조건 메뉴 추천",
        "general",
        1,
        "menu_recommendation_pattern",
        exclude_patterns=[r"무조건.{0,8}(?:먹지말|시키지말|주문하지말|가지말)"],
    ),
    build_generalized_rule(
        "menu_must_try",
        r"(?:공기밥|볶음밥|밥|찌개|된장찌개|김치찌개|사리(?:추가)?|면사리|소스|반찬)"
        r"(?:은|는|이|가)?필수",
        "메뉴는 필수",
        "food",
        1,
        "menu_must_try_pattern",
        exclude_patterns=[
            r"(?:예약|웨이팅|대기|주의|개선|확인|주문|방문).{0,6}필수",
            r"필수(?:는|가)?(?:아니|아닌|않)",
        ],
    ),
    build_generalized_rule(
        "menu_omission_positive_idiom",
        r"(?:공기밥|볶음밥|밥|찌개|된장찌개|김치찌개|사리(?:추가)?|면사리|소스|반찬)"
        r".{0,10}(?:빠트리|빼먹|빼면|안먹|안시키|빠지면).{0,8}서운",
        "빠트리면 서운한 메뉴",
        "food",
        1,
        "menu_omission_positive_pattern",
        suppress_terms={"서운하다"},
    ),
    build_generalized_rule(
        "worth_waiting",
        rf"{slot_pattern('wait_target')}.{{0,24}}{slot_pattern('worth_predicate')}",
        "기다릴 가치가 있다",
        "general",
        1,
        "worth_waiting_pattern",
        exclude_patterns=[r"(?:가치없|할만하지않|아깝)"],
    ),
    build_generalized_rule(
        "food_drink_craving",
        r"한입먹는순간.{0,18}(?:바로)?(?:소주|술).{0,10}(?:한병|시키|주문|생각)",
        "한 입 먹고 술을 주문하다",
        "food",
        1,
        "food_drink_craving_pattern",
        exclude_patterns=[r"한입먹는순간.{0,20}(?:별로|실망|토할|못먹)"],
    ),
    build_generalized_rule(
        "waiting_revisit",
        rf"{slot_pattern('wait_target')}(?:(?:도|해도)|.{{0,12}}(?:지만|그래도|하지만|(?:했|하|였|이었)?는데도|해도)).{{0,20}}{slot_pattern('revisit_time')}.{{0,8}}{slot_pattern('revisit_action')}",
        "기다렸어도 다시 이용하다",
        "general",
        2,
        "waiting_but_revisit_positive",
    ),
    build_generalized_rule(
        "waiting_speed",
        r"(?:회전율|회전률|줄|대기|웨이팅).{0,16}(?:빠르|빨리줄|빨리빠지|금방빠지|금방들어가|금방입장|좋.{0,6}금방)",
        "대기가 금방 줄다",
        "general",
        1,
        "wait_turnover_positive_pattern",
        suppress_terms={"빠르다", "길다"},
    ),
    build_generalized_rule(
        "line_popularity",
        r"(?:괜히.{0,5}줄.{0,8}(?:길|서)|줄.{0,8}(?:긴|서는)이유.{0,5}있)",
        "줄이 긴 이유가 있다",
        "general",
        1,
        "popularity_positive_pattern",
        suppress_terms={"길다"},
    ),
    # 양보절의 단점 자체는 guard가 중립화하고, 뒤의 명확한 긍정 결론은
    # 별도 약한 evidence로 보존합니다. 이는 특정 표현이 아니라 담화 구조입니다.
    build_generalized_rule(
        "concession_positive_resolution_evidence",
        r"(?:(?:비싸|웨이팅|대기)(?:도|해도)|(?:투박|불편)(?:해도)|(?:가격.{0,8}사악|비싸|웨이팅|대기|줄.{0,6}길|투박|기복|불편).{0,14}(?:지만|그래도|하지만|(?:했|하|였|이었)?는데도|해도)).{0,28}(?:맛있|만족|가치있|최고|또가|다시가|괜찮|좋|있을맛.{0,6}다있)",
        "단점을 넘어선 긍정 결론",
        "general",
        1,
        "concession_positive_resolution_evidence",
        exclude_patterns=[
            r"(?:맛있|만족|괜찮|좋).{0,10}(?:지않|진않|않|아니|별로|실망|맛없)",
            r"(?:맛있|만족|괜찮|좋).{0,35}(?:하지만|그런데|다만).{0,25}(?:별로|실망|나쁘|불친절|문제|아쉽)",
        ],
    ),
    # 타인의 부정 평가를 명시적으로 반박하는 태도는 식당에 대한 약한 긍정
    # 지지로 해석하되, 원래 부정 토큰은 context guard에서 0점 처리합니다.
    build_generalized_rule(
        "negative_opinion_rebuttal_evidence",
        r"(?:별로|맛없|불친절).{0,12}(?:라는|라고하|라던|다는).{0,10}(?:사람|분|리뷰|평).{0,22}(?:맛을모르|모르|아니|틀리|좋|맛있|괜찮)",
        "타인 부정 의견을 반박하다",
        "general",
        1,
        "negative_opinion_rebuttal_evidence",
    ),

    # 맛/품질: 평가 대상과 긍정 서술어의 결합
    build_generalized_rule(
        "food_quality",
        rf"{slot_pattern('food_target')}(?:이|가|은|는|도)?.{{0,8}}{slot_pattern('positive_predicate')}",
        "음식 품질 긍정",
        "food",
        1,
        "food_quality_positive_pattern",
        suppress_terms={"좋다", "괜찮다", "맛있다", "맛있다 먹다", "맛있게 먹다"},
        exclude_patterns=[
            r"(?:좋|괜찮|훌륭|맛있).{0,4}(?:지않|진않|않|없)",
            r"맛있다는(?:리뷰|후기|말)",
            r"(?:좋|괜찮|훌륭|맛있).{0,10}(?:줄알|것같았|기대했).{0,12}(?:별로|아니|실망|맛없)",
        ],
    ),
    build_generalized_rule(
        "colloquial_tasty",
        r"(?:맛난|맛나요|맛나영|마싯(?:어|어요|던데|던|네|다|음)?)",
        "구어체 맛 표현",
        "food",
        1,
        "colloquial_tasty_positive_pattern",
        exclude_patterns=[
            r"(?:탄|쓴|비린|누린|고약한|이상한)맛(?:나요|나영)",
            r"(?:맛난|맛나요|맛나영|마싯(?:어|어요|던데|던|네|다|음)?).{0,20}"
            r"(?:줄알|별로|실망|맛없|아니|않)",
            r"(?:이게|이걸).{0,10}(?:맛난|맛나요|맛나영|마싯).{0,8}(?:다고|\?)",
        ],
    ),
    build_generalized_rule(
        "food_value_jjangjjang",
        r"(?:맛|음식|고기|반찬|구성|가성비|가성베|서비스|분위기).{0,18}짱짱",
        "구성과 맛이 짱짱하다",
        "food",
        1,
        "food_quality_positive_pattern",
        exclude_patterns=[r"짱짱.{0,12}(?:아니|않|별로|아쉽|실망)"],
    ),
    build_generalized_rule(
        "service_speed_raw_phrase",
        r"음식(?:이|은|도)?.{0,8}(?:엄청|정말|아주)?(?:빨리|빠르게|금방).{0,5}(?:나오|나옴|나와)",
        "음식이 빨리 나오다",
        "service",
        1,
        "speed_service_positive_pattern",
    ),
    build_generalized_rule(
        "well_made_food",
        r"(?:(?:음식|요리|고기|면|냉면|평양냉면|메뉴).{0,12}(?:정말|아주|잘)?잘만들|"
        r"(?:정말|아주|잘)?잘만들.{0,12}(?:음식|요리|고기|면|냉면|평양냉면|메뉴))",
        "음식을 잘 만들다",
        "food",
        1,
        "well_made_food_positive_pattern",
        exclude_patterns=[r"잘만들.{0,12}(?:아니|않|못)"],
    ),
    build_generalized_rule(
        "taste_realization",
        r"(?:평냉의)?맛.{0,5}깨달",
        "맛을 깨닫다",
        "food",
        1,
        "taste_realization_positive_pattern",
    ),
    build_generalized_rule(
        "comfortable_meal",
        r"속편한식사",
        "속 편한 식사",
        "food",
        1,
        "comfortable_meal_positive_pattern",
    ),
    build_generalized_rule(
        "clear_positive_acknowledgement",
        r"킹정(?:이지)?인정|킹정이지",
        "확실히 인정하다",
        "general",
        1,
        "positive_acknowledgement_pattern",
    ),
    build_generalized_rule(
        "upper_tier_evaluation",
        r"(?:여기|이정도|여기정도).{0,8}(?:상급|상위)",
        "상급 평가",
        "general",
        1,
        "upper_tier_positive_pattern",
        exclude_patterns=[r"(?:상급|상위).{0,8}(?:아니|않)"],
    ),
    build_generalized_rule(
        "food_acceptance",
        r"(?:있을맛(?:은)?다있|먹을만하|기본(?:은|도)하|나쁘지않|실패할수없(?:는)?맛)",
        "기본 이상의 맛",
        "food",
        1,
        "food_quality_acceptance_pattern",
        suppress_terms={"나쁘다"},
    ),
    build_generalized_rule(
        "flavor_harmony",
        r"(?:맛|향|풍미|구수함|매운맛).{0,12}(?:조화|밸런스|중화|살리|덜어내)",
        "맛의 조화와 균형",
        "food",
        1,
        "flavor_balance_positive_pattern",
        suppress_terms={"맵다", "느끼하다"},
    ),
    build_generalized_rule(
        "food_trust",
        r"(?:믿고|안심하고).{0,6}먹",
        "믿고 먹다",
        "food",
        2,
        "trust_positive_pattern",
    ),
    build_generalized_rule(
        "rank_tier",
        r"(?:[가-힣A-Za-z0-9]+)?1티어",
        "1티어",
        "food",
        2,
        "rank_tier_positive_pattern",
    ),

    # 가격/가성비/양: 대상 + 긍정 속성, 가격 단점 뒤 만족 양보
    build_generalized_rule(
        "price_costs_a_bit",
        r"가격대(?:가|는|도)?.{0,8}(?:좀|조금)?나가",
        "가격대가 조금 나가다",
        "price",
        -1,
        "price_cost_negative_pattern",
    ),
    build_generalized_rule(
        "value_positive",
        rf"{slot_pattern('value_target')}(?:이|가|은|는|도)?.{{0,8}}{slot_pattern('value_positive_predicate')}",
        "가격과 가성비 긍정",
        "price",
        1,
        "value_positive_pattern",
        suppress_terms={"좋다", "괜찮다", "가성비 좋다", "가격 괜찮다"},
        exclude_patterns=[r"(?:좋|괜찮|착하).{0,4}(?:지않|진않|않)"],
    ),
    build_generalized_rule(
        "expensive_concession",
        r"(?:(?:비싸|가격(?:은|이)?.{0,8}사악)(?:도|해도)|(?:비싸|가격(?:은|이)?.{0,8}사악).{0,10}(?:지만|그래도|하지만|(?:했|하|였|이었)?는데도|해도)).{0,20}(?:맛있|만족|가치있|재방문)",
        "가격은 높아도 만족",
        "price",
        1,
        "expensive_but_positive_pattern",
        suppress_terms={"비싸다", "사악하다", "가격 사악하다"},
    ),
    build_generalized_rule(
        "large_portion",
        rf"{slot_pattern('quantity_target')}(?:이|가|은|는|도)?.{{0,8}}{slot_pattern('quantity_positive')}|(?:다|전부|둘이서|2인이서).{{0,5}}못먹",
        "양이 넉넉하다",
        "food",
        1,
        "large_portion_pattern",
        suppress_terms={"많다"},
        exclude_patterns=[r"(?:맛없|상했|비려|느끼|매워).{0,12}못먹"],
    ),
    build_generalized_rule(
        "unlimited_refill",
        r"(?:[가-힣]+)?무한리필",
        "무한리필",
        "price",
        1,
        "value_positive_pattern",
    ),

    # 분위기/경험과 부정어 형태의 편의성 긍정
    build_generalized_rule(
        "atmosphere_positive",
        rf"{slot_pattern('atmosphere_target')}(?:이|가|은|는|도)?.{{0,8}}(?:굿|좋|괜찮|예쁘|넘예|낭만)",
        "분위기와 경험 긍정",
        "atmosphere",
        1,
        "atmosphere_positive_pattern",
        suppress_terms={"좋다", "괜찮다"},
        exclude_patterns=[r"(?:좋|괜찮|예쁘).{0,4}(?:지않|진않|않)"],
    ),
    build_generalized_rule(
        "convenience_positive",
        r"(?:(?:굳이|차타고)?.{0,10}멀리.{0,8}(?:갈|가야할)필요없|멀리안가도(?:되|돼))",
        "멀리 갈 필요가 없다",
        "general",
        1,
        "convenience_positive_pattern",
        suppress_terms={"굳이", "필요없다", "없다", "안되다"},
    ),
    build_generalized_rule(
        "positive_food_idiom",
        rf"{slot_pattern('positive_idiom')}",
        "음식이 빠르게 사라지다",
        "food",
        1,
        "positive_food_idiom_pattern",
        suppress_terms={"사라지다", "없다"},
    ),

    # 명시적 점수는 텍스트에 쓰인 평가값 자체가 강한 근거입니다.
    build_generalized_rule(
        "explicit_total_rating",
        r"(?:총점|전체점수)(?:은|는|이|가)?(?:4\.[0-9]|5(?:\.0)?)",
        "명시적 총점 고득점",
        "general",
        3,
        "explicit_rating_positive_pattern",
    ),
    build_generalized_rule(
        "explicit_taste_rating",
        r"(?:맛|음식|퀄리티)(?:은|는|이|가)?(?:4\.[0-9]|5(?:\.0)?)",
        "명시적 맛 고득점",
        "food",
        2,
        "explicit_taste_rating_positive_pattern",
    ),
    build_generalized_rule(
        "explicit_hundred_rating",
        r"(?:백점만점(?:에)?(?:9[0-9]|100)점?|(?:여기|총점|맛)?100점|만점)",
        "명시적 백점 고득점",
        "general",
        3,
        "explicit_rating_positive_pattern",
        exclude_patterns=[r"백점만점에(?:[0-8]?[0-9])점"],
    ),
]


def compact_review_for_patterns(review_text):
    """phrase pattern 탐지에서만 사용하는 보수적 원문 정규화."""
    return re.sub(
        r"[^0-9A-Za-z가-힣.]+",
        "",
        normalize_review_text(review_text),
    )


def spans_overlap(left, right):
    """두 phrase match가 같은 원문 구간을 설명하는지 확인."""
    return max(left[0], right[0]) < min(left[1], right[1])


TAG_SEMANTIC_FAMILIES = {
    # 같은 원문 구간의 구체 phrase와 일반 pattern이 이 family를 공유하면
    # 둘 중 더 강하고 구체적인 하나만 점수화합니다.
    "revisit_positive_phrase": "revisit",
    "revisit_intention_phrase": "revisit",
    "waiting_revisit_positive": "revisit",
    "revisit_positive_pattern": "revisit",
    "waiting_but_revisit_positive": "revisit",
    "repeat_visit_context": "repeat_visit",
    "repeat_visit_phrase": "repeat_visit",
    "repeat_visit_pattern": "repeat_visit",
    "craving_memory_phrase": "craving_memory",
    "craving_memory_pattern": "craving_memory",
    "takeout_intention_phrase": "takeout_intent",
    "takeout_intention_pattern": "takeout_intent",
    "worth_waiting_phrase": "worth_waiting",
    "worth_waiting_pattern": "worth_waiting",
    "recommendation_positive_phrase": "recommendation",
    "recommendation_positive_pattern": "recommendation",
    "menu_recommendation_phrase": "recommendation",
    "menu_recommendation_pattern": "recommendation",
    "wait_turnover_positive": "waiting_speed",
    "short_wait_positive": "waiting_speed",
    "wait_turnover_positive_pattern": "waiting_speed",
    "popularity_positive_context": "line_popularity",
    "popularity_positive_pattern": "line_popularity",
    "food_positive_phrase": "food_quality",
    "quality_positive_phrase": "food_quality",
    "food_quality_positive_pattern": "food_quality",
    "complete_flavor_positive": "food_acceptance",
    "food_quality_acceptance_pattern": "food_acceptance",
    "flavor_harmony_positive": "flavor_harmony",
    "balanced_spicy_context": "flavor_harmony",
    "flavor_balance_positive_pattern": "flavor_harmony",
    "trust_positive_phrase": "food_trust",
    "trust_positive_pattern": "food_trust",
    "rank_tier_positive": "rank_tier",
    "rank_tier_positive_pattern": "rank_tier",
    "price_positive_phrase": "value_positive",
    "strong_positive_phrase": "value_positive",
    "value_positive_pattern": "value_positive",
    "value_positive_phrase": "unlimited_refill",
    "large_portion_context": "large_portion",
    "large_portion_pattern": "large_portion",
    "atmosphere_positive_phrase": "atmosphere_positive",
    "atmosphere_positive_pattern": "atmosphere_positive",
    "convenience_positive_context": "convenience_positive",
    "convenience_positive_pattern": "convenience_positive",
    "positive_idiom": "positive_food_idiom",
    "positive_food_idiom_pattern": "positive_food_idiom",
    "explicit_rating_positive": "explicit_total_rating",
    "explicit_rating_positive_pattern": "explicit_total_rating",
    "explicit_taste_rating_positive": "explicit_taste_rating",
    "explicit_taste_rating_positive_pattern": "explicit_taste_rating",
    "expensive_but_tasty_context": "expensive_concession",
}

GENERALIZED_FAMILY_ALIASES = {
    "strong_revisit_explicit": "revisit",
    "revisit_intent": "revisit",
    "revisit_explicit": "revisit",
    "waiting_revisit": "revisit",
    "repeat_visit_count_action": "repeat_visit",
    "habitual_visit": "repeat_visit",
    "menu_recommendation": "recommendation",
}


def semantic_family_for_rule(rule):
    """일반 pattern과 기존 고확신 phrase를 공통 의미 family로 연결."""
    if rule.get("generalized") and rule.get("family"):
        return GENERALIZED_FAMILY_ALIASES.get(rule["family"], rule["family"])
    return TAG_SEMANTIC_FAMILIES.get(rule.get("tag"), rule.get("family", rule.get("tag", "")))


def detect_explicit_phrase_rules(review_text):
    """
    명확한 원문 phrase와 의미 slot 기반 일반화 pattern을 탐지합니다.

    같은 원문 구간에서 일반/구체 규칙이 겹치면 점수가 큰 규칙 하나만 남겨
    pattern layer 자체의 중복 가산을 방지합니다. 서로 떨어진 재방문/맛/가격
    근거는 독립 evidence이므로 각각 보존합니다.
    """
    compact_text = compact_review_for_patterns(review_text)
    candidates = []

    for rule in GENERALIZED_PATTERN_RULES + PHRASE_RULES:
        excluded = any(
            re.search(pattern, compact_text, flags=re.IGNORECASE)
            for pattern in rule.get("exclude_patterns", [])
        )

        if excluded:
            continue

        match = re.search(rule["pattern"], compact_text, flags=re.IGNORECASE)
        if not match:
            continue

        candidate = dict(rule)
        candidate["_match_span"] = match.span()
        candidate["_matched_surface"] = match.group(0)
        candidate["_semantic_family"] = semantic_family_for_rule(candidate)
        candidates.append(candidate)

    # 구체적이고 점수가 큰 패턴을 먼저 선택합니다.
    candidates.sort(
        key=lambda rule: (
            abs(rule["score"]),
            rule["_match_span"][1] - rule["_match_span"][0],
            not rule.get("generalized", False),
        ),
        reverse=True,
    )
    selected = []

    for candidate in candidates:
        duplicate = False
        for existing in selected:
            if not spans_overlap(candidate["_match_span"], existing["_match_span"]):
                continue

            same_semantic_signal = (
                candidate["_semantic_family"] == existing["_semantic_family"]
                or candidate["tag"] == existing["tag"]
                or (
                    candidate["category"] == existing["category"]
                    and candidate["score"] * existing["score"] > 0
                    and (
                        candidate["label"] in existing["label"]
                        or existing["label"] in candidate["label"]
                    )
                )
            )
            if same_semantic_signal:
                duplicate = True
                break

        if not duplicate:
            selected.append(candidate)

    return selected


def evidence_strength(score):
    """점수에서 evidence 강도를 일관되게 계산합니다."""
    if score >= 2:
        return "strong_positive"
    if score == 1:
        return "weak_positive"
    if score <= -2:
        return "strong_negative"
    if score == -1:
        return "weak_negative"
    return "neutralized"


def make_evidence_item(
    term,
    score,
    category,
    source,
    tag,
    original_score=None,
    **extra,
):
    """
    token/phrase/context/floor 결과를 하나의 표준 evidence item으로 만듭니다.

    최종 category score와 matched_words는 이 item 목록만 사용합니다.
    """
    item = {
        "term": str(term),
        "score": score,
        "category": category,
        "source": source,
        "tag": tag,
        "strength": evidence_strength(score),
        "original_score": score if original_score is None else original_score,
    }
    item.update(extra)
    return item


def apply_phrase_pattern_rules(review_text, tokens_fixed):
    """
    원문에서 명확한 phrase만 탐지해 실제 점수 후보로 반환합니다.

    tokens_fixed는 향후 token/phrase 위치 연계를 위한 인자로 유지하지만,
    이 단계에서는 넓은 문맥 추정 없이 원문의 명시적 정규식만 사용합니다.
    """
    del tokens_fixed
    matches = []

    for rule in detect_explicit_phrase_rules(review_text):
        matches.append(make_evidence_item(
            term=rule["label"],
            score=rule["score"],
            category=rule["category"],
            source="phrase_rule",
            tag=rule["tag"],
            suppress_terms=set(rule.get("suppress_terms", set())),
            semantic_family=rule.get("_semantic_family", semantic_family_for_rule(rule)),
            start_idx=None,
            end_idx=None,
        ))

    return matches


def apply_phrase_rules(review_text, tokens_fixed):
    """이전 함수명과의 호환용 별칭."""
    return apply_phrase_pattern_rules(review_text, tokens_fixed)


def phrase_match_text(label, score, tag):
    score_text = f"+{score}" if score > 0 else str(score)
    return f"{label}:{score_text}[{tag}]"


def has_revised_by_later_positive_context(review_text):
    """
    앞의 평가를 명시적으로 수정한 뒤 강한 긍정 결론이 나오는 경우만 확인.

    리뷰 전체의 긍정을 넓게 추정하지 않고, '찾기 힘들다' 같은 모호한 감점이
    '수정/그래도/하지만' 이후의 명확한 긍정 결론보다 앞에 있을 때만 사용합니다.
    """
    compact_text = re.sub(r"\s+", "", str(review_text))
    difficulty_positions = [
        compact_text.find(pattern)
        for pattern in ["찾기힘", "찾기어렵"]
        if compact_text.find(pattern) != -1
    ]

    if not difficulty_positions:
        return False

    difficulty_idx = min(difficulty_positions)
    revision_boundaries = [
        "수정", "생각이짧았", "그래도", "하지만", "그렇지만",
        "기복이있지만", "결론적으로", "결국",
    ]
    strong_late_positive_patterns = [
        "서울최고", "최고김치찌개집", "최고보쌈집", "최고냉면집",
        "인생맛집", "후회없는선택", "강추", "재방문OK",
        "맛좋", "가성비좋", "가성비굿",
    ]

    for boundary in revision_boundaries:
        boundary_idx = compact_text.find(boundary, difficulty_idx + 1)

        if boundary_idx == -1:
            continue

        tail = compact_text[boundary_idx + len(boundary):]
        if any(pattern in tail for pattern in strong_late_positive_patterns):
            return True

    return False


GENERAL_CONTEXT_GUARD_PATTERNS = {
    # 부정 표현이 리뷰어의 결론이 아니라 타인의 평가 인용/반박인 구조.
    "other_people_negative_opinion": [
        r"(?:별로|맛없|불친절).{0,8}(?:라는|라고하|라던|다는).{0,8}(?:사람|분|리뷰|평).{0,18}(?:모르|아니|틀리|이해못)",
        r"(?:사람|분|리뷰).{0,8}(?:별로|맛없).{0,18}(?:나는|전|저는).{0,10}(?:좋|맛있|괜찮)",
        r"(?:별로|맛없).{0,8}(?:라던데|라고했지만).{0,12}(?:나는|전|저는).{0,10}(?:좋|맛있|괜찮)",
    ],
    # 부정 평가 자체를 부인하거나 유보하는 구조.
    "negated_negative_evaluation": [
        r"(?:불친절|별로|맛없|나쁘).{0,10}(?:모르겠|아니|않|없진않|그렇진않)",
        r"(?:불친절|별로|맛없|나쁘)(?:은|는|한건|지는)?.{0,5}아니",
    ],
    # 가격/대기/외형 단점이 양보 접속 뒤 명확한 긍정 결론으로 해소되는 구조.
    "concession_positive_resolution": [
        r"(?:(?:비싸|웨이팅|대기)(?:도|해도)|(?:투박|불편)(?:해도)|(?:가격.{0,8}사악|비싸|웨이팅|대기|줄.{0,6}길|투박|기복|불편).{0,14}(?:지만|그래도|하지만|(?:했|하|였|이었)?는데도|해도)).{0,28}(?:맛있|만족|가치있|최고|또가|다시가|괜찮|좋|있을맛.{0,6}다있)",
    ],
    "convenience_positive": [
        r"(?:(?:굳이|차타고)?.{0,10}멀리.{0,8}(?:갈|가야할)필요없|멀리안가도(?:되|돼))",
    ],
    "flavor_balance": [
        r"(?:맵|매운맛|매콤|느끼).{0,18}(?:중화|조화|밸런스|덜어내|물리지않|맛있|괜찮)",
        r"(?:중화|조화|밸런스).{0,18}(?:맵|매운맛|매콤|느끼)",
    ],
    "positive_food_idiom": [
        r"(?:게눈감추듯사라|순삭|계속들어가|젓가락이멈추지않)",
    ],
    "not_restaurant_negative": [
        r"(?:먹기싫|싫으면).{0,10}(?:딴데|다른데).{0,6}(?:가|가라)",
        r"(?:별로|맛없).{0,8}(?:라는|다고하는).{0,8}(?:사람|분)",
        r"그거.{0,8}(?:가지고|갖고).{0,6}난리",
    ],
    "waiting_positive_resolution": [
        r"(?:회전율|회전률|줄|대기|웨이팅).{0,16}(?:빠르|빨리줄|빨리빠지|금방빠지|금방들어가|금방입장|좋.{0,6}금방)",
        r"(?:괜히.{0,5}줄.{0,8}(?:길|서)|줄.{0,8}(?:긴|서는)이유.{0,5}있)",
    ],
}


def has_general_context_guard(review_text, guard_name):
    """일반화된 문맥 guard 패턴 중 하나가 원문에서 확인되는지 검사."""
    compact_text = compact_review_for_patterns(review_text)
    return any(
        re.search(pattern, compact_text, flags=re.IGNORECASE)
        for pattern in GENERAL_CONTEXT_GUARD_PATTERNS[guard_name]
    )


def adjust_score_by_context(
    term,
    score,
    category,
    tokens,
    start_idx,
    end_idx,
    review_text,
):
    """
    문맥 의존 감성어의 점수를 토큰 주변 문맥으로 보수적으로 조정.

    반환값:
    - adjusted_score: 최종 반영할 점수
    - tag: matched_words에 기록할 조정 이유. 조정이 없으면 None

    원칙:
    - 앞 5토큰, 뒤 8토큰만 확인
    - 확실한 문맥이면 기존 점수를 유지하거나 0점 처리
    - 사전 점수와 반대 극성이 보여도 곧바로 +/-를 뒤집지 않음
    """
    if not is_context_dependent_term(term):
        return score, None

    left = tokens[max(0, start_idx - 5):start_idx]
    right = tokens[end_idx:min(len(tokens), end_idx + 8)]
    direct_left = left[-2:]
    direct_right = right[:3]
    context = left + tokens[start_idx:end_idx] + right
    term_compact = str(term).replace(" ", "")
    review_compact = re.sub(r"\s+", "", str(review_text))

    def has(patterns):
        return token_window_has(context, patterns)

    def term_has(patterns):
        return any(str(pattern).replace(" ", "") in term_compact for pattern in patterns)

    def neutral(tag):
        return 0, tag

    if term_has(["힘들다"]) and has_revised_by_later_positive_context(review_text):
        return neutral("revised_by_later_positive_context")

    # 의미 구조 기반 공통 guard. 특정 예문이 아니라 부정 대상/양보 접속/
    # 후행 결론의 역할을 조합해 같은 구조의 새로운 리뷰에도 적용합니다.
    if score < 0 and term_has(["별로", "맛없다", "불친절"]):
        if has_general_context_guard(review_text, "other_people_negative_opinion"):
            return neutral("other_people_negative_opinion_pattern")

    if score < 0 and term_has(["불친절", "별로", "맛없다", "나쁘다"]):
        if has_general_context_guard(review_text, "negated_negative_evaluation"):
            return neutral("negated_negative_evaluation_pattern")

    if score < 0 and term_has(["비싸다", "길다", "투박하다", "웨이팅", "힘들다"]):
        if has_general_context_guard(review_text, "concession_positive_resolution"):
            return neutral("concession_positive_resolution_pattern")

    if score < 0 and term_compact in {"굳이", "필요없다", "없다", "안되다"}:
        if has_general_context_guard(review_text, "convenience_positive"):
            return neutral("convenience_positive_context_pattern")

    if score < 0 and term_has(["맵다", "매콤하다", "느끼하다"]):
        if has_general_context_guard(review_text, "flavor_balance"):
            return neutral("flavor_balance_context_pattern")

    if score < 0 and term_compact in {"사라지다", "없다", "멈추다"}:
        if has_general_context_guard(review_text, "positive_food_idiom"):
            return neutral("positive_food_idiom_context_pattern")

    if score < 0 and term_has(["싫다", "별로", "맛없다", "이상하다"]):
        if has_general_context_guard(review_text, "not_restaurant_negative"):
            return neutral("not_restaurant_negative_pattern")

    if term_has(["빠르다", "금방", "길다"]) and has_general_context_guard(
        review_text,
        "waiting_positive_resolution",
    ):
        if score < 0:
            return neutral("waiting_positive_resolution_pattern")
        return score, "wait_turnover_positive_pattern"

    # 메뉴 사이즈의 소짜/중짜/대짜가 짠맛 '짜다'로 깨진 경우.
    strong_salty_context = any(
        pattern in review_compact
        for pattern in ["너무짜", "간이짜", "짜서", "짠맛", "짜게"]
    )
    if (
        term_has(["짜다"])
        and re.search(r"(?:소짜|중짜|대짜)", review_compact)
        and not strong_salty_context
    ):
        return neutral("size_word_not_taste")

    # 소스가 매운맛/느끼함을 중화하는 명확한 균형 설명 문맥.
    strong_spicy_negative = any(
        pattern in review_compact
        for pattern in ["너무맵", "매워서혼", "못먹", "먹기힘들", "매워죽"]
    )
    sauce_balance_context = (
        any(pattern in review_compact for pattern in ["느끼한소스", "크리미"])
        and any(pattern in review_compact for pattern in ["찍어먹", "중화", "물리지도않", "물리지않"])
    )
    balanced_spicy_context = (
        any(pattern in review_compact for pattern in ["중화", "조화", "매콤하게", "조금매콤", "약간매콤"])
        and any(pattern in review_compact for pattern in ["소스", "맛있", "좋"])
    )

    if term_has(["느끼하다"]) and sauce_balance_context:
        return neutral("sauce_balance_context")
    if term_has(["맵다", "매콤하다"]) and balanced_spicy_context and not strong_spicy_negative:
        return neutral("balanced_spicy_context")

    # 타인의 부정 평가를 반박하는 표현은 식당 자체에 대한 부정이 아닙니다.
    if term_has(["별로"]) and any(
        pattern in review_compact
        for pattern in [
            "별로라는사람", "별로라고하는사람", "별로라하는사람",
            "별로라는분", "맛을모르는", "맛모르는",
        ]
    ):
        return neutral("other_people_negative_opinion")

    # 불친절하다는 평가 자체를 부인하거나 유보하는 문맥.
    if term_has(["불친절"]) and any(
        pattern in review_compact
        for pattern in ["불친절은모르겠", "불친절한지는모르겠", "불친절모르겠"]
    ):
        return neutral("negated_negative_service")

    # 투박함이 양보절이고 뒤에 명확한 맛 긍정 결론이 있는 경우.
    if term_has(["투박하다"]) and (
        "투박하지만있을맛은다있" in review_compact
        or "투박해도있을맛은다있" in review_compact
    ):
        return neutral("rough_but_complete_flavor")

    if term_has(["길다"]) and any(
        pattern in review_compact
        for pattern in ["괜히줄길게설까", "줄긴이유가있", "줄이긴이유가있"]
    ):
        return neutral("popularity_positive_context")

    if term_has(["비싸다"]) and any(
        pattern in review_compact
        for pattern in ["비싸도맛있", "비싸지만맛있"]
    ):
        return neutral("expensive_but_tasty_context")

    # 혼잡/소음을 식당의 단점으로 비판하는 것이 아니라 옹호하는 명확한 문맥.
    defensive_context = any(
        pattern in review_compact
        for pattern in [
            "감안하세요", "감안해", "그럴수도있지", "혼잡스러울수도있지",
            "그거가지고난리", "먹기싫음딴데가라", "먹기싫으면딴데가라",
            "가격괜찮고맛좋",
        ]
    )
    strong_noise_complaint = any(
        pattern in review_compact
        for pattern in ["소음이너무심", "목소리가지나치게크", "불편을주", "울렁거릴정도"]
    )

    if defensive_context and not strong_noise_complaint:
        if term_has(["시끄럽다"]):
            return neutral("defensive_noise_context")
        if term_has(["혼잡", "붐비다", "사람많다"]):
            return neutral("defensive_crowded_context")
        if term_has(["싫다", "이상하다"]) and (
            "딴데가라" in review_compact
            or "한국사람들참이상" in review_compact
        ):
            return neutral("not_restaurant_negative")

    if term_has(["붐비다"]) and any(
        pattern in review_compact
        for pattern in ["붐비지않는시간", "붐비지않을시간", "붐비지않을때"]
    ):
        return neutral("crowd_avoidance_advice")

    # '왠만하면 글을 안 쓰는데'의 왠만하다/웬만하다는 품질 평가가 아닙니다.
    if term_has(["왠만하다", "웬만하다"]):
        return neutral("non_evaluative_expression")

    # 검색어/방문 전 기대는 실제 식당 품질 평가가 아닙니다.
    if term_has(["맛집"]) and has(["검색", "찾아보다", "후기", "맛집이라 해서", "맛집이라하다"]):
        return neutral("search_or_expectation_context")
    if term_has(["설레다"]) and token_window_has(right, ["가다", "방문", "찾아가다"]):
        return neutral("pre_visit_expectation_context")

    # 특별하지 않다/특별할 것 없다는 긍정으로 계산하지 않습니다.
    if term_has(["특별하다"]) and (
        term_contains_negation(term)
        or token_window_has(direct_left, ["안", "못"])
        or token_window_has(direct_right, ["않다", "아니다", "없다", "별것 없다", "건 없다", "것 없다"])
    ):
        return neutral("negated_special_context")

    # 사전에 이미 '빠르다 않다', '저렴하다 않다'처럼 긴 부정 표현이 있으면 유지.
    if score < 0 and term_contains_negation(term):
        return score, None

    # 크다: 공간/양의 크기는 긍정 가능, 목소리/소음 크기는 긍정 금지.
    if term_has(["크다"]):
        noise_context = has([
            "목소리", "소리", "소음", "시끄럽", "떠들", "울리다", "불편",
            "지나치다", "너무 크다",
        ])
        positive_size_context = has([
            "매장", "공간", "내부", "자리", "테이블", "양", "사이즈", "큼직",
        ])

        if noise_context:
            return neutral("noise_context")
        if positive_size_context:
            return score, "positive_size_context" if score > 0 else None
        return neutral("size_ambiguous_context")

    # 바쁘다/걱정은 식당 품질의 직접 평가가 아닌 경우가 많아 중립을 우선.
    if term_has(["바쁘다"]):
        direct_service_failure = has([
            "응대 늦", "응대 안", "응대 않", "주문 늦", "주문 누락",
            "직원 무시", "서비스 늦", "서비스 엉망", "안 오다", "오지 않다",
            "불친절", "정신없",
        ])
        popularity_or_indirect = has([
            "매장", "손님", "사람", "인기", "웨이팅", "대기", "많다",
            "먹을 수", "주말", "점심", "저녁",
        ])

        if direct_service_failure:
            return score, None
        if popularity_or_indirect or score < 0:
            return neutral("busy_indirect_context")

    if term_has(["걱정"]):
        direct_risk = has([
            "위생", "안전", "상하다", "상한", "배탈", "식중독", "건강", "재사용",
        ])
        if direct_risk:
            return score, None
        return neutral("indirect_concern_context")

    # 속도: 서비스 흐름의 속도만 긍정. 먹는 행동이나 급히 떠난 상황은 중립.
    if term_has(["빠르게", "빠르다", "빨리", "금방", "바로"]):
        hard_non_service_speed = has([
            "욱여넣", "급하게", "대충", "허겁지겁", "불쾌", "찝찝", "도망",
        ])
        eating_or_leaving_action = has(["먹다", "먹고", "마시다", "나가다", "양보"])
        service_target = has([
            "음식", "메뉴", "주문", "서빙", "응대", "입장", "착석",
            "자리", "웨이팅", "웨이", "대기", "회전율", "회전",
            "조리", "초벌", "구워", "직원",
        ])
        service_result = has([
            "나오다", "받다", "입장", "앉다", "착석", "처리", "서빙", "응대",
            "초벌", "구워",
        ])

        if hard_non_service_speed:
            return neutral("speed_not_service_context")
        if service_target and service_result:
            return score, "speed_service_context" if score > 0 else None
        if eating_or_leaving_action:
            return neutral("speed_not_service_context")
        return neutral("speed_ambiguous_context")

    # 가격: 만족/값어치가 뒤따르는 비쌈은 부정 완화. 싼 맛/싸구려는 긍정 금지.
    if term_has(["비싸다"]):
        if has(["만족", "만족스럽", "마음에 들", "맘에 들"]):
            return neutral("expensive_but_satisfied_context")
        if has(["값어치", "값을 하다", "값하다", "가치", "납득", "아깝지 않"]):
            return neutral("expensive_but_worth_context")
        return score, None

    if term_has(["싸다", "저렴하다"]):
        if has(["싼 맛", "싸구려", "싸서 그런지", "저렴해서 그런지", "질 낮", "저품질"]):
            return neutral("cheap_quality_risk_context")
        if has(["가격", "가성비", "합리적", "부담 없"]):
            return score, "affordable_price_context" if score > 0 else None
        return neutral("cheap_ambiguous_context")

    # 웨이팅/사람 많음: 가치가 명시되면 음수 금지, 불편이 명시되면 음수 유지.
    if term_has(["웨이팅", "웨이", "사람많다", "많다"]) and has([
        "웨이팅", "웨이", "대기", "기다리", "사람", "손님",
    ]):
        if score > 0 and term_has(["없이"]):
            return score, None
        if has(["길다", "오래", "한참", "정신없", "복잡", "불편", "힘들", "지치"]):
            return (score, None) if score < 0 else neutral("crowded_negative_context")
        if score < 0 and has(["할 만", "기다릴 만", "가치", "이유가 있다", "이유 있다", "납득", "맛있"]):
            return neutral("worth_waiting_context")
        return neutral("waiting_or_crowd_ambiguous_context")

    # 강도: 취향/장점으로 해소되면 부정 점수를 지우고, 간/냄새 불만은 유지.
    if term_has(["약하다", "강하다", "세다"]):
        positive_preference = has([
            "마음에 들", "맘에 들", "좋다", "취향", "선호", "매력",
            "조화", "불맛", "향", "풍미", "탄력",
        ])
        explicit_negative = has([
            "간", "냄새", "잡내", "너무", "과하다", "부담", "아쉽", "별로", "맛없",
        ])

        if positive_preference and not explicit_negative:
            return neutral("positive_preference_context") if score < 0 else (score, None)
        if explicit_negative:
            return (score, None) if score < 0 else neutral("intensity_negative_context")
        return neutral("intensity_ambiguous_context")

    # 양: 대상에 따라 많다/적다의 극성이 달라집니다.
    if term_has(["많다", "적다"]):
        good_quantity_target = has(["양", "고기", "반찬", "토핑", "서비스", "구성", "건더기"])
        bad_quantity_target = has(["사람", "손님", "소음", "대기", "웨이팅", "웨이", "기름", "냄새", "잡내"])

        if term_has(["많다"]):
            if good_quantity_target and not bad_quantity_target:
                return (score, None) if score > 0 else neutral("many_positive_target_context")
            if bad_quantity_target:
                return (score, None) if score < 0 else neutral("many_negative_target_context")
        else:
            if bad_quantity_target and not good_quantity_target:
                return (score, None) if score > 0 else neutral("little_positive_target_context")
            if good_quantity_target:
                return (score, None) if score < 0 else neutral("little_negative_target_context")

        return neutral("quantity_ambiguous_context")

    # 크기/두께: 음식 종류와 속성 대상을 함께 봅니다.
    if term_has(["작다", "얇다", "두껍다"]):
        easy_or_crispy = has(["전", "튀김", "면", "바삭", "먹기 편", "한입", "한 입"])
        size_complaint = has(["양", "사이즈", "크기", "가격 대비", "가격대비", "너무 작"])
        meat_target = has(["고기", "고깃", "육질"])
        batter_target = has(["튀김옷", "옷이 두껍", "반죽"])

        if term_has(["두껍다"]) and meat_target and not batter_target:
            return (score, None) if score > 0 else neutral("thick_meat_context")
        if term_has(["두껍다"]) and batter_target:
            return (score, None) if score < 0 else neutral("thick_batter_context")
        if easy_or_crispy and not size_complaint:
            return neutral("size_food_style_context") if score < 0 else (score, None)
        if size_complaint or (meat_target and term_has(["작다", "얇다"])):
            return (score, None) if score < 0 else neutral("size_negative_context")
        return neutral("size_ambiguous_context")

    # 온도: 냉메뉴의 차가움과 뜨거운 국물은 정상 특성. 과도해 먹기 힘들면 부정.
    if term_has(["차갑다", "뜨겁다"]):
        too_hot_or_cold = has(["너무", "지나치", "과하", "먹기 힘들", "입천장", "데다", "별로", "아쉽"])
        cold_target = has(["냉면", "평냉", "육수", "음료", "아이스", "시원", "막국수", "소바", "콩국수", "거냉"])
        hot_target = has(["국물", "찌개", "탕", "국밥", "전골", "뚝배기"])
        cold_service = category in {"service", "atmosphere"} or has(["응대", "직원", "서비스", "태도"])

        if too_hot_or_cold:
            return (score, None) if score < 0 else neutral("temperature_excess_context")
        if term_has(["차갑다"]) and cold_target and not cold_service:
            return neutral("cold_dish_temperature") if score < 0 else (score, None)
        if term_has(["뜨겁다"]) and hot_target:
            return neutral("hot_dish_temperature") if score < 0 else (score, None)
        if term_has(["차갑다"]) and cold_service:
            return (score, None) if score < 0 else neutral("cold_service_context")
        return neutral("temperature_ambiguous_context")

    # 맛 성향: 긍정 취향이면 음수를 지우고, 명시적 불만이면 음수를 유지.
    if term_has(["기름지다", "자극적이다", "슴슴하다", "심심하다"]):
        positive_taste = has([
            "고소", "매력", "중독", "좋다", "취향", "평냉", "한식", "깔끔",
            "담백", "개운", "조화", "마음에 들", "맘에 들",
        ])
        negative_taste = has(["너무", "부담", "느끼", "별로", "맛없", "아쉽", "과하"])

        if positive_taste and not negative_taste:
            return neutral("positive_taste_preference_context") if score < 0 else (score, None)
        if negative_taste:
            return (score, None) if score < 0 else neutral("negative_taste_context")
        return neutral("taste_style_ambiguous_context")

    # 평가 강도: 무난/평범은 결론 표현을 보고, 특별함은 부정되지 않았을 때만 긍정.
    if term_has(["무난하다", "평범하다", "특별하다", "깔끔하다"]):
        positive_result = has(["맛있", "괜찮", "좋다", "만족", "편안", "추천"])
        plain_or_negative = has(["그냥", "너무 평범", "별로", "아쉽", "특별할 것 없", "특별할 건 없"])

        if term_has(["특별하다"]):
            return (score, None) if score > 0 else neutral("special_positive_context")
        if term_has(["평범하다", "무난하다"]):
            if positive_result:
                return neutral("mild_positive_evaluation_context") if score < 0 else (score, None)
            if plain_or_negative and score < 0:
                return score, None
            return neutral("mild_evaluation_ambiguous_context")
        if term_has(["깔끔하다"]):
            if (
                token_window_has(direct_left, ["안", "못"])
                or token_window_has(direct_right, ["않다", "아니다"])
                or has(["별로 깔끔", "지저분", "불결"])
            ):
                return neutral("clean_negated_context")
            return score, None

    # 오래: 대기/서비스 지연은 부정, 가게가 오래 지속되길 바라는 표현은 중립.
    if term_has(["오래"]):
        if has(["기다리", "대기", "웨이팅", "웨이", "걸리다", "늦다"]):
            return (score, None) if score < 0 else neutral("long_wait_context")
        if has(["오래오래", "해주", "영업", "남아", "유지", "단골"]):
            return neutral("positive_longevity_context")
        return neutral("duration_ambiguous_context")

    return score, None


def derive_overall_sentiment_score(review_text, category_total_score):
    """
    세부 카테고리 합계와 별개로 리뷰의 전체 결론 점수를 계산.
    강한 긍정/부정 결론은 전체 점수의 최소/최대 수준을 결정합니다.
    """
    score = float(category_total_score)
    compact_text = str(review_text).replace(" ", "")

    strong_positive_patterns = [
        "맛있는집을드디어찾", "드디어맛있는집을찾",
        "드디어찾았다", "드디어찾았",
        "가치가있", "가치있",
        "재방문의사있", "꼭다시오", "또올게",
        "추천드", "인생맛집", "사랑임",
    ]

    strong_negative_patterns = [
        "다신안가", "두번다시안가", "재방문의사없",
        "비추천", "최악", "돈아깝", "아까운돈",
    ]

    positive_positions = [
        compact_text.rfind(pattern)
        for pattern in strong_positive_patterns
        if compact_text.rfind(pattern) != -1
    ]
    negative_positions = [
        compact_text.rfind(pattern)
        for pattern in strong_negative_patterns
        if compact_text.rfind(pattern) != -1
    ]

    last_positive_idx = max(positive_positions) if positive_positions else -1
    last_negative_idx = max(negative_positions) if negative_positions else -1

    reasons = []

    if last_positive_idx > last_negative_idx:
        score = max(score, 2.0)
        reasons.append("strong_positive_conclusion")
    elif last_negative_idx > last_positive_idx:
        score = min(score, -2.0)
        reasons.append("strong_negative_conclusion")

    # 양보절 뒤에 긍정 결론이 오면 전체 평가는 긍정 쪽으로 해석.
    for boundary in CONCESSIVE_BOUNDARIES:
        boundary_idx = compact_text.rfind(boundary)

        if boundary_idx == -1:
            continue

        tail = compact_text[boundary_idx + len(boundary):]

        if (
            "strong_negative_conclusion" not in reasons
            and last_negative_idx < boundary_idx
            and any(pattern in tail for pattern in POSITIVE_RESOLUTION_PATTERNS)
        ):
            score = max(score, 2.0)
            reasons.append("positive_concession_resolution")
            break

    return pd.Series({
        "overall_sentiment_score": score,
        "overall_context_reason": " | ".join(dict.fromkeys(reasons)),
    })


def sentiment_to_star(score, score_min, score_max):
    """
    category_total_score를 1~5점 별점으로 변환.
    카카오 리뷰의 category_total_score 5%~95% 분위수를 기준으로 정규화.
    """
    if pd.isna(score):
        return np.nan

    if score_max == score_min:
        return np.nan

    normalized = (score - score_min) / (score_max - score_min)
    normalized = max(0, min(1, normalized))
    star = 1 + normalized * 4

    return round(star, 2)


# =========================================================
# 8. 리뷰별 감성점수 계산
# =========================================================

def _legacy_analyze_review_v20(tokens_text, tokens_pos_text, review_text):
    raw_tokens = str(tokens_text).split()
    tokens = [clean_token(t) for t in raw_tokens if clean_token(t) != ""]
    tokens = normalize_analysis_tokens(tokens, review_text)
    pos_tokens = parse_pos_tokens(tokens_pos_text)

    if len(pos_tokens) == len(tokens):
        pos_tokens = [
            (tokens[i], pos_tokens[i][1])
            for i in range(len(tokens))
        ]
    else:
        pos_tokens = []

    scores = {cat: 0 for cat in categories}
    matched = {cat: [] for cat in categories}
    phrase_rules = detect_explicit_phrase_rules(review_text)
    phrase_suppressed_terms = set().union(
        *(rule["suppress_terms"] for rule in phrase_rules)
    ) if phrase_rules else set()

    matched_terms = set()
    used_token_idx = set()
    stronger_phrase_spans = []

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

            # 같은 표현은 리뷰 안에서 한 번만 기록/점수화.
            if term in matched_terms:
                continue

            # '너무 맵다'처럼 강한 n-gram이 이미 잡혔으면 '맵다' 단독은 억제.
            if n == 1 and is_suppressed_by_stronger_phrase(
                term,
                i,
                stronger_phrase_spans,
            ):
                for category, score in sentiment_map[term]:
                    if category in scores:
                        matched[category].append(
                            f"{term}:{score}->0[suppressed_by_stronger_phrase]"
                        )

                matched_terms.add(term)
                used_token_idx.update(idx_range)
                continue

            # 위험한 일반 동사는 단독 토큰일 때만 0점 차단.
            # 더 긴 복합 표현은 이미 앞선 n-gram 순회에서 정상 매칭됩니다.
            if n == 1 and term in RISKY_SINGLE_TOKENS:
                if term not in matched_terms:
                    for category, score in sentiment_map[term]:
                        if category in scores:
                            matched[category].append(
                                f"{term}:{score}->0[blocked_risky_single_token]"
                            )

                    matched_terms.add(term)
                    used_token_idx.update(idx_range)

                continue

            if not is_standalone_negative_term(review_text, term):
                continue

            if not is_contextually_valid_ambiguous_term(review_text, term):
                continue

            if should_decompose_negated_mixed_term(review_text, term):
                continue

            for category, score in sentiment_map[term]:
                if category in scores:
                    context_dependent_term = is_context_dependent_term(term)

                    neg_context_found = False
                    comparative_reference_found = False
                    comparative_context_found = False

                    direct_negative_absence_found = False
                    cold_dish_temperature_found = False
                    concessive_preference_found = False
                    resolved_positive_concession_found = False

                    # =================================================
                    # A. 긍정어 보정
                    # =================================================
                    if score > 0:
                        # 긍정어 바로 앞뒤에 직접 연결된 부정만 인정.
                        # 원문 앞뒤 35자 검색은 다른 대상의 부정을 전파하므로 사용하지 않음.
                        if (
                            not context_dependent_term
                            and not term_contains_negation(term)
                        ):
                            if has_direct_negative_context(review_text, term):
                                neg_context_found = True

                        # 감성어가 '바삭보다는'처럼 비교 기준으로만 쓰이면 중립 처리
                        if (
                            not context_dependent_term
                            and not neg_context_found
                            and has_direct_comparative_reference(review_text, term)
                        ):
                            comparative_reference_found = True

                        # 비교/반어/대상 전환 문맥 검사
                        if (
                            not context_dependent_term
                            and not neg_context_found
                            and not comparative_reference_found
                        ):
                            if has_comparative_or_ironic_context(
                                review_text=review_text,
                                keyword=term,
                                tokens=tokens,
                                start_i=i,
                                end_i=i + n,
                                window=55
                            ):
                                comparative_context_found = True

                    # =================================================
                    # B. 부정어 보정
                    # =================================================
                    if score < 0:
                        # 1. 부정어가 직접 부정되면 해당 음수 점수만 무효화
                        # 예: 질기다 않다, 잡내 없다
                        if (
                            (
                                not context_dependent_term
                                and has_direct_negative_context(review_text, term)
                            )
                            or has_direct_absence_of_negative_token_context(
                                tokens=tokens,
                                start_i=i,
                                end_i=i + n,
                            )
                        ):
                            direct_negative_absence_found = True

                        # 2. 냉면류의 정상적인 차가운 온도는 중립 처리
                        if (
                            not context_dependent_term
                            and not direct_negative_absence_found
                            and is_cold_dish_temperature_context(review_text, term)
                        ):
                            cold_dish_temperature_found = True

                        # 3. 원래 취향을 설명하는 양보절의 '별로'는 중립 처리
                        if (
                            not direct_negative_absence_found
                            and not cold_dish_temperature_found
                            and has_concessive_preference_context(review_text, term)
                        ):
                            concessive_preference_found = True

                        # 4. general 부정이 양보절 뒤 긍정 결론으로 해소되면 중립 처리
                        if (
                            not context_dependent_term
                            and category == "general"
                            and not direct_negative_absence_found
                            and not cold_dish_temperature_found
                            and not concessive_preference_found
                            and has_resolved_positive_concession(review_text, term)
                        ):
                            resolved_positive_concession_found = True

                        # 주변의 긍정어만으로 음수 점수를 지우지 않습니다.
                        # 예: '맛있지 않았다 ... 도움이 됐으면 좋겠습니다'에서
                        # 뒤쪽 '좋겠습니다'가 '맛있지 않았다'를 무효화하면 안 됩니다.

                    # =================================================
                    # C. 최종 점수 반영
                    # =================================================
                    context_adjusted_score, context_tag = adjust_score_by_context(
                        term=term,
                        score=score,
                        category=category,
                        tokens=tokens,
                        start_idx=i,
                        end_idx=i + n,
                        review_text=review_text,
                    )

                    if term in phrase_suppressed_terms:
                        adjusted_score = 0
                        matched[category].append(
                            f"{term}:{score}->0[suppressed_by_phrase_rule]"
                        )

                    elif context_tag:
                        adjusted_score = context_adjusted_score
                        matched[category].append(
                            adjusted_match_text(
                                term,
                                score,
                                adjusted_score,
                                context_tag,
                            )
                        )

                    elif score > 0 and neg_context_found:
                        adjusted_score = 0
                        matched[category].append(f"{term}:{score}->0[direct_neg_context]")

                    elif score > 0 and comparative_reference_found:
                        adjusted_score = 0
                        matched[category].append(f"{term}:{score}->0[comparative_reference]")

                    elif score > 0 and comparative_context_found:
                        adjusted_score = 0
                        matched[category].append(f"{term}:{score}->0[comparative_context]")

                    elif score < 0 and direct_negative_absence_found:
                        adjusted_score = 0
                        matched[category].append(f"{term}:{score}->0[direct_negative_absence]")

                    elif score < 0 and cold_dish_temperature_found:
                        adjusted_score = 0
                        matched[category].append(f"{term}:{score}->0[cold_dish_temperature]")

                    elif score < 0 and concessive_preference_found:
                        adjusted_score = 0
                        matched[category].append(f"{term}:{score}->0[concessive_preference]")

                    elif score < 0 and resolved_positive_concession_found:
                        adjusted_score = 0
                        matched[category].append(f"{term}:{score}->0[resolved_positive_concession]")

                    else:
                        adjusted_score = score
                        matched[category].append(f"{term}:{score}[token]")

                    scores[category] += adjusted_score

            matched_terms.add(term)
            used_token_idx.update(idx_range)

            term_parts = term.split()
            if (
                len(term_parts) >= 2
                and any(part in INTENSITY_MODIFIERS for part in term_parts[:-1])
            ):
                stronger_phrase_spans.append((term_parts[-1], i, i + n))

    # 사전 매칭으로 잡기 어려운 명확한 원문 phrase를 마지막에 한 번만 보완.
    for rule in phrase_rules:
        category = rule["category"]
        score = rule["score"]

        if category not in scores:
            continue

        scores[category] += score
        matched[category].append(
            phrase_match_text(rule["label"], score, rule["tag"])
        )

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


RAW_EVIDENCE_REQUIRED_NEGATIVE_TERMS = {
    "덥다", "맵다", "느끼하다", "짜다", "힘들다", "사라지다",
    "굳이", "필요없다", "안되다", "없다",
}

RAW_EVIDENCE_PATTERNS = {
    "덥다": [r"덥", r"더워", r"더웠", r"더운", r"더위"],
    "맵다": [r"맵", r"매워", r"매웠", r"매운", r"매움"],
    "느끼하다": [r"느끼"],
    "짜다": [r"짜", r"짠", r"짜서", r"짜게"],
    "힘들다": [r"힘들", r"힘듦", r"힘듬"],
    "사라지다": [r"사라"],
    "굳이": [r"굳이"],
    "필요없다": [r"필요없"],
    "안되다": [r"안되", r"안돼"],
    "없다": [r"없"],
}

NON_EVALUATIVE_STANDALONE_NEGATIVES = {
    "굳이", "필요없다", "없다", "안되다",
}


def raw_text_supports_sentiment_term(term, review_text):
    """위험 감성 토큰이 실제 원문 표면형에서 확인되는지 검사."""
    compact_text = re.sub(r"\s+", "", normalize_review_text(review_text))
    patterns = RAW_EVIDENCE_PATTERNS.get(term, [re.escape(str(term).replace(" ", ""))])
    return any(re.search(pattern, compact_text) for pattern in patterns)


def apply_token_dictionary_matching(tokens_fixed):
    """
    4단계: tokens_fixed에서 사전 후보만 추출합니다.

    이 단계에서는 감성 점수를 아직 확정하지 않습니다. 긴 n-gram 우선,
    used_token_idx, 리뷰 내 동일 표현 한 번 매칭 규칙만 담당합니다.
    """
    tokens = [
        clean_token(token)
        for token in str(tokens_fixed).split()
        if clean_token(token)
    ]
    matches = []
    matched_terms = set()
    used_token_idx = set()

    for n in range(max_ngram, 0, -1):
        for i in range(len(tokens) - n + 1):
            idx_range = set(range(i, i + n))
            if used_token_idx & idx_range:
                continue

            term = " ".join(tokens[i:i + n])
            if term not in sentiment_map or term in matched_terms:
                continue

            for category, score in sentiment_map[term]:
                if category not in categories:
                    continue

                matches.append(make_evidence_item(
                    term=term,
                    score=score,
                    category=category,
                    source="dictionary",
                    tag="token",
                    start_idx=i,
                    end_idx=i + n,
                    ngram_size=n,
                ))

            matched_terms.add(term)
            used_token_idx.update(idx_range)

    return matches


def apply_context_guards(matched_terms, review_text, tokens_fixed, phrase_matches=None):
    """
    5단계: token 후보에 원문 증거와 문맥 guard를 적용합니다.

    원문에 없는 위험 부정 토큰, 단독 일반 동사/부정 표현, phrase와 중복되는
    token 점수를 0점 처리합니다. 애매한 경우 반전하지 않고 중립화합니다.
    """
    phrase_matches = phrase_matches or []
    phrase_replacements = defaultdict(list)
    for phrase_item in phrase_matches:
        for suppressed_term in phrase_item.get("suppress_terms", set()):
            phrase_replacements[suppressed_term].append(phrase_item)

    tokens = [clean_token(token) for token in str(tokens_fixed).split() if clean_token(token)]
    adjusted = []

    for match in matched_terms:
        item = dict(match)
        term = item["term"]
        score = item["original_score"]
        n = item.get("ngram_size", len(term.split()))
        start_idx = item["start_idx"]
        end_idx = item["end_idx"]
        adjusted_score = score
        tag = "token"

        replacement_items = phrase_replacements.get(term, [])
        positive_replacement_score = sum(
            item["score"]
            for item in replacement_items
            if item["score"] > 0
        )
        negative_replacement_score = sum(
            item["score"]
            for item in replacement_items
            if item["score"] < 0
        )
        replacement_score_is_sufficient = (
            (score > 0 and positive_replacement_score >= score)
            or (score < 0 and negative_replacement_score <= score)
            or (score < 0 and positive_replacement_score > 0)
        )

        if replacement_items and replacement_score_is_sufficient:
            adjusted_score = 0
            tag = "suppressed_by_phrase_rule"
            item["replacement_terms"] = [entry["term"] for entry in replacement_items]
            item["replacement_score"] = sum(entry["score"] for entry in replacement_items)

        elif replacement_items and score > 0:
            # 긍정 token을 지울 만큼 대체 phrase 점수가 충분하지 않으면
            # 원래 dictionary evidence를 유지해 총점 손실을 막습니다.
            adjusted_score = score
            tag = "suppression_skipped_insufficient_replacement"
            item["replacement_terms"] = [entry["term"] for entry in replacement_items]
            item["replacement_score"] = positive_replacement_score

        elif n == 1 and term in NON_EVALUATIVE_STANDALONE_NEGATIVES:
            adjusted_score = 0
            tag = "non_evaluative_standalone_negative"

        elif n == 1 and term in RISKY_SINGLE_TOKENS:
            adjusted_score = 0
            tag = "blocked_risky_single_token"

        elif (
            score < 0
            and n == 1
            and term in RAW_EVIDENCE_REQUIRED_NEGATIVE_TERMS
            and not raw_text_supports_sentiment_term(term, review_text)
        ):
            adjusted_score = 0
            tag = "token_not_supported_by_text"

        elif not is_standalone_negative_term(review_text, term):
            adjusted_score = 0
            tag = "embedded_non_standalone"

        elif not is_contextually_valid_ambiguous_term(review_text, term):
            adjusted_score = 0
            tag = "ambiguous_target_context"

        elif should_decompose_negated_mixed_term(review_text, term):
            adjusted_score = 0
            tag = "decomposed_negated_mixed_term"

        else:
            context_score, context_tag = adjust_score_by_context(
                term=term,
                score=score,
                category=item["category"],
                tokens=tokens,
                start_idx=start_idx,
                end_idx=end_idx,
                review_text=review_text,
            )

            if context_tag:
                adjusted_score = context_score
                tag = context_tag
            elif (
                score > 0
                and not is_context_dependent_term(term)
                and not term_contains_negation(term)
                and has_direct_negative_context(review_text, term)
            ):
                adjusted_score = 0
                tag = "direct_neg_context"
            elif (
                score > 0
                and not is_context_dependent_term(term)
                and has_direct_comparative_reference(review_text, term)
            ):
                adjusted_score = 0
                tag = "comparative_reference"
            elif (
                score > 0
                and not is_context_dependent_term(term)
                and has_comparative_or_ironic_context(
                    review_text=review_text,
                    keyword=term,
                    tokens=tokens,
                    start_i=start_idx,
                    end_i=end_idx,
                    window=55,
                )
            ):
                adjusted_score = 0
                tag = "comparative_context"
            elif (
                score < 0
                and (
                    (
                        not is_context_dependent_term(term)
                        and has_direct_negative_context(review_text, term)
                    )
                    or has_direct_absence_of_negative_token_context(
                        tokens=tokens,
                        start_i=start_idx,
                        end_i=end_idx,
                    )
                )
            ):
                adjusted_score = 0
                tag = "direct_negative_absence"
            elif score < 0 and is_cold_dish_temperature_context(review_text, term):
                adjusted_score = 0
                tag = "cold_dish_temperature"
            elif score < 0 and has_concessive_preference_context(review_text, term):
                adjusted_score = 0
                tag = "concessive_preference"
            elif (
                score < 0
                and item["category"] == "general"
                and has_resolved_positive_concession(review_text, term)
            ):
                adjusted_score = 0
                tag = "resolved_positive_concession"

        item["score"] = adjusted_score
        item["tag"] = tag
        item["strength"] = evidence_strength(adjusted_score)
        if tag != "token" or adjusted_score != score:
            item["original_source"] = item.get("source", "dictionary")
            item["source"] = "context_guard"
        adjusted.append(item)

    adjusted.extend(phrase_matches)
    return adjusted


REVISIT_EVIDENCE_TAGS = {
    "revisit_positive_phrase", "revisit_intention_phrase",
    "repeat_visit_context", "repeat_visit_phrase", "loyal_customer_phrase",
    "waiting_revisit_positive", "craving_memory_phrase",
    "takeout_intention_phrase",
    "revisit_positive_pattern", "repeat_visit_pattern",
    "waiting_but_revisit_positive", "craving_memory_pattern",
    "takeout_intention_pattern", "strong_revisit_positive",
}

RECOMMENDATION_EVIDENCE_TAGS = {
    "recommendation_positive_phrase", "menu_recommendation_phrase",
    "trust_positive_phrase", "worth_trying_phrase",
    "recommendation_positive_pattern", "trust_positive_pattern",
    "menu_recommendation_pattern",
}

WAITING_POSITIVE_TAGS = {
    "worth_waiting_phrase", "waiting_revisit_positive",
    "wait_turnover_positive", "short_wait_positive",
    "popularity_positive_context",
    "worth_waiting_pattern", "waiting_but_revisit_positive",
    "wait_turnover_positive_pattern", "popularity_positive_pattern",
}

WAITING_NEGATIVE_TAGS = {
    "waiting_negative_phrase",
}


def public_evidence_item(item):
    """CSV와 디버그에서 확인할 evidence item의 핵심 필드만 정리합니다."""
    public = {
        "term": item["term"],
        "score": item["score"],
        "category": item["category"],
        "source": item["source"],
        "tag": item["tag"],
        "strength": item["strength"],
    }
    for optional_key in [
        "original_score", "original_source", "replacement_terms",
        "replacement_score", "star_floor",
    ]:
        if optional_key in item:
            public[optional_key] = item[optional_key]
    return public


def calculate_category_total_score(evidence_items):
    """
    6단계: 모든 최종 점수와 matched_words를 evidence_items에서만 계산합니다.

    dictionary/phrase_rule/context_guard의 score가 같은 단일 원장에 있으므로,
    reason만 남고 실제 점수에서 누락되는 경로가 생기지 않습니다.
    """
    scores = {cat: 0 for cat in categories}
    matched = {cat: [] for cat in categories}
    evidence = {
        "strong_positive_count": 0,
        "weak_positive_count": 0,
        "strong_negative_count": 0,
        "weak_negative_count": 0,
        "revisit_positive_count": 0,
        "recommendation_count": 0,
        "waiting_positive_count": 0,
        "waiting_negative_count": 0,
        "positive_phrase_count": 0,
        "phrase_score_total": 0,
        "dictionary_score_total": 0,
        "context_guard_score_total": 0,
    }

    for item in evidence_items:
        category = item["category"]
        if category not in scores:
            continue

        score = item["score"]
        scores[category] += score
        tag = item.get("tag", "")
        term_compact = str(item.get("term", "")).replace(" ", "")

        if item["strength"] == "strong_positive":
            evidence["strong_positive_count"] += 1
        elif item["strength"] == "weak_positive":
            evidence["weak_positive_count"] += 1
        elif item["strength"] == "strong_negative":
            evidence["strong_negative_count"] += 1
        elif item["strength"] == "weak_negative":
            evidence["weak_negative_count"] += 1

        if item["source"] == "phrase_rule" and score > 0:
            evidence["positive_phrase_count"] += 1
        if item["source"] == "phrase_rule":
            evidence["phrase_score_total"] += score
        if item["source"] == "dictionary":
            evidence["dictionary_score_total"] += score
        if item["source"] == "context_guard":
            evidence["context_guard_score_total"] += score
        if score > 0 and (
            tag in REVISIT_EVIDENCE_TAGS
            or any(pattern in term_compact for pattern in ["재방문", "다시갈", "또갈", "단골"])
        ):
            evidence["revisit_positive_count"] += 1
        if score > 0 and (
            tag in RECOMMENDATION_EVIDENCE_TAGS
            or any(pattern in term_compact for pattern in ["추천", "강추"])
        ):
            evidence["recommendation_count"] += 1
        if score > 0 and (
            tag in WAITING_POSITIVE_TAGS
            or (
                any(pattern in term_compact for pattern in ["웨이팅", "대기", "기다리", "줄설"])
                and any(pattern in term_compact for pattern in ["가치", "만하다", "빠르", "금방"])
            )
        ):
            evidence["waiting_positive_count"] += 1
        if score < 0 and (
            tag in WAITING_NEGATIVE_TAGS
            or any(word in str(item.get("term", "")) for word in ["웨이팅", "대기", "기다리"])
        ):
            evidence["waiting_negative_count"] += 1

        if item["source"] == "phrase_rule":
            text = phrase_match_text(item["term"], score, item["tag"])
        else:
            text = adjusted_match_text(
                item["term"],
                item["original_score"],
                score,
                None if item["tag"] == "token" else item["tag"],
            )
        matched[category].append(text)

    result = {}
    for cat in categories:
        result[f"{cat}_score"] = scores[cat]
        result[f"{cat}_matched_words"] = " ".join(matched[cat])
        result[f"{cat}_label"] = (
            "positive" if scores[cat] > 0
            else "negative" if scores[cat] < 0
            else "neutral"
        )

    result["category_total_score"] = sum(scores.values())
    result["category_total_label"] = (
        "positive" if result["category_total_score"] > 0
        else "negative" if result["category_total_score"] < 0
        else "neutral"
    )
    result.update(evidence)
    result["evidence_items"] = [
        public_evidence_item(item)
        for item in evidence_items
    ]
    return result


def calculate_sentiment_star(score, score_min, score_max):
    """7단계-A: category_total_score를 기존 카카오 분위수 기준 별점으로 변환."""
    return sentiment_to_star(score, score_min, score_max)


def calibrate_sentiment_star(raw_star, evidence_row):
    """
    7단계-B: 요청한 텍스트 evidence count만 사용해 별점 하한을 보정.

    rating은 절대 참조하지 않습니다. strong_negative_count가 0인 경우에만
    긍정 floor를 적용합니다. 따라서 strong negative가 2개 이상이면 물론,
    하나라도 존재하는 리뷰에는 긍정 floor가 적용되지 않습니다.
    """
    if pd.isna(raw_star):
        return pd.Series({
            "sentiment_star": np.nan,
            "sentiment_calibration_reason": "",
            "evidence_floor_items": [],
        })

    star = float(raw_star)
    reasons = []
    floor_items = []
    strong_positive = int(evidence_row.get("strong_positive_count", 0))
    weak_positive = int(evidence_row.get("weak_positive_count", 0))
    strong_negative = int(evidence_row.get("strong_negative_count", 0))
    revisit_positive = int(evidence_row.get("revisit_positive_count", 0))
    phrase_score_total = float(evidence_row.get("phrase_score_total", 0))

    if strong_negative == 0:
        floor = star

        def raise_floor(value, tag):
            nonlocal floor
            if value > floor:
                floor = value
                reason = f"sentiment_floor:{value:.1f}[{tag}]"
                reasons.append(reason)
                floor_items.append(make_evidence_item(
                    term=f"sentiment_floor:{value:.1f}",
                    score=0,
                    category="general",
                    source="evidence_floor",
                    tag=tag,
                    star_floor=value,
                ))

        if strong_positive >= 2:
            raise_floor(4.0, "strong_positive_evidence")

        if revisit_positive >= 1:
            raise_floor(3.8, "revisit_positive_evidence")

        if weak_positive >= 3:
            raise_floor(3.7, "positive_evidence_count")

        if phrase_score_total >= 3:
            raise_floor(3.8, "phrase_score_evidence")

        star = floor

    return pd.Series({
        "sentiment_star": round(min(5.0, star), 2),
        "sentiment_calibration_reason": " | ".join(dict.fromkeys(reasons)),
        "evidence_floor_items": [
            public_evidence_item(item)
            for item in floor_items
        ],
    })


def merge_floor_into_evidence_items(evidence_items, floor_items):
    """
    최종 evidence_items에 별점 floor 근거도 보존합니다.

    evidence_floor의 score는 0이라 category score를 바꾸지 않고, 별점 보정이
    어떤 텍스트 evidence count에서 발생했는지만 추적할 수 있게 합니다.
    """
    base = evidence_items if isinstance(evidence_items, list) else []
    floors = floor_items if isinstance(floor_items, list) else []
    return base + floors


EVIDENCE_ASPECTS = [
    "food", "service", "price", "atmosphere", "wait",
    "revisit", "recommendation", "hygiene", "general",
]

ASPECT_WEIGHTS = {
    "food": 1.25,
    "revisit": 1.25,
    "recommendation": 1.15,
    "hygiene": 1.50,
    "service": 1.20,
    "price": 0.80,
    "wait": 0.55,
    "atmosphere": 0.65,
    "general": 1.00,
}

VERY_STRONG_POSITIVE_TAGS = {
    "strong_final_positive", "rank_tier_positive",
    "rank_tier_positive_pattern", "explicit_rating_positive",
    "explicit_rating_positive_pattern", "strong_rating_phrase",
    "strong_revisit_positive",
}

STRONG_POSITIVE_TAGS = {
    "revisit_positive_phrase", "revisit_intention_phrase",
    "revisit_positive_pattern", "waiting_revisit_positive",
    "waiting_but_revisit_positive", "trust_positive_phrase",
    "trust_positive_pattern", "strong_positive_phrase",
    "experience_positive_phrase",
}

HYGIENE_TERMS = {
    "위생", "불결", "이물질", "재사용", "식중독", "배탈", "상하다",
    "상한", "벌레", "곰팡이", "머리카락", "세척", "악취",
}

SEVERE_SERVICE_TERMS = {
    "서비스 엉망", "응대 엉망", "무시", "욕설", "하대", "기분 나쁘다",
    "주문 누락", "사과 없다",
}

WAIT_TERMS = {"웨이팅", "대기", "기다리", "줄", "회전율", "회전률"}
NOISE_CROWD_TERMS = {"시끄럽", "소음", "혼잡", "붐비", "사람 많"}


def evidence_strength_name(score):
    """리팩토링 evidence의 강도 체계를 점수와 일치시킵니다."""
    magnitude = abs(float(score))
    if magnitude >= 3:
        return "very_strong"
    if magnitude >= 2:
        return "strong"
    if magnitude >= 1:
        return "normal"
    if magnitude > 0:
        return "weak"
    return "weak"


def infer_evidence_aspect(item):
    """기존 후보의 category/tag/text를 새 9개 aspect로 재분류합니다."""
    tag = str(item.get("tag", ""))
    text = str(item.get("term", ""))
    compact = text.replace(" ", "")
    category = str(item.get("category", "general"))

    if (
        tag in REVISIT_EVIDENCE_TAGS
        or any(term in compact for term in ["재방문", "다시갈", "또갈", "단골", "반복방문"])
    ):
        return "revisit"
    if tag in RECOMMENDATION_EVIDENCE_TAGS or any(
        term in compact for term in ["추천", "강추", "꼭가보", "무조건메뉴추천"]
    ):
        return "recommendation"
    if (
        tag in WAITING_POSITIVE_TAGS
        or tag in WAITING_NEGATIVE_TAGS
        or any(term in compact for term in WAIT_TERMS)
    ):
        return "wait"
    if any(term in compact for term in HYGIENE_TERMS):
        return "hygiene"
    if category in {"food", "service", "price", "atmosphere"}:
        return category
    return "general"


def normalize_evidence_score(item, aspect):
    """
    후보 점수를 evidence 강도 체계로 정규화합니다.

    rating은 보지 않습니다. wait/noise/일반 가격 불편은 약하게, 위생과
    명시적인 심각 서비스 문제는 강하게 반영합니다.
    """
    score = float(item.get("score", 0))
    if score == 0:
        return 0.0

    tag = str(item.get("tag", ""))
    text = str(item.get("term", ""))
    compact = text.replace(" ", "")

    if score > 0:
        if tag in VERY_STRONG_POSITIVE_TAGS:
            return max(score, 3.0)
        if tag in STRONG_POSITIVE_TAGS:
            return max(score, 2.0)
        return min(max(score, 0.5), 3.0)

    if aspect == "hygiene":
        return min(score, -2.0)
    if aspect == "service" and any(term.replace(" ", "") in compact for term in SEVERE_SERVICE_TERMS):
        return min(score, -2.0)
    if aspect == "wait":
        return max(score, -0.5)
    if aspect == "price" and any(term in compact for term in ["비싸", "사악", "가격"]):
        if not any(term in compact for term in ["돈아깝", "바가지", "사기"]):
            return max(score, -0.5)
    if aspect == "atmosphere" and any(term in compact for term in NOISE_CROWD_TERMS):
        return max(score, -0.5)
    return max(min(score, -0.5), -3.0)


def refactor_evidence_item(item, review_text):
    """기존 탐지 후보를 최종 evidence 표준 스키마로 변환합니다."""
    aspect = infer_evidence_aspect(item)
    original_score = float(item.get("original_score", item.get("score", 0)))
    adjusted_score = normalize_evidence_score(item, aspect)
    is_active = adjusted_score != 0
    tag = str(item.get("tag", ""))
    source = str(item.get("source", "dictionary"))

    if source == "phrase_rule" and ("pattern" in tag or tag.endswith("_evidence")):
        source = "pattern_rule"
    if source == "evidence_floor":
        source = "floor"

    polarity = (
        "positive" if adjusted_score > 0
        else "negative" if adjusted_score < 0
        else "neutral"
    )
    text = str(item.get("term", ""))
    normalized_text = normalize_review_text(text).lower()
    guard_reason = None
    if not is_active and original_score != 0:
        guard_reason = tag

    return {
        "text": text,
        "normalized_text": normalized_text,
        "score": adjusted_score,
        "original_score": original_score,
        "polarity": polarity,
        "strength": evidence_strength_name(adjusted_score),
        "aspect": aspect,
        "source": source,
        "tag": tag,
        "start": item.get("start_idx"),
        "end": item.get("end_idx"),
        "is_active": is_active,
        "guard_reason": guard_reason,
        # 이전 출력/검증 코드와의 호환용 별칭입니다.
        "term": text,
        "category": aspect,
    }


def make_refactored_context_evidence(text, score, aspect, tag):
    """원문의 명시적인 결론/저점/거절을 active context evidence로 생성."""
    return {
        "text": text,
        "normalized_text": normalize_review_text(text).lower(),
        "score": float(score),
        "original_score": float(score),
        "polarity": "positive" if score > 0 else "negative",
        "strength": evidence_strength_name(score),
        "aspect": aspect,
        "source": "context_guard",
        "tag": tag,
        "start": None,
        "end": None,
        "is_active": True,
        "guard_reason": None,
        "term": text,
        "category": aspect,
    }


def deactivate_refactored_evidence(item, reason):
    """긍정처럼 보이지만 실제 평가 근거가 아닌 evidence를 감사 가능하게 비활성화."""
    item["original_score"] = item["score"]
    item["score"] = 0.0
    item["polarity"] = "neutral"
    item["strength"] = "weak"
    item["is_active"] = False
    item["guard_reason"] = reason
    item["tag"] = reason
    item["source"] = "context_guard"


def resolve_refactored_context(evidence_items, review_text):
    """
    후보 추출 이후 최종 결론과 평가 대상을 해석합니다.

    기존 token guard가 놓친 부정 결론을 active negative evidence로 만들고,
    질문/부정/리뷰 이벤트/타 식당 비교 속 긍정 후보는 비활성화합니다.
    """
    compact = compact_review_for_patterns(review_text)

    rhetorical_taste_question = bool(
        re.search(r"(?:맛있|좋).{0,8}(?:나요|습니까|맞나|진짜인가)", compact)
    )
    negated_restaurant_praise = bool(
        re.search(r"(?:맛집|추천|만족).{0,8}(?:은|는|이|가)?(?:아니|아닙|않)", compact)
    )
    negated_value = bool(
        re.search(r"(?:가격|가성비|합리적).{0,18}(?:아닌|아니|않|없)", compact)
    )
    review_event_context = "리뷰이벤트" in compact or "영수증리뷰" in compact
    negative_rebuttal_context = bool(
        re.search(
            r"(?:별로라는사람|별로라던데.{0,12}(?:저는|나는).{0,12}(?:좋|맛있)|"
            r"맛없(?:진|지는|지않|을수가없|는게없|없)|"
            r"맛이없(?:진|지는|지않|을수가없|는게없)|맛이없는건아니|대체할맛이없|"
            r"안질기|질기지않|비린맛없이|비리지않|나쁘지않|"
            r"불친절(?:은|한지는).{0,8}(?:모르|아니)|"
            r"비싸도.{0,10}맛있)",
            compact,
        )
    )
    quoted_negative_context = bool(
        re.search(
            r"(?:(?:아래)?후기|리뷰|소문|말|사람들?|친구들?).{0,28}"
            r"(?:맛없|별로|불친절|비위생|냄새나|실망|안좋).{0,35}"
            r"(?:저는|나는|직접|먹어보니|가보니|그런데|근데|하지만|했으나).{0,30}"
            r"(?:맛있|좋|괜찮|친절|못느|안보|준수|감지덕지)|"
            r"(?:맛없|별로|불친절|비위생|냄새나|실망|안좋).{0,12}"
            r"(?:후기|리뷰|소문|말).{0,35}(?:맛있|좋|괜찮|친절|못느|안보|준수)",
            compact,
        )
    )
    negated_food_negative_context = bool(
        re.search(
            r"(?:맛없|맛이없)(?:진|지는|지않|을수가없|는게없|없)|"
            r"맛이없는건아니|대체할맛이없|안질기|질기지않|비린맛없이|비리지않|"
            r"누린내(?:가|도)?(?:없|안나|나지않)",
            compact,
        )
    )
    hygiene_denial_or_reported_context = bool(
        re.search(
            r"(?:위생.{0,20}(?:문제.{0,8}(?:없|느끼지못)|괜찮|깔끔|준수)|"
            r"비위생.{0,15}(?:안보|보이지않|아니)|"
            r"(?:리뷰|후기).{0,20}(?:위생|비위생).{0,20}(?:준수|괜찮|문제없)|"
            r"더럽다고(?:하는데|하던데|했는데))",
            compact,
        )
    )
    other_target_positive = bool(
        (
            re.search(
                r"(?:여기말고|이곳말고|본점은|다른(?:집|곳|지점|매장)|옆집|차라리)"
                r".{0,38}(?:맛있|좋|괜찮|추천|낫|맛집|양.{0,5}많|부드럽)",
                compact,
            )
            and re.search(
                r"(?:여기|이곳|이번지점|이매장).{0,25}(?:별로|아니|맛없|실망|안좋|아쉽|아쉬)",
                compact,
            )
        )
        or re.search(
            r"진짜맛집은.{0,45}(?:다른곳|옆집|본점|[가-힣]+(?:시내|동|점)).{0,70}"
            r"(?:맛있|좋|고소|부드럽|양.{0,5}많|추천)",
            compact,
        )
        or re.search(
            r"진짜맛집은.{0,24}(?:다른곳|옆집|본점|[가-힣]+(?:시내|동|점))",
            compact,
        )
    )
    past_positive_current_negative = bool(
        re.search(
            r"(?:예전|전에는|옛날|처음엔|과거|전엔).{0,28}(?:맛있|좋|괜찮|진하|자주)"
            r".{0,45}(?:지금|이제|요즘|갈수록|최근|현재).{0,30}"
            r"(?:별로|아니|변|실망|맛없|안좋|잃|힘들|밍밍|떨어)|"
            r"(?:예전|전에는|옛날|전엔).{0,42}(?:맛있|좋|괜찮).{0,48}"
            r"(?:했는데|였는데|좋았는데).{0,50}(?:변|초심잃|떨어|별로|아니|실망|안좋)",
            compact,
        )
    )
    past_negative_current_positive = bool(
        re.search(
            r"(?:예전|전에는|옛날|처음|지난|여름|전번).{0,35}"
            r"(?:별로|맛없|맛이없|실망|안좋|너무매워|다시는안|안오리|못느)"
            r".{0,55}(?:지금|이제|이번|오늘|겨울|다시|오니|와보니).{0,35}"
            r"(?:맛있|좋|괜찮|나아|만족|조화|확실|평며들)|"
            r"(?:첫번째|처음).{0,20}(?:맛없|맛이없|못느).{0,45}"
            r"(?:이제|이번|두번째).{0,30}(?:맛있|좋|조화|느껴)|"
            r"예전.{0,40}(?:너무매워|맛없|별로).{0,90}"
            r"(?:음식자체는맛있|종종갈|또갈|재방문)",
            compact,
        )
    )
    expectation_or_reputation_negative = bool(
        re.search(
            r"(?:유명|맛집(?:이라|이라고|이래서|해서)|평점.{0,6}좋|추천받|기대|블루리본)"
            r".{0,28}(?:갔|왔|했|먹어봤|했는데).{0,35}"
            r"(?:별로|실망|아쉽|맛없|안좋|평범|돈아깝|노이해)",
            compact,
        )
    )
    conditional_warning_context = bool(
        re.search(
            r"(?:시끄러운?(?:거|것)|웨이팅|대기|매운(?:거|것)|조용한분위기|혼잡한?(?:거|것))"
            r".{0,16}(?:싫어하|싫으면|못먹|원하|예민하).{0,20}"
            r"(?:가지마세요|피하세요|비추천|조심)",
            compact,
        )
    )
    ironic_or_expectation_positive_not_actual = bool(
        re.search(
            r"(?:존맛|맛있|좋|맛집).{0,28}(?:생각할까봐걱정|이라고\?|인가\?)|"
            r"(?:이걸|이게|이곳을|여기를).{0,18}(?:존맛|맛있|좋|맛집).{0,25}"
            r"(?:이해안|어이가없|말이되)|"
            r"(?:어쩌다|왜).{0,12}맛집.{0,12}(?:된|인)|"
            r"맛있는줄알았.{0,25}(?:실망|별로)|"
            r"평점.{0,10}좋.{0,25}(?:실망|별로|맛없)|"
            r"맛집.{0,10}(?:된건지|인건지|이라고생각할까봐)|"
            r"굿이라고.{0,12}(?:하기엔|보기엔|말하기엔).{0,15}(?:아쉽|아쉬|별로|아니)",
            compact,
        )
    )
    not_really_positive_evaluation = bool(
        re.search(
            r"(?:맛있|신선|다양|특별|훌륭|좋은|괜찮은)(?:은|는|한|하고|하고서)?"
            r"(?:것|점|맛)?도(?:아니|아닌)",
            compact,
        )
    )
    operational_service_failure = bool(
        re.search(
            r"(?:우리보다|저희보다).{0,18}뒷번호.{0,12}먼저(?:앉|입장)|"
            r"차례.{0,18}(?:아니라고)?나가라|"
            r"(?:안내|웨이팅|대기|순번|키오스크).{0,25}말도없이.{0,18}(?:바뀌|변경)|"
            r"직원.{0,18}(?:물어보니|문의하니).{0,18}마감|"
            r"인사도안(?:함|하|했)|주문누락|대답(?:도)?없|"
            r"(?:손님|저희|우리).{0,15}무시|휴대폰.{0,15}손님.{0,8}무시|"
            r"말한마디라도.{0,15}해줘야|다시방문하기어려",
            compact,
        )
    )
    operational_wait_failure = bool(
        re.search(
            r"(?:뒷번호.{0,12}먼저(?:앉|입장)|차례.{0,18}나가라|"
            r"(?:웨이팅|대기|순번|키오스크).{0,25}말도없이.{0,18}(?:바뀌|변경)|"
            r"기다리.{0,25}마감)",
            compact,
        )
    )
    negated_current_negative_context = bool(
        re.search(
            r"(?:실망안시|실망시키지않|불친절(?:도|은|한지는)?(?:아님|아니|않|모르)|"
            r"비위생.{0,12}(?:안보|보이지않|아니)|"
            r"실망할수도.{0,30}(?:하지만|그러나|만).{0,30}(?:좋|맛있|독보적)|"
            r"맛없고맛있고.{0,20}(?:어렵|모르).{0,25}(?:맛있|좋))",
            compact,
        )
    )
    rating_scale_explanation_context = bool(
        re.search(r"5점.{0,14}4점.{0,14}3점.{0,14}2점.{0,14}1점", compact)
    )
    strong_salty_context = bool(
        re.search(r"(?:너무|정말|엄청|간이)?(?:짜다|짰|짜서|짠맛|짜게)", compact)
    )
    soft_noodle_negative_context = bool(
        re.search(r"(?:면|면발).{0,24}(?:힘이없|퍼지|퍼졌|너무부드러).{0,25}(?:아쉽|아쉬|별로|계속|못)", compact)
    )

    for item in evidence_items:
        if not item["is_active"]:
            continue
        text_compact = item["normalized_text"].replace(" ", "")

        if item["polarity"] == "negative":
            if (
                item["aspect"] == "hygiene"
                and "이상하다" in text_compact
            ):
                deactivate_refactored_evidence(item, "ambiguous_strange_not_hygiene")
            elif (
                conditional_warning_context
                and any(term in text_compact for term in ["가지말", "비추천", "피하", "조심"])
            ):
                deactivate_refactored_evidence(
                    item,
                    "conditional_warning_not_revisit_rejection",
                )
            elif (
                quoted_negative_context
                and not other_target_positive
                and not past_positive_current_negative
                and any(term in text_compact for term in [
                    "맛없", "별로", "불친절", "비위생", "냄새", "실망", "안좋",
                ])
            ):
                deactivate_refactored_evidence(item, "quoted_negative_not_actual_negative")
            elif (
                past_negative_current_positive
                and any(term in text_compact for term in [
                    "맛없", "별로", "실망", "매워", "맵다", "불가",
                ])
            ):
                deactivate_refactored_evidence(item, "past_negative_not_current")
            elif (
                negated_food_negative_context
                and item["aspect"] == "food"
                and any(term in text_compact for term in ["맛없", "질기", "비리", "누린내"])
            ):
                deactivate_refactored_evidence(item, "explicitly_negated_food_negative")
            elif (
                hygiene_denial_or_reported_context
                and any(term in text_compact for term in ["더럽", "위생"])
            ):
                deactivate_refactored_evidence(item, "reported_or_denied_hygiene_negative")
            elif (
                negated_current_negative_context
                and any(term in text_compact for term in ["실망", "불친절", "비위생", "맛없"])
            ):
                deactivate_refactored_evidence(item, "negated_negative_not_actual_negative")
            elif "서운하다" in text_compact and re.search(
                r"(?:빠트리|빼먹|없으면|안먹으면).{0,10}서운", compact
            ):
                deactivate_refactored_evidence(item, "positive_omission_idiom")
            elif (
                "짜다" in text_compact
                and "짜글" in compact
                and not strong_salty_context
            ):
                deactivate_refactored_evidence(item, "embedded_menu_name_not_salty")
            continue

        if item["polarity"] != "positive":
            continue

        if rhetorical_taste_question and any(
            term in text_compact for term in ["맛있", "좋다", "좋아서"]
        ):
            deactivate_refactored_evidence(item, "rhetorical_positive_question")
        elif negated_restaurant_praise and any(
            term in text_compact for term in ["맛집", "추천", "만족"]
        ):
            deactivate_refactored_evidence(item, "negated_positive_evaluation")
        elif "한번으로만족" in compact and "만족" in text_compact:
            deactivate_refactored_evidence(item, "one_time_only_not_satisfaction")
        elif re.search(r"만족스러운.{0,8}(?:것|점|부분).{0,5}없", compact) and "만족" in text_compact:
            deactivate_refactored_evidence(item, "negated_satisfaction_context")
        elif negated_value and item["aspect"] == "price":
            deactivate_refactored_evidence(item, "negated_value_context")
        elif review_event_context and item["aspect"] == "recommendation":
            deactivate_refactored_evidence(item, "review_event_recommendation")
        elif (
            other_target_positive
            and item["aspect"] in {"food", "general", "recommendation"}
            and any(term in text_compact for term in [
                "맛있", "좋", "괜찮", "추천", "맛집", "낫", "많", "부드럽", "고소",
            ])
        ):
            deactivate_refactored_evidence(item, "other_target_positive")
        elif (
            past_positive_current_negative
            and item["aspect"] in {"food", "general"}
            and any(term in text_compact for term in ["맛있", "좋", "괜찮", "맛집", "고소", "부드럽"])
        ):
            deactivate_refactored_evidence(item, "past_positive_not_current")
        elif (
            expectation_or_reputation_negative
            and any(term in text_compact for term in ["유명", "맛집", "평점", "좋다", "기대"])
        ):
            deactivate_refactored_evidence(item, "expectation_not_actual_positive")
        elif (
            ironic_or_expectation_positive_not_actual
            and item["aspect"] in {"food", "general", "recommendation"}
            and any(term in text_compact for term in [
                "존맛", "맛있", "맛집", "좋", "싱싱", "추천", "굿",
            ])
        ):
            deactivate_refactored_evidence(
                item,
                "ironic_or_expectation_positive_not_actual",
            )
        elif (
            not_really_positive_evaluation
            and any(term in text_compact for term in ["맛있", "신선", "다양", "특별", "훌륭", "좋", "괜찮"])
        ):
            deactivate_refactored_evidence(item, "not_really_positive_evaluation")
        elif "돈맛" in text_compact:
            deactivate_refactored_evidence(item, "money_taste_not_food_positive")
        elif "달달" in text_compact and re.search(r"달달(?:외|하게외|암기|외운)", compact):
            deactivate_refactored_evidence(item, "non_food_sweet_expression")
        elif (
            "싱싱" in text_compact
            and re.search(r"싱싱한?.{0,18}(?:무슨짓|왜이렇게|망치|모르겠)", compact)
        ):
            deactivate_refactored_evidence(
                item,
                "ironic_or_expectation_positive_not_actual",
            )
        elif (
            "부드럽" in text_compact
            and soft_noodle_negative_context
        ):
            deactivate_refactored_evidence(item, "soft_noodle_negative_context")
        elif (
            text_compact == "크다"
            and not re.search(
                r"(?:매장|공간|내부|자리|테이블|양|사이즈|가게).{0,10}(?:크|넓)|"
                r"(?:크|넓).{0,10}(?:매장|공간|내부|자리|테이블|양|사이즈|가게)",
                compact,
            )
        ):
            deactivate_refactored_evidence(item, "size_ambiguous_context")

    generated = []

    if conditional_warning_context:
        conditional_warning_item = make_refactored_context_evidence(
            "조건부 이용 경고", -1.0, "revisit",
            "conditional_warning_not_revisit_rejection",
        )
        deactivate_refactored_evidence(
            conditional_warning_item,
            "conditional_warning_not_revisit_rejection",
        )
        generated.append(conditional_warning_item)

    if not (past_negative_current_positive or conditional_warning_context) and re.search(
        r"(?:재방문(?:의사)?(?:은|는|도)?(?:x|X|없|안함|않음|안할|않을)|"
        r"재방문.{0,15}(?:절대)?(?:안할|않을|없을|없음|안함|않음|안합|않습)|"
        r"다시방문.{0,15}(?:생각)?(?:없|안|않)|"
        r"다시(?:는)?방문.{0,8}(?:싫|안|않|없)|"
        r"다시(?:는)?(?:안|않).{0,10}(?:가|오)|다시(?:가|오).{0,10}(?:않|안)|"
        r"다시갈생각.{0,5}없|안갈(?:것)?같|가지않을듯|이제안갑|다신안|재방문은굳이안|"
        r"이제(?:는)?다른곳.{0,10}(?:가|낫|좋)|"
        r"(?:절대)?가지마세요|다시(?:는)?못찾아먹|다시갈이유.{0,5}없)",
        compact,
    ):
        generated.append(make_refactored_context_evidence(
            "재방문 거절 결론", -3.0, "revisit", "explicit_revisit_rejection",
        ))

    service_denial = bool(
        re.search(
            r"(?:서비스.{0,8})?불친절(?:도|은|한지는|한건)?(?:아님|아니|않|모르)|"
            r"응대나서비스.{0,10}불친절.{0,10}(?:없|아니)",
            compact,
        )
    )
    if not (service_denial or quoted_negative_context) and (
        operational_service_failure
        or re.search(
            r"(?:서비스|응대|직원|태도).{0,20}(?:개판|엉망|최악|별로|무시|하대|문제|불친절|"
            r"기분나쁘|싸가지|주문누락|사과없|화나|일을.{0,5}못|실수(?:하|했|로|때문)|"
            r"떨어지|설렁설렁|짜증|험담|"
            r"인사도안|대답도없|액션도없|말한마디|나가라|마감.{0,8}안맞|최저)",
            compact,
        )
    ):
        generated.append(make_refactored_context_evidence(
            "심각한 서비스 부정 결론", -2.0, "service", "active_service_negative_conclusion",
        ))

    if not quoted_negative_context and operational_wait_failure:
        generated.append(make_refactored_context_evidence(
            "운영 대기 순서 문제", -1.0, "wait", "active_wait_operation_negative",
        ))

    high_confidence_food_negative = bool(
        re.search(
            r"(?:맛(?:이|은|도)?(?:정말|너무)?(?:없|별로|변했|변하)|"
            r"음식.{0,12}(?:맛없|별로)|돈아깝|돈아까|누린내|"
            r"(?:매워|느끼|질겨|비려|냄새나).{0,12}(?:못먹|먹기힘들|남김|물리)|"
            r"느끼.{0,8}(?:못먹|물리)|너무매워.{0,8}(?:못먹|힘들)|"
            r"여기는(?:그냥)?별로|솔직히별로)",
            compact,
        )
        and not negative_rebuttal_context
        and not quoted_negative_context
        and not past_negative_current_positive
        and not negated_current_negative_context
        and not negated_food_negative_context
    )
    if high_confidence_food_negative or (
        not (quoted_negative_context or past_negative_current_positive or negated_current_negative_context)
        and re.search(
        r"(?:돈아깝|돈아까|맛과서비스.{0,8}잃|"
        r"도저히.{0,12}(?:힘들|못|아니)|개인적으로.{0,12}취향.{0,8}아니|"
        r"맛이존재하지않|진짜별로였|최악으로|"
        r"(?:특별히|딱히|그다지)?맛있(?:지는|지)?않)",
        compact,
        )
    ):
        generated.append(make_refactored_context_evidence(
            "명시적 음식 부정 결론", -2.0, "food", "active_food_negative_conclusion",
        ))

    if re.search(
        r"(?:(?:가격|가격대|가성비|가격을고려).{0,25}"
        r"(?:합리적|값어치|가치).{0,8}(?:아닌|아니|않|없)|"
        r"합리적인맛.{0,8}(?:아닌|아니|않))",
        compact,
    ):
        generated.append(make_refactored_context_evidence(
            "가격 대비 합리적이지 않다", -2.0, "price", "active_price_negative_conclusion",
        ))

    if not (hygiene_denial_or_reported_context or quoted_negative_context) and re.search(
        r"(?:이물질|재사용|벌레|곰팡이|머리카락|돼지털|달걀껍질|계란껍질|비위생|불결|식중독|배탈|세척안|"
        r"위생.{0,10}(?:별로|최악|문제|적이지않|않았|안좋)|"
        r"(?:너무|정말)?더럽(?:다(?!고)|었|고|음))",
        compact,
    ):
        generated.append(make_refactored_context_evidence(
            "명시적 위생 문제", -2.0, "hygiene", "active_hygiene_negative_conclusion",
        ))

    if not conditional_warning_context and re.search(
        r"(?:(?<![가-힣])비추천|추천(?:은|하기)?(?:안|않|어렵)|추천할수없|"
        r"(?:절대)?가지마세요)",
        compact,
    ):
        generated.append(make_refactored_context_evidence(
            "추천 거절 결론", -2.0, "recommendation", "explicit_recommendation_rejection",
        ))

    if (
        not (
            negative_rebuttal_context
            or quoted_negative_context
            or past_negative_current_positive
            or negated_current_negative_context
        )
        and re.search(r"(?:너무|정말|솔직히)?(?:실망|별로였|최악|돈아깝)", compact)
    ):
        generated.append(make_refactored_context_evidence(
            "명시적 현재 부정 결론", -2.0, "general", "active_current_negative_conclusion",
        ))

    if not_really_positive_evaluation:
        generated.append(make_refactored_context_evidence(
            "실질적 긍정 평가 아님", -1.0, "food", "not_really_positive_evaluation",
        ))

    if ironic_or_expectation_positive_not_actual:
        generated.append(make_refactored_context_evidence(
            "반어 또는 기대 불일치", -1.0, "general",
            "ironic_or_expectation_positive_not_actual",
        ))

    if past_positive_current_negative:
        generated.append(make_refactored_context_evidence(
            "과거 대비 현재 품질 저하", -1.0, "general",
            "past_positive_current_negative",
        ))

    if soft_noodle_negative_context:
        generated.append(make_refactored_context_evidence(
            "면발 탄력 저하", -1.0, "food", "soft_noodle_negative_context",
        ))

    if not rating_scale_explanation_context and re.search(
        r"(?:맛|서비스|고기|삼겹살|목살|직원).{0,8}(?:0(?:\.[0-9])?|1(?:\.[0-9])?|2(?:\.[0-9])?)점",
        compact,
    ):
        generated.append(make_refactored_context_evidence(
            "텍스트 내 명시적 저점", -2.0, "general", "explicit_low_rating_evidence",
        ))

    if re.search(r"(?:웨이팅|기다림|줄).{0,12}(?:값어치|가치).{0,8}없", compact):
        generated.append(make_refactored_context_evidence(
            "대기할 가치가 없다", -1.0, "wait", "active_wait_negative_conclusion",
        ))

    if re.search(
        r"(?:맛|서비스|위생).{0,18}(?:다|어느것하나|어느것도).{0,18}"
        r"(?:최저|만족스러운.{0,5}없|좋을수가없)|"
        r"만족스러운.{0,8}(?:것|점|부분).{0,5}없",
        compact,
    ):
        generated.append(make_refactored_context_evidence(
            "명시적 전반 부정 결론", -2.0, "general", "active_overall_negative_conclusion",
        ))

    if re.search(
        r"(?:가격.{0,15}(?:너무|정말).{0,5}(?:비싸|높|올랐)|"
        r"돈독기|추가금액.{0,8}(?:천원|비싸)|"
        r"가격대비.{0,15}(?:떨어|낮|별로))",
        compact,
    ):
        generated.append(make_refactored_context_evidence(
            "명시적 가격 부담 결론", -2.0, "price", "active_price_negative_conclusion",
        ))

    if re.search(
        r"(?:유명한이유가있|괜히맛집이아니|평점좋은이유를알|"
        r"추천받아.{0,12}(?:만족|좋|맛있))",
        compact,
    ):
        generated.append(make_refactored_context_evidence(
            "명성이 실제 경험으로 확인됨", 1.0, "general", "reputation_confirmed_positive",
        ))

    return evidence_items + generated


def deduplicate_evidence(evidence_items):
    """
    동일 의미의 active evidence만 중복 제거합니다.

    suppress된 후보는 감사 추적을 위해 inactive evidence로 남기며, active
    evidence는 없애지 않습니다. 따라서 긍정 근거를 잡고 스스로 잃는 구조가
    최종 점수 계산에 들어오지 않습니다.
    """
    active_best = {}
    inactive = []

    for item in evidence_items:
        if not item["is_active"]:
            inactive.append(item)
            continue

        compact = re.sub(r"\s+", "", item["normalized_text"])
        key = (compact, item["aspect"], item["polarity"])
        previous = active_best.get(key)
        if previous is None or abs(item["score"]) > abs(previous["score"]):
            active_best[key] = item

    return list(active_best.values()) + inactive


def calculate_aspect_scores(evidence_items):
    """active evidence만 사용해 9개 aspect 점수를 계산합니다."""
    scores = {aspect: 0.0 for aspect in EVIDENCE_ASPECTS}
    for item in evidence_items:
        if item["is_active"] and item["aspect"] in scores:
            scores[item["aspect"]] += float(item["score"])
    return {aspect: round(score, 3) for aspect, score in scores.items()}


def calculate_evidence_counts(evidence_items):
    """최종 active evidence의 극성/강도/aspect 개수를 계산합니다."""
    counts = {
        "weak_positive_count": 0,
        "normal_positive_count": 0,
        "strong_positive_count": 0,
        "very_strong_positive_count": 0,
        "weak_negative_count": 0,
        "normal_negative_count": 0,
        "strong_negative_count": 0,
        "very_strong_negative_count": 0,
        "positive_evidence_count": 0,
        "negative_evidence_count": 0,
        "revisit_positive_count": 0,
        "recommendation_count": 0,
        "waiting_positive_count": 0,
        "waiting_negative_count": 0,
        "hygiene_strong_negative_count": 0,
        "hygiene_very_strong_negative_count": 0,
        "service_strong_negative_count": 0,
        "service_very_strong_negative_count": 0,
        "food_strong_positive_count": 0,
        "food_strong_negative_count": 0,
        "food_very_strong_negative_count": 0,
        "revisit_negative_count": 0,
        "strong_revisit_negative_count": 0,
        "recommendation_negative_count": 0,
        "explicit_negative_conclusion_count": 0,
        "positive_aspect_count": 0,
        "negative_aspect_count": 0,
        "negative_aspects_with_strong_negative": 0,
        "non_tolerable_negative_count": 0,
    }
    positive_aspects = set()
    negative_aspects = set()
    strong_negative_aspects = set()

    for item in evidence_items:
        if not item["is_active"]:
            continue
        polarity = item["polarity"]
        strength = item["strength"]
        aspect = item["aspect"]

        if polarity == "positive":
            positive_aspects.add(aspect)
            counts["positive_evidence_count"] += 1
            counts[f"{strength}_positive_count"] += 1
            if aspect == "food" and strength in {"strong", "very_strong"}:
                counts["food_strong_positive_count"] += 1
            if aspect == "revisit":
                counts["revisit_positive_count"] += 1
            if aspect == "recommendation":
                counts["recommendation_count"] += 1
            if aspect == "wait":
                counts["waiting_positive_count"] += 1
        elif polarity == "negative":
            negative_aspects.add(aspect)
            counts["negative_evidence_count"] += 1
            counts[f"{strength}_negative_count"] += 1
            if not (
                aspect in {"wait", "price", "atmosphere"}
                and strength == "weak"
            ):
                counts["non_tolerable_negative_count"] += 1
            if strength in {"strong", "very_strong"}:
                strong_negative_aspects.add(aspect)
            if aspect == "wait":
                counts["waiting_negative_count"] += 1
            if aspect == "hygiene" and strength in {"strong", "very_strong"}:
                counts["hygiene_strong_negative_count"] += 1
            if aspect == "hygiene" and strength == "very_strong":
                counts["hygiene_very_strong_negative_count"] += 1
            if aspect == "service" and strength in {"strong", "very_strong"}:
                counts["service_strong_negative_count"] += 1
            if aspect == "service" and strength == "very_strong":
                counts["service_very_strong_negative_count"] += 1
            if aspect == "food" and strength in {"strong", "very_strong"}:
                counts["food_strong_negative_count"] += 1
            if aspect == "food" and strength == "very_strong":
                counts["food_very_strong_negative_count"] += 1
            if aspect == "revisit":
                counts["revisit_negative_count"] += 1
                if strength in {"strong", "very_strong"}:
                    counts["strong_revisit_negative_count"] += 1
            if aspect == "recommendation":
                counts["recommendation_negative_count"] += 1
            if "negative_conclusion" in item["tag"] or item["tag"] in {
                "explicit_revisit_rejection", "explicit_recommendation_rejection",
                "explicit_low_rating_evidence",
            }:
                counts["explicit_negative_conclusion_count"] += 1

    counts["positive_aspect_count"] = len(positive_aspects)
    counts["negative_aspect_count"] = len(negative_aspects)
    counts["negative_aspects_with_strong_negative"] = len(strong_negative_aspects)

    # 기존 weak count 컬럼은 floor 호환을 위해 non-strong evidence 합계로 둡니다.
    counts["weak_positive_count"] += counts["normal_positive_count"]
    counts["weak_negative_count"] += counts["normal_negative_count"]
    return counts


def calculate_evidence_sentiment_star(aspect_scores, counts):
    """
    rating과 분위수 없이 텍스트 evidence만으로 별점을 계산합니다.

    aspect별 기여는 극단값을 막기 위해 +/-4로 제한합니다. food/revisit/
    recommendation은 크게, wait/noise/price 불편은 작게 반영합니다.
    """
    weighted_score = 0.0
    for aspect, score in aspect_scores.items():
        clipped = max(-4.0, min(4.0, float(score)))
        weighted_score += clipped * ASPECT_WEIGHTS[aspect]

    raw_star = max(1.0, min(5.0, 3.0 + 0.34 * weighted_score))
    star = raw_star
    reasons = []
    floor_items = []
    strong_negative_total = (
        counts["strong_negative_count"] + counts["very_strong_negative_count"]
    )

    def apply_floor(value, tag):
        nonlocal star
        if value > star:
            star = value
            reasons.append(f"sentiment_floor:{value:.1f}[{tag}]")
            floor_items.append({
                "text": f"sentiment_floor:{value:.1f}",
                "normalized_text": f"sentiment_floor:{value:.1f}",
                "score": 0.0,
                "original_score": 0.0,
                "polarity": "neutral",
                "strength": "weak",
                "aspect": "general",
                "source": "floor",
                "tag": tag,
                "start": None,
                "end": None,
                "is_active": False,
                "guard_reason": tag,
                "term": f"sentiment_floor:{value:.1f}",
                "category": "general",
            })

    floor_candidate_exists = (
        counts["very_strong_positive_count"] >= 1
        or counts["strong_positive_count"] >= 2
        or counts["revisit_positive_count"] >= 1
        or counts["weak_positive_count"] >= 3
    )
    positive_floor_allowed = (
        strong_negative_total == 0
        and counts["explicit_negative_conclusion_count"] == 0
        and counts["revisit_negative_count"] == 0
        and counts["recommendation_negative_count"] == 0
        and counts["food_strong_negative_count"] == 0
        and counts["service_strong_negative_count"] == 0
        and counts["hygiene_strong_negative_count"] == 0
        # 약한 긍정 3개만으로 여러 개의 실제 불만을 덮지 못하게 합니다.
        and counts["negative_evidence_count"] <= 1
        and counts["non_tolerable_negative_count"] <= 1
    )

    # 3단계: positive floor. 핵심 부정이 active이면 floor를 적용하지 않습니다.
    if positive_floor_allowed:
        if counts["very_strong_positive_count"] >= 1:
            apply_floor(4.2, "very_strong_positive_evidence")
        if counts["strong_positive_count"] >= 2:
            apply_floor(4.0, "strong_positive_evidence")
        if counts["revisit_positive_count"] >= 1:
            apply_floor(3.8, "revisit_positive_evidence")
        if counts["weak_positive_count"] >= 3:
            apply_floor(3.7, "positive_evidence_count")
    elif floor_candidate_exists:
        reasons.append("positive_floor_blocked[active_negative_evidence]")

    def apply_cap(value, tag, prefix="sentiment_cap"):
        nonlocal star
        if star > value:
            star = value
            reasons.append(f"{prefix}:{value:.1f}[{tag}]")

    # 약한 긍정 evidence만으로 4점대에 진입하지 않도록 제한합니다.
    actual_weak_positive_count = max(
        0,
        counts["weak_positive_count"] - counts["normal_positive_count"],
    )
    if (
        counts["very_strong_positive_count"] == 0
        and counts["strong_positive_count"] == 0
        and counts["normal_positive_count"] == 0
        and actual_weak_positive_count > 0
    ):
        weak_only_cap = 3.8 if (
            actual_weak_positive_count >= 5
            and counts["negative_evidence_count"] == 0
        ) else 3.7
        apply_cap(weak_only_cap, "weak_positive_only")

    # 4단계: negative cap. floor보다 나중에 적용해 핵심 부정이 최종 상한을
    # 반드시 제한하도록 합니다.
    if counts["hygiene_very_strong_negative_count"] >= 1:
        apply_cap(2.5, "hygiene_very_strong_negative")
    elif counts["hygiene_strong_negative_count"] >= 1:
        apply_cap(3.0, "hygiene_strong_negative")

    if counts["service_strong_negative_count"] >= 2:
        apply_cap(3.0, "multiple_service_strong_negative")
    elif (
        counts["service_strong_negative_count"] >= 1
        and counts["food_strong_positive_count"] == 0
    ):
        apply_cap(3.2, "service_strong_negative")

    if counts["strong_revisit_negative_count"] >= 1:
        apply_cap(2.7, "strong_revisit_rejection")
    elif counts["revisit_negative_count"] >= 1:
        apply_cap(3.0, "revisit_rejection")

    if counts["recommendation_negative_count"] >= 1:
        apply_cap(3.0, "recommendation_rejection")

    if counts["food_very_strong_negative_count"] >= 1:
        apply_cap(2.5, "food_very_strong_negative")
    elif (
        counts["food_strong_negative_count"] >= 1
        and counts["food_strong_positive_count"] == 0
    ):
        apply_cap(3.0, "food_strong_negative")

    if counts["negative_aspects_with_strong_negative"] >= 2:
        apply_cap(2.8, "multiple_strong_negative_aspects")

    if counts["explicit_negative_conclusion_count"] >= 1:
        apply_cap(3.5, "explicit_negative_conclusion")

    if (
        counts["negative_evidence_count"] >= 3
        and counts["very_strong_positive_count"] == 0
    ):
        apply_cap(3.9, "mixed_negative_evidence")

    # 5단계: mixed review cap. 한 aspect의 긍정이 다른 aspect의 명확한
    # 부정을 완전히 덮지 못하도록 하되, 단순한 약한 불편에는 적용하지 않습니다.
    if counts["positive_aspect_count"] >= 1 and counts["negative_aspect_count"] >= 1:
        if (
            aspect_scores["service"] <= -2
            or aspect_scores["hygiene"] <= -2
            or aspect_scores["revisit"] < 0
        ):
            apply_cap(3.5, "mixed_core_negative", "mixed_review_cap")
        if aspect_scores["food"] > 0 and aspect_scores["service"] < 0:
            apply_cap(3.8, "food_positive_service_negative", "mixed_review_cap")
        if aspect_scores["food"] > 0 and aspect_scores["revisit"] < 0:
            apply_cap(3.3, "food_positive_revisit_negative", "mixed_review_cap")
        if aspect_scores["atmosphere"] > 0 and aspect_scores["food"] < 0:
            apply_cap(3.2, "atmosphere_positive_food_negative", "mixed_review_cap")
        if (
            aspect_scores["food"] > 0
            and (
                aspect_scores["service"] <= -1.5
                or aspect_scores["hygiene"] < 0
                or aspect_scores["revisit"] < 0
            )
        ):
            apply_cap(3.5, "food_positive_cannot_override_core_negative", "mixed_review_cap")

    return {
        "weighted_evidence_score": round(weighted_score, 3),
        "sentiment_star_raw": round(raw_star, 2),
        "sentiment_star": round(max(1.0, min(5.0, star)), 2),
        "sentiment_calibration_reason": " | ".join(dict.fromkeys(reasons)),
        "evidence_floor_items": floor_items,
    }


def format_refactored_evidence(item):
    """evidence_items를 사람이 검토하기 쉬운 matched_words 문자열로 변환."""
    score = float(item["score"])
    score_text = f"+{score:g}" if score > 0 else f"{score:g}"
    if not item["is_active"] and item["original_score"] != 0:
        original = float(item["original_score"])
        return f"{item['text']}:{original:g}->0[{item['guard_reason']}]"
    return f"{item['text']}:{score_text}[{item['tag']}]"


def build_evidence_result(evidence_items):
    """aspect 점수, evidence count, 별점, matched_words를 한 원장에서 생성."""
    aspect_scores = calculate_aspect_scores(evidence_items)
    counts = calculate_evidence_counts(evidence_items)
    star_result = calculate_evidence_sentiment_star(aspect_scores, counts)
    all_items = evidence_items + star_result["evidence_floor_items"]

    result = {}
    for aspect in EVIDENCE_ASPECTS:
        score = aspect_scores[aspect]
        result[f"{aspect}_score"] = score
        result[f"{aspect}_label"] = (
            "positive" if score > 0 else "negative" if score < 0 else "neutral"
        )
        result[f"{aspect}_matched_words"] = " ".join(
            format_refactored_evidence(item)
            for item in all_items
            if item["aspect"] == aspect
        )

    category_total_score = round(sum(aspect_scores.values()), 3)
    result["category_total_score"] = category_total_score
    result["category_total_label"] = (
        "positive" if category_total_score > 0
        else "negative" if category_total_score < 0
        else "neutral"
    )
    result["matched_words"] = " | ".join(
        format_refactored_evidence(item) for item in all_items
    )
    result.update(counts)
    result.update(star_result)
    result["positive_phrase_count"] = sum(
        1
        for item in evidence_items
        if item["is_active"]
        and item["polarity"] == "positive"
        and item["source"] in {"phrase_rule", "pattern_rule", "explicit_rating"}
    )
    result["phrase_score_total"] = round(sum(
        item["score"]
        for item in evidence_items
        if item["is_active"]
        and item["source"] in {"phrase_rule", "pattern_rule", "explicit_rating"}
    ), 3)
    result["dictionary_score_total"] = round(sum(
        item["score"]
        for item in evidence_items
        if item["is_active"] and item["source"] == "dictionary"
    ), 3)
    result["context_guard_score_total"] = round(sum(
        item["score"]
        for item in evidence_items
        if item["is_active"] and item["source"] == "context_guard"
    ), 3)
    result["overall_sentiment_score"] = star_result["weighted_evidence_score"]
    result["overall_context_reason"] = "evidence_weighted_aspect_score"
    result["evidence_items"] = all_items
    return result


def analyze_review(tokens_text, tokens_pos_text, review_text):
    """
    evidence 기반 리팩토링 파이프라인.

    기존 token/phrase/pattern 로직은 후보 추출기로만 사용합니다. 최종 점수는
    표준 evidence_items -> context-resolved active evidence -> aspect score ->
    weighted sentiment_star 순서로만 계산합니다.
    """
    del tokens_pos_text
    normalized_review = normalize_review_text(review_text)
    tokens_fixed = fix_tokens_for_sentiment(tokens_text, normalized_review)
    phrase_candidates = apply_phrase_pattern_rules(normalized_review, tokens_fixed)
    dictionary_candidates = apply_token_dictionary_matching(tokens_fixed)
    resolved_candidates = apply_context_guards(
        dictionary_candidates,
        normalized_review,
        tokens_fixed,
        phrase_candidates,
    )
    evidence_items = [
        refactor_evidence_item(item, normalized_review)
        for item in resolved_candidates
    ]
    evidence_items = resolve_refactored_context(evidence_items, normalized_review)
    evidence_items = deduplicate_evidence(evidence_items)
    return pd.Series(build_evidence_result(evidence_items))


DEBUG_SENTIMENT_PROBES = [
    {
        "term": "맛나다",
        "review_text": "김치가 맛나다",
        "tokens": "김치 나다",
    },
    {
        "term": "맛있음",
        "review_text": "음식 맛있음",
        "tokens": "음식 맛있다",
    },
    {
        "term": "굿",
        "review_text": "분위기 굿",
        "tokens": "분위기 굿",
    },
    {
        "term": "최고",
        "review_text": "서울 최고 맛집",
        "tokens": "서울 최고 맛집",
    },
    {
        "term": "재방문",
        "review_text": "재방문 OK",
        "tokens": "재방문 ok",
    },
    {
        "term": "가성비",
        "review_text": "가성비가 좋다",
        "tokens": "가성비 좋다",
    },
    {
        "term": "줄 설만하다",
        "review_text": "이 정도 맛이면 줄 설만하다",
        "tokens": "정도 맛 줄 설 만하다",
    },
]


def inspect_sentiment_flow(review_text, tokens_text):
    """
    한 리뷰가 phrase/token 후보에서 guard와 최종 점수까지 이동하는 과정을 반환.

    실제 분석 함수와 같은 함수를 호출하므로 debug 결과와 본 분석 결과 사이에
    별도의 간이 로직 차이가 생기지 않습니다.
    """
    normalized_review = normalize_review_text(review_text)
    tokens_fixed = fix_tokens_for_sentiment(tokens_text, normalized_review)
    phrase_matches = apply_phrase_pattern_rules(normalized_review, tokens_fixed)
    token_matches = apply_token_dictionary_matching(tokens_fixed)
    evidence_items = apply_context_guards(
        token_matches,
        normalized_review,
        tokens_fixed,
        phrase_matches,
    )
    result = calculate_category_total_score(evidence_items)

    return {
        "normalized_review": normalized_review,
        "tokens_fixed": tokens_fixed,
        "phrase_matches": phrase_matches,
        "token_matches": token_matches,
        "evidence_items": evidence_items,
        "result": result,
    }


def debug_sentiment_pipeline(score_min=None, score_max=None):
    """
    현재 감성분석 파이프라인의 탐지와 실제 점수 반영 상태를 진단합니다.

    이 함수는 규칙이나 점수를 수정하지 않습니다. 감성사전/phrase/token 탐지,
    evidence 증가, matched_words와 실제 점수의 일치, 원문 미지원 토큰 guard,
    실제 분위수 기준 별점 변환만 출력합니다.
    """
    print("\n" + "=" * 72)
    print("[DEBUG SENTIMENT PIPELINE]")
    print("=" * 72)

    # 1. 실제 감성사전 파일
    print(f"[DEBUG 1] 실제 사용 감성사전: {dict_file.resolve()}")
    print("[DEBUG 1] 감성사전 후보:")
    for candidate in dict_candidates:
        selected_text = " <- SELECTED" if candidate == dict_file else ""
        print(f"  - {candidate} / exists={candidate.exists()}{selected_text}")

    # 2~4. 핵심 표현의 사전/phrase 탐지, 점수 반영, evidence 증가
    print("\n[DEBUG 2-4] 핵심 표현 탐지와 실제 점수/evidence 반영")
    debug_flows = []
    for probe in DEBUG_SENTIMENT_PROBES:
        flow = inspect_sentiment_flow(probe["review_text"], probe["tokens"])
        result = flow["result"]
        debug_flows.append((probe["term"], flow))

        dictionary_exact = probe["term"] in sentiment_map
        token_hits = [
            f"{item['term']}:{item['score']}[{item['category']}]"
            for item in flow["token_matches"]
        ]
        phrase_hits = [
            f"{item['term']}:{item['score']}[{item['tag']}]"
            for item in flow["phrase_matches"]
        ]
        nonzero_categories = {
            cat: result[f"{cat}_score"]
            for cat in categories
            if result[f"{cat}_score"] != 0
        }
        captured = bool(token_hits or phrase_hits)

        print(f"  [{probe['term']}] captured={captured} / dictionary_exact={dictionary_exact}")
        print(f"    token_hits={token_hits or '없음'}")
        print(f"    phrase_hits={phrase_hits or '없음'}")
        print(
            "    "
            f"phrase_score_total={result['phrase_score_total']} / "
            f"category_scores={nonzero_categories or '모두 0'} / "
            f"category_total_score={result['category_total_score']}"
        )
        print(
            "    "
            f"positive_phrase_count={result['positive_phrase_count']} / "
            f"strong_positive_count={result['strong_positive_count']} / "
            f"revisit_positive_count={result['revisit_positive_count']}"
        )

    # 3. phrase-only 입력으로 phrase 합계가 category score에 실제 반영되는지 검사
    phrase_only_flow = inspect_sentiment_flow(
        "재방문 OK 분위기 굿 이 정도 맛이면 줄 설만하다",
        "",
    )
    phrase_only_result = phrase_only_flow["result"]
    expected_phrase_total = sum(
        item["score"]
        for item in phrase_only_flow["evidence_items"]
        if item["source"] == "phrase_rule"
    )
    expected_category_scores = {
        cat: sum(
            item["score"]
            for item in phrase_only_flow["evidence_items"]
            if item["category"] == cat
        )
        for cat in categories
    }
    phrase_score_reflected = (
        expected_phrase_total == phrase_only_result["phrase_score_total"]
        and all(
            expected_category_scores[cat] == phrase_only_result[f"{cat}_score"]
            for cat in categories
        )
        and sum(expected_category_scores.values())
        == phrase_only_result["category_total_score"]
    )
    print("\n[DEBUG 3] phrase rule 실제 점수 반영")
    print(f"  expected_phrase_score_total={expected_phrase_total}")
    print(f"  actual_phrase_score_total={phrase_only_result['phrase_score_total']}")
    print(f"  expected_category_scores={expected_category_scores}")
    print(
        "  actual_category_scores="
        + str({
            cat: phrase_only_result[f"{cat}_score"]
            for cat in categories
        })
    )
    print(f"  category_total_score={phrase_only_result['category_total_score']}")
    print(f"  phrase_score_reflected={phrase_score_reflected}")

    # 6. matched_words에만 남고 실제 점수 합계에 반영되지 않은 항목 검사
    score_mismatches = []
    missing_matched_entries = []
    for probe_name, flow in debug_flows + [("phrase_only_integration", phrase_only_flow)]:
        result = flow["result"]
        expected_scores = {
            cat: sum(
                item["score"]
                for item in flow["evidence_items"]
                if item["category"] == cat
            )
            for cat in categories
        }
        for cat in categories:
            if expected_scores[cat] != result[f"{cat}_score"]:
                score_mismatches.append(
                    {
                        "probe": probe_name,
                        "category": cat,
                        "expected": expected_scores[cat],
                        "actual": result[f"{cat}_score"],
                    }
                )

        for item in flow["evidence_items"]:
            if item["source"] == "phrase_rule":
                rendered = phrase_match_text(item["term"], item["score"], item["tag"])
            else:
                rendered = adjusted_match_text(
                    item["term"],
                    item["original_score"],
                    item["score"],
                    None if item["tag"] == "token" else item["tag"],
                )
            if rendered not in result[f"{item['category']}_matched_words"]:
                missing_matched_entries.append(
                    {
                        "probe": probe_name,
                        "category": item["category"],
                        "entry": rendered,
                    }
                )

    print("\n[DEBUG 6] matched_words와 실제 점수 일치 검사")
    print(f"  category_score_mismatch_count={len(score_mismatches)}")
    print(f"  adjusted_match_missing_from_matched_words_count={len(missing_matched_entries)}")
    print(
        "  matched_words_only_scoring_issue="
        + ("발견됨" if score_mismatches or missing_matched_entries else "없음")
    )
    if score_mismatches:
        print(f"  score_mismatches={score_mismatches}")
    if missing_matched_entries:
        print(f"  missing_matched_entries={missing_matched_entries}")

    # 7. 원문에 없는 위험 부정 토큰 guard
    unsupported_flow = inspect_sentiment_flow(
        "플레이트가 예쁘고 분위기 굿이에요",
        "플레이트 덥다 분위기 굿",
    )
    unsupported_result = unsupported_flow["result"]
    unsupported_items = [
        item
        for item in unsupported_flow["evidence_items"]
        if item["term"] == "덥다"
    ]
    unsupported_negative_score = sum(item["score"] for item in unsupported_items)
    print("\n[DEBUG 7] 원문에 없는 token 감성어 감점 guard")
    print("  review_text=플레이트가 예쁘고 분위기 굿이에요")
    print("  tokens=플레이트 덥다 분위기 굿")
    print(
        "  덥다 처리="
        + str([
            {
                "original_score": item["original_score"],
                "score": item["score"],
                "tag": item["tag"],
            }
            for item in unsupported_items
        ])
    )
    print(f"  unsupported_token_negative_score={unsupported_negative_score}")
    print(
        "  unsupported_negative_blocked="
        + str(bool(unsupported_items) and unsupported_negative_score == 0)
    )
    print(f"  category_total_score={unsupported_result['category_total_score']}")

    # 5. 실제 카카오 5%~95% 기준값이 계산된 뒤 score 0~3의 별점 출력
    print("\n[DEBUG 5] category_total_score별 sentiment_star 변환")
    if score_min is None or score_max is None:
        print("  실제 score_min/score_max가 아직 없어 별점 변환 진단을 보류합니다.")
    else:
        print(f"  score_min={score_min} / score_max={score_max}")
        for score in [0, 1, 2, 3]:
            raw_star = calculate_sentiment_star(score, score_min, score_max)
            no_evidence_row = {
                "category_total_score": score,
                "strong_positive_count": 0,
                "weak_positive_count": 0,
                "strong_negative_count": 0,
                "weak_negative_count": 0,
                "revisit_positive_count": 0,
                "positive_phrase_count": 0,
                "phrase_score_total": 0,
            }
            calibrated = calibrate_sentiment_star(raw_star, no_evidence_row)
            print(
                f"  category_total_score={score} -> "
                f"sentiment_star_raw={raw_star} / "
                f"sentiment_star(no_evidence)={calibrated['sentiment_star']}"
            )

    print("=" * 72 + "\n")


def run_structure_diagnostics():
    """
    전체 CSV 실행 전에 구조적 실패를 짧게 진단합니다.

    rating은 사용하지 않으며, 사전 로딩/일반화 pattern 실점수 반영/evidence
    증가/원문에 없는 위험 토큰 차단이 실제 파이프라인에서 동작하는지 봅니다.
    """
    dictionary_probes = ["맛나다", "맛있다", "굿", "최고", "재방문"]
    dictionary_status = {
        term: term in sentiment_map
        for term in dictionary_probes
    }
    print(f"[구조 진단] 사용 감성사전: {dict_file.name}")
    print(f"[구조 진단] 핵심 사전 표현 로딩: {dictionary_status}")
    print(
        "[구조 진단] 일반화 pattern 규칙 수:",
        len(GENERALIZED_PATTERN_RULES),
        "/ 고확신 phrase 규칙 수:",
        len(PHRASE_RULES),
    )

    pattern_probe = analyze_review(
        "다음 방문 가격 사악하다 만족",
        "",
        "다음에 다시 방문할 예정이고 가격은 사악하지만 만족합니다",
    )
    unsupported_probe = analyze_review(
        "플레이트 덥다 분위기 좋다",
        "",
        "플레이트가 예쁘고 분위기가 좋아요",
    )
    pattern_matched = " ".join(
        str(pattern_probe.get(f"{cat}_matched_words", ""))
        for cat in categories
    )
    unsupported_matched = " ".join(
        str(unsupported_probe.get(f"{cat}_matched_words", ""))
        for cat in categories
    )

    if pattern_probe["phrase_score_total"] <= 0:
        raise AssertionError("일반화 phrase pattern 점수가 실제 점수에 반영되지 않았습니다.")
    if pattern_probe["strong_positive_count"] + pattern_probe["weak_positive_count"] <= 0:
        raise AssertionError("일반화 phrase pattern이 positive evidence count를 증가시키지 않았습니다.")
    if "덥다:-1[token]" in unsupported_matched:
        raise AssertionError("원문에 없는 위험 부정 토큰이 감점에 사용되었습니다.")

    print(
        "[구조 진단] pattern score/evidence 반영:",
        f"phrase_score_total={pattern_probe['phrase_score_total']},",
        f"positive_evidence={pattern_probe['strong_positive_count'] + pattern_probe['weak_positive_count']},",
        f"matched={pattern_matched}",
    )
    print(
        "[구조 진단] 원문 미지원 위험 토큰 차단:",
        "통과" if "덥다:-1[token]" not in unsupported_matched else "실패",
    )


def run_preflight_unit_tests():
    """
    전체 CSV 실행 전 사전 로딩/정규화/n-gram/원문 guard/필수 회귀 사례 검사.
    하나라도 실패하면 전체 처리를 중단해 잘못된 결과 파일 생성을 막습니다.
    """
    failures = []
    test_results = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    # 사전 로딩과 정확한 token n-gram 매칭 자체가 정상인지 먼저 확인.
    check("맛나다" in sentiment_map, "감성사전에 '맛나다'가 로딩되지 않았습니다.")
    exact_matches = apply_token_dictionary_matching("맛나다")
    check(
        any(
            item["term"] == "맛나다"
            and item["category"] == "food"
            and item["score"] > 0
            for item in exact_matches
        ),
        "token n-gram 매칭이 사전의 '맛나다'를 찾지 못했습니다.",
    )
    repaired = fix_tokens_for_sentiment("김치 나다", "김치가 맛나다")
    check("맛나다" in repaired.split(), "'김치가 맛나다' 토큰 복원에 실패했습니다.")

    # suppression은 대체 phrase가 기존 긍정 점수를 충분히 보장할 때만 허용.
    synthetic_token = make_evidence_item(
        term="강한긍정",
        score=2,
        category="general",
        source="dictionary",
        tag="token",
        start_idx=0,
        end_idx=1,
        ngram_size=1,
    )
    weak_replacement = make_evidence_item(
        term="약한 대체 phrase",
        score=1,
        category="general",
        source="phrase_rule",
        tag="synthetic_weak_replacement",
        suppress_terms={"강한긍정"},
        start_idx=None,
        end_idx=None,
    )
    sufficient_replacement = make_evidence_item(
        term="충분한 대체 phrase",
        score=2,
        category="general",
        source="phrase_rule",
        tag="synthetic_sufficient_replacement",
        suppress_terms={"강한긍정"},
        start_idx=None,
        end_idx=None,
    )
    weak_suppression_items = apply_context_guards(
        [synthetic_token],
        "강한긍정",
        "강한긍정",
        [weak_replacement],
    )
    sufficient_suppression_items = apply_context_guards(
        [synthetic_token],
        "강한긍정",
        "강한긍정",
        [sufficient_replacement],
    )
    check(
        weak_suppression_items[0]["score"] == 2
        and weak_suppression_items[0]["tag"] == "suppression_skipped_insufficient_replacement",
        "대체 phrase 점수가 부족한데 기존 긍정 token이 suppress되었습니다.",
    )
    check(
        sufficient_suppression_items[0]["score"] == 0
        and sufficient_suppression_items[0]["tag"] == "suppressed_by_phrase_rule",
        "충분한 대체 phrase가 있는데 중복 긍정 token suppression이 동작하지 않았습니다.",
    )

    # 요청한 sentiment_star evidence floor 조건을 서로 독립적으로 검증.
    floor_base = {
        "strong_positive_count": 0,
        "weak_positive_count": 0,
        "strong_negative_count": 0,
        "weak_negative_count": 0,
        "revisit_positive_count": 0,
        "recommendation_count": 0,
        "waiting_positive_count": 0,
        "waiting_negative_count": 0,
        "phrase_score_total": 0,
    }

    def check_floor(overrides, expected_star, expected_tag=None):
        evidence = {**floor_base, **overrides}
        calibrated = calibrate_sentiment_star(2.0, evidence)
        check(
            calibrated["sentiment_star"] == expected_star,
            f"floor 테스트 {overrides}: {calibrated['sentiment_star']} != {expected_star}",
        )
        floor_tags = {
            item.get("tag")
            for item in calibrated["evidence_floor_items"]
        }
        if expected_tag is None:
            check(
                not calibrated["evidence_floor_items"],
                f"floor 테스트 {overrides}: 적용 금지 조건인데 floor evidence가 생성되었습니다.",
            )
        else:
            check(
                expected_tag in floor_tags,
                f"floor 테스트 {overrides}: 기대 태그 [{expected_tag}]가 없습니다.",
            )
            check(
                all(
                    item.get("source") == "evidence_floor"
                    for item in calibrated["evidence_floor_items"]
                ),
                f"floor 테스트 {overrides}: evidence_floor source가 아닌 항목이 있습니다.",
            )

    check_floor(
        {"strong_positive_count": 2},
        4.0,
        "strong_positive_evidence",
    )
    check_floor(
        {"revisit_positive_count": 1},
        3.8,
        "revisit_positive_evidence",
    )
    check_floor(
        {"weak_positive_count": 3},
        3.7,
        "positive_evidence_count",
    )
    check_floor(
        {"phrase_score_total": 3},
        3.8,
        "phrase_score_evidence",
    )
    check_floor(
        {
            "strong_positive_count": 3,
            "revisit_positive_count": 1,
            "weak_positive_count": 4,
            "phrase_score_total": 7,
            "strong_negative_count": 1,
        },
        2.0,
    )
    check_floor(
        {
            "strong_positive_count": 3,
            "revisit_positive_count": 1,
            "weak_positive_count": 4,
            "phrase_score_total": 7,
            "strong_negative_count": 2,
        },
        2.0,
    )

    # 일반화 pattern이 후행 부정 결론이나 우연한 음절을 긍정으로 오판하지
    # 않는지 확인합니다. 이 반례들은 전체 CSV 실행 전에 반드시 통과해야 합니다.
    pattern_guard_cases = [
        {
            "name": "repeat_visit_then_negative",
            "text": "2번 방문 연속 불은면 받고 많이 속상합니다",
            "tokens": "번 방문 연속 불다 면 받다 많이 속상하다",
            "forbidden_positive_tag": "repeat_visit_pattern",
        },
        {
            "name": "repeat_visit_continued_problem",
            "text": "3번정도 방문했는데 계속 그러네요 아쉽네요",
            "tokens": "번 정도 방문 계속 그렇다 아쉽다",
            "forbidden_positive_tag": "repeat_visit_pattern",
        },
        {
            "name": "habitual_visit_final_rejection",
            "text": "여기 좋아해서 자주왔는데 마음 상해서 이제 안갑니다",
            "tokens": "여기 좋아하다 자주 오다 마음 상하다 이제 안 가다",
            "forbidden_positive_tag": "repeat_visit_pattern",
        },
        {
            "name": "habitual_visit_then_deterioration",
            "text": "자주 갔었지만 웨이팅 길어지면서 맛과 서비스를 잃음",
            "tokens": "자주 가다 웨이팅 길어지다 맛 서비스 잃다",
            "forbidden_positive_tag": "repeat_visit_pattern",
        },
        {
            "name": "degree_do_not_concession",
            "text": "웨이팅 40분 정도 했고 음식은 빨리 나왔어요 맛있게 먹었습니다",
            "tokens": "웨이팅 분 정도 음식 빨리 나오다 맛있다 먹다",
            "forbidden_positive_tag": "concession_positive_resolution_evidence",
        },
    ]

    for guard_case in pattern_guard_cases:
        guard_result = analyze_review(
            guard_case["tokens"],
            "",
            guard_case["text"],
        )
        check(
            not any(
                item["score"] > 0
                and item["tag"] == guard_case["forbidden_positive_tag"]
                for item in guard_result["evidence_items"]
            ),
            (
                f"{guard_case['name']}: 후행 부정/비양보 문맥인데 "
                f"[{guard_case['forbidden_positive_tag']}]가 긍정으로 잡혔습니다."
            ),
        )

    cases = [
        {
            "name": "tier_repeat_tasty",
            "text": "곰탕 1티어 무조건 특으로 먹을 것 2번째 방문 동식이형네는 참 김치가 맛나다",
            "tokens": "곰탕 티어 무조건 먹다 방문 동식 이형 차다 김치 나다",
            "required": ["rank_tier_positive", "menu_recommendation_phrase", "repeat_visit_context", "food_positive_phrase"],
            "forbidden_negative": [],
            "print_result": True,
        },
        {
            "name": "flavor_harmony",
            "text": "고기국물의 구수함 십리향 쌀의 구수함 서로 다른 두 구수함의 조화 고추향을 살리고 매운맛은 덜어낸 독특한 양념",
            "tokens": "국물 구수하다 십리 구수하다 서로 다른 구수하다 조화 고추 살리다 맵다 덜다 내다 독특하다 양념",
            "required": ["food_depth_positive", "flavor_harmony_positive", "balanced_spicy_context"],
            "forbidden_negative": ["맵다:-1[token]"],
        },
        {
            "name": "camping_convenience",
            "text": "날씨 좋은 요즘 홍대에서 야장 느낌을 내면서 고기 먹을 수 있는게 너무 좋은 것 같아요 굳이 차타고 멀리 갈 필요없다 이게 바로 캠핑이지",
            "tokens": "날씨 좋다 요즘 홍대 야장 느낌 내면 고기 먹다 너무 좋다 같다 굳이 타고 멀리 갈다 필요없다 바로 캠핑",
            "required": ["outdoor_atmosphere_positive", "convenience_positive_pattern", "experience_positive_phrase"],
            "forbidden_negative": ["굳이:-1[token]", "필요없다:-1[token]"],
            "print_result": True,
        },
        {
            "name": "pretty_romantic",
            "text": "플레이트부터 넘예더라구요 밖에도 낭만넘치는데 실내도 분위기 굿이에요 맛있게 잘 먹고 가겠습니다",
            "tokens": "플레이 넘다 덥다 낭만 넘치다 실내 분위기 맛있다 자다 먹다 가다",
            "required": ["atmosphere_positive_phrase", "food_positive_phrase", "token_not_supported_by_text"],
            "forbidden_negative": ["덥다:-1[token]"],
        },
        {
            "name": "explicit_rating",
            "text": "총점 : 4 5 맛 : 4 5 분위기 : 3 5 을밀대 2회차 방문 국물은 생각보다 간이 되어있고 면에서는 메밀 냄새가 많이 남 무엇보다 이곳에서 평냉에 소주 맛을 알아버림",
            "tokens": "분위기 을밀대 방문 국물 생각 간이 되어다 메밀 냄새 많이 무엇 평냉 소주 버리다",
            "required": ["explicit_rating_positive", "explicit_taste_rating_positive", "repeat_visit_context", "taste_discovery_positive"],
            "forbidden_negative": [],
        },
        {
            "name": "v22_expensive_queue_rebuttal",
            "text": "비싸도 맛있음 데이터는 거짓말하지 않음 괜히 줄 길게 설까 별로라는 사람들은 맛을 모르는겨",
            "tokens": "비싸다 맛있다 데이터 거짓말 않다 괜히 줄 길다 별로 사람 맛 모르다",
            "required": ["expensive_but_tasty_context", "popularity_positive_context", "other_people_negative_opinion_pattern"],
            "forbidden_negative": ["비싸다:-1[token]", "별로:-1[token]", "길다:-1[token]"],
            "minimum_star": 3.8,
            "print_result": True,
        },
        {
            "name": "v22_service_denial_complete_flavor",
            "text": "불친절은 모르겠구요 맛은 투박하지만 있을 맛은 다 있어요",
            "tokens": "불친절 모르다 맛 투박하다 있다 맛 다 있다",
            "required": ["negated_negative_service", "complete_flavor_positive"],
            "forbidden_negative": ["불친절:-1[token]", "투박하다:-1[token]"],
            "minimum_star": 3.8,
            "print_result": True,
        },
        {
            "name": "v22_waiting_revisit_memory",
            "text": "평냉에 녹두전에 소주 밖에서 웨이팅을 했는데도 또 가고 싶습니다 집 가는 길에 생각이 나요",
            "tokens": "평냉 녹두전 소주 밖 웨이팅 하다 또 가다 싶다 집 가다 길 생각 나다",
            "required": ["waiting_but_revisit_positive", "craving_memory_pattern"],
            "forbidden_negative": [],
            "minimum_star": 3.8,
            "print_result": True,
        },
        {
            "name": "v22_fast_turnover",
            "text": "웨이팅이 있긴했는데 테이블회전율이 빠른편이라 금방 들어갈 수 있었어요 평냉 처음먹어보는 거였는데 거부감없이 먹을 수 있었습니다",
            "tokens": "웨이팅 있다 테이블 회전율 빠르다 편 금방 들어가다 평냉 처음 먹다 거부감 없이 먹다",
            "required": ["wait_turnover_positive", "short_wait_positive", "approachable_taste_positive"],
            "forbidden_negative": [],
            "minimum_star": 3.7,
            "print_result": True,
        },
        {
            "name": "v22_takeout_popularity",
            "text": "일요일 아침에 오픈 직후에 갔는데 홀은 반 정도 찼고 20분 밥 먹는 동안 냄비 3개 김치통 2개 테이크아웃 하는 거 봄 1회용 포장 나가는 건 다 못 셈 ㅋㅋ 나도 다음에 남비 들고 가야겠다",
            "tokens": "일요일 아침 오픈 직후 가다 홀 반 차다 밥 먹다 동안 냄비 김치통 테이크아웃 보다 포장 나가다 다 못 세다 다음 남비 들다 가다",
            "required": ["takeout_intention_phrase", "popularity_positive_context"],
            "forbidden_negative": [],
            "minimum_star": 3.0,
            "print_result": True,
        },
        {
            "name": "v22_cheese_trust_refill",
            "text": "홍대에서 치즈 폭포에 제대로 빠지고 싶다면 여기 100 국내산 닭이라 믿고 먹는데 쌈채소 무한리필이라 고삐 풀릴 수 있으니 주의하세요 ㅋㅋ 마지막 장인볶음밥까지 먹으면 끝나쥬",
            "tokens": "홍대 치즈 폭포 제대로 빠지다 싶다 여기 국내산 닭 믿다 먹다 쌈채소 무한 리필 고삐 풀리다 주의 마지막 장인 볶음밥 먹다 끝나다",
            "required": ["food_experience_positive", "trust_positive_phrase", "value_positive_pattern", "menu_must_try_phrase", "satisfying_finish_positive"],
            "forbidden_negative": [],
            "minimum_star": 4.0,
            "print_result": True,
        },
        {
            "name": "v23_deep_taste_idiom_revisit",
            "text": "후배랑 둘이서 3메뉴 한 개씩 주문함 찌개는 묵은지로 해서 깊은 맛이 나고 밑반찬들도 다 맛있어서 밥이 게눈 감추듯 사라짐 재방문 OK",
            "tokens": "후배 둘 메뉴 주문 찌개 묵은지 깊다 맛 나다 밑반찬 다 맛있다 밥 게눈 감추다 사라지다 재방문 ok",
            "required": ["food_depth_positive", "positive_idiom", "revisit_positive_phrase"],
            "forbidden_negative": ["사라지다:-1[token]"],
            "minimum_star": 3.8,
            "print_result": True,
        },
        {
            "name": "v23_revision_final_positive",
            "text": "5회 이상 방문 이 집만큼 과대평가된 집 찾기 힘듬 수정 생각이 짧았습니다 기복이 좀 있는 편이긴하지만 그래도 서울 최고 김치찌개집이다",
            "tokens": "회 이상 방문 집 과대평가 집 찾다 힘들다 수정 생각 짧다 기복 있다 편 하지만 그래도 서울 최고 김치찌개 집",
            "required": ["repeat_visit_phrase", "strong_final_positive", "revised_by_later_positive_context"],
            "forbidden_negative": ["힘들다:-1[token]"],
            "minimum_star": 3.8,
            "print_result": True,
        },
        {
            "name": "v23_generalized_revisit_variant",
            "text": "다음에는 친구들과 다시 먹으러 가야겠어요",
            "tokens": "다음 친구 다시 먹다 가다",
            "required": ["revisit_positive_pattern"],
            "forbidden_negative": [],
            "minimum_star": 3.8,
            "print_result": True,
        },
        {
            "name": "v23_generalized_price_concession_variant",
            "text": "가격은 조금 사악하지만 재방문할 듯합니다",
            "tokens": "가격 조금 사악하다 하지만 재방문 하다",
            "required": ["expensive_but_positive_pattern", "revisit_positive_phrase"],
            "forbidden_negative": [],
            "minimum_star": 3.8,
            "print_result": True,
        },
        {
            "name": "v23_generalized_rebuttal_variant",
            "text": "별로라던데 저는 맛이 좋았어요",
            "tokens": "별로 저 맛 좋다",
            "required": ["other_people_negative_opinion_pattern", "food_positive_phrase"],
            "forbidden_negative": ["별로:-1[token]"],
            "minimum_star": 2.4,
            "print_result": True,
        },
    ]

    for case in cases:
        result = analyze_review(case["tokens"], "", case["text"])
        required_evidence_keys = {"term", "score", "category", "source", "tag", "strength"}
        check(
            all(
                required_evidence_keys.issubset(item)
                for item in result["evidence_items"]
            ),
            f"{case['name']}: evidence_items 필수 필드가 누락되었습니다.",
        )
        check(
            sum(item["score"] for item in result["evidence_items"])
            == result["category_total_score"],
            f"{case['name']}: evidence_items 합계와 category_total_score가 다릅니다.",
        )
        matched_text = " ".join(
            str(result.get(f"{cat}_matched_words", ""))
            for cat in EVIDENCE_ASPECTS
        )
        missing_required_tags = [
            required
            for required in case["required"]
            if f"[{required}]" not in matched_text
        ]
        forbidden_hits = [
            forbidden
            for forbidden in case["forbidden_negative"]
            if forbidden in matched_text
        ]
        for required in case["required"]:
            check(
                f"[{required}]" in matched_text,
                f"{case['name']}: 필수 태그 [{required}]가 없습니다.",
            )
        for forbidden in case["forbidden_negative"]:
            check(
                forbidden not in matched_text,
                f"{case['name']}: 금지된 감점 '{forbidden}'이 남았습니다.",
            )
        check(
            result["category_total_score"] > 0,
            f"{case['name']}: 최종 category_total_score가 양수가 아닙니다.",
        )

        test_star = result["sentiment_star"]
        if "minimum_star" in case:
            check(
                test_star >= case["minimum_star"],
                f"{case['name']}: 테스트 sentiment_star {test_star}가 기대 하한 {case['minimum_star']}보다 낮습니다.",
            )

        test_results.append({
            "test_name": case["name"],
            "review_text": case["text"],
            "tokens": case["tokens"],
            "status": (
                "PASS"
                if (
                    not missing_required_tags
                    and not forbidden_hits
                    and result["category_total_score"] > 0
                    and test_star >= case.get("minimum_star", 0)
                )
                else "FAIL"
            ),
            "required_tags": " | ".join(case["required"]),
            "missing_required_tags": " | ".join(missing_required_tags),
            "forbidden_hits": " | ".join(forbidden_hits),
            "category_total_score": result["category_total_score"],
            "sentiment_star": test_star,
            "phrase_score_total": result["phrase_score_total"],
            "positive_evidence_count": result["positive_evidence_count"],
            "negative_evidence_count": result["negative_evidence_count"],
            "matched_words": matched_text,
        })

        if case.get("print_result"):
            print_console_safely(
                f"[테스트 {case['name']}] "
                f"review_text={case['text']} / "
                f"phrase_score_total={result['phrase_score_total']} / "
                f"strong_positive_count={result['strong_positive_count']} / "
                f"weak_positive_count={result['weak_positive_count']} / "
                f"strong_negative_count={result['strong_negative_count']} / "
                f"category_total_score={result['category_total_score']} / "
                f"sentiment_star={test_star} / "
                f"calibration_reason={result['sentiment_calibration_reason']} / "
                f"matched_words={matched_text}"
            )

    pd.DataFrame(test_results).to_csv(
        test_results_output,
        index=False,
        encoding="utf-8-sig",
    )

    if failures:
        raise AssertionError(
            "[사전 실행 단위 테스트 실패]\n- " + "\n- ".join(failures)
        )

    print(f"[사전 실행 단위 테스트 통과] {16 + len(cases)}개 검사")
    print("[사전 실행 리뷰 테스트 결과]", test_results_output)


def run_overestimation_prevention_tests():
    """전체 CSV 실행 전 과대평가 방지 회귀 테스트 10개를 검사합니다."""
    cases = [
        {
            "name": "food_positive_service_revisit_negative",
            "text": "맛은 괜찮았는데 직원이 너무 불친절해서 다시는 안 갈 것 같아요",
            "tokens": "맛 괜찮다 직원 너무 불친절하다 다시 안 가다 같다",
            "max_star": 3.0,
            "required_active_tag": "explicit_revisit_rejection",
        },
        {
            "name": "reputation_then_disappointment",
            "text": "유명하대서 갔는데 솔직히 별로였고 돈 아까웠어요",
            "tokens": "유명하다 가다 솔직히 별로 돈 아깝다",
            "max_star": 3.0,
            "required_active_tag": "active_food_negative_conclusion",
        },
        {
            "name": "atmosphere_positive_food_revisit_negative",
            "text": "분위기는 좋았는데 음식이 너무 맛없어서 재방문은 안 할 듯",
            "tokens": "분위기 좋다 음식 너무 맛없다 재방문 안 하다 듯",
            "max_star": 3.0,
            "required_active_tag": "explicit_revisit_rejection",
        },
        {
            "name": "past_positive_current_negative",
            "text": "예전엔 맛있었는데 지금은 맛이 변했고 다시 갈 생각은 없어요",
            "tokens": "예전 맛있다 지금 맛 변하다 다시 가다 생각 없다",
            "max_star": 3.0,
            "required_guard": "past_positive_not_current",
        },
        {
            "name": "other_target_positive",
            "text": "여기 말고 옆집이 훨씬 맛있어요 여기는 그냥 별로",
            "tokens": "여기 말고 옆집 훨씬 맛있다 여기 그냥 별로",
            "max_star": 3.0,
            "required_guard": "other_target_positive",
        },
        {
            "name": "weak_food_hygiene_negative",
            "text": "음식은 먹을 만했지만 위생이 너무 별로라 추천하기 어렵습니다",
            "tokens": "음식 먹다 만하다 위생 너무 별로 추천 어렵다",
            "max_star": 3.0,
            "required_active_tag": "active_hygiene_negative_conclusion",
        },
        {
            "name": "wait_service_negative_plain_food",
            "text": "웨이팅도 길고 직원 응대도 별로였어요 음식은 평범했습니다",
            "tokens": "웨이팅 길다 직원 응대 별로 음식 평범하다",
            "max_star": 3.3,
            "required_active_tag": "active_service_negative_conclusion",
        },
        {
            "name": "food_positive_price_quantity_revisit_negative",
            "text": "맛있긴 한데 가격이 너무 비싸고 양도 적어서 다시 가진 않을 듯",
            "tokens": "맛있다 가격 너무 비싸다 양도 적다 다시 가다 않다 듯",
            "max_star": 3.5,
            "required_active_tag": "explicit_revisit_rejection",
        },
        {
            "name": "rating_expectation_disappointment",
            "text": "평점 좋아서 기대했는데 너무 실망했습니다",
            "tokens": "평점 좋다 기대하다 너무 실망하다",
            "max_star": 3.0,
            "required_guard": "expectation_not_actual_positive",
        },
        {
            "name": "photo_positive_food_service_negative",
            "text": "사진은 예쁘게 나오는데 맛은 없고 서비스도 별로였어요",
            "tokens": "사진 예쁘다 나오다 맛 없다 서비스 별로",
            "max_star": 3.0,
            "required_active_tag": "active_service_negative_conclusion",
        },
    ]

    records = []
    failures = []

    for case in cases:
        result = analyze_review(case["tokens"], "", case["text"])
        active_positive = [
            item for item in result["evidence_items"]
            if item["is_active"] and item["polarity"] == "positive"
        ]
        active_negative = [
            item for item in result["evidence_items"]
            if item["is_active"] and item["polarity"] == "negative"
        ]
        inactive_guards = {
            item.get("guard_reason")
            for item in result["evidence_items"]
            if not item["is_active"]
        }
        active_tags = {
            item.get("tag")
            for item in result["evidence_items"]
            if item["is_active"]
        }
        reason = str(result["sentiment_calibration_reason"])
        passed = result["sentiment_star"] <= case["max_star"]

        if case.get("required_active_tag"):
            passed = passed and case["required_active_tag"] in active_tags
        if case.get("required_guard"):
            passed = passed and case["required_guard"] in inactive_guards

        records.append({
            "test_name": case["name"],
            "review_text": case["text"],
            "status": "PASS" if passed else "FAIL",
            "max_expected_star": case["max_star"],
            "sentiment_star": result["sentiment_star"],
            "category_total_score": result["category_total_score"],
            "positive_floor_applied": "sentiment_floor:" in reason,
            "negative_cap_applied": "sentiment_cap:" in reason,
            "mixed_review_cap_applied": "mixed_review_cap:" in reason,
            "sentiment_calibration_reason": reason,
            "active_positive_evidence": json.dumps(active_positive, ensure_ascii=False),
            "active_negative_evidence": json.dumps(active_negative, ensure_ascii=False),
            "evidence_items": json.dumps(result["evidence_items"], ensure_ascii=False),
            **{
                f"{aspect}_score": result[f"{aspect}_score"]
                for aspect in EVIDENCE_ASPECTS
            },
        })

        if not passed:
            failures.append(
                f"{case['name']}: star={result['sentiment_star']}, "
                f"reason={reason}, guards={sorted(str(x) for x in inactive_guards if x)}"
            )

        print_console_safely(
            f"[과대평가 방지 테스트 {case['name']}] "
            f"star={result['sentiment_star']} / max={case['max_star']} / "
            f"reason={reason} / active_positive={len(active_positive)} / "
            f"active_negative={len(active_negative)}"
        )

    pd.DataFrame(records).to_csv(
        overcap_test_results_output,
        index=False,
        encoding="utf-8-sig",
    )

    if failures:
        raise AssertionError(
            "[과대평가 방지 테스트 실패]\n- " + "\n- ".join(failures)
        )

    print("[과대평가 방지 테스트 통과] 10개")
    print("[과대평가 방지 테스트 결과]", overcap_test_results_output)


def run_final_tuning_tests():
    """인용/시점/타깃 분리와 cap/floor 정밀 보정 회귀 테스트 10개."""
    cases = [
        {
            "name": "quoted_food_negative",
            "text": "맛없다는 후기가 있던데 저는 맛있게 먹었어요",
            "tokens": "맛없다 후기 있다 저 맛있다 먹다",
            "min_star": 3.4,
            "required_guard": "quoted_negative_not_actual_negative",
        },
        {
            "name": "quoted_service_negative",
            "text": "불친절하다는 말이 있는데 저는 전혀 못 느꼈고 친절했어요",
            "tokens": "불친절하다 말 있다 저 전혀 못 느끼다 친절하다",
            "min_star": 3.4,
            "required_guard": "quoted_negative_not_actual_negative",
        },
        {
            "name": "other_people_negative_opinion",
            "text": "별로라는 사람들도 있던데 제 입맛에는 딱이었어요",
            "tokens": "별로 사람 있다 제 입맛 맞다",
            "min_star": 3.0,
            "required_guards_any": {
                "other_people_negative_opinion_pattern",
                "not_restaurant_negative_pattern",
                "quoted_negative_not_actual_negative",
            },
        },
        {
            "name": "past_positive_current_negative_direction",
            "text": "예전엔 맛있었는데 지금은 별로였어요",
            "tokens": "예전 맛있다 지금 별로",
            "max_star": 3.0,
            "required_guard": "past_positive_not_current",
        },
        {
            "name": "past_negative_current_positive_direction",
            "text": "예전엔 별로였는데 이번에는 진짜 맛있었어요",
            "tokens": "예전 별로 이번 진짜 맛있다",
            "min_star": 3.5,
            "required_guard": "past_negative_not_current",
        },
        {
            "name": "expectation_not_actual_positive",
            "text": "유명하대서 갔는데 솔직히 실망했습니다",
            "tokens": "유명하다 가다 솔직히 실망하다",
            "max_star": 3.0,
            "required_active_tag": "active_current_negative_conclusion",
        },
        {
            "name": "reputation_confirmed_positive",
            "text": "유명한 이유가 있네요 정말 맛있었습니다",
            "tokens": "유명하다 이유 있다 정말 맛있다",
            "min_star": 3.5,
            "required_active_tag": "reputation_confirmed_positive",
        },
        {
            "name": "other_target_positive_excluded",
            "text": "여기 말고 옆집이 훨씬 맛있고 여기는 별로였어요",
            "tokens": "여기 말고 옆집 훨씬 맛있다 여기 별로",
            "max_star": 3.0,
            "required_guard": "other_target_positive",
        },
        {
            "name": "branch_positive_current_negative",
            "text": "본점은 맛있는데 여기는 맛이 아쉬웠어요",
            "tokens": "본점 맛있다 여기 맛 아쉽다",
            "max_star": 3.5,
            "required_guard": "other_target_positive",
        },
        {
            "name": "mixed_food_service_revisit_negative",
            "text": "맛은 괜찮았지만 직원 응대가 너무 별로라 다시 가진 않을 것 같아요",
            "tokens": "맛 괜찮다 직원 응대 너무 별로 다시 가다 않다 같다",
            "max_star": 3.5,
            "required_active_tag": "explicit_revisit_rejection",
        },
    ]

    records = []
    failures = []
    for case in cases:
        result = analyze_review(case["tokens"], "", case["text"])
        evidence = result["evidence_items"]
        active_positive = [
            item for item in evidence
            if item["is_active"] and item["polarity"] == "positive"
        ]
        active_negative = [
            item for item in evidence
            if item["is_active"] and item["polarity"] == "negative"
        ]
        inactive = [item for item in evidence if not item["is_active"]]
        inactive_guards = {item.get("guard_reason") for item in inactive}
        active_tags = {item.get("tag") for item in evidence if item["is_active"]}
        passed = True

        if "min_star" in case:
            passed = passed and result["sentiment_star"] >= case["min_star"]
        if "max_star" in case:
            passed = passed and result["sentiment_star"] <= case["max_star"]
        if case.get("required_guard"):
            passed = passed and case["required_guard"] in inactive_guards
        if case.get("required_guards_any"):
            passed = passed and bool(case["required_guards_any"] & inactive_guards)
        if case.get("required_active_tag"):
            passed = passed and case["required_active_tag"] in active_tags

        records.append({
            "test_name": case["name"],
            "review_text": case["text"],
            "status": "PASS" if passed else "FAIL",
            "min_expected_star": case.get("min_star"),
            "max_expected_star": case.get("max_star"),
            "sentiment_star": result["sentiment_star"],
            "category_total_score": result["category_total_score"],
            "active_positive_evidence": json.dumps(active_positive, ensure_ascii=False),
            "inactive_evidence_with_guard_reason": json.dumps(inactive, ensure_ascii=False),
            "active_negative_evidence": json.dumps(active_negative, ensure_ascii=False),
            "aspect_scores": json.dumps(
                {aspect: result[f"{aspect}_score"] for aspect in EVIDENCE_ASPECTS},
                ensure_ascii=False,
            ),
            "applied_floor_cap_reason": result["sentiment_calibration_reason"],
        })

        if not passed:
            failures.append(
                f"{case['name']}: star={result['sentiment_star']}, "
                f"guards={sorted(str(x) for x in inactive_guards if x)}, "
                f"active_tags={sorted(str(x) for x in active_tags if x)}"
            )

    pd.DataFrame(records).to_csv(
        final_tuning_test_results_output,
        index=False,
        encoding="utf-8-sig",
    )
    if failures:
        raise AssertionError("[최종 튜닝 테스트 실패]\n- " + "\n- ".join(failures))
    print("[최종 튜닝 추가 테스트 통과] 10개")
    print("[최종 튜닝 추가 테스트 결과]", final_tuning_test_results_output)


def run_minimal_error_tuning_tests():
    """남은 오류 유형만 겨냥한 최소 수정 회귀 테스트 10개."""
    cases = [
        {
            "name": "conditional_warning_not_revisit_rejection",
            "text": "시끄러운 거 싫어하면 평일 저녁엔 가지마세요 저는 보쌈은 맛있게 먹었습니다",
            "tokens": "시끄럽다 싫어하다 평일 저녁 가다 말다 저 보쌈 맛있다 먹다",
            "min_star": 2.0,
            "required_guard": "conditional_warning_not_revisit_rejection",
            "forbidden_active_tags": {
                "explicit_revisit_rejection",
                "explicit_recommendation_rejection",
            },
        },
        {
            "name": "other_target_positive_excluded_precisely",
            "text": "진짜 맛집은 옆집이고 여기는 그냥 별로였어요",
            "tokens": "진짜 맛집 옆집 여기 그냥 별로",
            "max_star": 3.0,
            "required_guard": "other_target_positive",
        },
        {
            "name": "ironic_johnmat_not_actual",
            "text": "외국인들이 이게 존맛이라고 생각할까봐 걱정됩니다",
            "tokens": "외국인 이게 존맛 생각하다 걱정",
            "max_star": 3.0,
            "required_guard": "ironic_or_expectation_positive_not_actual",
        },
        {
            "name": "past_positive_current_negative_precise",
            "text": "예전에는 맛있고 좋았는데 지금은 초심 잃고 변했네요",
            "tokens": "예전 맛있다 좋다 지금 초심 잃다 변하다",
            "max_star": 3.0,
            "required_guard": "past_positive_not_current",
            "required_active_tag": "past_positive_current_negative",
        },
        {
            "name": "past_negative_current_positive_precise",
            "text": "예전엔 별로였는데 이번에는 진짜 맛있었어요",
            "tokens": "예전 별로 이번 진짜 맛있다",
            "min_star": 3.5,
            "required_guard": "past_negative_not_current",
        },
        {
            "name": "not_really_positive_evaluation",
            "text": "설렁탕이 맛있는 것도 아니고 반찬이 다양한 것도 아니고 가격만 비싸요",
            "tokens": "설렁탕 맛있다 아니다 반찬 다양하다 아니다 가격 비싸다",
            "max_star": 3.0,
            "required_guard": "not_really_positive_evaluation",
            "required_active_tag": "not_really_positive_evaluation",
        },
        {
            "name": "queue_order_operation_failure",
            "text": "우리보다 뒷번호를 먼저 앉히고 차례가 됐는데 아니라고 나가라 하더라고요",
            "tokens": "우리 뒷번호 먼저 앉히다 차례 되다 아니다 나가라",
            "max_star": 2.5,
            "required_active_tag": "active_service_negative_conclusion",
        },
        {
            "name": "silent_waiting_change_and_close",
            "text": "웨이팅 안내가 말도 없이 바뀌고 직원에게 물어보니 마감됐다고 하네요",
            "tokens": "웨이팅 안내 말 없이 바뀌다 직원 물어보다 마감되다",
            "max_star": 2.5,
            "required_active_tag": "active_wait_operation_negative",
        },
        {
            "name": "non_food_sweet_expression",
            "text": "달달 외운 영어로만 응대하네요",
            "tokens": "달달 외우다 영어 응대하다",
            "max_star": 3.0,
            "required_guard": "non_food_sweet_expression",
        },
        {
            "name": "soft_noodle_negative",
            "text": "면발이 힘이 없어지고 너무 부드러워져서 아쉬웠어요",
            "tokens": "면발 힘 없다 너무 부드럽다 아쉽다",
            "max_star": 3.0,
            "required_guard": "soft_noodle_negative_context",
            "required_active_tag": "soft_noodle_negative_context",
        },
    ]

    records = []
    failures = []
    for case in cases:
        result = analyze_review(case["tokens"], "", case["text"])
        evidence = result["evidence_items"]
        active_positive = [
            item for item in evidence
            if item["is_active"] and item["polarity"] == "positive"
        ]
        active_negative = [
            item for item in evidence
            if item["is_active"] and item["polarity"] == "negative"
        ]
        inactive = [item for item in evidence if not item["is_active"]]
        inactive_guards = {item.get("guard_reason") for item in inactive}
        active_tags = {item.get("tag") for item in evidence if item["is_active"]}
        passed = True

        if "min_star" in case:
            passed = passed and result["sentiment_star"] >= case["min_star"]
        if "max_star" in case:
            passed = passed and result["sentiment_star"] <= case["max_star"]
        if case.get("required_guard"):
            passed = passed and case["required_guard"] in inactive_guards
        if case.get("required_active_tag"):
            passed = passed and case["required_active_tag"] in active_tags
        if case.get("forbidden_active_tags"):
            passed = passed and not (case["forbidden_active_tags"] & active_tags)

        records.append({
            "test_name": case["name"],
            "review_text": case["text"],
            "status": "PASS" if passed else "FAIL",
            "sentiment_star": result["sentiment_star"],
            "active_positive_evidence": json.dumps(active_positive, ensure_ascii=False),
            "inactive_evidence_with_guard_reason": json.dumps(inactive, ensure_ascii=False),
            "active_negative_evidence": json.dumps(active_negative, ensure_ascii=False),
            "aspect_scores": json.dumps(
                {aspect: result[f"{aspect}_score"] for aspect in EVIDENCE_ASPECTS},
                ensure_ascii=False,
            ),
            "calibration_reason": result["sentiment_calibration_reason"],
        })

        if not passed:
            failures.append(
                f"{case['name']}: star={result['sentiment_star']}, "
                f"guards={sorted(str(x) for x in inactive_guards if x)}, "
                f"active_tags={sorted(str(x) for x in active_tags if x)}"
            )

    pd.DataFrame(records).to_csv(
        minimal_tuning_test_results_output,
        index=False,
        encoding="utf-8-sig",
    )
    if failures:
        raise AssertionError("[최소 오류 튜닝 테스트 실패]\n- " + "\n- ".join(failures))
    print("[최소 오류 튜닝 테스트 통과] 10개")
    print("[최소 오류 튜닝 테스트 결과]", minimal_tuning_test_results_output)


def run_positive_evidence_recovery_tests():
    """고평점 positive evidence 0 사례의 안전 회복 및 오탐 방지 테스트 15개."""
    positive_tags = {
        "colloquial_tasty_positive_pattern",
        "menu_must_try_pattern",
        "menu_omission_positive_pattern",
        "revisit_positive_pattern",
        "food_drink_craving_pattern",
        "food_quality_positive_pattern",
        "speed_service_positive_pattern",
    }
    cases = [
        {
            "name": "colloquial_tasty_menu_must_try",
            "text": "맛난 갈비에 공기밥은 필수고 찌개는 빠트리면 서운해요",
            "tokens": "갈비 공기밥 필수 찌개 빠트리다 서운하다",
            "min_positive": 2,
            "required_tags": {"colloquial_tasty_positive_pattern", "menu_must_try_pattern"},
        },
        {
            "name": "tasty_nayo",
            "text": "고기 맛나요 반찬도 맛나요",
            "tokens": "고기 반찬",
            "min_positive": 1,
            "required_tags": {"colloquial_tasty_positive_pattern"},
        },
        {
            "name": "tasty_nayoung_fast_service",
            "text": "가격대가 좀 나가지만 맛나영 음식도 빨리 나옴",
            "tokens": "가격대 나가다 음식 빨리 나오다",
            "min_positive": 2,
            "required_tags": {
                "colloquial_tasty_positive_pattern",
                "speed_service_positive_pattern",
            },
            "required_negative_tag": "price_cost_negative_pattern",
        },
        {
            "name": "misspelled_tasty",
            "text": "넘 마싯던데요",
            "tokens": "넘다",
            "min_positive": 1,
            "required_tags": {"colloquial_tasty_positive_pattern"},
        },
        {
            "name": "revisit_intent_good",
            "text": "재방문의사 굿입니다",
            "tokens": "재방문 의사 굿",
            "min_positive": 1,
            "required_tags": {"revisit_positive_pattern"},
        },
        {
            "name": "worth_traveling",
            "text": "찾아서 올 만한 집이에요",
            "tokens": "찾다 오다 만하다 집",
            "min_positive": 1,
            "required_tags": {"revisit_positive_pattern"},
        },
        {
            "name": "food_drink_craving",
            "text": "한 입 먹는 순간 바로 소주 한 병 생각났어요",
            "tokens": "한 입 먹다 순간 바로 소주 한 병 생각나다",
            "min_positive": 1,
            "required_tags": {"food_drink_craving_pattern"},
        },
        {
            "name": "jjangjjang_food_value",
            "text": "가성비 구성 맛 진짜 짱짱입니다",
            "tokens": "가성비 구성 맛 진짜",
            "min_positive": 1,
            "required_tags": {"food_quality_positive_pattern"},
        },
        {
            "name": "reservation_required_not_positive",
            "text": "예약 필수라서 불편했어요",
            "tokens": "예약 필수 불편하다",
            "max_recovery_positive": 0,
        },
        {
            "name": "waiting_required_revisit_negative",
            "text": "웨이팅 필수인 곳이라 다시 가긴 힘들 것 같아요",
            "tokens": "웨이팅 필수 곳 다시 가다 힘들다 같다",
            "max_recovery_positive": 0,
            "max_star": 3.0,
        },
        {
            "name": "tasty_expectation_disappointment",
            "text": "맛난 줄 알았는데 별로였어요",
            "tokens": "맛나다 줄 알다 별로",
            "max_recovery_positive": 0,
            "max_star": 3.0,
        },
        {
            "name": "worth_traveling_negated",
            "text": "찾아서 올 만하진 않네요",
            "tokens": "찾다 오다 만하다 않다",
            "max_recovery_positive": 0,
        },
        {
            "name": "revisit_intent_absent",
            "text": "재방문 의사 없음",
            "tokens": "재방문 의사 없다",
            "max_recovery_positive": 0,
            "max_star": 3.0,
        },
        {
            "name": "good_ironic_disappointment",
            "text": "굿이라고 하기엔 아쉬웠어요",
            "tokens": "굿 아쉽다",
            "max_recovery_positive": 0,
            "required_guard": "ironic_or_expectation_positive_not_actual",
        },
        {
            "name": "improvement_required_not_positive",
            "text": "개선 필수입니다",
            "tokens": "개선 필수",
            "max_recovery_positive": 0,
        },
    ]

    records = []
    failures = []
    for case in cases:
        result = analyze_review(case["tokens"], "", case["text"])
        evidence = result["evidence_items"]
        active_positive = [
            item for item in evidence
            if item["is_active"] and item["polarity"] == "positive"
        ]
        active_negative = [
            item for item in evidence
            if item["is_active"] and item["polarity"] == "negative"
        ]
        inactive = [item for item in evidence if not item["is_active"]]
        active_tags = {item.get("tag") for item in evidence if item["is_active"]}
        inactive_guards = {item.get("guard_reason") for item in inactive}
        recovery_positive = [
            item for item in active_positive if item.get("tag") in positive_tags
        ]
        passed = True

        if "min_positive" in case:
            passed = passed and len(active_positive) >= case["min_positive"]
        if "max_recovery_positive" in case:
            passed = passed and len(recovery_positive) <= case["max_recovery_positive"]
        if case.get("required_tags"):
            passed = passed and case["required_tags"].issubset(active_tags)
        if case.get("required_negative_tag"):
            passed = passed and case["required_negative_tag"] in active_tags
        if case.get("required_guard"):
            passed = passed and case["required_guard"] in inactive_guards
        if "max_star" in case:
            passed = passed and result["sentiment_star"] <= case["max_star"]

        records.append({
            "test_name": case["name"],
            "review_text": case["text"],
            "status": "PASS" if passed else "FAIL",
            "sentiment_star": result["sentiment_star"],
            "active_positive_evidence": json.dumps(active_positive, ensure_ascii=False),
            "inactive_evidence_with_guard_reason": json.dumps(inactive, ensure_ascii=False),
            "active_negative_evidence": json.dumps(active_negative, ensure_ascii=False),
            "aspect_scores": json.dumps(
                {aspect: result[f"{aspect}_score"] for aspect in EVIDENCE_ASPECTS},
                ensure_ascii=False,
            ),
            "calibration_reason": result["sentiment_calibration_reason"],
        })

        if not passed:
            failures.append(
                f"{case['name']}: star={result['sentiment_star']}, "
                f"active_tags={sorted(str(x) for x in active_tags if x)}, "
                f"guards={sorted(str(x) for x in inactive_guards if x)}"
            )

    pd.DataFrame(records).to_csv(
        positive_recovery_test_results_output,
        index=False,
        encoding="utf-8-sig",
    )
    if failures:
        raise AssertionError("[positive evidence 회복 테스트 실패]\n- " + "\n- ".join(failures))
    print("[positive evidence 회복 테스트 통과] 15개")
    print("[positive evidence 회복 테스트 결과]", positive_recovery_test_results_output)


run_structure_diagnostics()
run_preflight_unit_tests()
run_overestimation_prevention_tests()
run_final_tuning_tests()
run_minimal_error_tuning_tests()
run_positive_evidence_recovery_tests()


df["tokens_fixed"] = df.apply(
    lambda row: fix_tokens_for_sentiment(row["tokens"], row["review_text"]),
    axis=1,
)

result_df = df.apply(
    lambda row: analyze_review(
        row["tokens_fixed"],
        row["tokens_pos"],
        row["review_text"],
    ),
    axis=1
)

df = pd.concat([df, result_df], axis=1)

# =========================================================
# 8-1. evidence 기반 sentiment_star 확정
# =========================================================
# analyze_review 단계에서 aspect score와 evidence count를 모두 반영했습니다.
# rating과 카카오 분위수는 sentiment_star 계산에 사용하지 않습니다.
print("[evidence refactor] sentiment_star는 rating/분위수 없이 active evidence로 계산했습니다.")

if DEBUG_ONLY:
    print("[DEBUG ONLY] 진단 완료. 결과 CSV 저장 없이 종료합니다.")
    sys.exit(0)

# CSV에서 파싱 가능한 JSON 문자열로 보존합니다.
df["evidence_items"] = df["evidence_items"].apply(
    lambda items: json.dumps(items, ensure_ascii=False)
)
df["evidence_floor_items"] = df["evidence_floor_items"].apply(
    lambda items: json.dumps(items, ensure_ascii=False)
)

if "rating" in df.columns:
    df["diff"] = df["rating"] - df["sentiment_star"]
    df["diff_abs"] = df["diff"].abs()
    df["diff_squared"] = df["diff"] ** 2

# =========================================================
# 8-2. 높은 실제 별점인데 분석점수가 낮은 사례와 사전 누락 후보 저장
# =========================================================

CANDIDATE_STOP_TOKENS = {
    "그리고", "그래서", "하지만", "그렇지만", "그냥", "정도", "정말",
    "진짜", "너무", "매우", "아주", "조금", "좀", "같다", "이다",
    "저희", "제가", "나는", "오늘", "이번", "여기", "거기", "이곳",
    "하다", "있다", "없다", "되다", "보다", "가다", "오다", "먹다",
    "들다", "나다", "주다", "그렇다", "아니다", "않다",
}

POSITIVE_CANDIDATE_HINTS = {
    "좋", "맛있", "괜찮", "훌륭", "추천", "감동", "최고", "만족",
    "친절", "깔끔", "부드럽", "푸짐", "신선", "고소", "쫄깃",
    "든든", "완벽", "재방문", "단골", "극락", "신세계", "혜자",
    "순삭", "가성비", "퀄리티", "깊은맛", "진한맛", "후회없",
}


def build_unmatched_positive_candidates(case_df, max_rows=3000):
    """
    높은 실제 별점인데 분석점수가 낮은 리뷰에서 미매칭 1~3-gram 후보를 추출.

    감성사전에 이미 있는 표현은 제외하고, 리뷰별 중복 횟수도 한 번만 세어
    반복적으로 놓치는 단어/phrase가 위쪽에 오도록 정렬합니다.
    """
    columns = [
        "candidate_term",
        "candidate_type",
        "ngram_size",
        "positive_hint",
        "review_count",
        "avg_rating",
        "avg_category_total_score",
        "example_review",
    ]

    if case_df.empty:
        return pd.DataFrame(columns=columns)

    stats = {}

    for _, row in case_df.iterrows():
        tokens = [
            token
            for token in str(row.get("tokens_fixed", "")).split()
            if token
        ]
        review_candidates = set()

        for n in (3, 2, 1):
            for i in range(len(tokens) - n + 1):
                parts = tokens[i:i + n]
                term = " ".join(parts)
                compact_term = "".join(parts)

                if term in sentiment_map:
                    continue
                if len(compact_term) < 2 or len(compact_term) > 30:
                    continue
                if not re.fullmatch(r"[0-9A-Za-z가-힣]+", compact_term):
                    continue
                if all(part in CANDIDATE_STOP_TOKENS for part in parts):
                    continue
                if n == 1 and (
                    term in CANDIDATE_STOP_TOKENS
                    or term in RISKY_SINGLE_TOKENS
                    or term.isdigit()
                ):
                    continue

                review_candidates.add((term, n))

        for term, n in review_candidates:
            key = (term, n)
            if key not in stats:
                stats[key] = {
                    "review_count": 0,
                    "rating_sum": 0.0,
                    "score_sum": 0.0,
                    "example_review": str(row.get("review_text", ""))[:300],
                }

            stats[key]["review_count"] += 1
            stats[key]["rating_sum"] += float(row.get("rating", 0) or 0)
            stats[key]["score_sum"] += float(row.get("category_total_score", 0) or 0)

    records = []
    for (term, n), values in stats.items():
        count = values["review_count"]
        compact_term = term.replace(" ", "")
        records.append({
            "candidate_term": term,
            "candidate_type": "token" if n == 1 else "phrase",
            "ngram_size": n,
            "positive_hint": any(
                hint in compact_term
                for hint in POSITIVE_CANDIDATE_HINTS
            ),
            "review_count": count,
            "avg_rating": round(values["rating_sum"] / count, 3),
            "avg_category_total_score": round(values["score_sum"] / count, 3),
            "example_review": values["example_review"],
        })

    candidate_df = pd.DataFrame(records, columns=columns)
    if candidate_df.empty:
        return candidate_df

    return (
        candidate_df
        .sort_values(
            by=["positive_hint", "review_count", "ngram_size", "avg_rating"],
            ascending=[False, False, False, False],
        )
        .head(max_rows)
        .reset_index(drop=True)
    )


def diagnose_overestimation_case(row):
    """rating을 점수 계산에 쓰지 않고, 사후 오류 원인만 구조적으로 분류합니다."""
    try:
        evidence = json.loads(row.get("evidence_items", "[]"))
    except (TypeError, json.JSONDecodeError):
        evidence = []

    review_compact = compact_review_for_patterns(row.get("review_text", ""))
    reason = str(row.get("sentiment_calibration_reason", ""))
    active_positive = [
        item for item in evidence
        if item.get("is_active") and float(item.get("score", 0)) > 0
    ]
    inactive_reasons = {
        item.get("guard_reason")
        for item in evidence
        if not item.get("is_active")
    }
    types = []

    if (
        float(row.get("food_score", 0)) > 0
        and float(row.get("service_score", 0)) < 0
    ):
        types.append("food_positive_overrides_service_negative")
    if (
        "sentiment_floor:" in reason
        and int(row.get("negative_evidence_count", 0)) > 0
    ):
        types.append("positive_floor_applied_despite_negative")
    if (
        int(row.get("strong_positive_count", 0))
        + int(row.get("very_strong_positive_count", 0)) == 0
        and float(row.get("sentiment_star", 0)) >= 3.5
    ):
        types.append("weak_positive_too_strong")
    if {"other_target_positive", "past_positive_not_current"} & inactive_reasons:
        types.append("competitor_or_other_target_positive")
    if (
        int(row.get("positive_aspect_count", 0)) >= 1
        and int(row.get("negative_aspect_count", 0)) >= 1
        and float(row.get("sentiment_star", 0)) > 3.5
        and "mixed_review_cap:" not in reason
    ):
        types.append("mixed_review_not_capped")
    if (
        re.search(r"(?:재방문.{0,10}(?:없|안|않)|다시(?:는)?안.{0,8}(?:가|오)|다신안)", review_compact)
        and float(row.get("revisit_score", 0)) >= 0
    ):
        types.append("revisit_negative_missing")
    if (
        re.search(r"(?:위생|불친절|서비스.{0,10}(?:최악|별로|엉망)|이물질|벌레)", review_compact)
        and min(
            float(row.get("hygiene_score", 0)),
            float(row.get("service_score", 0)),
        ) > -2
    ):
        types.append("hygiene_or_service_negative_too_weak")
    if (
        re.search(r"(?:맛없|별로였|불친절|돈아깝|실망|최악)", review_compact)
        and int(row.get("negative_evidence_count", 0)) == 0
    ):
        types.append("negation_guard_too_aggressive")
    if (
        any(item.get("source") == "dictionary" for item in active_positive)
        and (
            "expectation_not_actual_positive" in inactive_reasons
            or "other_target_positive" in inactive_reasons
        )
    ):
        types.append("dictionary_false_positive")

    return " | ".join(dict.fromkeys(types or ["unknown"]))


def parse_evidence_items(value):
    """CSV 출력용 JSON evidence를 안전하게 다시 읽습니다."""
    try:
        parsed = json.loads(value if isinstance(value, str) else "[]")
        return parsed if isinstance(parsed, list) else []
    except (TypeError, json.JSONDecodeError):
        return []


def inactive_guard_reasons(row):
    return {
        str(item.get("guard_reason", ""))
        for item in parse_evidence_items(row.get("evidence_items", "[]"))
        if not item.get("is_active") and item.get("guard_reason")
    }


def diagnose_remaining_error_case(row):
    """남은 큰 오차와 양방향 오류를 사후 진단합니다. rating은 점수 계산에 쓰지 않습니다."""
    evidence = parse_evidence_items(row.get("evidence_items", "[]"))
    inactive_reasons = inactive_guard_reasons(row)
    active_items = [item for item in evidence if item.get("is_active")]
    active_positive = [
        item for item in active_items if float(item.get("score", 0) or 0) > 0
    ]
    active_negative = [
        item for item in active_items if float(item.get("score", 0) or 0) < 0
    ]
    review_compact = compact_review_for_patterns(row.get("review_text", ""))
    reason = str(row.get("sentiment_calibration_reason", ""))
    diff = float(row.get("diff", 0) or 0)
    types = []

    if diff > 0:
        if "quoted_negative_not_actual_negative" in inactive_reasons:
            types.append("quoted_negative_not_actual_negative")
        if "past_negative_not_current" in inactive_reasons:
            types.append("past_negative_current_positive")
        if (
            "sentiment_cap:" in reason or "mixed_review_cap:" in reason
        ) and active_negative:
            types.append("cap_misfire_on_positive_review")
        if int(row.get("positive_evidence_count", 0) or 0) == 0:
            types.append("positive_phrase_missing")
        elif float(row.get("category_total_score", 0) or 0) <= 1:
            types.append("weak_positive_underweighted")
        elif int(row.get("positive_aspect_count", 0) or 0) <= 1:
            types.append("aspect_weight_too_low")
    elif diff < 0:
        if "other_target_positive" in inactive_reasons:
            types.append("other_target_positive")
        if "past_positive_not_current" in inactive_reasons:
            types.append("past_positive_current_negative")
        if "expectation_not_actual_positive" in inactive_reasons:
            types.append("expectation_positive_not_actual")
        if (
            "sentiment_floor:" in reason
            and int(row.get("negative_evidence_count", 0) or 0) > 0
        ):
            types.append("weak_positive_overweighted")
        if (
            int(row.get("positive_aspect_count", 0) or 0) >= 1
            and int(row.get("negative_aspect_count", 0) or 0) >= 1
            and "mixed_review_cap:" not in reason
        ):
            types.append("mixed_review_not_capped_enough")
        if (
            re.search(
                r"(?:재방문.{0,10}(?:없|안|않)|다시(?:는)?안.{0,8}(?:가|오)|다신안)",
                review_compact,
            )
            and float(row.get("revisit_score", 0) or 0) >= 0
        ):
            types.append("revisit_negative_missing")
        if (
            re.search(r"(?:위생|불친절|이물질|벌레|계란껍질|서비스.{0,10}(?:최악|별로|엉망))", review_compact)
            and min(
                float(row.get("hygiene_score", 0) or 0),
                float(row.get("service_score", 0) or 0),
            ) > -2
        ):
            types.append("service_or_hygiene_negative_too_weak")
        if (
            active_positive
            and int(row.get("strong_positive_count", 0) or 0)
            + int(row.get("very_strong_positive_count", 0) or 0) == 0
        ):
            types.append("weak_positive_overweighted")
        if any(
            item.get("source") == "dictionary" for item in active_positive
        ) and re.search(r"(?:기대|유명|명성|소문|본점|다른집|타매장)", review_compact):
            types.append("dictionary_false_positive")

    return " | ".join(dict.fromkeys(types or ["unknown"]))


def count_inactive_guard_reason(frame, guard_names):
    """요청한 target/time/quote guard가 실제로 작동한 횟수를 집계합니다."""
    guard_names = set(guard_names)
    count = 0
    for value in frame.get("evidence_items", pd.Series(dtype=str)).fillna("[]"):
        for item in parse_evidence_items(value):
            if not item.get("is_active") and item.get("guard_reason") in guard_names:
                count += 1
    return count


if "rating" in df.columns:
    error_cases = df.loc[df["diff_abs"] > 2].copy()
    error_case_cols = [
        "platform", "store_name", "review_text", "tokens", "tokens_fixed",
        "rating", "category_total_score", "category_total_label",
        "sentiment_star_raw", "sentiment_star", "sentiment_calibration_reason",
        "diff", "diff_abs",
        "very_strong_positive_count", "strong_positive_count",
        "normal_positive_count", "weak_positive_count",
        "very_strong_negative_count", "strong_negative_count",
        "normal_negative_count", "weak_negative_count",
        "positive_evidence_count", "negative_evidence_count",
        "revisit_positive_count", "recommendation_count",
        "waiting_positive_count", "waiting_negative_count",
        "positive_phrase_count", "phrase_score_total",
        "dictionary_score_total", "context_guard_score_total",
        "evidence_items", "evidence_floor_items",
    ]
    error_case_cols += ["matched_words"]
    error_case_cols += [f"{aspect}_score" for aspect in EVIDENCE_ASPECTS]
    error_case_cols += [f"{aspect}_matched_words" for aspect in EVIDENCE_ASPECTS]
    error_case_cols = [col for col in error_case_cols if col in error_cases.columns]
    error_cases[error_case_cols].to_csv(
        error_cases_output,
        index=False,
        encoding="utf-8-sig",
    )

    low_score_high_rating_cases = df.loc[
        (df["rating"] >= 4)
        & (df["sentiment_star"] <= 3)
    ].copy()

    low_case_cols = [
        "platform", "store_name", "review_text", "tokens", "tokens_fixed",
        "rating", "category_total_score", "category_total_label",
        "sentiment_star_raw", "sentiment_star", "sentiment_calibration_reason",
        "diff", "diff_abs",
        "very_strong_positive_count", "strong_positive_count",
        "normal_positive_count", "weak_positive_count",
        "very_strong_negative_count", "strong_negative_count",
        "normal_negative_count", "weak_negative_count",
        "positive_evidence_count", "negative_evidence_count",
        "revisit_positive_count", "recommendation_count",
        "waiting_positive_count", "waiting_negative_count",
        "positive_phrase_count", "phrase_score_total",
        "dictionary_score_total", "context_guard_score_total",
        "evidence_items", "evidence_floor_items",
    ]
    low_case_cols += ["matched_words"]
    low_case_cols += [f"{aspect}_score" for aspect in EVIDENCE_ASPECTS]
    low_case_cols += [f"{aspect}_matched_words" for aspect in EVIDENCE_ASPECTS]
    low_case_cols = [col for col in low_case_cols if col in low_score_high_rating_cases.columns]

    low_score_high_rating_cases[low_case_cols].to_csv(
        low_score_high_rating_output,
        index=False,
        encoding="utf-8-sig",
    )

    high_score_low_rating_cases = df.loc[
        (df["rating"] <= 2)
        & (df["sentiment_star"] >= 4)
    ].copy()
    high_score_low_rating_cases[low_case_cols].to_csv(
        high_score_low_rating_output,
        index=False,
        encoding="utf-8-sig",
    )

    overestimation_cases = df.loc[
        (df["rating"] <= 2)
        & (df["sentiment_star"] >= 3.5)
    ].copy()
    overestimation_cases[low_case_cols].to_csv(
        overestimation_cases_output,
        index=False,
        encoding="utf-8-sig",
    )

    remaining_error_diagnosis = pd.concat(
        [error_cases, low_score_high_rating_cases, overestimation_cases],
        ignore_index=True,
    ).drop_duplicates(
        subset=[
            col for col in
            ["platform", "store_name", "review_text", "rating", "sentiment_star"]
            if col in df.columns
        ]
    )
    remaining_error_diagnosis["error_direction"] = np.select(
        [
            remaining_error_diagnosis["diff"] > 0,
            remaining_error_diagnosis["diff"] < 0,
        ],
        ["underestimation", "overestimation"],
        default="aligned",
    )
    remaining_error_diagnosis["error_type"] = remaining_error_diagnosis.apply(
        diagnose_remaining_error_case,
        axis=1,
    )
    diagnosis_cols = [
        "platform", "store_name", "review_text", "rating", "sentiment_star",
        "diff", "diff_abs", "category_total_score",
        "positive_evidence_count", "negative_evidence_count",
        "positive_aspect_count", "negative_aspect_count",
        "sentiment_calibration_reason", "error_direction", "error_type",
        "evidence_items", "matched_words",
    ]
    diagnosis_cols += [f"{aspect}_score" for aspect in EVIDENCE_ASPECTS]
    diagnosis_cols = [
        col for col in diagnosis_cols
        if col in remaining_error_diagnosis.columns
    ]
    remaining_error_diagnosis[diagnosis_cols].to_csv(
        overestimation_diagnosis_output,
        index=False,
        encoding="utf-8-sig",
    )

    calibration_reason = df["sentiment_calibration_reason"].fillna("")
    possible_cap_misfire_cases = df.loc[
        (df["rating"] >= 4)
        & (df["sentiment_star"] <= 3.5)
        & (
            calibration_reason.str.contains("sentiment_cap:|mixed_review_cap:")
            | (df["negative_evidence_count"] >= 1)
            | (df["service_score"] < 0)
            | (df["hygiene_score"] < 0)
            | (df["food_score"] < 0)
        )
    ].copy()
    possible_cap_misfire_cases[low_case_cols].to_csv(
        possible_cap_misfire_output,
        index=False,
        encoding="utf-8-sig",
    )

    possible_floor_misfire_cases = df.loc[
        (df["rating"] <= 2)
        & (df["sentiment_star"] >= 3.5)
        & (
            calibration_reason.str.contains("sentiment_floor:")
            | (df["strong_positive_count"] >= 1)
            | (df["very_strong_positive_count"] >= 1)
            | (df["weak_positive_count"] >= 2)
        )
    ].copy()
    possible_floor_misfire_cases[low_case_cols].to_csv(
        possible_floor_misfire_output,
        index=False,
        encoding="utf-8-sig",
    )

    unmatched_positive_candidates = build_unmatched_positive_candidates(
        low_score_high_rating_cases
    )
    unmatched_positive_candidates.to_csv(
        unmatched_candidate_output,
        index=False,
        encoding="utf-8-sig",
    )

    rated_rows = df.loc[df["rating"].notna()].copy()
    previous_metrics = {}
    previous_result_file = BASE_DIR / "final_high_trust_reviews_with_sentiment_star_minimal_tuned.csv"
    if previous_result_file.exists():
        previous_df, _ = read_csv_safely(previous_result_file)
        if {"rating", "sentiment_star", "diff_abs"}.issubset(previous_df.columns):
            previous_rated = previous_df.loc[previous_df["rating"].notna()].copy()
            previous_diff = previous_rated["rating"] - previous_rated["sentiment_star"]
            previous_metrics = {
                "previous_version": "minimal_tuned",
                "previous_mae": round(previous_rated["diff_abs"].mean(), 4),
                "previous_rmse": round(np.sqrt((previous_diff ** 2).mean()), 4),
                "previous_diff_abs_mean": round(previous_rated["diff_abs"].mean(), 4),
                "previous_diff_abs_median": round(previous_rated["diff_abs"].median(), 4),
                "previous_diff_abs_over_2_count": int((previous_rated["diff_abs"] > 2).sum()),
                "previous_high_rating_low_star_count": int(
                    (
                        (previous_rated["rating"] >= 4)
                        & (previous_rated["sentiment_star"] <= 3)
                    ).sum()
                ),
                "previous_high_score_low_rating_count": int(
                    (
                        (previous_rated["rating"] <= 2)
                        & (previous_rated["sentiment_star"] >= 4)
                    ).sum()
                ),
                "previous_overestimation_candidate_count": int(
                    (
                        (previous_rated["rating"] <= 2)
                        & (previous_rated["sentiment_star"] >= 3.5)
                    ).sum()
                ),
                "previous_high_rating_without_positive_evidence_count": int(
                    (
                        (previous_rated["rating"] >= 4)
                        & (previous_rated["positive_evidence_count"] == 0)
                    ).sum()
                ) if "positive_evidence_count" in previous_rated.columns else None,
            }

            print(
                "[minimal tuned 대비 변화 요약] "
                f"diff_abs 평균 {previous_metrics['previous_diff_abs_mean']} -> "
                f"{round(rated_rows['diff_abs'].mean(), 4)}, "
                f"diff_abs > 2 {previous_metrics['previous_diff_abs_over_2_count']} -> "
                f"{int((rated_rows['diff_abs'] > 2).sum())}, "
                f"고별점-저분석점수 {previous_metrics['previous_high_rating_low_star_count']} -> "
                f"{int(((rated_rows['rating'] >= 4) & (rated_rows['sentiment_star'] <= 3)).sum())}, "
                f"저별점-고분석점수 {previous_metrics['previous_high_score_low_rating_count']} -> "
                f"{int(((rated_rows['rating'] <= 2) & (rated_rows['sentiment_star'] >= 4)).sum())}, "
                f"과대평가 후보 {previous_metrics['previous_overestimation_candidate_count']} -> "
                f"{len(overestimation_cases)}"
            )

    inactive_quoted_negative_count = count_inactive_guard_reason(
        rated_rows,
        {"quoted_negative_not_actual_negative"},
    )
    inactive_other_target_positive_count = count_inactive_guard_reason(
        rated_rows,
        {"other_target_positive"},
    )
    inactive_expectation_positive_count = count_inactive_guard_reason(
        rated_rows,
        {"expectation_not_actual_positive"},
    )
    inactive_past_direction_count = count_inactive_guard_reason(
        rated_rows,
        {"past_positive_not_current", "past_negative_not_current"},
    )

    sentiment_error_summary = pd.DataFrame([
        {
            "total_review_count": len(df),
            "rated_review_count": len(rated_rows),
            "mae": round(rated_rows["diff_abs"].mean(), 4),
            "rmse": round(np.sqrt(rated_rows["diff_squared"].mean()), 4),
            "diff_abs_mean": round(rated_rows["diff_abs"].mean(), 4),
            "diff_abs_median": round(rated_rows["diff_abs"].median(), 4),
            "diff_abs_over_2_count": int((rated_rows["diff_abs"] > 2).sum()),
            "high_rating_low_star_count": int(
                (
                    (rated_rows["rating"] >= 4)
                    & (rated_rows["sentiment_star"] <= 3)
                ).sum()
            ),
            "high_rating_without_positive_phrase_count": int(
                (
                    (rated_rows["rating"] >= 4)
                    & (rated_rows["positive_evidence_count"] == 0)
                ).sum()
            ),
            "strong_positive_but_low_star_count": int(
                (
                    (
                        (
                            rated_rows["strong_positive_count"]
                            + rated_rows["very_strong_positive_count"]
                        ) >= 1
                    )
                    & (rated_rows["sentiment_star"] <= 3)
                ).sum()
            ),
            "high_score_low_rating_count": int(
                (
                    (rated_rows["rating"] <= 2)
                    & (rated_rows["sentiment_star"] >= 4)
                ).sum()
            ),
            "overestimation_candidate_count": len(overestimation_cases),
            "underestimation_diff_over_2_count": int((rated_rows["diff"] > 2).sum()),
            "overestimation_diff_over_2_count": int((rated_rows["diff"] < -2).sum()),
            "positive_floor_applied_count": int(
                rated_rows["sentiment_calibration_reason"]
                .fillna("")
                .str.contains("sentiment_floor:")
                .sum()
            ),
            "negative_cap_applied_count": int(
                rated_rows["sentiment_calibration_reason"]
                .fillna("")
                .str.contains("sentiment_cap:")
                .sum()
            ),
            "mixed_review_cap_applied_count": int(
                rated_rows["sentiment_calibration_reason"]
                .fillna("")
                .str.contains("mixed_review_cap:")
                .sum()
            ),
            "possible_cap_misfire_count": len(possible_cap_misfire_cases),
            "possible_floor_misfire_count": len(possible_floor_misfire_cases),
            "remaining_error_diagnosis_count": len(remaining_error_diagnosis),
            "inactive_quoted_negative_count": inactive_quoted_negative_count,
            "inactive_other_target_positive_count": inactive_other_target_positive_count,
            "inactive_expectation_positive_count": inactive_expectation_positive_count,
            "inactive_past_direction_count": inactive_past_direction_count,
            "generalized_pattern_rule_count": len(GENERALIZED_PATTERN_RULES),
            "explicit_phrase_rule_count": len(PHRASE_RULES),
            "dictionary_file": dict_file.name,
            **previous_metrics,
        }
    ])
    sentiment_error_summary.to_csv(
        error_summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    current_mae = round(rated_rows["diff_abs"].mean(), 4)
    current_rmse = round(np.sqrt(rated_rows["diff_squared"].mean()), 4)
    current_large_error_count = int((rated_rows["diff_abs"] > 2).sum())
    current_low_score_high_rating_count = len(low_score_high_rating_cases)
    current_high_score_low_rating_count = len(high_score_low_rating_cases)
    current_overestimation_candidate_count = len(overestimation_cases)
    missing_positive_evidence_count = int(
        (low_score_high_rating_cases["positive_evidence_count"] == 0).sum()
    )
    mixed_negative_underestimate_count = int(
        (low_score_high_rating_cases["negative_evidence_count"] >= 2).sum()
    )
    error_type_counts = (
        remaining_error_diagnosis["error_type"]
        .fillna("unknown")
        .str.split(r"\s*\|\s*")
        .explode()
        .value_counts()
    )

    report_lines = [
        "# Positive Evidence Recovery 실행 보고서",
        "",
        "## 구조 검증",
        "",
        "- 최종 점수 원장: `evidence_items`",
        "- aspect: food, service, price, atmosphere, wait, revisit, recommendation, hygiene, general",
        "- `sentiment_star`는 rating과 분위수 없이 active evidence와 aspect 가중치로 계산",
        "- positive floor 이후 negative cap, mixed review cap 순서로 적용",
        "- rating은 오류 비교와 사후 진단에만 사용하며 점수 계산에는 사용하지 않음",
        f"- 기존 긍정 회귀 테스트: `{test_results_output.name}`",
        f"- 과대평가 방지 테스트: `{overcap_test_results_output.name}`",
        f"- 최종 튜닝 추가 테스트: `{final_tuning_test_results_output.name}`",
        f"- 최소 오류 튜닝 테스트: `{minimal_tuning_test_results_output.name}`",
        f"- positive evidence 회복 테스트: `{positive_recovery_test_results_output.name}`",
        "",
        "## Minimal Tuned 대비 결과",
        "",
        f"- 평가 가능한 리뷰: {len(rated_rows):,}개",
        f"- MAE: {previous_metrics.get('previous_mae', 'N/A')} -> {current_mae}",
        f"- RMSE: {previous_metrics.get('previous_rmse', 'N/A')} -> {current_rmse}",
        (
            "- `diff_abs > 2`: "
            f"{previous_metrics.get('previous_diff_abs_over_2_count', 'N/A')} "
            f"-> {current_large_error_count}"
        ),
        (
            "- 고별점-저분석점수: "
            f"{previous_metrics.get('previous_high_rating_low_star_count', 'N/A')} "
            f"-> {current_low_score_high_rating_count}"
        ),
        (
            "- 저별점-고분석점수: "
            f"{previous_metrics.get('previous_high_score_low_rating_count', 'N/A')} "
            f"-> {current_high_score_low_rating_count}"
        ),
        (
            "- 과대평가 후보(`rating <= 2`, `sentiment_star >= 3.5`): "
            f"{previous_metrics.get('previous_overestimation_candidate_count', 'N/A')} "
            f"-> {current_overestimation_candidate_count}"
        ),
        (
            "- 고평점 positive evidence 0: "
            f"{previous_metrics.get('previous_high_rating_without_positive_evidence_count', 'N/A')} "
            f"-> {int(((rated_rows['rating'] >= 4) & (rated_rows['positive_evidence_count'] == 0)).sum())}"
        ),
        (
            "- positive floor / negative cap / mixed cap 적용: "
            f"{int(rated_rows['sentiment_calibration_reason'].fillna('').str.contains('sentiment_floor:').sum())} / "
            f"{int(rated_rows['sentiment_calibration_reason'].fillna('').str.contains('sentiment_cap:').sum())} / "
            f"{int(rated_rows['sentiment_calibration_reason'].fillna('').str.contains('mixed_review_cap:').sum())}"
        ),
        (
            "- 인용 부정 / 타 대상 긍정 / 기대·명성 긍정 / 과거·현재 방향 guard 비활성화: "
            f"{inactive_quoted_negative_count} / {inactive_other_target_positive_count} / "
            f"{inactive_expectation_positive_count} / {inactive_past_direction_count}"
        ),
        "",
        "## 안전 회복 방식",
        "",
        "- 감성사전 파일은 수정하지 않음",
        "- 제한적 pattern: 구어체 맛 표현, 음식 대상 필수/서운 관용구, 명시적 재방문 의사, 찾아올 가치, 음식 제공 속도, 맛 만족 관용구",
        "- context guard: 부정 맛 명사에 붙은 `맛나요`, 기대-실망, `굿이라고 하기엔 아쉽다`, 직접 부정 재방문 문맥 제외",
        "- 애매한 고평점 리뷰와 rating-text mismatch는 회수하지 않음",
        "",
        "## 남은 주요 오류 유형",
        "",
        (
            f"- 고별점-저분석점수 중 positive evidence가 전혀 없는 사례: "
            f"{missing_positive_evidence_count}개. 사전/형태 정규화 누락 후보를 우선 검토해야 합니다."
        ),
        (
            f"- 고별점-저분석점수 중 negative evidence가 2개 이상인 사례: "
            f"{mixed_negative_underestimate_count}개. 별점과 리뷰 문장의 실제 불만이 충돌하는 사례가 포함됩니다."
        ),
        (
            f"- 저별점-고분석점수 사례: {current_high_score_low_rating_count}개. "
            "명시적 최종 부정 결론 누락 또는 리뷰 본문과 rating 불일치를 구분해 검토해야 합니다."
        ),
        (
            f"- 미매칭 긍정 후보 파일: {len(unmatched_positive_candidates):,}개 후보. "
            "빈도와 예문을 확인한 뒤 일반화 가능한 형태만 사전/정규화 계층에 반영하는 것이 안전합니다."
        ),
        "",
        "### 사후 진단 유형 빈도",
        "",
        *[
            f"- {error_type}: {count}개"
            for error_type, count in error_type_counts.items()
        ],
        "",
        "## 산출물",
        "",
        f"- 전체 결과: `{output_file.name}`",
        f"- 기존 긍정 테스트: `{test_results_output.name}`",
        f"- 과대평가 방지 테스트: `{overcap_test_results_output.name}`",
        f"- 최종 튜닝 추가 테스트: `{final_tuning_test_results_output.name}`",
        f"- 최소 오류 튜닝 테스트: `{minimal_tuning_test_results_output.name}`",
        f"- positive evidence 회복 테스트: `{positive_recovery_test_results_output.name}`",
        f"- 남은 저별점-고분석점수: `{overestimation_cases_output.name}`",
        f"- 남은 오류 통합 진단: `{overestimation_diagnosis_output.name}`",
        f"- 큰 오차 사례: `{error_cases_output.name}`",
        f"- 고별점-저분석점수: `{low_score_high_rating_output.name}`",
        f"- 저별점-고분석점수: `{high_score_low_rating_output.name}`",
        f"- cap 오작동 후보: `{possible_cap_misfire_output.name}`",
        f"- floor 오작동 후보: `{possible_floor_misfire_output.name}`",
        f"- 미매칭 후보: `{unmatched_candidate_output.name}`",
        f"- 오류 요약: `{error_summary_output.name}`",
    ]
    refactor_report_output.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

# =========================================================
# 9. 플랫폼별 요약
# =========================================================

if "platform" in df.columns:
    agg_dict = {
        "review_count": ("review_text", "count"),
        "avg_total_score": ("category_total_score", "mean"),
        "avg_overall_sentiment_score": ("overall_sentiment_score", "mean"),
        "avg_sentiment_star": ("sentiment_star", "mean"),
        "positive_ratio": ("category_total_label", lambda x: (x == "positive").mean()),
        "neutral_ratio": ("category_total_label", lambda x: (x == "neutral").mean()),
        "negative_ratio": ("category_total_label", lambda x: (x == "negative").mean()),
    }

    if "rating" in df.columns:
        agg_dict["avg_rating"] = ("rating", "mean")
        agg_dict["avg_diff_abs"] = ("diff_abs", "mean")

    for aspect in EVIDENCE_ASPECTS:
        agg_dict[f"avg_{aspect}_score"] = (f"{aspect}_score", "mean")

    platform_summary = df.groupby("platform").agg(**agg_dict).reset_index()

    platform_cols = (
        ["platform", "review_count"]
        + [f"avg_{aspect}_score" for aspect in EVIDENCE_ASPECTS]
        + ["avg_total_score", "avg_overall_sentiment_score", "avg_sentiment_star"]
    )

    if "rating" in df.columns:
        platform_cols += ["avg_rating", "avg_diff_abs"]

    platform_cols += ["positive_ratio", "neutral_ratio", "negative_ratio"]

    platform_summary = platform_summary[platform_cols]
    platform_summary.to_csv(platform_output, index=False, encoding="utf-8-sig")

# =========================================================
# 10. 식당별 요약
# =========================================================

if "store_name" in df.columns and "platform" in df.columns:
    agg_dict = {
        "review_count": ("review_text", "count"),
        "avg_total_score": ("category_total_score", "mean"),
        "avg_overall_sentiment_score": ("overall_sentiment_score", "mean"),
        "avg_sentiment_star": ("sentiment_star", "mean"),
        "positive_ratio": ("category_total_label", lambda x: (x == "positive").mean()),
        "neutral_ratio": ("category_total_label", lambda x: (x == "neutral").mean()),
        "negative_ratio": ("category_total_label", lambda x: (x == "negative").mean()),
    }

    if "rating" in df.columns:
        agg_dict["avg_rating"] = ("rating", "mean")
        agg_dict["avg_diff_abs"] = ("diff_abs", "mean")

    for aspect in EVIDENCE_ASPECTS:
        agg_dict[f"avg_{aspect}_score"] = (f"{aspect}_score", "mean")

    store_summary = df.groupby(["store_name", "platform"]).agg(**agg_dict).reset_index()

    store_cols = (
        ["store_name", "platform", "review_count"]
        + [f"avg_{aspect}_score" for aspect in EVIDENCE_ASPECTS]
        + ["avg_total_score", "avg_overall_sentiment_score", "avg_sentiment_star"]
    )

    if "rating" in df.columns:
        store_cols += ["avg_rating", "avg_diff_abs"]

    store_cols += ["positive_ratio", "neutral_ratio", "negative_ratio"]

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
    "tokens_fixed",
]

if "rating" in df.columns:
    review_score_cols += ["rating"]

review_score_cols += [f"{aspect}_score" for aspect in EVIDENCE_ASPECTS]

review_score_cols += [
    "category_total_score",
    "category_total_label",
    "strong_positive_count",
    "very_strong_positive_count",
    "normal_positive_count",
    "weak_positive_count",
    "strong_negative_count",
    "very_strong_negative_count",
    "normal_negative_count",
    "weak_negative_count",
    "positive_evidence_count",
    "negative_evidence_count",
    "revisit_positive_count",
    "recommendation_count",
    "waiting_positive_count",
    "waiting_negative_count",
    "positive_phrase_count",
    "phrase_score_total",
    "dictionary_score_total",
    "context_guard_score_total",
    "evidence_items",
    "evidence_floor_items",
    "overall_sentiment_score",
    "overall_context_reason",
    "sentiment_star_raw",
    "sentiment_star",
    "sentiment_calibration_reason",
    "matched_words",
]

if "rating" in df.columns:
    review_score_cols += [
        "diff",
        "diff_abs",
        "diff_squared",
    ]

review_score_cols += [f"{aspect}_label" for aspect in EVIDENCE_ASPECTS]
review_score_cols += [f"{aspect}_matched_words" for aspect in EVIDENCE_ASPECTS]

review_score_cols = [col for col in review_score_cols if col in df.columns]

df[review_score_cols].to_csv(
    review_score_output,
    index=False,
    encoding="utf-8-sig",
)

# =========================================================
# 12. 전체 결과 저장
# =========================================================

df.to_csv(output_file, index=False, encoding="utf-8-sig")

# 원본 CSV 열은 그대로 유지하고, 별점 확인에 필요한 핵심 결과만 추가합니다.
# rating 바로 뒤에 분석 별점과 오차를 배치해 사람이 두 별점을 쉽게 비교할 수
# 있도록 하며, 상세 진단용 evidence_items는 전체 결과 파일에만 보존합니다.
compact_cols = []
for col in original_input_columns:
    if col in df.columns and col not in compact_cols:
        compact_cols.append(col)
    if col == "rating":
        compact_cols.extend([
            candidate
            for candidate in ["sentiment_star", "diff", "diff_abs", "diff_squared"]
            if candidate in df.columns and candidate not in compact_cols
        ])

compact_analysis_cols = [
    "tokens_fixed",
    "category_total_score",
    "category_total_label",
    "sentiment_star_raw",
    "weighted_evidence_score",
    "sentiment_calibration_reason",
    "food_score",
    "service_score",
    "price_score",
    "atmosphere_score",
    "wait_score",
    "revisit_score",
    "recommendation_score",
    "hygiene_score",
    "general_score",
    "positive_evidence_count",
    "negative_evidence_count",
    "matched_words",
]
compact_cols.extend([
    col
    for col in compact_analysis_cols
    if col in df.columns and col not in compact_cols
])
df[compact_cols].to_csv(compact_output, index=False, encoding="utf-8-sig")

# 별점 결과를 한눈에 검토하기 위한 20열 요약 파일입니다.
# 원본 데이터 전체 보존은 compact/full 파일이 담당하고, 이 파일은 사람이
# rating과 sentiment_star의 차이 및 그 근거를 빠르게 확인하는 데 집중합니다.
review_view_cols = [
    "platform",
    "store_name",
    "visit_date",
    "review_text",
    "rating",
    "sentiment_star",
    "diff",
    "diff_abs",
    "category_total_score",
    "food_score",
    "service_score",
    "price_score",
    "atmosphere_score",
    "wait_score",
    "revisit_score",
    "hygiene_score",
    "positive_evidence_count",
    "negative_evidence_count",
    "sentiment_calibration_reason",
    "matched_words",
]
review_view_cols = [col for col in review_view_cols if col in df.columns]
df[review_view_cols].to_csv(
    review_view_output,
    index=False,
    encoding="utf-8-sig",
)

# =========================================================
# 13. 실행 결과 출력
# =========================================================

print("\n완료:", output_file)
print("원본 열 구조 호환 별점 결과:", compact_output)
print("한눈에 보는 별점 검토 결과:", review_view_output)
print("리뷰별 감성점수 확인 파일:", review_score_output)

if "rating" in df.columns:
    print("diff_abs > 2 오류 사례:", error_cases_output)
    print("고별점-저분석점수 사례:", low_score_high_rating_output)
    print("저별점-고분석점수 사례:", high_score_low_rating_output)
    print("남은 오류 통합 진단:", overestimation_diagnosis_output)
    print("cap 오작동 후보:", possible_cap_misfire_output)
    print("floor 오작동 후보:", possible_floor_misfire_output)
    print("미매칭 긍정 후보:", unmatched_candidate_output)
    print("오류 요약:", error_summary_output)
    print("positive evidence 회복 보고서:", refactor_report_output)

print("사전 실행 리뷰 테스트 결과:", test_results_output)
print("과대평가 방지 테스트 결과:", overcap_test_results_output)
print("최종 튜닝 추가 테스트 결과:", final_tuning_test_results_output)
print("최소 오류 튜닝 테스트 결과:", minimal_tuning_test_results_output)
print("positive evidence 회복 테스트 결과:", positive_recovery_test_results_output)

if "platform" in df.columns:
    print("플랫폼 요약:", platform_output)

if "store_name" in df.columns and "platform" in df.columns:
    print("식당 요약:", store_output)

print("\n[사용 카테고리]")
print(EVIDENCE_ASPECTS)

print("\n[사전에 들어 있는 카테고리]")
print(dict_categories)

print("\n[감성분석 결과 미리보기]")

preview_cols = [
    "review_text",
    "tokens",
    "tokens_fixed",
]

if "rating" in df.columns:
    preview_cols += ["rating"]

preview_cols += [f"{aspect}_score" for aspect in EVIDENCE_ASPECTS]

preview_cols += [
    "category_total_score",
    "category_total_label",
    "overall_sentiment_score",
    "overall_context_reason",
    "sentiment_star",
]

if "rating" in df.columns:
    preview_cols += [
        "diff",
        "diff_abs",
    ]

preview_cols += [f"{aspect}_matched_words" for aspect in EVIDENCE_ASPECTS]

preview_cols = [col for col in preview_cols if col in df.columns]

print_console_safely(df[preview_cols].head(20).to_string())