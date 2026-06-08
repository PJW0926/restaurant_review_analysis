import pandas as pd
import re
from konlpy.tag import Okt

# =========================
# 1. 파일 불러오기
# =========================

input_file = "final_high_trust_reviews.csv"
output_file = "final_high_trust_reviews_pos.csv"

df = pd.read_csv(input_file, encoding="utf-8-sig")

print("데이터 크기:", df.shape)
print("컬럼 목록:")
print(df.columns.tolist())

# 리뷰 텍스트 열
text_col = "review_text"

# 결측치 처리
df[text_col] = df[text_col].fillna("").astype(str)

# =========================
# 2. 형태소 분석기 준비
# =========================

okt = Okt()

# 감성분석에 사용할 품사
USE_POS = {"Noun", "Adjective", "Verb", "Adverb"}

# 불용어: 너무 많이 빼면 의미가 사라지므로 최소한만 설정
STOPWORDS = {
    "이", "가", "은", "는", "을", "를", "에", "의", "도", "로", "으로",
    "와", "과", "하고", "에서", "에게", "께서",
    "것", "거", "수", "때", "좀",
    "하다", "되다", "있다", "없다"
}

# =========================
# 3. 전처리 함수
# =========================

def clean_text(text):
    text = str(text)
    
    # URL 제거
    text = re.sub(r"http\S+|www\S+", " ", text)
    
    # 한글, 영어, 숫자, 공백, 일부 감정표현만 유지
    text = re.sub(r"[^가-힣a-zA-Z0-9\sㅋㅎㅠㅜ]", " ", text)
    
    # 반복 공백 제거
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def get_pos_tags(text):
    text = clean_text(text)
    
    # norm=True: 정규화
    # stem=True: 원형 복원
    # 예: 맛있었어요 -> 맛있다, 친절했어요 -> 친절하다
    return okt.pos(text, norm=True, stem=True)


def extract_tokens(text):
    pos_tags = get_pos_tags(text)
    
    tokens = [
        word for word, pos in pos_tags
        if pos in USE_POS
        and word not in STOPWORDS
        and len(word) > 1
    ]
    
    return tokens


def extract_tokens_with_pos(text):
    pos_tags = get_pos_tags(text)
    
    tokens_pos = [
        f"{word}/{pos}" for word, pos in pos_tags
        if pos in USE_POS
        and word not in STOPWORDS
        and len(word) > 1
    ]
    
    return tokens_pos


# =========================
# 4. 전체 리뷰에 적용
# =========================

df["clean_text"] = df[text_col].apply(clean_text)

df["tokens"] = df[text_col].apply(
    lambda x: " ".join(extract_tokens(x))
)

df["tokens_pos"] = df[text_col].apply(
    lambda x: " ".join(extract_tokens_with_pos(x))
)

# 토큰 개수
df["token_count"] = df["tokens"].apply(lambda x: len(x.split()) if x else 0)

# =========================
# 5. 저장
# =========================

df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("완료:", output_file)
print(df[[text_col, "tokens", "tokens_pos", "token_count"]].head())