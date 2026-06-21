# Restaurant Review Analysis

네이버지도와 카카오맵 음식점 리뷰를 대상으로 리뷰 신뢰도 필터링, 감성분석, 플랫폼별 비교 분석을 수행한 데이터마이닝 프로젝트입니다.

본 프로젝트의 목적은 단순 별점이나 리뷰 수만으로 음식점을 평가하는 방식의 한계를 보완하고, 리뷰 텍스트의 신뢰도와 감성 정보를 반영한 새로운 맛집 평가 지표를 제안하는 것입니다.

---

## 1. 프로젝트 개요

온라인 음식점 리뷰는 이용자가 식당을 선택할 때 중요한 참고 자료로 활용됩니다. 그러나 리뷰는 플랫폼별 구조, 작성자 특성, 리뷰 품질에 따라 평가 경향이 다르게 나타날 수 있습니다.

특히 네이버지도와 카카오맵은 리뷰 작성 방식과 평가 구조가 다르기 때문에 두 플랫폼의 리뷰를 단순히 같은 기준으로 비교하기 어렵습니다.

본 프로젝트에서는 다음 세 가지 질문을 중심으로 분석을 진행했습니다.

1. 리뷰 텍스트와 작성자 정보를 바탕으로 신뢰도 높은 리뷰를 선별할 수 있는가?
2. 신뢰도 높은 리뷰를 기반으로 텍스트 감성점수를 산출할 수 있는가?
3. 네이버와 카카오의 플랫폼 차이를 고려한 맛집 평가 지표를 제안할 수 있는가?

전체 분석 흐름은 다음과 같습니다.

```text
리뷰 수집
→ 데이터 전처리
→ 신뢰도 필터링
→ 감성분석
→ 플랫폼별 비교 및 인사이트 도출
```

---

## 2. 데이터 설명

본 프로젝트에서는 네이버지도와 카카오맵의 음식점 리뷰 데이터를 사용했습니다.

### 데이터 범위

* 대상 플랫폼: 네이버지도, 카카오맵
* 대상 업종: 음식점
* 대상 지역: 성수, 마포
* 수집 기간: 2025-01-01 ~ 2026-05-09
* 전체 리뷰 수: 6,881건

  * 네이버 리뷰: 4,175건
  * 카카오 리뷰: 2,526건

### 주요 변수

공통 변수는 다음과 같습니다.

* `platform`: 리뷰 플랫폼
* `store_name`: 식당명
* `account_id`: 익명화된 계정 ID
* `account_review_count`: 계정의 리뷰 작성 수
* `visit_date`: 방문일
* `verification_method`: 방문 인증 방식
* `review_text`: 리뷰 본문
* `review_length`: 리뷰 길이
* `has_photo`: 사진 포함 여부

플랫폼별 추가 변수는 다음과 같습니다.

* 네이버: `visit_count`
* 카카오: `rating`, `account_avg_rating`, `reviewer_level`

개인정보 보호를 위해 계정 정보는 익명화하여 사용했습니다. 익명화된 전체 데이터는 `2. preprocessing` 폴더에 포함되어 있습니다.

---

## 3. 폴더 구조

```text
restaurant_review_analysis/
├── 1. crawling/
│   ├── naver_crawler/
│   │   ├── config.py
│   │   ├── crawler.py
│   │   └── main.py
│   └── kakao_crawler.py
│
├── 2. preprocessing/
│   ├── all_크롤링_익명화.csv
│   ├── missing_value.py
│   ├── outlier_iqr.py
│   ├── model_compare_ml.py
│   └── model_compare_result/
│       ├── logistic_regression_coefficients.csv
│       ├── model_comparison_metrics.csv
│       ├── random_forest_feature_importance.csv
│       └── test_predictions.csv
│
├── 3. trust_filtering/
│   ├── label_analysis_all_labeled.py
│   ├── label_analysis_all_crawling.py
│   ├── labeled_all_filter_result.csv
│   └── crawling_all_filter_result.csv
│
├── 4. sentiment_analysis/
│   ├── final_high_trust_reviews_pos.csv
│   ├── 감성분석_토큰화.py
│   ├── 감성분석_최종.py
│   └── 통합_감성사전_v10.csv
│
├── 5. insights/
│   ├── figures/
│   │   ├── fig_insight1_kakao_rating_text_sentiment_diff_final.png
│   │   ├── fig_insight1_platform_before_after_text_sentiment_final.png
│   │   ├── fig_insight2_aspect_heatmap_final.png
│   │   ├── fig_platform_avg_sentiment_star.png
│   │   ├── fig_sentiment_rmse_before_after.png
│   │   └── fig_store_naver_kakao_sentiment_comparison.png
│   │
│   ├── 04_insight1_review_decision_visualization_checked.py
│   ├── 04_platform_avg_only_with_summary_box.py
│   ├── 05_insight2_aspect_check_visualization.py
│   ├── 05_platform_sentiment_dumbbell_only_clean_v2.py
│   └── 08_rmse_before_after_visualization.py
│
├── docs/
│   └── 데이터마이닝_최종_데마초.pdf
│
└── README.md
```

---

## 4. 분석 방법

### 4.1 리뷰 수집

`1. crawling` 폴더에는 네이버지도와 카카오맵 리뷰 수집 코드가 포함되어 있습니다.

* `naver_crawler/`: 네이버지도 리뷰 수집 코드
* `kakao_crawler.py`: 카카오맵 리뷰 수집 코드

수집한 데이터에는 식당명, 리뷰 본문, 작성자 정보, 방문일, 별점 정보, 사진 여부 등의 정보가 포함됩니다.

---

### 4.2 데이터 전처리

`2. preprocessing` 폴더에서는 수집한 데이터를 분석에 적합한 형태로 정리했습니다.

주요 전처리 과정은 다음과 같습니다.

* 네이버·카카오 리뷰 데이터 통합
* 컬럼명 표준화
* 계정 정보 익명화
* 결측치 확인
* 이상치 확인
* 모델 비교용 변수 생성

주요 파일은 다음과 같습니다.

```text
2. preprocessing/all_크롤링_익명화.csv
2. preprocessing/missing_value.py
2. preprocessing/outlier_iqr.py
2. preprocessing/model_compare_ml.py
```

`model_compare_result` 폴더에는 모델 비교 결과가 저장되어 있습니다.

```text
model_comparison_metrics.csv
logistic_regression_coefficients.csv
random_forest_feature_importance.csv
test_predictions.csv
```

---

### 4.3 신뢰도 필터링

`3. trust_filtering` 폴더에서는 리뷰의 신뢰도를 판단하기 위한 규칙 기반 필터를 적용했습니다.

신뢰도 점수는 다음 요소를 반영하여 산출했습니다.

* 리뷰 길이
* 사진 포함 여부
* 메뉴명 포함 여부
* 구체적인 경험 표현 포함 여부
* 숫자 정보 포함 여부
* 재방문 표현
* 계정 리뷰 수
* 일반적인 칭찬어 사용 여부
* 이벤트·광고성 표현 여부
* 구체적인 부정 경험 표현 여부

신뢰도 기준은 다음과 같이 사용했습니다.

* `trust_score >= 5.7`: 수동 라벨 기준 평가용 high 예측 기준
* `trust_score >= 3.5`: 최종 감성분석에 포함한 high trust 리뷰 기준

주요 파일은 다음과 같습니다.

```text
3. trust_filtering/label_analysis_all_labeled.py
3. trust_filtering/label_analysis_all_crawling.py
3. trust_filtering/labeled_all_filter_result.csv
3. trust_filtering/crawling_all_filter_result.csv
```

최종 분석에는 high trust 리뷰 4,268건을 사용했습니다.

---

### 4.4 감성분석

`4. sentiment_analysis` 폴더에서는 신뢰도 필터를 통과한 리뷰를 대상으로 감성분석을 수행했습니다.

감성분석은 자체 구축한 감성사전을 기반으로 진행했습니다. 감성사전은 감성 표현, aspect, polarity, score로 구성되어 있습니다.

주요 aspect는 다음과 같습니다.

* `food`: 음식
* `price`: 가격
* `service`: 서비스
* `atmosphere`: 분위기
* `general`: 일반 평가

감성분석 과정은 다음과 같습니다.

1. 리뷰 텍스트 토큰화
2. 감성사전과 리뷰 토큰 매칭
3. aspect별 감성점수 산출
4. 리뷰별 총 감성점수 산출
5. 감성점수를 1~5점 감성별점으로 변환

주요 파일은 다음과 같습니다.

```text
4. sentiment_analysis/final_high_trust_reviews_pos.csv
4. sentiment_analysis/감성분석_토큰화.py
4. sentiment_analysis/감성분석_최종.py
4. sentiment_analysis/통합_감성사전_v10.csv
```

---

## 5. 주요 결과

### 5.1 High Trust 리뷰 선별 결과

신뢰도 필터 적용 후 최종 high trust 리뷰 수는 다음과 같습니다.

* 전체 high trust 리뷰: 4,268건
* 네이버 high trust 리뷰: 2,786건
* 카카오 high trust 리뷰: 1,482건

이 리뷰들은 감성분석과 플랫폼 비교 분석의 주요 입력 데이터로 사용되었습니다.

---

### 5.2 플랫폼별 감성별점 차이

네이버와 카카오의 평균 감성별점을 비교한 결과, 두 플랫폼 간 평가 경향에 차이가 있었습니다.

* 네이버 평균 감성별점: 4.193
* 카카오 평균 감성별점: 3.716

이는 네이버와 카카오 리뷰를 단순 통합하여 비교하기보다, 플랫폼별 리뷰 생성 구조와 평가 문화를 고려해야 함을 보여줍니다.

관련 figure는 다음 위치에 있습니다.

```text
5. insights/figures/fig_platform_avg_sentiment_star.png
```

---

### 5.3 카카오 실제 별점과 텍스트 감성별점 비교

카카오 리뷰의 실제 별점과 텍스트 기반 감성별점을 비교하여, 별점만으로는 드러나지 않는 평가 차이를 확인했습니다.

관련 figure는 다음 위치에 있습니다.

```text
5. insights/figures/fig_insight1_kakao_rating_text_sentiment_diff_final.png
```

---

### 5.4 동일 식당의 네이버·카카오 비교

동일 식당이라도 네이버와 카카오에서 감성별점이 다르게 나타나는지 비교했습니다.

관련 figure는 다음 위치에 있습니다.

```text
5. insights/figures/fig_store_naver_kakao_sentiment_comparison.png
```

---

### 5.5 Aspect 기반 식당 비교

리뷰 텍스트를 음식, 가격, 서비스, 분위기 등 aspect별로 나누어 식당별 강점과 약점을 확인했습니다.

관련 figure는 다음 위치에 있습니다.

```text
5. insights/figures/fig_insight2_aspect_heatmap_final.png
```

---

### 5.6 RMSE 개선

카카오 실제 별점과 텍스트 감성별점의 오차를 비교한 결과, 신뢰도 필터 적용 후 RMSE가 감소했습니다.

* 필터 적용 전 RMSE: 1.0605
* 필터 적용 후 RMSE: 0.8609

이는 신뢰도 필터링이 텍스트 기반 평가 지표의 안정성을 높이는 데 기여했음을 보여줍니다.

관련 figure는 다음 위치에 있습니다.

```text
5. insights/figures/fig_sentiment_rmse_before_after.png
```

---

## 6. 인사이트 시각화 파일

`5. insights/figures` 폴더에는 최종 발표 및 보고서에 사용한 주요 시각화 결과물이 포함되어 있습니다.

```text
fig_insight1_kakao_rating_text_sentiment_diff_final.png
fig_insight1_platform_before_after_text_sentiment_final.png
fig_insight2_aspect_heatmap_final.png
fig_platform_avg_sentiment_star.png
fig_sentiment_rmse_before_after.png
fig_store_naver_kakao_sentiment_comparison.png
```

각 figure는 다음 내용을 보여줍니다.

* 전체 리뷰와 high trust 리뷰의 감성별점 차이
* 카카오 실제 별점과 텍스트 감성별점의 차이
* 식당별 aspect 평가 차이
* 플랫폼별 평균 감성별점 차이
* 동일 식당의 네이버·카카오 감성별점 차이
* 신뢰도 필터 적용 전후 RMSE 변화

---

## 7. 실행 방법

### 7.1 요구 패키지

본 프로젝트는 Python 기반으로 작성되었습니다.

주요 요구 패키지는 다음과 같습니다.

```text
pandas
numpy
matplotlib
scikit-learn
openpyxl
```

패키지는 다음 명령어로 설치할 수 있습니다.

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

일부 환경에서는 한글 폰트 설정 또는 한국어 텍스트 처리를 위한 추가 설정이 필요할 수 있습니다.

---

### 7.2 실행 순서

기본 실행 순서는 다음과 같습니다.

```bash
python "2. preprocessing/missing_value.py"
python "2. preprocessing/outlier_iqr.py"
python "2. preprocessing/model_compare_ml.py"

python "3. trust_filtering/label_analysis_all_labeled.py"
python "3. trust_filtering/label_analysis_all_crawling.py"

python "4. sentiment_analysis/감성분석_토큰화.py"
python "4. sentiment_analysis/감성분석_최종.py"

python "5. insights/04_insight1_review_decision_visualization_checked.py"
python "5. insights/04_platform_avg_only_with_summary_box.py"
python "5. insights/05_insight2_aspect_check_visualization.py"
python "5. insights/05_platform_sentiment_dumbbell_only_clean_v2.py"
python "5. insights/08_rmse_before_after_visualization.py"
```

폴더명에 공백과 마침표가 포함되어 있으므로, 실행할 때 경로를 따옴표로 감싸야 합니다.

일부 코드는 팀원 로컬 환경에서 작성되어 파일 경로가 다를 수 있습니다. 실행 전 각 코드 상단의 다음 변수들을 확인해야 합니다.

```python
INPUT_FILE
OUTPUT_DIR
DATA_DIR
BASE_DIR
```

---

## 8. 데이터 및 재현성 관련 유의사항

* 본 저장소는 수업 과제 목적의 데이터마이닝 프로젝트입니다.
* 원본 리뷰 데이터는 플랫폼별 특성과 계정 정보를 포함하므로, 공유 데이터는 익명화하여 사용했습니다.
* 네이버의 별점 부재는 단순 결측치가 아니라 플랫폼 구조에 따른 구조적 결측으로 보았습니다.
* 카카오 별점은 텍스트 감성별점의 검증 및 비교 기준으로 활용했습니다.
* 일부 중간 산출물은 저장소를 간결하게 유지하기 위해 제외했습니다.
* 실행 환경에 따라 한글 파일명, 한글 폰트, CSV 인코딩 관련 오류가 발생할 수 있습니다.

---

## 9. 최종 보고서

최종 보고서는 `docs` 폴더에 포함되어 있습니다.

```text
docs/데이터마이닝_최종_데마초.pdf
```

---

## 10. 프로젝트 정보

* 프로젝트명: 네이버·카카오 음식점 리뷰 기반 신뢰도 보정 감성 평가 지표 개발
* 수업명: 데이터마이닝실습
* 팀명: 데마초
