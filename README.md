# Restaurant Review Analysis

네이버지도와 카카오맵 음식점 리뷰를 기반으로 리뷰 신뢰도 필터링, 감성분석, 플랫폼별 비교 분석을 수행한 데이터마이닝 프로젝트입니다.

프로젝트의 목적은 단순 별점이나 리뷰 수만으로 음식점을 평가하는 방식의 한계를 보완하고, 리뷰 텍스트의 신뢰도와 감성 정보를 반영한 새로운 맛집 평가 지표를 제안하는 것입니다.

---

## 1. Project Overview

네이버지도와 카카오맵의 음식점 리뷰 데이터를 수집한 뒤, 다음과 같은 분석 절차를 수행했습니다.

1. 리뷰 데이터 수집
2. 데이터 전처리 및 익명화
3. 리뷰 신뢰도 필터링
4. 고신뢰 리뷰 기반 감성분석
5. 감성점수의 1~5점 별점 변환
6. 플랫폼별 감성별점 및 aspect 비교
7. 신뢰도 필터 적용 전후 성능 비교

분석 대상은 서울 성수·마포 지역 한식 음식점 리뷰이며, 네이버와 카카오의 플랫폼 구조 차이를 고려하여 비교 분석을 진행했습니다.

---

## 2. Data Description

네이버지도와 카카오맵 음식점 리뷰 데이터를 사용했습니다.

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

개인정보 보호를 위해 계정 정보는 익명화하여 사용했습니다.

---

## 3. Repository Structure

```text
restaurant_review_analysis/
├── crawling/
│   └── 리뷰 수집 코드
│
├── preprocessing/
│   └── 데이터 정리 및 전처리 코드
│
├── trust_filtering/
│   └── 리뷰 신뢰도 점수 산출 및 필터링 코드
│
├── sentiment_analysis/
│   └── 감성사전 기반 감성점수 및 감성별점 산출 코드
│
├── insights/
│   ├── figures/
│   │   └── 최종 발표 및 보고서에 사용한 시각화 이미지
│   └── 인사이트 시각화 코드
│
├── docs/
│   └── 발표 자료 및 보고서
│
└── README.md
```

---

## 4. Methodology

### 4.1 Trust Filtering

리뷰의 신뢰도를 판단하기 위해 규칙 기반 신뢰도 점수를 산출했습니다.

신뢰도 점수는 다음과 같은 요소를 반영했습니다.

* 메뉴명 포함 여부
* 구체적인 경험 표현 포함 여부
* 숫자 정보 포함 여부
* 사진 포함 여부
* 리뷰 길이
* 계정 리뷰 수
* 재방문 표현
* 광고성·이벤트성 표현
* 지나치게 일반적인 칭찬 표현
* 구체적인 부정 경험 표현

신뢰도 필터의 주요 기준은 다음과 같습니다.

* `trust_score >= 5.7`: 수동 라벨 기준 평가용 high 예측 기준
* `trust_score >= 3.5`: 최종 분석에 포함한 high trust 리뷰 기준

최종 분석에는 high trust 리뷰 4,268건을 사용했습니다.

---

### 4.2 Sentiment Analysis

신뢰도 필터를 통과한 리뷰를 대상으로 감성사전 기반 감성분석을 수행했습니다.

감성분석은 다음 aspect를 기준으로 진행했습니다.

* `food`: 음식
* `price`: 가격
* `service`: 서비스
* `atmosphere`: 분위기
* `general`: 일반 평가

각 감성 표현은 aspect, polarity, score를 기준으로 사전에 등록했으며, 리뷰 텍스트 내 표현과 매칭하여 감성점수를 산출했습니다.

이후 감성점수를 1~5점 범위의 감성별점으로 변환했습니다.

---

### 4.3 Platform Comparison

네이버와 카카오의 플랫폼 차이를 고려하여 다음 항목을 비교했습니다.

* 플랫폼별 평균 감성별점
* 동일 식당의 네이버·카카오 감성별점 차이
* 식당별 aspect 강점 및 약점
* 카카오 실제 별점과 텍스트 감성별점의 차이
* 신뢰도 필터 적용 전후 RMSE 변화

---

## 5. Main Results

### 5.1 High Trust Review Count

신뢰도 필터 적용 후 high trust 리뷰 수는 다음과 같습니다.

* 전체 high trust 리뷰: 4,268건
* 네이버 high trust 리뷰: 2,786건
* 카카오 high trust 리뷰: 1,482건

---

### 5.2 Platform Sentiment Difference

동일 식당 기준으로 네이버와 카카오의 감성별점을 비교한 결과, 플랫폼별 리뷰 분위기와 평가 경향에 차이가 있음을 확인했습니다.

* 네이버 평균 감성별점: 4.193
* 카카오 평균 감성별점: 3.716

이는 네이버와 카카오 리뷰를 단순히 통합하여 비교하기보다, 플랫폼별 리뷰 생성 구조와 평가 문화를 고려해야 함을 보여줍니다.

---

### 5.3 RMSE Improvement

카카오 실제 별점과 텍스트 기반 감성별점을 비교한 결과, 신뢰도 필터 적용 후 오차가 감소했습니다.

* 필터 적용 전 RMSE: 1.0605
* 필터 적용 후 RMSE: 0.8609

이를 통해 신뢰도 필터가 리뷰 기반 평가 지표의 안정성을 높이는 데 기여했음을 확인했습니다.

---

## 6. How to Run

### 6.1 Required Packages

본 프로젝트는 Python 기반으로 작성되었습니다.

주요 패키지는 다음과 같습니다.

```text
pandas
numpy
matplotlib
scikit-learn
openpyxl
```

필요한 패키지는 다음 명령어로 설치할 수 있습니다.

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
```

---

### 6.2 Execution Order

프로젝트 코드는 다음 순서로 실행합니다.

```bash
python trust_filtering/01_label_analysis_all_labeled.py
python trust_filtering/02_label_analysis_all_crawling.py
python sentiment_analysis/03_sentiment_analysis.py
python insights/04_insight1_review_decision_visualization_checked.py
python insights/05_insight2_aspect_check_visualization.py
python insights/04_platform_avg_only_with_summary_box.py
python insights/05_platform_sentiment_dumbbell_only_clean_v2.py
python insights/08_rmse_before_after_visualization.py
```

## 7. Output Files

최종 결과물은 주로 `insights/figures/` 폴더에 정리했습니다.

주요 figure는 다음과 같습니다.

* 플랫폼별 평균 감성별점 비교
* 동일 식당의 네이버·카카오 감성별점 차이
* 전체 리뷰와 high trust 리뷰의 감성별점 비교
* 식당별 aspect 히트맵
* 감성별점 보정 전후 RMSE 비교

---

## 8. Notes

* 본 프로젝트는 수업 과제 목적의 분석 프로젝트입니다.
* 크롤링 데이터는 연구 및 학습 목적으로만 사용했습니다.
* 계정 정보는 익명화하여 분석했습니다.
* 네이버의 별점 부재는 단순 결측치가 아니라 플랫폼 구조에 따른 구조적 결측으로 보았습니다.
* 카카오 별점은 텍스트 감성별점의 검증 및 비교 기준으로 활용했습니다.

---

## 9. Contributors

팀명: 데마초

본 프로젝트는 데이터마이닝실습 팀 프로젝트로 수행되었습니다.
