import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. 파일 경로 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# 여기를 네 파일명으로 바꿔
INPUT_FILE = BASE_DIR / "labeled_all_filter_result.csv"

OUT_DIR = BASE_DIR / "model_compare_result"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("전체 데이터 크기:", df.shape)
print("컬럼 목록:")
print(df.columns.tolist())


# =========================================================
# 2. manual_label 정리
# =========================================================

if "manual_label" not in df.columns:
    raise ValueError("manual_label 컬럼이 없습니다. 수동 라벨링된 파일을 사용해야 합니다.")

# manual_label 결측 제거
df = df[df["manual_label"].notna()].copy()

# manual_label이 문자일 경우를 대비한 변환
def normalize_label(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, str):
        x_clean = x.strip().lower()

        real_values = ["1", "real", "genuine", "true", "high", "신뢰", "진짜", "일반"]
        fake_values = ["0", "fake", "false", "low", "광고", "가짜", "저신뢰"]

        if x_clean in real_values:
            return 1
        if x_clean in fake_values:
            return 0

    try:
        return int(float(x))
    except:
        return np.nan

df["manual_label"] = df["manual_label"].apply(normalize_label)
df = df[df["manual_label"].isin([0, 1])].copy()
df["manual_label"] = df["manual_label"].astype(int)

print("\n라벨 분포:")
print(df["manual_label"].value_counts())


# =========================================================
# 3. Feature / Target 분리
# =========================================================

y = df["manual_label"]

# 데이터 누수 방지를 위해 제외할 컬럼
# manual_label: 정답
# pred_label, trust_level: 기존 모델 결과
# trust_score: 최종 규칙 모델 점수이므로 ML 입력에서 제외
# trust_reasons: 규칙 결과 설명
# review_text: 여기서는 TF-IDF 없이 이미 추출된 feature만 사용
# store_name, account_id: 식당/계정 식별자라 일반화에 방해될 수 있음
drop_cols = [
    "manual_label",
    "pred_label",
    "trust_level",
    "trust_score",
    "trust_weight",
    "rating_x_trust_weight",
    "trust_reasons",
    "review_text",
    "store_name",
    "account_id",
    "visit_date"
]

feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()

# 전부 결측인 컬럼 제거
X = X.dropna(axis=1, how="all")

# 숫자형 / 범주형 컬럼 분리
numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

print("\n사용 feature 수:", X.shape[1])
print("숫자형 feature 수:", len(numeric_cols))
print("범주형 feature 수:", len(categorical_cols))


# =========================================================
# 4. Train / Test split
# =========================================================

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X,
    y,
    df,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain 크기:", X_train.shape[0])
print("Test 크기:", X_test.shape[0])
print("\nTrain 라벨 분포:")
print(y_train.value_counts())
print("\nTest 라벨 분포:")
print(y_test.value_counts())


# =========================================================
# 5. 전처리 파이프라인
# =========================================================

try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_preprocess_scaled = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_preprocess_plain = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_preprocess = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", onehot)
])

preprocess_scaled = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess_scaled, numeric_cols),
        ("cat", categorical_preprocess, categorical_cols)
    ],
    remainder="drop"
)

preprocess_plain = ColumnTransformer(
    transformers=[
        ("num", numeric_preprocess_plain, numeric_cols),
        ("cat", categorical_preprocess, categorical_cols)
    ],
    remainder="drop"
)


# =========================================================
# 6. 평가 함수
# =========================================================

def evaluate_model(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_real": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_real": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_real": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_fake": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1]
    }

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)
    print("Confusion Matrix [[TN, FP], [FN, TP]]")
    print(cm)
    print(classification_report(y_true, y_pred, target_names=["fake(0)", "real(1)"], zero_division=0))

    return result


results = []


# =========================================================
# 7. Baseline 0: Dummy Classifier
#    다수 클래스를 찍는 최소 기준선
# =========================================================

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

results.append(evaluate_model("Dummy_Baseline", y_test, dummy_pred))


# =========================================================
# 8. Logistic Regression
# =========================================================

logreg_pipe = Pipeline(steps=[
    ("preprocess", preprocess_scaled),
    ("clf", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ))
])

logreg_params = {
    "clf__C": [0.01, 0.1, 1, 10]
}

logreg_grid = GridSearchCV(
    estimator=logreg_pipe,
    param_grid=logreg_params,
    scoring="f1",
    cv=3,
    n_jobs=-1
)

logreg_grid.fit(X_train, y_train)

print("\nLogistic Regression Best Params:")
print(logreg_grid.best_params_)

logreg_pred = logreg_grid.predict(X_test)
results.append(evaluate_model("Logistic_Regression", y_test, logreg_pred))


# =========================================================
# 9. Random Forest
# =========================================================

rf_pipe = Pipeline(steps=[
    ("preprocess", preprocess_plain),
    ("clf", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

rf_params = {
    "clf__n_estimators": [200, 500],
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_leaf": [1, 3, 5]
}

rf_grid = GridSearchCV(
    estimator=rf_pipe,
    param_grid=rf_params,
    scoring="f1",
    cv=3,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

print("\nRandom Forest Best Params:")
print(rf_grid.best_params_)

rf_pred = rf_grid.predict(X_test)
results.append(evaluate_model("Random_Forest", y_test, rf_pred))


# =========================================================
# 10. 최종 규칙 기반 trust_score 모델
#     ML 입력에는 trust_score를 안 넣었지만,
#     최종 제안 모델 평가에는 trust_score threshold 사용
# =========================================================

if "trust_score" in df_test.columns:
    # 보고서에서 binary threshold로 사용한 5.7 기준
    rule_pred_57 = (df_test["trust_score"] >= 5.7).astype(int)
    results.append(evaluate_model("Rule_TrustScore_threshold_5.7", y_test, rule_pred_57))

    # 참고용: High Trust 운영 기준 3.5
    rule_pred_35 = (df_test["trust_score"] >= 3.5).astype(int)
    results.append(evaluate_model("Rule_TrustScore_threshold_3.5", y_test, rule_pred_35))
else:
    print("\n주의: trust_score 컬럼이 없어 규칙 기반 모델 평가는 건너뜁니다.")


# =========================================================
# 11. 결과 저장
# =========================================================

result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by="f1_real", ascending=False)

print("\n최종 모델 비교표:")
print(result_df)

result_df.to_csv(OUT_DIR / "model_comparison_metrics.csv", index=False, encoding="utf-8-sig")

# Test 예측 결과 저장
test_output = df_test.copy()
test_output["pred_dummy"] = dummy_pred
test_output["pred_logistic_regression"] = logreg_pred
test_output["pred_random_forest"] = rf_pred

if "trust_score" in test_output.columns:
    test_output["pred_rule_5_7"] = (test_output["trust_score"] >= 5.7).astype(int)
    test_output["pred_rule_3_5"] = (test_output["trust_score"] >= 3.5).astype(int)

test_output.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")


# =========================================================
# 12. Feature Importance 저장
# =========================================================

# 전처리 후 feature 이름 가져오기
def get_feature_names(preprocessor):
    names = []

    if len(numeric_cols) > 0:
        names.extend(numeric_cols)

    if len(categorical_cols) > 0:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        names.extend(cat_names)

    return names


# Logistic Regression 계수
try:
    logreg_best = logreg_grid.best_estimator_
    logreg_feature_names = get_feature_names(logreg_best.named_steps["preprocess"])
    logreg_coef = logreg_best.named_steps["clf"].coef_[0]

    logreg_importance = pd.DataFrame({
        "feature": logreg_feature_names,
        "coefficient": logreg_coef,
        "abs_coefficient": np.abs(logreg_coef)
    }).sort_values(by="abs_coefficient", ascending=False)

    logreg_importance.to_csv(
        OUT_DIR / "logistic_regression_coefficients.csv",
        index=False,
        encoding="utf-8-sig"
    )

except Exception as e:
    print("Logistic Regression 계수 저장 실패:", e)


# Random Forest feature importance
try:
    rf_best = rf_grid.best_estimator_
    rf_feature_names = get_feature_names(rf_best.named_steps["preprocess"])
    rf_importance_values = rf_best.named_steps["clf"].feature_importances_

    rf_importance = pd.DataFrame({
        "feature": rf_feature_names,
        "importance": rf_importance_values
    }).sort_values(by="importance", ascending=False)

    rf_importance.to_csv(
        OUT_DIR / "random_forest_feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

except Exception as e:
    print("Random Forest 중요도 저장 실패:", e)


print("\n저장 완료:")
print(OUT_DIR / "model_comparison_metrics.csv")
print(OUT_DIR / "test_predictions.csv")
print(OUT_DIR / "logistic_regression_coefficients.csv")
print(OUT_DIR / "random_forest_feature_importance.csv")