"""
같은 식당의 네이버·카카오 감성 별점 차이 dumbbell chart만 생성하는 코드.

입력:
- final_high_trust_reviews_sentiment_star_positive_recovery_review_view.csv

출력:
- fig_store_naver_kakao_sentiment_comparison.png

주의:
- 다른 PNG, CSV, TXT 파일은 생성하지 않습니다.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / 'outputs' / 'final_high_trust_reviews_sentiment_star_positive_recovery_review_view.csv'
if not INPUT_FILE.exists():
    INPUT_FILE = PROJECT_DIR / 'outputs' / 'sentiment' / 'final_high_trust_reviews_sentiment_star_positive_recovery_review_view.csv'
OUTPUT_DIR = PROJECT_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_REVIEWS_PER_PLATFORM = 5
TOP_N_GAP_STORES = 15

PLATFORM_COLORS = {
    "네이버": "#2DB400",
    "카카오": "#F5C400",
}


def read_csv_safely(path: Path):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding), encoding
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"CSV 인코딩을 읽지 못했습니다: {path} / 마지막 오류: {last_error}",
    )


def configure_korean_font():
    preferred_fonts = ["Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in preferred_fonts if font in installed), None)

    if selected:
        plt.rcParams["font.family"] = selected

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    return selected or "default"


def normalize_platform(value):
    text = str(value).strip().lower()

    if text in {"naver", "네이버"}:
        return "네이버"

    if text in {"kakao", "카카오"}:
        return "카카오"

    return str(value).strip()


def build_store_comparison(df):
    grouped = (
        df.groupby(["store_name", "platform"])["sentiment_star"]
        .agg(["mean", "count"])
        .reset_index()
    )

    means = grouped.pivot(index="store_name", columns="platform", values="mean")
    counts = grouped.pivot(index="store_name", columns="platform", values="count")

    required = {"네이버", "카카오"}
    if not required.issubset(means.columns):
        raise ValueError("네이버와 카카오 양쪽 플랫폼 데이터가 모두 필요합니다.")

    comparison = pd.DataFrame(
        {
            "store_name": means.index,
            "naver_mean": means["네이버"],
            "kakao_mean": means["카카오"],
            "naver_review_count": counts["네이버"],
            "kakao_review_count": counts["카카오"],
        }
    ).dropna(subset=["naver_mean", "kakao_mean"])

    comparison["platform_gap"] = comparison["naver_mean"] - comparison["kakao_mean"]
    comparison["absolute_gap"] = comparison["platform_gap"].abs()
    comparison["eligible_min_reviews"] = (
        (comparison["naver_review_count"] >= MIN_REVIEWS_PER_PLATFORM)
        & (comparison["kakao_review_count"] >= MIN_REVIEWS_PER_PLATFORM)
    )

    return comparison.sort_values("absolute_gap", ascending=False).reset_index(drop=True)


def save_store_dumbbell_figure(comparison):
    eligible = comparison.loc[comparison["eligible_min_reviews"]].copy()

    if eligible.empty:
        eligible = comparison.copy()
        subtitle = "양쪽 플랫폼 리뷰 보유 식당"
    else:
        subtitle = f"플랫폼별 리뷰 {MIN_REVIEWS_PER_PLATFORM}개 이상 식당"

    plot_df = (
        eligible.nlargest(TOP_N_GAP_STORES, "absolute_gap")
        .sort_values("platform_gap")
        .reset_index(drop=True)
    )

    y = np.arange(len(plot_df))
    fig_height = max(7, 0.48 * len(plot_df) + 2.7)

    fig, ax = plt.subplots(figsize=(12.5, fig_height))

    for idx, row in plot_df.iterrows():
        ax.plot(
            [row["naver_mean"], row["kakao_mean"]],
            [idx, idx],
            color="#B7B7B7",
            linewidth=2.2,
            zorder=1,
        )

    ax.scatter(
        plot_df["naver_mean"],
        y,
        s=78,
        color=PLATFORM_COLORS["네이버"],
        edgecolor="#333333",
        linewidth=0.6,
        label="네이버 평균",
        zorder=3,
    )

    ax.scatter(
        plot_df["kakao_mean"],
        y,
        s=78,
        color=PLATFORM_COLORS["카카오"],
        edgecolor="#333333",
        linewidth=0.6,
        label="카카오 평균",
        zorder=3,
    )

    for idx, row in plot_df.iterrows():
        right = max(row["naver_mean"], row["kakao_mean"])
        ax.text(
            min(5.08, right + 0.08),
            idx,
            f"차이 {row['platform_gap']:+.2f}",
            va="center",
            fontsize=10,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["store_name"], fontsize=11)
    ax.set_xlim(1, 5.35)
    ax.set_xlabel("평균 감성 별점", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)

    # 메인 제목과 부제목이 너무 떨어져 보이지 않도록 위치를 가깝게 조정
    fig.suptitle(
        "같은 식당의 네이버·카카오 감성 별점 차이",
        fontsize=28,
        fontweight="bold",
        y=0.965,
    )

    # "절대" 표현은 제거하고, 부제목 글자도 약간 키움
    ax.set_title(
        f"차이가 큰 상위 {len(plot_df)}개 식당 · {subtitle}",
        fontsize=15,
        color="#555555",
        pad=8,
    )

    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.legend(loc="upper left", frameon=True, fontsize=12)

    # 제목과 부제목이 더 붙어 보이도록 위쪽 여백을 조정
    fig.tight_layout(rect=[0, 0, 1, 0.955])

    output_path = OUTPUT_DIR / "fig_store_naver_kakao_sentiment_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 CSV를 찾을 수 없습니다: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_font = configure_korean_font()

    df, encoding = read_csv_safely(INPUT_FILE)

    required_columns = {"platform", "store_name", "sentiment_star"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = df.copy()
    df["platform"] = df["platform"].apply(normalize_platform)
    df["store_name"] = df["store_name"].fillna("").astype(str).str.strip()
    df["sentiment_star"] = pd.to_numeric(df["sentiment_star"], errors="coerce")

    df = df.loc[
        df["platform"].isin(["네이버", "카카오"])
        & df["sentiment_star"].notna()
        & df["store_name"].ne("")
    ].copy()

    store_comparison = build_store_comparison(df)
    output_path = save_store_dumbbell_figure(store_comparison)

    print(f"[입력 인코딩] {encoding}")
    print(f"[한글 폰트] {selected_font}")
    print(f"[분석 리뷰 수] {len(df):,}")
    print(f"[양쪽 플랫폼 공통 식당 수] {len(store_comparison):,}")
    print(f"[생성된 차트] {output_path}")


if __name__ == "__main__":
    main()
