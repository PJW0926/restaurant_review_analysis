"""
플랫폼별 평균 감성 별점 비교 막대그래프만 생성하는 코드

기능
- 네이버 / 카카오 평균 감성 별점 계산
- 막대그래프 생성
- 그래프 안에 요약 박스 표시
  (네이버 점수 / 카카오 점수 / 차이)
- 다른 PNG, CSV, TXT 파일은 생성하지 않음

입력 파일
- final_high_trust_reviews_sentiment_star_positive_recovery_review_view.csv

출력 파일
- fig_platform_avg_sentiment_star.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / 'outputs' / 'final_high_trust_reviews_sentiment_star_positive_recovery_review_view.csv'
if not INPUT_FILE.exists():
    INPUT_FILE = PROJECT_DIR / 'outputs' / 'sentiment' / 'final_high_trust_reviews_sentiment_star_positive_recovery_review_view.csv'
OUTPUT_DIR = PROJECT_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def build_platform_summary(df):
    summary = (
        df.groupby("platform")["sentiment_star"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_sentiment_star", "count": "review_count"})
    )

    # 순서를 네이버 -> 카카오로 고정
    order = ["네이버", "카카오"]
    summary["platform"] = pd.Categorical(summary["platform"], categories=order, ordered=True)
    summary = summary.sort_values("platform").reset_index(drop=True)

    return summary


def save_platform_average_figure(summary):
    required_platforms = {"네이버", "카카오"}
    if not required_platforms.issubset(set(summary["platform"].astype(str))):
        raise ValueError("네이버와 카카오 두 플랫폼이 모두 있어야 그래프를 생성할 수 있습니다.")

    fig, ax = plt.subplots(figsize=(8.2, 6.3))

    x = summary["platform"].tolist()
    y = summary["avg_sentiment_star"].tolist()
    counts = summary["review_count"].tolist()
    colors = [PLATFORM_COLORS.get(platform, "#999999") for platform in x]

    bars = ax.bar(x, y, color=colors, width=0.58)

    # 막대 위 값 라벨
    for bar, mean_value, count_value in zip(bars, y, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            f"{mean_value:.3f}점\n(n={count_value:,})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # 차이 계산
    naver_mean = float(summary.loc[summary["platform"] == "네이버", "avg_sentiment_star"].iloc[0])
    kakao_mean = float(summary.loc[summary["platform"] == "카카오", "avg_sentiment_star"].iloc[0])
    gap = naver_mean - kakao_mean

    # 제목 / 부제목
    ax.set_title(
        "플랫폼별 평균 감성 별점 비교",
        fontsize=20,
        fontweight="bold",
        pad=16,
    )
    ax.text(
        0.5,
        1.02,
        "네이버와 카카오의 평균 감성 별점 차이",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#666666",
    )

    ax.set_ylabel("평균 감성 별점", fontsize=13)
    ax.set_ylim(0, 5.3)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    # 그래프 안 요약 박스 (오른쪽 위 안쪽)
    summary_text = (
        f"네이버 {naver_mean:.3f}점\n"
        f"카카오 {kakao_mean:.3f}점\n"
        f"차이 {gap:+.3f}점"
    )
    ax.text(
        0.97,
        0.80,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#666666",
            linewidth=1.2,
            alpha=0.95,
        ),
    )

    fig.tight_layout()

    output_path = OUTPUT_DIR / "fig_platform_avg_sentiment_star.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path, naver_mean, kakao_mean, gap


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"입력 CSV를 찾을 수 없습니다: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_font = configure_korean_font()
    df, encoding = read_csv_safely(INPUT_FILE)

    required_columns = {"platform", "sentiment_star"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = df.copy()
    df["platform"] = df["platform"].apply(normalize_platform)
    df["sentiment_star"] = pd.to_numeric(df["sentiment_star"], errors="coerce")

    df = df.loc[
        df["platform"].isin(["네이버", "카카오"])
        & df["sentiment_star"].notna()
    ].copy()

    summary = build_platform_summary(df)
    output_path, naver_mean, kakao_mean, gap = save_platform_average_figure(summary)

    print(f"[입력 인코딩] {encoding}")
    print(f"[한글 폰트] {selected_font}")
    print(f"[분석 리뷰 수] {len(df):,}")
    print(f"[생성된 그래프] {output_path}")
    print(f"[네이버 평균] {naver_mean:.3f}")
    print(f"[카카오 평균] {kakao_mean:.3f}")
    print(f"[차이] {gap:+.3f}")


if __name__ == "__main__":
    main()
