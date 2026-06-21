"""
final 버전: 별점만으로는 부족한 리뷰 의사결정 시각화

생성 파일
- fig_insight1_platform_before_after_text_sentiment_final.png
- fig_insight1_kakao_rating_text_sentiment_diff_final.png
- summary_insight1_platform_before_after_final.csv
- summary_insight1_kakao_rating_text_sentiment_diff_final.csv

특징
- 원데이터에서 숫자를 다시 계산합니다.
- 기존 summary CSV와 재계산값이 일치하는지 검증합니다.
- 발표자가 설명하기 쉽도록 그래프 안 문구를 짧게 유지합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR
KNOWN_DATA_DIR = Path(r"C:\Users\eunse\OneDrive\바탕 화면\CSV 파일 합친거")

ALL_FILE_CANDIDATES = [
    KNOWN_DATA_DIR / "all_reviews_sentiment_star_positive_recovery_review_view.csv",
    SCRIPT_DIR / "all_reviews_sentiment_star_positive_recovery_review_view.csv",
]
HIGH_TRUST_FILE_CANDIDATES = [
    KNOWN_DATA_DIR / "final_high_trust_reviews_sentiment_star_positive_recovery_review_view.csv",
    SCRIPT_DIR / "final_high_trust_reviews_sentiment_star_positive_recovery_review_view.csv",
]

ENCODINGS = ["utf-8-sig", "cp949", "utf-8"]

PLATFORM_COL_CANDIDATES = ["platform", "source", "플랫폼", "Platform", "SOURCE"]
RATING_COL_CANDIDATES = ["rating", "star", "별점", "actual_rating", "actual_star"]
TEXT_SENTIMENT_COL_CANDIDATES = [
    "sentiment_star",
    "text_sentiment_star",
    "감성별점",
    "텍스트감성점수",
    "sentiment_score",
]

PLATFORM_ORDER = ["네이버", "카카오"]

OVERALL_COLOR = "#777777"
HIGHTRUST_COLOR = "#287FA3"
NEGATIVE_COLOR = "#D95F59"
POSITIVE_COLOR = "#287FA3"
GRID_COLOR = "#D6D6D6"


def read_csv_safely(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
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


def configure_korean_font() -> str:
    preferred = ["Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in preferred if name in installed), None)
    if selected:
        plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    return selected or "default"


def normalized_name(value: str) -> str:
    return str(value).strip().lower().replace(" ", "").replace("_", "")


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    lookup = {normalized_name(column): column for column in df.columns}
    for candidate in candidates:
        key = normalized_name(candidate)
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(f"후보 컬럼을 찾지 못했습니다: {list(candidates)}")
    return None


def discover_file(explicit_candidates: list[Path], include_patterns: list[str]) -> Path:
    for path in explicit_candidates:
        if path.exists():
            return path

    matches: list[Path] = []
    for directory in [SCRIPT_DIR, KNOWN_DATA_DIR]:
        if not directory.exists():
            continue
        for path in directory.glob("*.csv"):
            name = path.name.lower()
            if all(pattern.lower() in name for pattern in include_patterns):
                matches.append(path)

    if matches:
        return max(matches, key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(f"CSV 파일을 찾지 못했습니다. 포함 패턴: {include_patterns}")


def normalize_platform(value) -> str:
    text = str(value).strip().lower()
    if text in {"naver", "네이버"}:
        return "네이버"
    if text in {"kakao", "카카오"}:
        return "카카오"
    return str(value).strip()


def prepare_frame(
    df: pd.DataFrame,
    platform_col: str,
    sentiment_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    result = df.copy()
    result["_platform"] = result[platform_col].apply(normalize_platform)
    result["_sentiment_star"] = pd.to_numeric(result[sentiment_col], errors="coerce")

    valid_platform = result["_platform"].isin(PLATFORM_ORDER)
    valid_sentiment = result["_sentiment_star"].notna()
    clean = result.loc[valid_platform & valid_sentiment].copy()

    excluded = {
        "raw_rows": int(len(result)),
        "used_rows": int(len(clean)),
        "excluded_unknown_platform": int((~valid_platform).sum()),
        "excluded_missing_or_invalid_sentiment": int((valid_platform & ~valid_sentiment).sum()),
    }
    return clean, excluded


def build_platform_before_after_summary(
    all_df: pd.DataFrame,
    high_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for platform in PLATFORM_ORDER:
        before = all_df.loc[all_df["_platform"] == platform, "_sentiment_star"].dropna()
        after = high_df.loc[high_df["_platform"] == platform, "_sentiment_star"].dropna()
        if before.empty or after.empty:
            continue
        rows.append(
            {
                "platform": platform,
                "all_review_count": int(len(before)),
                "high_trust_review_count": int(len(after)),
                "all_text_sentiment_mean": float(before.mean()),
                "high_trust_text_sentiment_mean": float(after.mean()),
                "score_change": float(after.mean() - before.mean()),
                "retention_rate": float(len(after) / len(before)),
                "all_text_sentiment_std": float(before.std(ddof=1)) if len(before) > 1 else 0.0,
                "high_trust_text_sentiment_std": float(after.std(ddof=1)) if len(after) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_kakao_diff_frame(high_df: pd.DataFrame, rating_col: str) -> tuple[pd.DataFrame, dict[str, int]]:
    high = high_df.copy()
    high["_actual_rating"] = pd.to_numeric(high[rating_col], errors="coerce")

    kakao_all = high.loc[high["_platform"] == "카카오"].copy()
    valid_rating = kakao_all["_actual_rating"].notna()
    valid_sentiment = kakao_all["_sentiment_star"].notna()
    kakao = kakao_all.loc[valid_rating & valid_sentiment].copy()
    kakao["diff"] = kakao["_sentiment_star"] - kakao["_actual_rating"]
    kakao["diff_abs"] = kakao["diff"].abs()

    excluded = {
        "kakao_rows_before_rating_sentiment_filter": int(len(kakao_all)),
        "kakao_rows_used_for_diff": int(len(kakao)),
        "excluded_missing_or_invalid_rating": int((~valid_rating).sum()),
        "excluded_missing_or_invalid_text_sentiment": int((valid_rating & ~valid_sentiment).sum()),
    }
    return kakao, excluded


def build_kakao_diff_summary(kakao: pd.DataFrame) -> pd.DataFrame:
    diff = kakao["diff"].dropna()
    return pd.DataFrame(
        [
            {
                "source_dataset": "High Trust 카카오 리뷰",
                "kakao_review_count": int(len(diff)),
                "mean_actual_rating": float(kakao["_actual_rating"].mean()),
                "mean_text_sentiment_score": float(kakao["_sentiment_star"].mean()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(diff.median()),
                "mean_absolute_diff": float(diff.abs().mean()),
                "ratio_abs_diff_ge_0_5": float((diff.abs() >= 0.5).mean()),
                "ratio_abs_diff_ge_1_0": float((diff.abs() >= 1.0).mean()),
                "ratio_text_more_positive": float((diff > 0).mean()),
                "ratio_text_more_negative": float((diff < 0).mean()),
                "ratio_exact_or_near_match_abs_lt_0_25": float((diff.abs() < 0.25).mean()),
                "min_diff": float(diff.min()),
                "max_diff": float(diff.max()),
            }
        ]
    )


def compare_with_existing_summary(
    recalculated: pd.DataFrame,
    existing_path: Path,
    key_columns: list[str],
) -> bool | None:
    if not existing_path.exists():
        return None
    existing = pd.read_csv(existing_path, encoding="utf-8-sig")
    recalculated_sorted = recalculated.sort_values(key_columns).reset_index(drop=True)
    existing_sorted = existing.sort_values(key_columns).reset_index(drop=True)
    common_columns = [
        column
        for column in recalculated_sorted.columns
        if column in existing_sorted.columns
    ]
    if len(recalculated_sorted) != len(existing_sorted):
        return False
    for column in common_columns:
        left = recalculated_sorted[column]
        right = existing_sorted[column]
        if pd.api.types.is_numeric_dtype(left):
            if not np.allclose(left.to_numpy(float), right.to_numpy(float), equal_nan=True):
                return False
        elif not left.astype(str).equals(right.astype(str)):
            return False
    return True


def save_platform_before_after_figure(summary: pd.DataFrame) -> Path:
    plot_df = summary.set_index("platform").loc[
        [platform for platform in PLATFORM_ORDER if platform in set(summary["platform"])]
    ].reset_index()

    x = np.arange(len(plot_df))
    width = 0.34
    before_values = plot_df["all_text_sentiment_mean"].to_numpy()
    after_values = plot_df["high_trust_text_sentiment_mean"].to_numpy()

    y_min = 3.35
    y_max = 4.52
    fig, ax = plt.subplots(figsize=(12.6, 6.9))

    ax.bar(
        x - width / 2,
        before_values,
        width=width,
        color="#8A8A8A",
        edgecolor="white",
        linewidth=1.2,
        label="전체 리뷰",
    )
    ax.bar(
        x + width / 2,
        after_values,
        width=width,
        color=HIGHTRUST_COLOR,
        edgecolor="white",
        linewidth=1.2,
        label="High Trust 리뷰",
    )

    for idx, row in plot_df.iterrows():
        before = float(row["all_text_sentiment_mean"])
        after = float(row["high_trust_text_sentiment_mean"])
        change = float(row["score_change"])
        ax.text(
            x[idx] - width / 2,
            before + 0.026,
            f"{before:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            x[idx] + width / 2,
            after + 0.026,
            f"{after:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            x[idx],
            max(before, after) + 0.12,
            f"{change:+.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            color=HIGHTRUST_COLOR if change >= 0 else NEGATIVE_COLOR,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D5D5D5"},
        )
        arrow_y = max(before, after) + 0.085
        ax.annotate(
            "",
            xy=(x[idx] + width / 2 - 0.04, arrow_y),
            xytext=(x[idx] - width / 2 + 0.04, arrow_y),
            arrowprops={
                "arrowstyle": "->",
                "color": HIGHTRUST_COLOR,
                "linewidth": 1.8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=4,
        )

    all_gap = np.nan
    high_gap = np.nan
    if {"네이버", "카카오"}.issubset(set(plot_df["platform"])):
        all_gap = (
            float(plot_df.loc[plot_df["platform"] == "네이버", "all_text_sentiment_mean"].iloc[0])
            - float(plot_df.loc[plot_df["platform"] == "카카오", "all_text_sentiment_mean"].iloc[0])
        )
        high_gap = (
            float(plot_df.loc[plot_df["platform"] == "네이버", "high_trust_text_sentiment_mean"].iloc[0])
            - float(plot_df.loc[plot_df["platform"] == "카카오", "high_trust_text_sentiment_mean"].iloc[0])
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["platform"], fontsize=14)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("평균 텍스트 감성 점수", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title(
        "필터 전·후 플랫폼별 평균 텍스트 감성 점수",
        fontsize=22,
        fontweight="bold",
        pad=28,
    )
    ax.text(
        0.5,
        1.018,
        "막대 위 숫자는 평균값, 박스는 High Trust 적용 후 변화량",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color="#666666",
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.18, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    card_ax = fig.add_axes([0.742, 0.37, 0.228, 0.285])
    card_ax.set_axis_off()
    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.045,rounding_size=0.08",
        transform=card_ax.transAxes,
        facecolor="white",
        edgecolor="#888888",
        linewidth=1.25,
        alpha=0.98,
        clip_on=False,
    )
    card_ax.add_patch(card)
    card_ax.plot([0.055, 0.055], [0.15, 0.86], color=HIGHTRUST_COLOR, linewidth=4.0)
    card_ax.text(0.10, 0.84, "핵심 해석", fontsize=11.5, fontweight="bold", color="#555555")
    card_ax.plot([0.08, 0.92], [0.72, 0.72], color="#DDDDDD", linewidth=1)
    card_ax.text(
        0.10,
        0.54,
        "High Trust 이후에도\n플랫폼별 감성 점수 차이 유지",
        ha="left",
        va="center",
        fontsize=11.2,
        fontweight="bold",
    )
    if not np.isnan(all_gap) and not np.isnan(high_gap):
        card_ax.text(
            0.10,
            0.26,
            f"전체 차이 {all_gap:+.3f}점\nHigh Trust 차이 {high_gap:+.3f}점",
            ha="left",
            va="center",
            fontsize=10.5,
            color="#333333",
            linespacing=1.5,
        )

    fig.text(
        0.5,
        0.04,
        "소비자 관점: 별점뿐 아니라 고신뢰 리뷰의 텍스트 감성 흐름도 함께 확인할 필요가 있음",
        ha="center",
        fontsize=11.2,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.70, bottom=0.16, top=0.80)

    output_path = OUTPUT_DIR / "fig_insight1_platform_before_after_text_sentiment_final.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_kakao_diff_figure(kakao: pd.DataFrame, summary: pd.DataFrame) -> Path:
    diff = kakao["diff"].dropna().to_numpy(dtype=float)
    row = summary.iloc[0]

    lower = np.floor((diff.min() - 0.05) / 0.25) * 0.25
    upper = np.ceil((diff.max() + 0.05) / 0.25) * 0.25
    lower = min(lower, -3.0)
    upper = max(upper, 3.0)
    bins = np.arange(lower, upper + 0.25, 0.25)
    counts, edges = np.histogram(diff, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    colors = [NEGATIVE_COLOR if center < -1e-12 else POSITIVE_COLOR for center in centers]

    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    ax.axvspan(lower, 0, color=NEGATIVE_COLOR, alpha=0.055, zorder=0)
    ax.axvspan(0, upper, color=POSITIVE_COLOR, alpha=0.055, zorder=0)
    ax.bar(
        centers,
        counts,
        width=widths * 0.92,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.88,
        zorder=2,
    )
    ax.axvline(0, color="#222222", linestyle="--", linewidth=2.4, zorder=4)
    ax.axvline(row["mean_diff"], color="#111111", linewidth=1.5, alpha=0.75, zorder=4)

    ymax = max(counts) * 1.18
    ax.set_ylim(0, ymax)
    ax.text(
        0,
        ymax * 0.98,
        "0: 별점과 텍스트 감성이 비슷함",
        ha="center",
        va="top",
        fontsize=11.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D8D8D8"},
        zorder=5,
    )
    ax.text(
        row["mean_diff"],
        ymax * 0.80,
        f"mean diff {row['mean_diff']:+.3f}",
        ha="center",
        va="top",
        fontsize=11.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D8D8D8"},
        zorder=5,
    )

    ax.text(
        0.02,
        0.95,
        "음수: 텍스트가\n별점보다 부정적",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        color=NEGATIVE_COLOR,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.95,
        "양수: 텍스트가\n별점보다 긍정적",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.5,
        color=POSITIVE_COLOR,
        fontweight="bold",
    )

    summary_text = (
        f"n = {int(row['kakao_review_count']):,}\n"
        f"mean diff = {row['mean_diff']:+.3f}\n"
        f"mean |diff| = {row['mean_absolute_diff']:.3f}\n"
        f"|diff| ≥ 0.5 = {row['ratio_abs_diff_ge_0_5']:.1%}\n"
        f"|diff| ≥ 1.0 = {row['ratio_abs_diff_ge_1_0']:.1%}"
    )
    ax.text(
        0.975,
        0.61,
        summary_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11.8,
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.5",
            "fc": "white",
            "ec": "#888888",
            "lw": 1.1,
            "alpha": 0.96,
        },
        zorder=5,
    )

    ax.set_title(
        "카카오 별점과 텍스트 감성 점수의 차이 분포",
        fontsize=22,
        fontweight="bold",
        pad=28,
    )
    ax.text(
        0.5,
        1.018,
        "diff = 텍스트 감성 점수 - 실제 별점",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        color="#666666",
    )
    ax.set_xlabel("텍스트 감성 점수 - 실제 별점", fontsize=15)
    ax.set_ylabel("리뷰 수", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.18, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.04,
        "소비자 관점: 같은 별점이라도 텍스트 내용은 더 긍정적이거나 부정적으로 해석될 수 있음",
        ha="center",
        fontsize=11.2,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    output_path = OUTPUT_DIR / "fig_insight1_kakao_rating_text_sentiment_diff_final.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_font = configure_korean_font()

    all_path = discover_file(ALL_FILE_CANDIDATES, ["all_reviews", "review_view"])
    high_path = discover_file(HIGH_TRUST_FILE_CANDIDATES, ["final_high_trust", "review_view"])

    all_raw, all_encoding = read_csv_safely(all_path)
    high_raw, high_encoding = read_csv_safely(high_path)

    platform_col_all = find_column(all_raw, PLATFORM_COL_CANDIDATES)
    platform_col_high = find_column(high_raw, PLATFORM_COL_CANDIDATES)
    sentiment_col_all = find_column(all_raw, TEXT_SENTIMENT_COL_CANDIDATES)
    sentiment_col_high = find_column(high_raw, TEXT_SENTIMENT_COL_CANDIDATES)
    rating_col_high = find_column(high_raw, RATING_COL_CANDIDATES)

    all_df, all_excluded = prepare_frame(all_raw, platform_col_all, sentiment_col_all)
    high_df, high_excluded = prepare_frame(high_raw, platform_col_high, sentiment_col_high)

    platform_summary = build_platform_before_after_summary(all_df, high_df)
    platform_summary_path = OUTPUT_DIR / "summary_insight1_platform_before_after_final.csv"
    platform_summary.to_csv(platform_summary_path, index=False, encoding="utf-8-sig")

    kakao_diff, kakao_excluded = build_kakao_diff_frame(high_df, rating_col_high)
    if kakao_diff.empty:
        raise ValueError("카카오 리뷰 중 실제 별점과 텍스트 감성 점수가 모두 있는 행이 없습니다.")
    kakao_summary = build_kakao_diff_summary(kakao_diff)
    kakao_summary_path = OUTPUT_DIR / "summary_insight1_kakao_rating_text_sentiment_diff_final.csv"
    kakao_summary.to_csv(kakao_summary_path, index=False, encoding="utf-8-sig")

    previous_platform_ok = compare_with_existing_summary(
        platform_summary,
        OUTPUT_DIR / "summary_insight1_platform_before_after.csv",
        ["platform"],
    )
    previous_kakao_ok = compare_with_existing_summary(
        kakao_summary,
        OUTPUT_DIR / "summary_insight1_kakao_rating_text_sentiment_diff.csv",
        ["source_dataset"],
    )

    fig1_path = save_platform_before_after_figure(platform_summary)
    fig2_path = save_kakao_diff_figure(kakao_diff, kakao_summary)

    print("[사용 파일]")
    print(f"- 전체 리뷰: {all_path} / encoding={all_encoding} / rows={len(all_raw):,}")
    print(f"- High Trust 리뷰: {high_path} / encoding={high_encoding} / rows={len(high_raw):,}")
    print(f"[한글 폰트] {selected_font}")

    print("\n[매칭된 컬럼]")
    print(f"- 전체 리뷰 platform: {platform_col_all}")
    print(f"- 전체 리뷰 text sentiment: {sentiment_col_all}")
    print(f"- High Trust platform: {platform_col_high}")
    print(f"- High Trust text sentiment: {sentiment_col_high}")
    print(f"- High Trust actual rating: {rating_col_high}")

    print("\n[제외/사용 행 수]")
    print(f"- 전체 리뷰 사용 행 수: {all_excluded['used_rows']:,} / 원본 {all_excluded['raw_rows']:,}")
    print(f"  · 플랫폼 제외: {all_excluded['excluded_unknown_platform']:,}")
    print(f"  · 텍스트 감성 점수 결측/변환 오류 제외: {all_excluded['excluded_missing_or_invalid_sentiment']:,}")
    print(f"- High Trust 리뷰 사용 행 수: {high_excluded['used_rows']:,} / 원본 {high_excluded['raw_rows']:,}")
    print(f"  · 플랫폼 제외: {high_excluded['excluded_unknown_platform']:,}")
    print(f"  · 텍스트 감성 점수 결측/변환 오류 제외: {high_excluded['excluded_missing_or_invalid_sentiment']:,}")
    print(
        "- 카카오 실제 별점 vs 텍스트 감성 비교 최종 행 수: "
        f"{kakao_excluded['kakao_rows_used_for_diff']:,} / 카카오 High Trust "
        f"{kakao_excluded['kakao_rows_before_rating_sentiment_filter']:,}"
    )
    print(f"  · 실제 별점 결측/변환 오류 제외: {kakao_excluded['excluded_missing_or_invalid_rating']:,}")
    print(
        "  · 텍스트 감성 점수 결측/변환 오류 제외: "
        f"{kakao_excluded['excluded_missing_or_invalid_text_sentiment']:,}"
    )

    print("\n[플랫폼별 필터 전후 요약]")
    print(platform_summary.to_string(index=False))
    print("\n[카카오 별점-텍스트 감성 차이 요약]")
    print(kakao_summary.to_string(index=False))

    print("\n[기존 summary와 재계산값 일치 여부]")
    print(f"- 플랫폼 전후 summary: {previous_platform_ok}")
    print(f"- 카카오 diff summary: {previous_kakao_ok}")

    print("\n[저장 확인]")
    for path in [fig1_path, fig2_path, platform_summary_path, kakao_summary_path]:
        print(f"- {path} / exists={path.exists()} / bytes={path.stat().st_size if path.exists() else 0:,}")


if __name__ == "__main__":
    main()
