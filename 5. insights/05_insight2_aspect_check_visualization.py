from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyBboxPatch


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR

ENCODINGS = ("utf-8-sig", "cp949", "utf-8", "euc-kr")
MIN_REVIEWS = 50
MIN_EVIDENCE_COUNT = 5
MIN_EVIDENCE_RATE = 0.03
CHECK_SCORE_THRESHOLD = -0.03
STRENGTH_SCORE_THRESHOLD = 0.05

STORE_CANDIDATES = ["store_name", "restaurant_name", "가게명", "식당명"]
PLATFORM_CANDIDATES = ["platform", "source", "플랫폼"]
SENTIMENT_CANDIDATES = ["sentiment_star", "text_sentiment_star", "감성별점"]

ASPECT_CANDIDATES = {
    "food": ["food_score", "food", "음식", "맛"],
    "price": ["price_score", "price", "가격", "가성비"],
    "service": ["service_score", "service", "서비스", "응대"],
    "atmosphere": ["atmosphere_score", "atmosphere", "분위기"],
    "wait": ["wait_score", "waiting_score", "wait", "웨이팅", "대기"],
    "revisit": ["revisit_score", "revisit", "재방문", "추천"],
    "hygiene": ["hygiene_score", "hygiene", "위생", "청결"],
}

ASPECT_LABELS = {
    "food": "음식",
    "price": "가격",
    "service": "서비스",
    "atmosphere": "분위기",
    "wait": "웨이팅",
    "revisit": "재방문",
    "hygiene": "위생",
}

NON_FOOD_ASPECTS = ["price", "service", "atmosphere", "wait", "revisit", "hygiene"]


def set_korean_font() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def read_csv_safely(path: Path) -> tuple[pd.DataFrame, str]:
    last_error: Exception | None = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False), enc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"CSV를 읽을 수 없습니다: {path} ({last_error})")


def norm_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "")


def find_column(columns: Iterable[str], candidates: list[str], required: bool = True) -> str | None:
    columns = list(columns)
    norm_map = {norm_col(col): col for col in columns}
    for cand in candidates:
        if norm_col(cand) in norm_map:
            return norm_map[norm_col(cand)]
    for cand in candidates:
        cand_norm = norm_col(cand)
        for col in columns:
            if cand_norm in norm_col(col):
                return col
    if required:
        raise KeyError(f"컬럼을 찾지 못했습니다. 후보: {candidates}")
    return None


def find_input_file() -> tuple[Path, pd.DataFrame, str, dict[str, str]]:
    priority = [
        "final_high_trust_reviews_sentiment_star_positive_recovery_review_view.csv",
        "final_high_trust_reviews_sentiment_star_minimal_tuned_review_view.csv",
        "final_high_trust_reviews_sentiment_star_final_tuned_review_view.csv",
        "final_high_trust_reviews_sentiment_star_overcap_review_view.csv",
    ]
    search_dirs = [
        SCRIPT_DIR,
        Path(r"C:\Users\eunse\OneDrive\바탕 화면\CSV 파일 합친거"),
    ]

    files: list[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for name in priority:
            path = directory / name
            if path.exists():
                files.append(path)
        files.extend(sorted(directory.glob("*.csv")))

    best: tuple[int, Path, pd.DataFrame, str, dict[str, str]] | None = None
    seen: set[Path] = set()
    for path in files:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            df, enc = read_csv_safely(path)
            store_col = find_column(df.columns, STORE_CANDIDATES, required=False)
            aspect_cols = {
                aspect: find_column(df.columns, cands, required=False)
                for aspect, cands in ASPECT_CANDIDATES.items()
            }
            matched = {k: v for k, v in aspect_cols.items() if v is not None}
            score = len(matched) * 10 + (30 if path.name in priority else 0) + (10 if store_col else 0)
            if store_col and len(matched) >= 4:
                mapping = {"store": store_col, **{f"aspect_{k}": v for k, v in matched.items()}}
                platform_col = find_column(df.columns, PLATFORM_CANDIDATES, required=False)
                sentiment_col = find_column(df.columns, SENTIMENT_CANDIDATES, required=False)
                if platform_col:
                    mapping["platform"] = platform_col
                if sentiment_col:
                    mapping["sentiment"] = sentiment_col
                if best is None or score > best[0]:
                    best = (score, path, df, enc, mapping)
        except Exception as exc:  # noqa: BLE001
            print(f"[건너뜀] {path.name}: {exc}")

    if best is None:
        raise FileNotFoundError("식당명과 aspect 점수 컬럼을 가진 CSV를 찾지 못했습니다.")
    _, path, df, enc, mapping = best
    return path, df, enc, mapping


def aspect_long_summary(df: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    store_col = mapping["store"]
    sentiment_col = mapping.get("sentiment")
    aspect_keys = [a for a in ASPECT_LABELS if f"aspect_{a}" in mapping]
    aspect_cols = [mapping[f"aspect_{a}"] for a in aspect_keys]

    work_cols = [store_col] + ([sentiment_col] if sentiment_col else []) + aspect_cols
    work = df[work_cols].copy()
    work[store_col] = work[store_col].astype(str).str.strip()
    work = work[work[store_col] != ""]
    for col in ([sentiment_col] if sentiment_col else []) + aspect_cols:
        if col:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    base = work.groupby(store_col).size().rename("review_count").reset_index()
    base = base.rename(columns={store_col: "store_name"})
    if sentiment_col:
        sent = work.groupby(store_col)[sentiment_col].mean().rename("sentiment_star").reset_index()
        sent = sent.rename(columns={store_col: "store_name"})
        base = base.merge(sent, on="store_name", how="left")

    rows = []
    for aspect, col in zip(aspect_keys, aspect_cols):
        grouped = work.groupby(store_col)[col].agg(
            aspect_score_mean="mean",
            aspect_evidence_count=lambda s: int((s.fillna(0) != 0).sum()),
        ).reset_index()
        grouped = grouped.rename(columns={store_col: "store_name"})
        grouped["aspect"] = aspect
        grouped["aspect_label"] = ASPECT_LABELS[aspect]
        rows.append(grouped)
    long = pd.concat(rows, ignore_index=True)
    long = long.merge(base[["store_name", "review_count"]], on="store_name", how="left")
    long["aspect_evidence_rate"] = long["aspect_evidence_count"] / long["review_count"].replace(0, np.nan)
    long["enough_evidence"] = (
        (long["aspect_evidence_count"] >= MIN_EVIDENCE_COUNT)
        & (long["aspect_evidence_rate"] >= MIN_EVIDENCE_RATE)
    )
    long["is_check_candidate"] = (
        (long["aspect_score_mean"] <= CHECK_SCORE_THRESHOLD)
        & long["enough_evidence"]
    )
    long["is_strength"] = (
        (long["aspect_score_mean"] >= STRENGTH_SCORE_THRESHOLD)
        & long["enough_evidence"]
    )
    long["interpretation"] = np.select(
        [
            long["is_check_candidate"],
            long["is_strength"],
            ~long["enough_evidence"],
        ],
        [
            "점검 후보",
            "강점 가능",
            "판단 유보(evidence 부족)",
        ],
        default="중립/관찰",
    )

    wide_scores = long.pivot(index="store_name", columns="aspect", values="aspect_score_mean").reset_index()
    wide_evidence_count = long.pivot(index="store_name", columns="aspect", values="aspect_evidence_count").reset_index()
    wide_evidence_rate = long.pivot(index="store_name", columns="aspect", values="aspect_evidence_rate").reset_index()
    wide_enough = long.pivot(index="store_name", columns="aspect", values="enough_evidence").reset_index()

    summary = base.copy()
    for aspect in aspect_keys:
        summary[aspect] = wide_scores[aspect].values
        summary[f"{aspect}_evidence_count"] = wide_evidence_count[aspect].values
        summary[f"{aspect}_evidence_rate"] = wide_evidence_rate[aspect].values
        summary[f"{aspect}_enough_evidence"] = wide_enough[aspect].values.astype(bool)

    summary["aspect_gap"] = summary[aspect_keys].max(axis=1) - summary[aspect_keys].min(axis=1)
    return long, summary, aspect_keys


def choose_stores(long: pd.DataFrame, summary: pd.DataFrame, aspect_keys: list[str], max_stores: int = 12) -> pd.DataFrame:
    candidates = summary[summary["review_count"] >= MIN_REVIEWS].copy()
    if candidates.empty:
        candidates = summary.copy()

    check_pressure = (
        long[long["is_check_candidate"]]
        .groupby("store_name")["aspect_score_mean"]
        .min()
        .abs()
        .rename("check_pressure")
        .reset_index()
    )
    candidates = candidates.merge(check_pressure, on="store_name", how="left")
    candidates["check_pressure"] = candidates["check_pressure"].fillna(0)

    high_gap = candidates.sort_values(["aspect_gap", "review_count"], ascending=[False, False]).head(7)
    check_cases = candidates.sort_values(["check_pressure", "aspect_gap"], ascending=[False, False]).head(5)
    selected = pd.concat([high_gap, check_cases], ignore_index=True).drop_duplicates("store_name")
    if len(selected) < max_stores:
        filler = candidates.sort_values(["review_count", "aspect_gap"], ascending=[False, False])
        selected = pd.concat([selected, filler], ignore_index=True).drop_duplicates("store_name")

    selected = selected.head(max_stores).copy()
    selected["_sort"] = selected["aspect_gap"] + selected["check_pressure"]
    return selected.sort_values(["_sort", "review_count"], ascending=[False, False]).drop(columns="_sort")


def matrixes_for_selected(long: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_names = selected["store_name"].tolist()
    sub = long[long["store_name"].isin(selected_names)].copy()
    sub["store_name"] = pd.Categorical(sub["store_name"], categories=selected_names, ordered=True)
    scores = sub.pivot(index="store_name", columns="aspect_label", values="aspect_score_mean")
    evidence_rate = sub.pivot(index="store_name", columns="aspect_label", values="aspect_evidence_rate")
    enough = sub.pivot(index="store_name", columns="aspect_label", values="enough_evidence")
    ordered_cols = [ASPECT_LABELS[a] for a in ASPECT_LABELS if ASPECT_LABELS[a] in scores.columns]
    return scores[ordered_cols], evidence_rate[ordered_cols], enough[ordered_cols]


def text_color_for_value(value: float, enough: bool) -> str:
    if not enough:
        return "#6b7280"
    if value >= 1.35:
        return "white"
    if value <= -0.08:
        return "#9a3412"
    return "#1f2937"


def draw_heatmap(
    matrix: pd.DataFrame,
    evidence_rate: pd.DataFrame,
    enough: pd.DataFrame,
    selected: pd.DataFrame,
    output_path: Path,
    final: bool,
) -> None:
    set_korean_font()
    if final:
        fig = plt.figure(figsize=(18.8, 10.8), dpi=150)
        gs = fig.add_gridspec(1, 2, width_ratios=[5.55, 1.45], wspace=0.10)
        ax = fig.add_subplot(gs[0, 0])
        side = fig.add_subplot(gs[0, 1])
        side.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(17.2, 10.4), dpi=150)
        side = None

    cmap = LinearSegmentedColormap.from_list(
        "check_diverging",
        ["#c2410c", "#f4a261", "#f7f7f4", "#9bd5cf", "#008891", "#064e63"],
    )
    vmin = min(-0.3, float(np.nanmin(matrix.values)))
    vmax = max(2.7, float(np.nanmax(matrix.values)))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    alpha = np.where(enough.to_numpy(dtype=bool), 1.0, 0.45)
    im = ax.imshow(matrix.values, cmap=cmap, norm=norm, aspect="auto", alpha=alpha)

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, fontsize=13.0, fontweight="bold")
    ax.set_yticks(np.arange(matrix.shape[0]))
    n_lookup = selected.set_index("store_name")["review_count"].to_dict()
    ax.set_yticklabels([f"{name}  (n={int(n_lookup[name])})" for name in matrix.index], fontsize=11.6, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)

    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = float(matrix.iloc[y, x])
            ok = bool(enough.iloc[y, x])
            rate = float(evidence_rate.iloc[y, x])
            label = f"{value:.2f}"
            if not ok:
                label += "*"
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="#9ca3af",
                        hatch="///",
                        linewidth=0,
                        alpha=0.22,
                    )
                )
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=9.6,
                fontweight="bold",
                color=text_color_for_value(value, ok),
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.027 if final else 0.025, pad=0.015)
    cbar.ax.set_title("점수", fontsize=11.5, fontweight="bold", pad=8)
    cbar.ax.tick_params(labelsize=9.5)
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels(["낮음\n점검 후보", "중립", "높음\n강점"])

    title = "식당별 Aspect 프로필: 강점과 점검 후보" if final else "대표 식당별 Aspect 강점·점검 후보 히트맵"
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=24.5, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "낮은 점수는 evidence가 충분할 때만 점검 후보로 해석합니다.",
        ha="center",
        va="top",
        fontsize=13.0,
        color="#5f6368",
    )
    fig.text(
        0.5,
        0.035,
        "* 또는 해칭 셀 = evidence 부족으로 판단 유보. 낮은 aspect 점수는 곧바로 약점으로 단정하지 않습니다.",
        ha="center",
        va="bottom",
        fontsize=11.3,
        color="#666666",
    )

    if side is not None:
        card = FancyBboxPatch(
            (0.03, 0.10),
            0.92,
            0.80,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            linewidth=1.8,
            edgecolor="#d7e0e7",
            facecolor="#f8fbfd",
            transform=side.transAxes,
        )
        side.add_patch(card)
        side.text(0.10, 0.82, "읽는 법", fontsize=20, fontweight="bold", color="#073b4c", transform=side.transAxes)
        side.text(
            0.10,
            0.70,
            "진한 청록색\n= 강점 aspect",
            fontsize=14.5,
            fontweight="bold",
            color="#075985",
            transform=side.transAxes,
            linespacing=1.45,
        )
        side.text(
            0.10,
            0.52,
            "낮은 값 + 충분한 evidence\n= 점검 후보",
            fontsize=14,
            fontweight="bold",
            color="#9a3412",
            transform=side.transAxes,
            linespacing=1.45,
        )
        side.text(
            0.10,
            0.36,
            "* 해칭/흐린 셀\n= evidence 부족,\n  판단 유보",
            fontsize=13.3,
            fontweight="bold",
            color="#4b5563",
            transform=side.transAxes,
            linespacing=1.45,
        )
        side.plot([0.10, 0.86], [0.29, 0.29], color="#d7dee5", lw=1.5, transform=side.transAxes)
        side.text(0.10, 0.21, "운영자 관점", fontsize=16.5, fontweight="bold", color="#111827", transform=side.transAxes)
        side.text(
            0.10,
            0.115,
            "강점은 홍보 포인트로,\n점검 후보는 현장 확인\n우선순위로 활용",
            fontsize=12.7,
            color="#374151",
            transform=side.transAxes,
            linespacing=1.42,
        )

    if final:
        fig.subplots_adjust(left=0.135, right=0.975, top=0.865, bottom=0.095, wspace=0.10)
    else:
        fig.subplots_adjust(left=0.175, right=0.93, top=0.865, bottom=0.095)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pick_strength_and_check(long: pd.DataFrame, selected: pd.DataFrame, exclude_food: bool = False) -> pd.DataFrame:
    rows = []
    selected_names = selected["store_name"].tolist()
    aspects_allowed = [a for a in ASPECT_LABELS if not (exclude_food and a == "food")]
    for store in selected_names:
        sub = long[(long["store_name"] == store) & (long["aspect"].isin(aspects_allowed))].copy()
        strong_pool = sub[(sub["is_strength"])].sort_values("aspect_score_mean", ascending=False)
        if strong_pool.empty:
            strength = sub.sort_values("aspect_score_mean", ascending=False).iloc[0]
            strength_label = f"{strength['aspect_label']} {strength['aspect_score_mean']:+.2f}"
            strength_note = "강점 후보(evidence 확인 필요)"
        else:
            strength = strong_pool.iloc[0]
            strength_label = f"{strength['aspect_label']} {strength['aspect_score_mean']:+.2f}"
            strength_note = "강점"

        check_pool = sub[sub["is_check_candidate"]].sort_values("aspect_score_mean", ascending=True)
        if check_pool.empty:
            low = sub.sort_values("aspect_score_mean", ascending=True).iloc[0]
            if not bool(low["enough_evidence"]):
                check_label = "판단 유보"
                check_note = f"{low['aspect_label']} 낮지만 evidence 부족"
            else:
                check_label = "뚜렷한 점검 후보 없음"
                check_note = f"최저 {low['aspect_label']} {low['aspect_score_mean']:+.2f}"
            check_aspect = str(low["aspect_label"])
            check_score = float(low["aspect_score_mean"])
            check_rate = float(low["aspect_evidence_rate"])
            check_count = int(low["aspect_evidence_count"])
            is_check = False
        else:
            check = check_pool.iloc[0]
            check_label = f"{check['aspect_label']} {check['aspect_score_mean']:+.2f}"
            check_note = "점검 후보"
            check_aspect = str(check["aspect_label"])
            check_score = float(check["aspect_score_mean"])
            check_rate = float(check["aspect_evidence_rate"])
            check_count = int(check["aspect_evidence_count"])
            is_check = True

        review_count = int(selected.loc[selected["store_name"] == store, "review_count"].iloc[0])
        rows.append(
            {
                "store_name": store,
                "review_count": review_count,
                "strength_aspect": str(strength["aspect_label"]),
                "strength_score": float(strength["aspect_score_mean"]),
                "strength_evidence_count": int(strength["aspect_evidence_count"]),
                "strength_evidence_rate": float(strength["aspect_evidence_rate"]),
                "strength_label": strength_label,
                "strength_note": strength_note,
                "check_aspect": check_aspect,
                "check_score": check_score,
                "check_evidence_count": check_count,
                "check_evidence_rate": check_rate,
                "check_label": check_label,
                "check_note": check_note,
                "is_check_candidate": is_check,
                "mode": "음식 제외 운영 점검" if exclude_food else "전체 aspect",
            }
        )
    return pd.DataFrame(rows)


def draw_summary_table(summary: pd.DataFrame, output_path: Path, exclude_food: bool = False) -> None:
    set_korean_font()
    n = len(summary)
    fig, ax = plt.subplots(figsize=(16.4, 9.8), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 2.55)
    ax.axis("off")

    title = "식당별 운영 점검 포인트 요약" if exclude_food else "식당별 핵심 강점과 점검 후보 요약"
    subtitle = (
        "음식 aspect를 제외하고, 가격·서비스·분위기·웨이팅·재방문·위생 중심으로 계산"
        if exclude_food
        else "점검 후보는 낮은 점수와 충분한 evidence가 함께 있을 때만 표시했습니다."
    )
    ax.text(0.5, n + 2.08, title, ha="center", va="center", fontsize=25.5, fontweight="bold")
    ax.text(0.5, n + 1.56, subtitle, ha="center", va="center", fontsize=13.7, color="#777777")

    headers = ["식당", "상대 강점 aspect" if exclude_food else "강점 aspect", "점검 후보", "해석"]
    xs = [0.035, 0.335, 0.57, 0.805]
    widths = [0.265, 0.195, 0.195, 0.13]
    for x, w, h in zip(xs, widths, headers):
        ax.add_patch(Rectangle((x, n + 0.78), w, 0.50, facecolor="#eef2f6", edgecolor="white"))
        ax.text(x + w / 2, n + 1.03, h, ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111827")

    row_height = 0.84
    for i, row in summary.reset_index(drop=True).iterrows():
        y = n - i + 0.05
        bg = "#ffffff" if i % 2 == 0 else "#fafafa"
        ax.add_patch(Rectangle((0.025, y - 0.48), 0.95, row_height, facecolor=bg, edgecolor="#e5e7eb", linewidth=0.85))
        ax.text(0.045, y - 0.06, f"{row['store_name']}  (n={int(row['review_count'])})", ha="left", va="center", fontsize=12.3, fontweight="bold", color="#111827")

        ax.add_patch(FancyBboxPatch((0.335, y - 0.34), 0.195, 0.52, boxstyle="round,pad=0.018,rounding_size=0.025", fc="#e4f4f1", ec="#acdcd4"))
        ax.text(0.4325, y - 0.055, row["strength_label"], ha="center", va="center", fontsize=12.0, fontweight="bold", color="#006d77")
        ax.text(0.4325, y - 0.285, f"evidence {row['strength_evidence_rate']*100:.1f}%", ha="center", va="center", fontsize=9.8, color="#37716f")

        if row["is_check_candidate"]:
            fc, ec, color = "#fff1e8", "#f3c5a6", "#ba3a1d"
            interp = "우선 확인"
        else:
            fc, ec, color = "#f3f4f6", "#d1d5db", "#4b5563"
            if row["check_label"] == "판단 유보":
                interp = "근거 부족"
            else:
                interp = "판단 유보"
        ax.add_patch(FancyBboxPatch((0.57, y - 0.34), 0.195, 0.52, boxstyle="round,pad=0.018,rounding_size=0.025", fc=fc, ec=ec))
        ax.text(0.6675, y - 0.055, row["check_label"], ha="center", va="center", fontsize=11.7, fontweight="bold", color=color)
        ax.text(0.6675, y - 0.285, f"evidence {row['check_evidence_rate']*100:.1f}%", ha="center", va="center", fontsize=9.8, color="#6b7280")

        interp_fc = "#fff1e8" if interp == "우선 확인" else "#f3f4f6"
        interp_ec = "#f3c5a6" if interp == "우선 확인" else "#d1d5db"
        interp_color = "#ba3a1d" if interp == "우선 확인" else "#4b5563"
        ax.add_patch(FancyBboxPatch((0.805, y - 0.28), 0.13, 0.40, boxstyle="round,pad=0.015,rounding_size=0.025", fc=interp_fc, ec=interp_ec))
        ax.text(0.87, y - 0.08, interp, ha="center", va="center", fontsize=12.2, fontweight="bold", color=interp_color)

    ax.text(
        0.5,
        0.18,
        "낮은 점수는 바로 약점이 아니라, 리뷰에서 충분히 언급된 경우에만 점검 후보로 보았습니다.",
        ha="center",
        va="center",
        fontsize=12.2,
        color="#555555",
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.96])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    input_path, df, encoding, mapping = find_input_file()
    long, store_summary, aspect_keys = aspect_long_summary(df, mapping)
    selected = choose_stores(long, store_summary, aspect_keys, max_stores=12)
    matrix, evidence_rate, enough = matrixes_for_selected(long, selected)

    aspect_evidence_csv = OUTPUT_DIR / "summary_insight2_aspect_evidence.csv"
    strength_csv = OUTPUT_DIR / "summary_insight2_strength_check.csv"
    heatmap_readable = OUTPUT_DIR / "fig_insight2_aspect_heatmap_readable.png"
    heatmap_final = OUTPUT_DIR / "fig_insight2_aspect_heatmap_final.png"
    summary_readable_png = OUTPUT_DIR / "fig_insight2_strength_check_summary_readable.png"
    summary_no_food_png = OUTPUT_DIR / "fig_insight2_strength_check_summary_no_food.png"
    operation_final_png = OUTPUT_DIR / "fig_insight2_operation_check_summary_final.png"
    operation_final_csv = OUTPUT_DIR / "summary_insight2_operation_check_summary_final.csv"

    long_out = long[long["store_name"].isin(selected["store_name"])].copy()
    long_out = long_out.sort_values(["store_name", "aspect"])
    long_out.to_csv(aspect_evidence_csv, index=False, encoding="utf-8-sig")

    strength_all = pick_strength_and_check(long, selected, exclude_food=False)
    strength_no_food = pick_strength_and_check(long, selected, exclude_food=True)
    strength_out = pd.concat([strength_all, strength_no_food], ignore_index=True)
    strength_out.to_csv(strength_csv, index=False, encoding="utf-8-sig")
    strength_no_food.to_csv(operation_final_csv, index=False, encoding="utf-8-sig")

    draw_heatmap(matrix, evidence_rate, enough, selected, heatmap_readable, final=False)
    draw_heatmap(matrix, evidence_rate, enough, selected, heatmap_final, final=True)
    draw_summary_table(strength_all, summary_readable_png, exclude_food=False)
    draw_summary_table(strength_no_food, summary_no_food_png, exclude_food=True)
    draw_summary_table(strength_no_food, operation_final_png, exclude_food=True)

    print("\n[사용 데이터]")
    print(f"- 파일명: {input_path}")
    print(f"- 인코딩: {encoding}")
    print("\n[사용 컬럼]")
    print(f"- 식당명 컬럼: {mapping['store']}")
    if mapping.get("platform"):
        print(f"- 플랫폼 컬럼: {mapping['platform']}")
    if mapping.get("sentiment"):
        print(f"- 감성 점수 컬럼: {mapping['sentiment']}")
    for aspect in aspect_keys:
        print(f"- {ASPECT_LABELS[aspect]}: {mapping[f'aspect_{aspect}']}")

    print("\n[식당 선정 기준]")
    print(f"- 최소 리뷰 수 기준: High Trust 리뷰 {MIN_REVIEWS}개 이상")
    print("- aspect 차이가 큰 식당 + 점검 후보 압력이 큰 식당을 결합해 12개 선정")
    print("- 선정 식당:", ", ".join(selected["store_name"].tolist()))

    print("\n[evidence_count 계산 방식]")
    print("- 리뷰별 aspect_score가 0이 아니면 해당 aspect evidence가 탐지된 리뷰로 계산했습니다.")
    print(f"- 충분한 evidence 기준: evidence_count >= {MIN_EVIDENCE_COUNT} AND evidence_rate >= {MIN_EVIDENCE_RATE:.0%}")

    print("\n[점검 후보 판정 기준]")
    print(f"- aspect 평균 점수 <= {CHECK_SCORE_THRESHOLD:+.2f} AND 충분한 evidence")
    print("- 낮은 점수라도 evidence가 부족하면 점검 후보가 아니라 판단 유보로 표시했습니다.")

    print("\n[생성된 그래프/CSV]")
    for path in [
        heatmap_readable,
        heatmap_final,
        summary_readable_png,
        summary_no_food_png,
        operation_final_png,
        aspect_evidence_csv,
        strength_csv,
        operation_final_csv,
    ]:
        print(f"- {path.name}: {'저장 완료' if path.exists() else '저장 실패'}")

    print("\n[음식 aspect 제외 여부]")
    print("- fig_insight2_operation_check_summary_final.png: 음식 aspect 제외")
    print("\n[상대 강점 aspect 계산 기준]")
    print("- 음식을 제외한 가격·서비스·분위기·웨이팅·재방문·위생 중 evidence가 충분하고 점수가 가장 높은 aspect를 우선 선택했습니다.")
    print("- 충분한 evidence가 없으면 가장 높은 점수의 aspect를 강점 후보로 표시하되, evidence 비율을 함께 표기했습니다.")
    print("\n[관련 언급 부족 처리 기준]")
    print("- 점검 후보 조건을 만족하지 않고 evidence가 부족한 경우 해석 컬럼에 '근거 부족' 또는 '판단 유보'로 표시했습니다.")

    print("\n[발표 슬라이드 추천]")
    print("- 메인: fig_insight2_aspect_heatmap_final.png")
    print("- 보조: fig_insight2_strength_check_summary_no_food.png")

    print("\n[기존 시각화 대비 개선점]")
    print("- '개선 후보' 표현을 모두 '점검 후보'로 바꾸어 문제 확정처럼 보이지 않게 했습니다.")
    print("- 0 기준 diverging colormap을 적용해 낮은 값/중립/높은 값이 더 명확하게 보입니다.")
    print("- 식당별 n은 유지하고, aspect별 evidence_count/evidence_rate를 계산했습니다.")
    print("- evidence 부족 셀은 해칭과 별표로 판단 유보를 표시했습니다.")
    print("- 음식이 항상 강점으로 보이는 문제를 줄이기 위해 음식 제외 운영 점검용 요약 차트를 추가했습니다.")


if __name__ == "__main__":
    main()
