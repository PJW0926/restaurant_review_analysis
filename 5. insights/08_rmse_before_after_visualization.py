from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


OUTPUT_DIR = Path(__file__).resolve().parent

# Values from the project evaluation note.
BEFORE_RMSE = 1.0605
AFTER_RMSE = 0.8609


def configure_korean_font() -> None:
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        preferred = ["Malgun Gothic", "AppleGothic", "NanumGothic"]
        installed = {font.name for font in font_manager.fontManager.ttflist}
        selected = next((name for name in preferred if name in installed), None)
        if selected:
            plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def main() -> None:
    configure_korean_font()

    reduction = BEFORE_RMSE - AFTER_RMSE
    improvement_pct = reduction / BEFORE_RMSE * 100

    summary = pd.DataFrame(
        [
            {"version": "previous", "version_ko": "\uc774\uc804 \ubc84\uc804", "rmse": BEFORE_RMSE},
            {"version": "current", "version_ko": "\ud604\uc7ac \ubc84\uc804", "rmse": AFTER_RMSE},
        ]
    )
    summary["rmse_reduction_from_before"] = [0.0, reduction]
    summary["improvement_pct_from_before"] = [0.0, improvement_pct]
    summary_path = OUTPUT_DIR / "summary_sentiment_rmse_before_after.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.8, 6.0), dpi=180)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.14)
    x = [0, 1]
    values = [BEFORE_RMSE, AFTER_RMSE]
    colors = ["#A6A6A6", "#2E8B57"]

    bars = ax.bar(
        x,
        values,
        width=0.55,
        color=colors,
        edgecolor="none",
    )

    ax.set_title(
        "\uac10\uc131\ubcc4\uc810 \ubcf4\uc815 \uc804\u00b7\ud6c4 RMSE \ube44\uad50",
        fontsize=19,
        fontweight="bold",
        pad=16,
    )
    ax.set_ylabel("RMSE", fontsize=13, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["\uc774\uc804 \ubc84\uc804", "\ud604\uc7ac \ubc84\uc804"],
        fontsize=13,
    )
    ax.set_ylim(0, 1.2)
    ax.set_xlim(-0.35, 1.35)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="-", alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#222222",
        )

    fig_path = OUTPUT_DIR / "fig_sentiment_rmse_before_after_clean.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    # Also overwrite the simple canonical filename so the latest version is easy to find.
    canonical_path = OUTPUT_DIR / "fig_sentiment_rmse_before_after.png"
    fig.savefig(canonical_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {fig_path}")
    print(f"saved: {canonical_path}")
    print(f"saved: {summary_path}")
    print(f"before_rmse={BEFORE_RMSE:.4f}")
    print(f"after_rmse={AFTER_RMSE:.4f}")
    print(f"abs_reduction={reduction:.4f}")
    print(f"improvement_pct={improvement_pct:.1f}%")


if __name__ == "__main__":
    main()
