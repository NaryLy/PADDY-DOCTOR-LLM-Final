"""Generate the comparison bar chart and confusion matrix heatmap for the report."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.dirname(__file__)

# Validated categorical palette (fixed order) and sequential blue ramp, from the dataviz skill.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"

MODES = ["baseline", "frozen", "finetuned"]
LABELS = ["Baseline\n(CNN from scratch)", "Ablation A\n(frozen backbone)", "Ablation B\n(fine-tuned, deployed)"]
COLORS = [BLUE, ORANGE, AQUA]


def load(mode):
    with open(os.path.join(RESULTS_DIR, f"{mode}_test_eval.json")) as f:
        return json.load(f)


def bar_chart():
    data = [load(m) for m in MODES]
    accs = [d["test_acc"] * 100 for d in data]
    f1s = [d["macro_f1"] * 100 for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, values, title in zip(axes, [accs, f1s], ["Test accuracy", "Macro F1"]):
        bars = ax.bar(LABELS, values, color=COLORS, width=0.6)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center",
                    fontsize=10.5, color=TEXT_PRIMARY, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=12, color=TEXT_PRIMARY, fontweight="bold", pad=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", labelsize=9, colors=TEXT_SECONDARY)
        ax.tick_params(axis="y", labelsize=9, colors=TEXT_SECONDARY)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_yticks(range(0, 101, 20))

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "..", "comparison_chart.png")
    plt.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


def confusion_matrix():
    d = load("finetuned")
    names = d["class_names"]
    cm = np.array(d["confusion_matrix"], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)  # row-normalized (recall view)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_yticklabels(names, fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_xlabel("Predicted label", fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylabel("True label", fontsize=10, color=TEXT_PRIMARY)
    ax.set_title("Confusion matrix -- deployed model (test set, row-normalized)",
                 fontsize=11.5, color=TEXT_PRIMARY, fontweight="bold", pad=12)

    for i in range(len(names)):
        for j in range(len(names)):
            val = cm_norm[i, j]
            if val < 0.01:
                continue
            color = "white" if val > 0.55 else TEXT_PRIMARY
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    cbar.set_label("Share of true-class samples", fontsize=9, color=TEXT_SECONDARY)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "..", "confusion_matrix.png")
    plt.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    bar_chart()
    confusion_matrix()
