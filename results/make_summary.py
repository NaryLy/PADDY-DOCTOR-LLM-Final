"""Read the three *_test_eval.json files and produce a markdown summary table."""
import json
import os

RESULTS_DIR = os.path.dirname(__file__)
MODES = ["baseline", "frozen", "finetuned"]
NAMES = {
    "baseline": "Baseline (CNN from scratch)",
    "frozen": "Ablation A -- frozen ResNet18 backbone",
    "finetuned": "Ablation B / deployed -- fine-tuned ResNet18",
}


def main():
    rows = []
    for mode in MODES:
        path = os.path.join(RESULTS_DIR, f"{mode}_test_eval.json")
        if not os.path.exists(path):
            print(f"(missing: {path} -- run training first)")
            continue
        with open(path) as f:
            data = json.load(f)
        rows.append((mode, data))

    lines = []
    lines.append("| Model | Trainable params | Best val acc | Test accuracy | Macro F1 |")
    lines.append("|---|---|---|---|---|")
    for mode, data in rows:
        lines.append(
            f"| {NAMES[mode]} | {data['trainable_params']:,} | "
            f"{data['best_val_acc']:.4f} | {data['test_acc']:.4f} | {data['macro_f1']:.4f} |"
        )

    print("\n".join(lines))

    # Per-class table for the deployed (finetuned) model
    finetuned = next((d for m, d in rows if m == "finetuned"), None)
    if finetuned:
        print("\n\n### Per-class performance (deployed model)\n")
        print("| Class | Precision | Recall | F1 | Support |")
        print("|---|---|---|---|---|")
        for cls, stats in finetuned["per_class"].items():
            print(f"| {cls} | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1']:.3f} | {stats['support']} |")

    out_path = os.path.join(RESULTS_DIR, "summary_table.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        if finetuned:
            f.write("\n\n### Per-class performance (deployed model)\n\n")
            f.write("| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|\n")
            for cls, stats in finetuned["per_class"].items():
                f.write(f"| {cls} | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1']:.3f} | {stats['support']} |\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
