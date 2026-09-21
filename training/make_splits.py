"""
Build a stratified train/val/test manifest for the resized Paddy Doctor dataset.

Input:  data/paddy_resized_160/<class>/<image>.jpg
Output: data/manifest.csv with columns: path,label,split
"""
import csv
import os
import random

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "paddy_resized_160")
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "manifest.csv")

SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining ~0.15 goes to test

random.seed(SEED)


def main():
    classes = sorted(
        d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))
    )
    print(f"Found {len(classes)} classes: {classes}")

    rows = []
    for label in classes:
        class_dir = os.path.join(DATA_ROOT, label)
        files = sorted(os.listdir(class_dir))
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        for i, fname in enumerate(files):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            rel_path = os.path.join(label, fname)
            rows.append((rel_path, label, split))

        print(f"  {label}: {n} total -> train {n_train}, val {n_val}, test {n - n_train - n_val}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "split"])
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
    split_counts = {}
    for _, _, s in rows:
        split_counts[s] = split_counts.get(s, 0) + 1
    print("Split totals:", split_counts)


if __name__ == "__main__":
    main()
