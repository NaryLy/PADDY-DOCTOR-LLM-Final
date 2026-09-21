"""Shared Dataset class + label map for the rice disease pipeline."""
import csv
import os

from PIL import Image
from torch.utils.data import Dataset

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
IMG_ROOT = os.path.join(DATA_ROOT, "paddy_resized_160")
MANIFEST = os.path.join(DATA_ROOT, "manifest.csv")
 

def load_manifest():
    rows = []
    with open(MANIFEST, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def build_label_map(rows):
    labels = sorted(set(r["label"] for r in rows))
    return {label: i for i, label in enumerate(labels)}


class RiceDiseaseDataset(Dataset):
    def __init__(self, split, label_map, transform=None):
        rows = load_manifest()
        self.rows = [r for r in rows if r["split"] == split]
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = os.path.join(IMG_ROOT, row["path"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.label_map[row["label"]]
        return image, label

    def class_counts(self):
        counts = {}
        for r in self.rows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        return counts
