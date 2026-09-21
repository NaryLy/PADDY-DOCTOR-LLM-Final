"""
Resize the raw Paddy Doctor training images (from the Kaggle download) down to 160x160 JPEGs,
which is what make_splits.py / dataset.py expect under data/paddy_resized_160/<class>/.

Usage:
    python3 resize_dataset.py /path/to/paddy-disease-classification/train_images

(that's the folder containing bacterial_leaf_blight/, blast/, normal/, etc. -- the same
folder this project's README points you to inside your unzipped Kaggle download.)
"""
import os
import sys
from PIL import Image

SIZE = (160, 160)
QUALITY = 85
DST_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "paddy_resized_160")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    src_root = sys.argv[1]
    if not os.path.isdir(src_root):
        print(f"Not a directory: {src_root}")
        sys.exit(1)

    os.makedirs(DST_ROOT, exist_ok=True)
    classes = sorted(d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d)))
    if not classes:
        print(f"No class subfolders found under {src_root}")
        sys.exit(1)

    total = 0
    for cls in classes:
        src_dir = os.path.join(src_root, cls)
        dst_dir = os.path.join(DST_ROOT, cls)
        os.makedirs(dst_dir, exist_ok=True)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for fname in files:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            try:
                im = Image.open(src_path).convert("RGB")
                im = im.resize(SIZE, Image.BILINEAR)
                im.save(dst_path, "JPEG", quality=QUALITY)
                total += 1
            except Exception as e:
                print(f"  failed on {src_path}: {e}")
        print(f"{cls}: {len(files)} images")

    print(f"\nDone. Resized {total} images into {DST_ROOT}")
    print("Next: python3 make_splits.py")


if __name__ == "__main__":
    main()
