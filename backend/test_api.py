"""
Minimal smoke tests for the running API. Not a full test suite -- just enough to
catch obvious regressions before a demo. Requires the server to already be running
(`uvicorn app:app`) and at least one image per class under
../data/paddy_resized_160/<class>/.

Usage: python3 test_api.py [base_url]
"""
import glob
import os
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "paddy_resized_160")


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    return cond


def main():
    ok = True

    r = requests.get(f"{BASE}/api/health", timeout=10)
    ok &= check("health endpoint returns 200", r.status_code == 200)
    body = r.json()
    ok &= check("model is loaded", body.get("model_loaded") is True)
    ok &= check("10 classes reported", len(body.get("classes", [])) == 10)

    # frontend served
    r = requests.get(f"{BASE}/", timeout=10)
    ok &= check("frontend root returns 200", r.status_code == 200)
    ok &= check("frontend root looks like html", "<html" in r.text.lower())

    # predict on one real image per class -- just check the endpoint responds sanely,
    # not that every prediction is correct (that's what results/*_test_eval.json is for).
    class_dirs = sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)))
    correct = 0
    for cls in class_dirs:
        files = glob.glob(os.path.join(DATA_ROOT, cls, "*.jpg"))
        if not files:
            continue
        with open(files[0], "rb") as f:
            r = requests.post(f"{BASE}/api/predict", files={"file": (os.path.basename(files[0]), f, "image/jpeg")}, timeout=30)
        ok &= check(f"predict responds 200 for class={cls}", r.status_code == 200)
        data = r.json()
        ok &= check(f"  has predicted_label + treatment for {cls}", "predicted_label" in data and data.get("treatment") is not None)
        if data.get("predicted_label") == cls:
            correct += 1

    print(f"\nSanity check: {correct}/{len(class_dirs)} single-sample predictions matched their folder's class label.")

    # bad input handling
    r = requests.post(f"{BASE}/api/predict", files={"file": ("not_an_image.txt", b"hello world", "text/plain")}, timeout=10)
    ok &= check("garbage upload returns 4xx (not 500)", 400 <= r.status_code < 500)

    # history
    r = requests.get(f"{BASE}/api/history", timeout=10)
    ok &= check("history endpoint returns 200", r.status_code == 200)
    ok &= check("history is a list", isinstance(r.json(), list))

    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
