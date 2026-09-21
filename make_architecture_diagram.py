"""Generate a simple system architecture diagram for the report."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6.5)
ax.axis("off")

GREEN_DARK = "#1b4332"
GREEN = "#2d6a4f"
GREEN_LIGHT = "#d8f3dc"
AMBER = "#e09f3e"
GREY = "#5a6b62"


def box(x, y, w, h, text, fc=GREEN_LIGHT, ec=GREEN_DARK, tc=GREEN_DARK, fontsize=10.5, bold=True):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=tc, fontweight="bold" if bold else "normal", wrap=True)
    return (x, y, w, h)


def arrow(b1, b2, label="", side1="right", side2="left", rad=0.0, label_dy=0.18, label_dx=0.0):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts = {
        "right": (x1 + w1, y1 + h1 / 2),
        "left": (x1, y1 + h1 / 2),
        "top": (x1 + w1 / 2, y1 + h1),
        "bottom": (x1 + w1 / 2, y1),
    }
    pts2 = {
        "right": (x2 + w2, y2 + h2 / 2),
        "left": (x2, y2 + h2 / 2),
        "top": (x2 + w2 / 2, y2 + h2),
        "bottom": (x2 + w2 / 2, y2),
    }
    p1 = pts[side1]
    p2 = pts2[side2]
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
                         color=GREY, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        va = "bottom" if label_dy >= 0 else "top"
        ax.text(mx + label_dx, my + label_dy, label, ha="center", va=va, fontsize=8.2, color=GREY)


# Row 1: user devices
user = box(0.3, 5.2, 2.0, 0.9, "Farmer /\nExtension officer\n(phone or laptop browser)", fc="#fff3d6", ec=AMBER, tc="#6b5300")

# Frontend
frontend = box(3.0, 5.2, 2.6, 0.9, "Frontend\nindex.html (vanilla JS)\nEN / KM toggle", fc=GREEN_LIGHT)

# Backend
backend = box(6.3, 5.2, 2.6, 0.9, "Backend API\nFastAPI (app.py)\n/api/predict /api/history", fc=GREEN_LIGHT)

# Model
model = box(6.3, 3.6, 2.6, 0.9, "AI Model\nResNet18 (ImageNet-pretrained,\nfine-tuned) -- 11.18M params", fc="white", ec=GREEN)

# Treatment DB
treat = box(3.0, 3.6, 2.6, 0.9, "Treatment DB\ntreatment_db.py\nEN/KM advice, 10 classes", fc="white", ec=GREEN)

# History store
hist = box(9.3, 3.6, 1.5, 0.9, "History\nSQLite\n(history.db)", fc="white", ec=GREEN)

# Dataset / training (offline, dashed)
dataset = box(0.3, 1.6, 2.6, 0.9, "Dataset\nPaddy Doctor\n10,407 images / 10 classes", fc="#eef4ef", ec=GREY, tc=GREY, bold=False)
training = box(3.4, 1.6, 2.6, 0.9, "Training pipeline\ntrain.py (baseline / frozen /\nfinetuned)", fc="#eef4ef", ec=GREY, tc=GREY, bold=False)
ckpt = box(6.5, 1.6, 2.4, 0.9, "Checkpoint\nmodels/finetuned_best.pt", fc="#eef4ef", ec=GREY, tc=GREY, bold=False)

arrow(user, frontend, "photo upload", rad=0.35, label_dy=0.32)
arrow(frontend, user, "diagnosis + advice", side1="left", side2="right", rad=0.35, label_dy=-0.32)
arrow(frontend, backend, "POST /api/predict (image)", rad=0.35, label_dy=0.32)
arrow(backend, frontend, "JSON: label, confidence,\ntop-3, treatment", side1="left", side2="right", rad=0.35, label_dy=-0.65, label_dx=0.9)
arrow(backend, model, "preprocess +\nforward pass", side1="bottom", side2="top")
arrow(backend, treat, "lookup by\npredicted label", side1="left", side2="right", label_dx=-1.1, label_dy=-0.1)
arrow(backend, hist, "log prediction", side1="right", side2="left")

arrow(dataset, training, "", side1="right", side2="left")
arrow(training, ckpt, "best val-acc\ncheckpoint", side1="right", side2="left")
arrow(ckpt, model, "loaded at\nserver startup", side1="top", side2="bottom", rad=0.15)

ax.text(0.3, 2.85, "Offline (once): dataset → training → checkpoint", fontsize=9, color=GREY, style="italic")
ax.text(0.3, 6.35, "Online (per request): user ↔ frontend ↔ backend ↔ model / treatment DB / history", fontsize=9, color=GREY, style="italic")

plt.tight_layout()
plt.savefig("/home/claude/rice-disease-app/architecture_diagram.png", dpi=170, bbox_inches="tight")
print("saved")
