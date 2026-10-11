"""สร้าง infographic ประกอบตัวอย่างโจทย์คำนวณ → public/images/calc/*.svg
รัน: python scripts/make_calc_figures.py  (ต้องมีฟอนต์ Prompt ในเครื่อง)
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "public" / "images" / "calc"
OUT.mkdir(parents=True, exist_ok=True)

# พาเลตอ้างอิง (ผ่านการตรวจสีแล้ว) + โทนข้อความ
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE_T, ORANGE_T, AQUA_T, YELLOW_T = "#dbe8f8", "#fce3d8", "#d3f1e6", "#fbecc8"

plt.rcParams.update({
    "font.family": ["Prompt", "DejaVu Sans"],  # DejaVu ใช้แทนสัญลักษณ์กรีก/คณิตศาสตร์ที่ Prompt ไม่มี
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "svg.fonttype": "path",
    "lines.linewidth": 2,
})


PREVIEW = os.environ.get("PREVIEW_DIR")  # ตั้งค่าเพื่อบันทึก PNG ไว้ตรวจดู


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.15)
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / f"{name}.png", dpi=80, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("saved", name)


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def box(ax, x, y, w, h, text, fill, edge, size=11, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fill, edgecolor=edge, linewidth=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight, color=color)


def arrow(ax, p, q, color=MUTED, style="-|>", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, linewidth=lw, linestyle=ls))


# ---------------------------------------------------------------- 6: Confusion matrix
def confusion_metrics():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    x0, y0, cw, ch = 2.6, 1.2, 2.6, 1.9
    ax.text(x0 + cw, y0 + 2 * ch + 0.75, "สิ่งที่โมเดลทำนาย", ha="center", fontsize=12, fontweight="bold")
    ax.text(x0 + cw / 2, y0 + 2 * ch + 0.25, "ทำนาย + (ป่วย)", ha="center", color=INK2)
    ax.text(x0 + 1.5 * cw, y0 + 2 * ch + 0.25, "ทำนาย − (ปกติ)", ha="center", color=INK2)
    ax.text(x0 - 0.3, y0 + 1.5 * ch, "จริง +\n(ป่วย 60)", ha="right", va="center", color=INK2)
    ax.text(x0 - 0.3, y0 + 0.5 * ch, "จริง −\n(ปกติ 140)", ha="right", va="center", color=INK2)
    cells = [((0, 1), "TP = 52", "ทายถูก", BLUE_T, BLUE), ((1, 1), "FN = 8", "ป่วยแต่ทายว่าปกติ", ORANGE_T, ORANGE),
             ((0, 0), "FP = 18", "ปกติแต่ทายว่าป่วย", ORANGE_T, ORANGE), ((1, 0), "TN = 122", "ทายถูก", BLUE_T, BLUE)]
    for (c, r), big, small, fill, edge in cells:
        x, y = x0 + c * cw, y0 + r * ch
        ax.add_patch(Rectangle((x + 0.04, y + 0.04), cw - 0.08, ch - 0.08, facecolor=fill, edgecolor=edge, linewidth=1.5))
        ax.text(x + cw / 2, y + ch / 2 + 0.25, big, ha="center", va="center", fontsize=17, fontweight="bold")
        ax.text(x + cw / 2, y + ch / 2 - 0.4, small, ha="center", va="center", fontsize=10, color=INK2)
    # วงเล็บแสดงตัวหาร
    ax.add_patch(Rectangle((x0 - 0.02, y0 + ch - 0.02), 2 * cw + 0.04, ch + 0.04, fill=False, edgecolor=AQUA, linewidth=2.5, linestyle="--"))
    ax.add_patch(Rectangle((x0 - 0.1, y0 - 0.1), cw + 0.2, 2 * ch + 0.2, fill=False, edgecolor=VIOLET, linewidth=2.5, linestyle=":"))
    tx = x0 + 2 * cw + 0.5
    rows = [
        (AQUA, "Recall = TP/(TP+FN)", "แถวป่วยจริง (กรอบเขียว)", "52/60 = 0.8667"),
        (VIOLET, "Precision = TP/(TP+FP)", "คอลัมน์ทำนายว่าป่วย (กรอบม่วง)", "52/70 = 0.7429"),
        (INK2, "Specificity = TN/(TN+FP)", "แถวปกติจริง", "122/140 = 0.8714"),
        (BLUE, "Accuracy = (TP+TN)/ทั้งหมด", "ช่องสีฟ้า = ทายถูก", "174/200 = 0.8700"),
        (INK2, "F1 = 2TP/(2TP+FP+FN)", "ค่าเฉลี่ยฮาร์มอนิกของ P และ R", "104/130 = 0.8000"),
    ]
    for i, (c, f, note, val) in enumerate(rows):
        y = 5.6 - i * 1.05
        ax.add_patch(Rectangle((tx, y - 0.05), 0.12, 0.62, facecolor=c, edgecolor="none"))
        ax.text(tx + 0.3, y + 0.38, f, fontsize=11.5, fontweight="bold")
        ax.text(tx + 0.3, y - 0.02, f"{note}  →  {val}", fontsize=10, color=INK2)
    save(fig, "q6-confusion-metrics")


def spam_tree():
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    box(ax, 4.6, 3.9, 2.8, 0.85, "อีเมลทั้งหมด 400 ฉบับ", "#f1f0ec", MUTED, 12, "bold")
    box(ax, 1.4, 2.3, 3.2, 0.85, "อีเมลขยะจริง 80 (คลาสบวก)", ORANGE_T, ORANGE, 11, "bold")
    box(ax, 7.4, 2.3, 3.2, 0.85, "อีเมลปกติจริง 320 (คลาสลบ)", BLUE_T, BLUE, 11, "bold")
    arrow(ax, (5.6, 3.9), (3.4, 3.15)); arrow(ax, (6.4, 3.9), (8.6, 3.15))
    leaves = [(0.05, "ระบบจับได้", "TP = 68", "โจทย์ให้มา", BLUE), (2.95, "หลุดรอด", "FN = 80 − 68 = 12", "คำนวณจากผลรวมแถว", ORANGE),
              (6.15, "ถูกกรองผิด", "FP = 16", "โจทย์ให้มา", ORANGE), (9.05, "ผ่านเข้ากล่องจดหมาย", "TN = 320 − 16 = 304", "คำนวณจากผลรวมแถว", BLUE)]
    for x, label, val, note, c in leaves:
        box(ax, x, 0.35, 2.7, 1.25, "", "#ffffff", c)
        ax.text(x + 1.35, 1.3, label, ha="center", fontsize=10, color=INK2)
        ax.text(x + 1.35, 0.9, val, ha="center", fontsize=11.5, fontweight="bold")
        ax.text(x + 1.35, 0.52, note, ha="center", fontsize=9, color=MUTED)
    arrow(ax, (2.4, 2.3), (1.6, 1.6)); arrow(ax, (3.6, 2.3), (4.4, 1.6))
    arrow(ax, (8.4, 2.3), (7.6, 1.6)); arrow(ax, (9.6, 2.3), (10.4, 1.6))
    save(fig, "q6-spam-tree")


# ---------------------------------------------------------------- 7: Regression
def linear_prediction():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.linspace(0, 4.2, 50)
    a.plot(x, 5 + 2 * x, color=BLUE)
    a.scatter([0], [5], s=60, color=AQUA, zorder=3)
    a.annotate("θ₀ = 5 (จุดตัดแกน)", (0, 5), (0.35, 3.3), color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
    a.plot([1, 2, 2], [7, 7, 9], color=YELLOW, linewidth=2)
    a.text(1.5, 6.45, "Δx = 1", ha="center", fontsize=10, color=INK2)
    a.text(2.1, 8, "Δŷ = 2 = θ₁", va="center", fontsize=10, color=INK2)
    a.scatter([3], [11], s=80, color=ORANGE, zorder=3)
    a.plot([3, 3], [0, 11], color=MUTED, linestyle="--", linewidth=1)
    a.annotate("x = 3 → ŷ = 5 + 2(3) = 11", (3, 11), (0.15, 13.2), fontweight="bold", arrowprops=dict(arrowstyle="-", color=MUTED))
    a.set_xlim(0, 4.2); a.set_ylim(0, 14.5); a.set_xlabel("x"); a.set_ylabel("ŷ")
    a.set_title("(ข) ŷ = 5 + 2x", loc="left", fontweight="bold"); ygrid(a)
    # waterfall ของแบบจำลองพหุคูณ
    steps = [("θ₀", 10, 0, BLUE), ("+2.5 × x₁ (4)", 10, 10, AQUA), ("−1.5 × x₂ (2)", -3, 20, ORANGE)]
    for i, (lab, v, base, c) in enumerate(steps):
        lo = base + min(v, 0)
        b.bar(i, abs(v), bottom=lo, color=c, width=0.6)
        b.text(i, base + v + (0.5 if v > 0 else -1.2), f"{v:+g}" if i else "10", ha="center", fontweight="bold")
    b.bar(3, 17, color=VIOLET, width=0.6)
    b.text(3, 17.5, "ŷ = 17", ha="center", fontweight="bold")
    for i in range(3):
        top = [10, 20, 17][i]
        b.plot([i + 0.3, i + 0.7], [top, top], color=MUTED, linewidth=1, linestyle=":")
    b.set_xticks(range(4), [s[0] for s in steps] + ["ผลรวม"], fontsize=9.5)
    b.set_ylim(0, 25.5); b.set_title("(ก) ŷ = 10 + 2.5x₁ − 1.5x₂", loc="left", fontweight="bold"); ygrid(b)
    b.text(1.5, 23.6, "θ₂ ติดลบ: x₂ เพิ่ม 1 → ŷ ลด 1.5", ha="center", fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, "q7-linear-prediction")


def error_metrics():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.3))
    idx = np.array([1, 2, 3]); y = np.array([4, 6, 8]); yh = np.array([3, 6, 11])
    a.scatter(idx, y, s=70, color=BLUE, zorder=3, label="ค่าจริง y")
    a.scatter(idx, yh, s=70, color=ORANGE, zorder=3, marker="D", label="ค่าทำนาย ŷ")
    for i, (t, p) in enumerate(zip(y, yh)):
        e = t - p
        if e:
            lo = min(t, p)
            a.add_patch(Rectangle((idx[i], lo), abs(e) * 0.12, abs(e), facecolor=YELLOW_T, edgecolor=YELLOW, linewidth=1.2))
            a.plot([idx[i], idx[i]], [t, p], color=YELLOW, linewidth=2)
            a.text(idx[i] + abs(e) * 0.12 + 0.05, lo + abs(e) / 2, f"e = {e:+d}\ne² = {e*e}", va="center", fontsize=10)
        else:
            a.text(idx[i] + 0.08, t, "e = 0", va="center", fontsize=10)
    a.set_xticks(idx, [f"ข้อมูล {i}" for i in idx]); a.set_xlim(0.6, 3.9); a.set_ylim(2, 12.5)
    a.legend(frameon=False, loc="upper left")
    a.set_title("(ข) MSE = (1 + 0 + 9)/3 = 3.3333,  RMSE = 1.8257", loc="left", fontweight="bold", fontsize=11); ygrid(a)
    e = np.array([-1, 0, 2, -2]); w = 0.36; pos = np.arange(4)
    b.bar(pos - w / 2 - 0.02, np.abs(e), width=w, color=AQUA, label="|e| (ใช้ใน MAE)")
    b.bar(pos + w / 2 + 0.02, e ** 2, width=w, color=VIOLET, label="e² (ใช้ใน MSE/RMSE)")
    for p, v1, v2 in zip(pos, np.abs(e), e ** 2):
        b.text(p - w / 2 - 0.02, v1 + 0.1, f"{v1}", ha="center", fontsize=10)
        b.text(p + w / 2 + 0.02, v2 + 0.1, f"{v2}", ha="center", fontsize=10)
    b.set_xticks(pos, [f"e = {v:+d}" if v else "e = 0" for v in e]); b.set_ylim(0, 5.2)
    b.legend(frameon=False, loc="upper left")
    b.set_title("(ก) MAE = 5/4 = 1.25  vs  RMSE = √(9/4) = 1.5", loc="left", fontweight="bold", fontsize=11); ygrid(b)
    b.text(2.5, 4.6, "e = ±2 ถูกขยายเป็น 4\nRMSE จึงไวต่อ error ใหญ่", ha="center", fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, "q7-error-metrics")


def sigmoid():
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    z = np.linspace(-6, 6, 300); s = 1 / (1 + np.exp(-z))
    ax.plot(z, s, color=BLUE)
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.axvline(0, color=MUTED, linestyle=":", linewidth=1)
    ax.text(-5.9, 0.53, "threshold 0.5", color=INK2, fontsize=10)
    pts = [(-2, ORANGE, "σ(−2) = 0.1192 → คลาส 0"), (1, AQUA, "σ(1) = 0.7311 → คลาส 1")]
    for zz, c, lab in pts:
        v = 1 / (1 + np.exp(-zz))
        ax.scatter([zz], [v], s=80, color=c, zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(lab, (zz, v), (zz + (0.6 if zz > 0 else -3.9), v + (0.12 if zz > 0 else 0.18)),
                    fontweight="bold", arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("z = θᵀx"); ax.set_ylabel("ความน่าจะเป็นคลาส 1"); ax.set_ylim(-0.03, 1.05); ygrid(ax)
    top = ax.secondary_xaxis("top", functions=(lambda zz: (zz + 4) / 0.05, lambda x: 0.05 * x - 4))
    top.set_xlabel("ระดับน้ำตาล x ของโจทย์ (ก): z = −4 + 0.05x", color=INK2)
    ax.annotate("z = 0 ⇔ x = 80 = decision boundary", (0, 0.5), (0.5, 0.25), fontsize=10, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.annotate("(ก) x = 100 → z = 1", (1, 0.7311), (2.2, 0.55), fontsize=10, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    save(fig, "q7-sigmoid")


# ---------------------------------------------------------------- 8: Tree & ensembles
def gini():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")

    def node(x, y, a_, b_, title, g):
        tot = a_ + b_; w = 3.0
        ax.add_patch(Rectangle((x, y), w * a_ / tot - 0.02, 0.55, facecolor=BLUE, edgecolor="white"))
        ax.add_patch(Rectangle((x + w * a_ / tot + 0.02, y), w * b_ / tot - 0.02, 0.55, facecolor=ORANGE, edgecolor="white"))
        ax.text(x + w * a_ / tot / 2, y + 0.27, f"A {a_}", ha="center", va="center", color="white", fontweight="bold")
        if b_:
            ax.text(x + w * a_ / tot + w * b_ / tot / 2, y + 0.27, f"B {b_}", ha="center", va="center", color="white", fontweight="bold")
        ax.text(x + w / 2, y + 0.85, title, ha="center", fontsize=10.5, color=INK2)
        ax.text(x + w / 2, y - 0.38, g, ha="center", fontsize=10.5, fontweight="bold")

    ax.text(0.1, 5.05, "(ก) Gini ก่อนและหลังแบ่งโหนด", fontweight="bold", fontsize=12)
    node(2.2, 3.4, 5, 5, "โหนดแม่ (10 ตัว)", "1 − (0.5² + 0.5²) = 0.50")
    node(0.4, 0.9, 4, 1, "โหนดซ้าย (5 ตัว)", "1 − (0.8² + 0.2²) = 0.32")
    node(4.0, 0.9, 1, 4, "โหนดขวา (5 ตัว)", "1 − (0.2² + 0.8²) = 0.32")
    arrow(ax, (3.2, 3.0), (1.9, 1.95)); arrow(ax, (4.2, 3.0), (5.5, 1.95))
    ax.text(3.7, 0.05, "ถ่วงน้ำหนัก = 5/10(0.32) + 5/10(0.32) = 0.32   (ลดลง 0.18)", ha="center", fontsize=10.5, color=INK2)
    ax.plot([7.6, 7.6], [0.3, 5.1], color=GRID, linewidth=1)
    ax.text(8.0, 5.05, "(ข) Gini ของโหนดเดียว", fontweight="bold", fontsize=12)
    node(8.3, 3.4, 6, 4, "A 6 ตัว, B 4 ตัว", "1 − (0.6² + 0.4²) = 0.48")
    ax.text(9.8, 1.6, "Gini = 0 → บริสุทธิ์ (คลาสเดียว)\nGini = 0.5 → ปนครึ่งต่อครึ่ง (2 คลาส)", ha="center", fontsize=10, color=INK2)
    save(fig, "q8-gini")


def voting():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [5, 3]}, sharey=True)
    cases = [("(ก) 5 โมเดล", [0.8, 0.45, 0.45, 0.45, 0.9]), ("(ข) 3 โมเดล", [0.2, 0.6, 0.6])]
    for ax, (title, p) in zip(axes, cases):
        n = len(p); x = np.arange(n)
        colors = [BLUE if v >= 0.5 else ORANGE for v in p]
        ax.bar(x, p, color=colors, width=0.6)
        for i, v in enumerate(p):
            ax.text(i, v + 0.02, f"{v}", ha="center", fontsize=10)
            ax.text(i, 0.03, "โหวต 1" if v >= 0.5 else "โหวต 0", ha="center", fontsize=9, color="white", fontweight="bold")
        m = np.mean(p)
        ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
        ax.axhline(m, color=VIOLET, linewidth=2)
        lx = int(np.argmin(p))  # วางป้ายเหนือแท่งที่เตี้ยที่สุด ไม่ให้ทับแท่ง
        ax.text(lx, m + (0.03 if p[lx] < m - 0.08 else -0.06), f"ค่าเฉลี่ย {m:.4g}", ha="center", color=VIOLET, fontweight="bold", fontsize=10)
        ones = sum(v >= 0.5 for v in p)
        hard = 1 if ones > n / 2 else 0
        ax.set_title(f"{title}\nHard: {ones} ต่อ {n-ones} → คลาส {hard} · Soft: {m:.4g} → คลาส {int(m >= 0.5)}",
                     loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xticks(x, [f"M{i+1}" for i in x]); ax.set_ylim(0, 1.05); ygrid(ax)
    axes[0].set_ylabel("P(คลาส 1)")
    fig.tight_layout()
    save(fig, "q8-voting")


# ---------------------------------------------------------------- 9: MNIST & California
def dense_mnist():
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.2); ax.axis("off")
    rng = np.random.default_rng(3)
    img = np.zeros((28, 28)); img[6:22, 12:16] = 1; img[6:9, 9:16] = 1
    img += rng.random((28, 28)) * 0.05
    ax.imshow(img, cmap="Greys", extent=(0.2, 2.6, 0.8, 3.2))
    ax.text(1.4, 0.45, "ภาพ 28 × 28\nค่า 0–255", ha="center", fontsize=10, color=INK2)
    arrow(ax, (2.8, 2.0), (3.8, 2.0)); ax.text(3.3, 2.25, "Flatten", ha="center", fontsize=9.5, color=INK2)
    for i in range(14):
        ax.add_patch(Rectangle((4.0, 0.8 + i * 0.17), 0.35, 0.15, facecolor=BLUE_T, edgecolor=BLUE, linewidth=0.8))
    ax.text(4.18, 0.45, "784 ค่า\n(÷255 → 0–1)", ha="center", fontsize=10, color=INK2)
    ax.scatter([7.4] * 8, [0.95 + i * 0.29 for i in range(8)], s=120, facecolor=AQUA_T, edgecolor=AQUA, zorder=2)
    for i in range(0, 14, 3):
        for j in range(8):
            ax.plot([4.35, 7.28], [0.87 + i * 0.17, 0.95 + j * 0.29], color=GRID, linewidth=0.6, zorder=0)
    ax.text(7.4, 0.45, "Dense 128 โหนด", ha="center", fontsize=10, color=INK2)
    ax.text(8.2, 3.3, "พารามิเตอร์ของชั้นนี้", fontweight="bold", fontsize=11.5)
    ax.text(8.2, 2.7, "ค่าน้ำหนัก = 784 × 128 = 100,352", fontsize=10.5)
    ax.text(8.2, 2.25, "ค่าเอนเอียง = 128 (โหนดละ 1)", fontsize=10.5)
    ax.text(8.2, 1.65, "รวม = 100,480", fontsize=13, fontweight="bold", color=BLUE)
    save(fig, "q9-dense-mnist")


def split_units():
    fig, (a, b) = plt.subplots(2, 1, figsize=(9.6, 4.4), gridspec_kw={"height_ratios": [1.4, 1]})
    rows = [("(ข) test_size = 0.25", 15480, 5160), ("(ก) test_size = 0.3", 14448, 6192)]
    for i, (lab, tr, te) in enumerate(rows):
        a.barh(i, tr, color=BLUE, height=0.55)
        a.barh(i, te, left=tr + 120, color=ORANGE, height=0.55)
        a.text(tr / 2, i, f"Train {tr:,}", ha="center", va="center", color="white", fontweight="bold")
        a.text(tr + 120 + te / 2, i, f"Test {te:,}", ha="center", va="center", color="white", fontweight="bold")
    a.set_yticks([0, 1], [r[0] for r in rows]); a.invert_yaxis()
    a.set_xlim(0, 20640 + 200); a.set_xticks([0, 5000, 10000, 15000, 20640], ["0", "5,000", "10,000", "15,000", "20,640"])
    a.set_title("California Housing 20,640 แถว: ชุดทดสอบ = ⌈20,640 × test_size⌉", loc="left", fontweight="bold", fontsize=11)
    b.axis("off"); b.set_xlim(0, 12); b.set_ylim(0, 2)
    b.text(0.1, 1.65, "target มีหน่วย 100,000 ดอลลาร์", fontweight="bold", fontsize=11)
    for x, left, right in [(0.2, "RMSE = 0.58  (ข)", "58,000 ดอลลาร์"), (6.2, "45,000 ดอลลาร์  (ก)", "MAE = 0.45")]:
        box(b, x, 0.25, 2.3, 0.8, left, BLUE_T, BLUE, 10.5, "bold")
        box(b, x + 3.3, 0.25, 2.3, 0.8, right, AQUA_T, AQUA, 10.5, "bold")
        arrow(b, (x + 2.35, 0.65), (x + 3.25, 0.65), color=INK2)
        b.text(x + 2.8, 1.18, "× 100,000" if x < 1 else "÷ 100,000", ha="center", fontsize=9.5, color=INK2)
    fig.tight_layout()
    save(fig, "q9-split-units")


# ---------------------------------------------------------------- 10: K-Means
def kmeans():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    cases = [
        ("(ข) 4 จุด", np.array([[2, 2], [3, 2], [6, 5], [7, 6]]), np.array([[2, 2], [7, 6]]), ["(2,2)", "(3,2)", "(6,5)", "(7,6)"]),
        ("(ก) 5 จุด", np.array([[1, 2], [2, 1], [2, 3], [6, 5], [7, 7]]), np.array([[2, 2], [6, 6]]), list("ABCDE")),
    ]
    for ax, (title, X, C, names) in zip(axes, cases):
        d = np.linalg.norm(X[:, None] - C[None], axis=2); lab = d.argmin(1)
        new = np.array([X[lab == k].mean(0) for k in range(2)])
        cols = [BLUE, ORANGE]
        for i, p in enumerate(X):
            c = C[lab[i]]
            ax.plot([p[0], c[0]], [p[1], c[1]], color=cols[lab[i]], linewidth=1, linestyle=":", zorder=1)
        for k in range(2):
            pts = X[lab == k]
            ax.scatter(pts[:, 0], pts[:, 1], s=70, color=cols[k], zorder=3, edgecolor="white", linewidth=1.5, label=f"กลุ่ม {k+1}")
            ax.scatter(*C[k], s=160, marker="X", color="white", edgecolor=cols[k], linewidth=2, zorder=4)
            ax.scatter(*new[k], s=220, marker="*", color=cols[k], edgecolor="white", linewidth=1, zorder=5)
            if np.linalg.norm(new[k] - C[k]) > 0.05:
                arrow(ax, tuple(C[k]), tuple(new[k]), color=cols[k], lw=1.5)
            ax.annotate(f"c{k+1} ใหม่ = ({new[k][0]:.4g}, {new[k][1]:.4g})", new[k], new[k] + np.array([0.6, 0.6] if k == 0 else [0.4, -0.9]),
                        fontsize=10, fontweight="bold", color=INK)
        for p, n in zip(X, names):
            ax.text(p[0] - 0.15, p[1] + 0.25, n, fontsize=9.5, color=INK2, ha="right")
        ax.set_xlim(0, 9); ax.set_ylim(0, 8.5); ax.set_aspect("equal"); ygrid(ax)
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.text(0.5, -0.02, "✕ = centroid เริ่มต้น   ★ = centroid ใหม่ (ค่าเฉลี่ยของสมาชิก)   เส้นประ = จัดเข้ากลุ่มที่ใกล้ที่สุด",
             ha="center", fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, "q10-kmeans")


# ---------------------------------------------------------------- 12: RL
def q_update():
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 3.8))
    cases = [("(ข) Q = 3, α = 0.2, r = 2, γ = 0.8, max Q(s′,a′) = 6", 3, 6.8, 0.2, "TD target = 2 + 0.8×6 = 6.8"),
             ("(ก) สถานะสุดท้าย: Q = 4, α = 0.5, r = 10", 4, 10, 0.5, "TD target = r = 10 (ไม่มีอนาคต)")]
    for ax, (title, q, t, alpha, tlabel) in zip(axes, cases):
        new = q + alpha * (t - q)
        ax.set_xlim(0, 11); ax.set_ylim(-1, 1.6); ax.axis("off")
        ax.plot([0.5, 10.5], [0, 0], color=MUTED, linewidth=1)
        for v in range(1, 11):
            ax.plot([v, v], [-0.08, 0.08], color=MUTED, linewidth=1); ax.text(v, -0.45, str(v), ha="center", fontsize=9, color=MUTED)
        ax.add_patch(Rectangle((q, -0.12), t - q, 0.24, facecolor=YELLOW_T, edgecolor="none"))
        ax.text((q + t) / 2, 0.32, f"TD error = {t - q:g}", ha="center", fontsize=10, color=INK2)
        ax.scatter([q], [0], s=90, color=ORANGE, zorder=3); ax.text(q, -0.85, f"Q เดิม {q:g}", ha="center", fontweight="bold", fontsize=10)
        ax.scatter([t], [0], s=90, color=VIOLET, zorder=3); ax.text(t, -0.85, tlabel, ha="center", fontsize=9.5)
        ax.scatter([new], [0], s=140, marker="D", color=BLUE, zorder=4)
        arrow(ax, (q, 0.7), (new, 0.7), color=BLUE, lw=2)
        ax.text((q + new) / 2 + 0.1, 0.85, f"ขยับ α = {alpha} ของระยะ → Q ใหม่ = {new:g}", ha="left", fontsize=10, fontweight="bold", color=BLUE)
        ax.text(0, 1.4, title, fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "q12-q-update")


def discount():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.1))
    t = np.arange(4); w = 0.9 ** t
    a.bar(t, w, color=[GRID, GRID, GRID, BLUE], width=0.6)
    for i, v in enumerate(w):
        a.text(i, v + 0.02, f"γ{'⁰¹²³'[i]} = {v:.3g}", ha="center", fontsize=10)
        a.text(i, -0.14, f"r = {[0, 0, 0, 10][i]}", ha="center", fontsize=10, color=INK2)
    a.set_xticks(t, [f"ก้าวที่ {i+1}" for i in t]); a.set_ylim(-0.2, 1.32); ygrid(a)
    a.set_title("(ก) G₀ = 0.729 × 10 = 7.29", loc="left", fontweight="bold")
    a.text(1.5, 1.2, "รางวัลไกลออกไปมีน้ำหนักลดลง", ha="center", fontsize=10, color=INK2)
    n = np.arange(1, 121)
    for g, c in [(0.95, BLUE), (0.9, AQUA)]:
        cum = (1 - g ** n) / (1 - g)
        b.plot(n, cum, color=c)
        b.axhline(1 / (1 - g), color=c, linestyle="--", linewidth=1)
        b.text(121, 1 / (1 - g), f"  γ = {g}: 1/(1−γ) = {1/(1-g):g}", va="center", fontsize=10, color=INK)
    b.set_xlabel("จำนวนก้าวที่ได้รางวัล +1"); b.set_ylabel("ผลตอบแทนสะสม"); b.set_xlim(0, 120); b.set_ylim(0, 22); ygrid(b)
    b.set_title("(ข) รางวัล +1 ทุกก้าว: ลู่เข้า 1/(1−γ)", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "q12-discount")


# ---------------------------------------------------------------- 13: Neural network
def nn_params():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    archs = [("(ข) 5 → 8 → 2", [5, 8, 2]), ("(ก) 3 → 5 → 4 → 2", [3, 5, 4, 2])]
    fills = [BLUE_T, AQUA_T, AQUA_T, YELLOW_T]; edges = [BLUE, AQUA, AQUA, YELLOW]
    for ax, (title, L) in zip(axes, archs):
        ax.axis("off"); ax.set_xlim(-0.6, len(L) - 0.4); ax.set_ylim(-1.6, 8.6)
        pos = []
        for li, n in enumerate(L):
            ys = np.linspace(4 - (n - 1) * 0.5, 4 + (n - 1) * 0.5, n)
            pos.append(ys)
        for li in range(len(L) - 1):
            for y1 in pos[li]:
                for y2 in pos[li + 1]:
                    ax.plot([li, li + 1], [y1, y2], color=GRID, linewidth=0.8, zorder=0)
            ax.text(li + 0.5, 8.0, f"{L[li]}×{L[li+1]} = {L[li]*L[li+1]}", ha="center", fontsize=10, color=INK)
            ax.text(li + 0.5, 7.4, f"+ bias {L[li+1]}", ha="center", fontsize=9.5, color=INK2)
        for li, ys in enumerate(pos):
            k = 0 if li == 0 else (3 if li == len(L) - 1 else 1)
            ax.scatter([li] * len(ys), ys, s=170, facecolor=fills[k], edgecolor=edges[k], linewidth=1.3, zorder=2)
            ax.text(li, -0.6, ["นำเข้า", "ซ่อน", "ซ่อน", "ผลลัพธ์"][k] + f"\n{L[li]}", ha="center", fontsize=9.5, color=INK2)
        w = sum(a * b for a, b in zip(L, L[1:])); bb = sum(L[1:])
        ax.set_title(f"{title}\nน้ำหนัก {w} + เอนเอียง {bb} = {w + bb} พารามิเตอร์", loc="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "q13-nn-params")


def softmax():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))
    z = np.array([3, 1, 0]); e = np.exp(z); p = e / e.sum()
    data = [("1) คะแนนดิบ z", z, "{:g}"), ("2) eᶻ (รวม 23.804)", e, "{:.3f}"), ("3) softmax = eᶻ / รวม", p, "{:.3f}")]
    cols = [BLUE, AQUA, YELLOW]
    for ax, (title, v, fmt) in zip(axes, data):
        ax.bar(range(3), v, color=cols, width=0.6)
        for i, x in enumerate(v):
            ax.text(i, x + max(v) * 0.03, fmt.format(x), ha="center", fontsize=10)
        ax.set_xticks(range(3), ["คลาส 1", "คลาส 2", "คลาส 3"]); ax.set_ylim(0, max(v) * 1.2); ygrid(ax)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    fig.suptitle("(ข) Softmax ของ [3, 1, 0] · (ก) [1, 3, 0] ได้ค่าเดียวกันแต่สลับตำแหน่ง → p(คลาสถูก) = 0.844, Cross-Entropy = −ln(0.844) = 0.170",
                 fontsize=10, color=INK2, y=0.02)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "q13-softmax")


# ---------------------------------------------------------------- 14: VRAM
def vram():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
    bits = [32, 16, 8, 4]; gb = [13 * x / 8 for x in bits]
    a.barh(range(4), gb, color=[GRID, BLUE, GRID, BLUE], height=0.55)
    for i, (bt, g) in enumerate(zip(bits, gb)):
        a.text(g + 0.8, i, f"{g:g} GB  (13B × {bt/8:g} ไบต์)", va="center", fontsize=10)
    a.axvline(24, color=ORANGE, linewidth=2); a.text(24.6, -0.62, "GPU 24 GB", color=ORANGE, fontweight="bold", fontsize=10)
    a.set_yticks(range(4), [f"{x} บิต" for x in bits]); a.set_ylim(3.5, -0.85); a.set_xlim(0, 75)
    a.set_xlabel("หน่วยความจำของค่าน้ำหนัก (GB)")
    a.set_title("(ข) โมเดล 13B ใช้ VRAM เท่าใด", loc="left", fontweight="bold")
    caps = [(16, 12), (8, 24), (4, 48)]
    b.bar(range(3), [c[1] for c in caps], color=[BLUE, GRID, BLUE], width=0.6)
    for i, (bt, n) in enumerate(caps):
        b.text(i, n + 1, f"{n}B", ha="center", fontweight="bold")
        b.text(i, -6, f"24 GB ÷ {bt/8:g}", ha="center", fontsize=9.5, color=INK2)
    b.set_xticks(range(3), [f"{c[0]} บิต" for c in caps]); b.set_ylim(-9, 56); ygrid(b)
    b.set_ylabel("พันล้านพารามิเตอร์ (B)")
    b.set_title("(ก) GPU 24 GB จุโมเดลได้ใหญ่สุด", loc="left", fontweight="bold")
    fig.text(0.5, -0.03, "สูตร: หน่วยความจำ = จำนวนพารามิเตอร์ × บิต/8 (เฉพาะค่าน้ำหนัก ใช้งานจริงต้องเผื่อ KV cache และ overhead)",
             ha="center", fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, "q14-vram")


for f in [confusion_metrics, spam_tree, linear_prediction, error_metrics, sigmoid, gini, voting, dense_mnist,
          split_units, kmeans, q_update, discount, nn_params, softmax, vram]:
    f()
