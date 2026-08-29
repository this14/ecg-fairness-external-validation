"""Figure 1 — internal versus external effect of race loss reweighting.

Dumbbell form: before -> after per item, one hue in two shades, with marker
shape as a second encoding so the figure survives greyscale printing.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LIGHT = "#5598e7"   # blue step 350 — Standard
DARK  = "#104281"   # blue step 650 — Reweighted
INK, INK2 = "#0b0b0b", "#52514e"
GRID, RULE = "#dedcd8", "#b8b7b2"

BLOCKS = [
    ("12 leads", [
        ("MIMIC-IV-ECG, internal test",  0.9849, 0.9757, "−0.0092", "—"),
        ("EchoNext, QRS ≥ 120 ms",       0.9392, 0.8203, "−0.1189", "***"),
        ("SNUH, complete block",         0.8865, 0.7288, "−0.1577", "***"),
    ]),
    ("3 leads", [
        ("MIMIC-IV-ECG, internal test",  0.9645, 0.9583, "−0.0062", "—"),
        ("EchoNext, QRS ≥ 120 ms",       0.9385, 0.9240, "−0.0145", "ns"),
        ("SNUH, complete block",         0.8947, 0.8051, "−0.0896", "***"),
    ]),
    ("1 lead", [
        ("MIMIC-IV-ECG, internal test",  0.9188, 0.9321, "+0.0133", "—"),
        ("EchoNext, QRS ≥ 120 ms",       0.8866, 0.8787, "−0.0079", "ns"),
        ("SNUH, complete block",         0.8145, 0.7691, "−0.0454", "***"),
    ]),
]

rows, heads, y = [], [], 0.0
for bname, items in BLOCKS:
    heads.append((y, bname)); y += 0.95
    for lab, s, r, d, sig in items:
        rows.append((y, lab, s, r, d, sig)); y += 1.0
    y += 0.55
ymax = y - 0.55

H = 0.26 * ymax + 1.35
fig, ax = plt.subplots(figsize=(7.1, H), dpi=300)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

for yy, lab, s, r, d, sig in rows:
    ax.plot([s, r], [yy, yy], color=RULE, lw=2.0, solid_capstyle="round", zorder=1)
    ax.plot([s], [yy], marker="o", ms=8.0, mfc=LIGHT, mec="white", mew=1.3, zorder=3)
    ax.plot([r], [yy], marker="s", ms=7.4, mfc=DARK,  mec="white", mew=1.3, zorder=3)
    strong = sig == "***"
    ax.text(1.115, yy, d, transform=ax.get_yaxis_transform(), va="center", ha="right",
            fontsize=7.6, color=INK if strong else INK2,
            fontweight="bold" if strong else "normal")
    ax.text(1.145, yy, sig, transform=ax.get_yaxis_transform(), va="center", ha="left",
            fontsize=7.0, color=INK if strong else INK2)

for yy, bname in heads:
    ax.text(-0.335, yy, bname, transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8.6, color=INK, fontweight="bold")

ax.set_yticks([r[0] for r in rows])
ax.set_yticklabels(["   " + r[1] for r in rows], fontsize=7.9, color=INK2)
ax.tick_params(axis="y", length=0, pad=2)
ax.set_ylim(ymax - 0.45, -0.55)

ax.set_xlim(0.70, 1.00)
ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
ax.set_xticklabels(["0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "1.00"],
                   fontsize=7.9, color=INK2)
ax.set_xlabel("Area under the receiver operating characteristic curve",
              fontsize=8.4, color=INK, labelpad=6)
ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(RULE); ax.spines["bottom"].set_linewidth(0.8)

ax.text(1.115, -0.9, "Δ", transform=ax.get_yaxis_transform(), va="center",
        ha="right", fontsize=7.6, color=INK2, style="italic")

ax.legend(handles=[
    Line2D([], [], marker="o", ms=8.0, mfc=LIGHT, mec="white", mew=1.3, ls="none",
           label="Standard"),
    Line2D([], [], marker="s", ms=7.4, mfc=DARK, mec="white", mew=1.3, ls="none",
           label="Reweighted"),
], loc="lower left", bbox_to_anchor=(-0.335, 1.012), ncol=2, frameon=False,
   fontsize=8.0, handletextpad=0.45, columnspacing=2.0, labelcolor=INK2)

plt.subplots_adjust(left=0.32, right=0.815, top=1 - 0.52 / H, bottom=0.62 / H)
fig.savefig("/home/claude/Figure1_internal_vs_external.pdf", facecolor="white")
fig.savefig("/home/claude/Figure1_internal_vs_external.png", facecolor="white")
print("written  (figure height %.2f in)" % H)
