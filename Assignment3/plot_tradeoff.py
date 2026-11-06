"""Task 10 figure: test AUC vs. total training time (incl. tuning).
Add the Task 7-8 models to MODELS when their results are ready, then rerun."""
import matplotlib.pyplot as plt

# (name, total training time incl. tuning [s], test AUC)
MODELS = [
    ("Naive Bayes", 0.25, 0.8495),
    ("Decision Tree", 35.9, 0.8719),
    ("MLP (Task 5)", 67.2, 0.8813),
    ("DNN (Task 9)", 20.8 * 60 + 1.9 * 60, 0.8821),
    # ("Bagging", ..., ...),
    # ("Boosting", ..., ...),
]

INK, MUTED, GRID, MARK = "#1f2328", "#59636e", "#e6e8eb", "#2f6fb0"

fig, ax = plt.subplots(figsize=(7, 4))
for name, t, auc in MODELS:
    ax.scatter(t, auc, s=60, color=MARK, edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(name, (t, auc), xytext=(7, -4), textcoords="offset points",
                fontsize=10, color=INK)

ax.set_xscale("log")
ax.set_xlim(0.1, 2e4)
ax.set_ylim(0.84, 0.89)
ax.set_xlabel("Total training time incl. tuning (seconds, log scale)", color=MUTED)
ax.set_ylabel("Test AUC", color=MUTED)
ax.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=MUTED)
plt.tight_layout()
plt.savefig("Overall_tradeoff.png", dpi=200)
print("saved Overall_tradeoff.png")
