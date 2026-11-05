"""Figure generation (matplotlib only, static PNGs for the paper).

Visual conventions used in every figure:
* class colours are fixed: ham = blue, spam = orange;
* model colours are fixed: Naive Bayes = blue, Logistic Regression = orange,
  Random Forest = aqua (a validated colour-blind-safe categorical order);
* hairline solid grid, light axes, a legend or direct labels on every
  multi-series plot so that identity never relies on colour alone.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .features import FEATURE_SETS  # noqa: E402
from .models import FEATURE_LABELS, MODEL_LABELS, MODEL_NAMES  # noqa: E402

# ---- palette ------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
HAM, SPAM = "#2a78d6", "#eb6834"
MODEL_COLORS = {"naive_bayes": "#2a78d6", "logistic_regression": "#eb6834", "random_forest": "#1baf7a"}
MODEL_MARKERS = {"naive_bayes": "o", "logistic_regression": "s", "random_forest": "D"}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DPI = 200


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelsize": 9.5, "axes.labelcolor": INK_2, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,
    })


def _save(fig, path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


# ---- 1. message length --------------------------------------------------
def length_distribution(df, path) -> None:
    """Density histograms of message length (characters, words) per class."""
    n_chars = df["text"].str.len().to_numpy()
    n_words = df["text"].str.split().str.len().to_numpy()
    y = df["label"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, values, unit, cap in [(axes[0], n_chars, "characters", 400), (axes[1], n_words, "words", 80)]:
        bins = np.linspace(0, cap, 41)
        for cls, color, name in [(0, HAM, "ham"), (1, SPAM, "spam")]:
            v = np.clip(values[y == cls], 0, cap)  # values above cap fold into the last bin
            ax.hist(v, bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.22)
            ax.hist(v, bins=bins, density=True, histtype="step", color=color, linewidth=1.6,
                    label=f"{name} (n={int((y == cls).sum()):,})")
            med = float(np.median(values[y == cls]))
            ax.axvline(med, color=color, linewidth=1.2)
            # label sits beside its line, ham lower than spam so they never overlap
            # ham label right of its line, spam label left of its line (away from the peak)
            ax.text(med, 0.62 if cls == 0 else 0.85, f" median\n {name} {med:.0f}" if cls == 0
                    else f"median \n{name} {med:.0f} ", transform=ax.get_xaxis_transform(),
                    color=INK_2, fontsize=8, va="center", ha="left" if cls == 0 else "right")
        ax.set_xlabel(f"Message length ({unit}; ≥{cap} shown in last bin)")
        ax.set_ylabel("Density")
        ax.set_xlim(0, cap)
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="upper right")
    axes[0].set_title("Message length in characters")
    axes[1].set_title("Message length in words")
    fig.tight_layout()
    _save(fig, path)


# ---- 2. model comparison -----------------------------------------------
def model_comparison(metrics, path) -> None:
    """Dot plot: test F1 with bootstrap 95% CI for every (features, model)."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ticks, labels, ypos = [], [], 0
    for g, feat in enumerate(FEATURE_SETS):
        for model in MODEL_NAMES:
            r = metrics[(metrics.features == feat) & (metrics.model == model)].iloc[0]
            ax.errorbar(r.f1, ypos, xerr=[[r.f1 - r.f1_ci_low], [r.f1_ci_high - r.f1]],
                        fmt=MODEL_MARKERS[model], color=MODEL_COLORS[model], markersize=7,
                        elinewidth=2, capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=MODEL_LABELS[model] if g == 0 else None)
            ax.annotate(f"{r.f1:.3f}", xy=(r.f1_ci_high, ypos), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
            ticks.append(ypos)
            labels.append(f"{FEATURE_LABELS[feat]} · {MODEL_LABELS[model]}")
            ypos += 1
        ypos += 0.6  # visual gap between feature-set groups
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test F1 (spam class) with bootstrap 95% CI")
    lo = max(0.0, np.floor(metrics.f1_ci_low.min() * 20) / 20 - 0.02)
    ax.set_xlim(lo, 1.0)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.3)
    ax.set_title("Spam detection F1 by feature set and classifier", pad=26)
    fig.tight_layout()
    _save(fig, path)


# ---- 3. confusion matrices ---------------------------------------------
def _draw_cm(ax, r, fontsize=11) -> None:
    cm = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=float)
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(row_pct, cmap=SEQ_BLUE, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "#ffffff" if row_pct[i, j] > 0.55 else INK
            ax.text(j, i, f"{int(cm[i, j]):,}\n{100 * row_pct[i, j]:.1f}%", ha="center", va="center",
                    color=color, fontsize=fontsize)
    ax.set_xticks([0, 1], ["ham", "spam"])
    ax.set_yticks([0, 1], ["ham", "spam"])
    ax.grid(False)
    ax.axhline(0.5, color=SURFACE, linewidth=2)  # 2px surface gap between cells
    ax.axvline(0.5, color=SURFACE, linewidth=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def confusion_matrix_single(r, path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    _draw_cm(ax, r, fontsize=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix (test set)\n{title}", fontsize=10)
    fig.text(0.5, -0.02, "Cell: count and % of the true class (row)", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def confusion_matrix_grid(metrics, path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 8.0))
    for i, feat in enumerate(FEATURE_SETS):
        for j, model in enumerate(MODEL_NAMES):
            r = metrics[(metrics.features == feat) & (metrics.model == model)].iloc[0]
            ax = axes[i, j]
            _draw_cm(ax, r, fontsize=9.5)
            if i == 0:
                ax.set_title(MODEL_LABELS[model])
            if j == 0:
                ax.set_ylabel(f"{FEATURE_LABELS[feat]}\n\nTrue label")
            if i == 2:
                ax.set_xlabel("Predicted label")
    fig.suptitle("Test-set confusion matrices (rows: feature set, columns: classifier)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    _save(fig, path)


# ---- 4. ROC & PR curves --------------------------------------------------
def _curve_panels(y_true, scores, metrics, path, kind: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    for ax, feat in zip(axes, FEATURE_SETS):
        if kind == "roc":
            inset = ax.inset_axes([0.42, 0.45, 0.53, 0.38])
            inset.set_facecolor(SURFACE)
        score_lines = []
        for model in MODEL_NAMES:
            r = metrics[(metrics.features == feat) & (metrics.model == model)].iloc[0]
            s = scores[(feat, model)]
            if kind == "roc":
                x, y, _ = roc_curve(y_true, s)
                score_lines.append(f"{MODEL_LABELS[model]}  {r.roc_auc:.3f}")
                inset.plot(x, y, color=MODEL_COLORS[model], linewidth=1.6)
            else:
                y, x, _ = precision_recall_curve(y_true, s)
                score_lines.append(f"{MODEL_LABELS[model]}  {r.pr_auc:.3f}")
            ax.plot(x, y, color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                    drawstyle="steps-post" if kind == "pr" else None)
        metric = "ROC-AUC" if kind == "roc" else "Average precision"
        ax.text(0.97, 0.04, metric + "\n" + "\n".join(score_lines), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.8, color=INK_2, linespacing=1.4,
                bbox={"facecolor": SURFACE, "edgecolor": GRID, "boxstyle": "round,pad=0.4"})
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
            ax.set_xlabel("False-positive rate (ham flagged as spam)")
            inset.set_xlim(0, 0.05)
            inset.set_ylim(0.6, 1.0)
            inset.set_title("zoom: FPR ≤ 5%", fontsize=7.5, fontweight="normal", color=INK_2)
            inset.tick_params(labelsize=7)
        else:
            ax.set_xlabel("Recall (spam caught)")
            ax.set_ylim(0.0, 1.02)
        ax.set_xlim(-0.01, 1.01)
        ax.set_title(FEATURE_LABELS[feat])
    axes[0].set_ylabel("True-positive rate (recall)" if kind == "roc" else "Precision")
    fig.suptitle("ROC curves on the test set" if kind == "roc" else "Precision–recall curves on the test set",
                 fontweight="bold", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    _save(fig, path)


def roc_curves(y_true, scores, metrics, path) -> None:
    _curve_panels(y_true, scores, metrics, path, "roc")


def pr_curves(y_true, scores, metrics, path) -> None:
    _curve_panels(y_true, scores, metrics, path, "pr")


# ---- 5. interpretability -------------------------------------------------
def top_words(top, path, k: int = 20) -> None:
    """Largest logistic-regression coefficients (TF-IDF): ham vs spam."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.6))
    for ax, direction, color, title in [(axes[0], "ham", HAM, "Strongest ham (legitimate) indicators"),
                                        (axes[1], "spam", SPAM, "Strongest spam indicators")]:
        d = top[top.direction == direction].head(k)
        vals = d.coefficient.abs().to_numpy()
        ax.barh(np.arange(len(d)), vals, color=color, height=0.72)
        ax.set_yticks(np.arange(len(d)), d.feature)
        ax.invert_yaxis()
        ax.set_xlabel("|coefficient| (logistic regression on TF-IDF)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.text(0.5, -0.01, "Entity placeholders: phonetoken = phone number or SMS short code, "
             "numtoken = other number, moneytoken = currency amount, urltoken = link",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def handcrafted_importance(imp, path) -> None:
    d = imp.sort_values("permutation_importance_f1_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    y = np.arange(len(d))
    ax.barh(y, d.permutation_importance_f1_mean, xerr=d.permutation_importance_f1_std,
            color=MODEL_COLORS["random_forest"], height=0.7,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 0})
    ax.set_yticks(y, d.feature)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("Drop in test F1 when the feature is shuffled (mean ± sd, 10 repeats)")
    ax.set_title("Hand-crafted features: permutation importance\n(Random Forest)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    _save(fig, path)


PREVALENCE_FEATURES = [
    "n_phone_numbers", "n_money", "n_urls", "n_emails", "reward_words", "call_to_action_words",
    "urgency_words", "credential_words", "n_allcaps_words", "n_exclamation", "n_question",
]


def feature_prevalence(prev, path) -> None:
    d = prev.set_index("feature").loc[PREVALENCE_FEATURES]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    y = np.arange(len(d))
    h = 0.38
    ax.barh(y - h / 2 - 0.01, d.ham_pct_nonzero, height=h, color=HAM, label="ham")
    ax.barh(y + h / 2 + 0.01, d.spam_pct_nonzero, height=h, color=SPAM, label="spam")
    for yi, (hv, sv) in enumerate(zip(d.ham_pct_nonzero, d.spam_pct_nonzero)):
        ax.text(hv + 1, yi - h / 2, f"{hv:.0f}%", va="center", fontsize=7.5, color=INK_2)
        ax.text(sv + 1, yi + h / 2, f"{sv:.0f}%", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, d.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Messages containing the cue (%) — training split")
    ax.set_title("Prevalence of hand-crafted spam/phishing cues by class")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    _save(fig, path)
