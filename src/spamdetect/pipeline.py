"""End-to-end experiment: data -> features -> models -> metrics -> figures.

Run from the repository root with::

    python run_pipeline.py            # full run (about 3-6 minutes on 4 CPU cores)
    python run_pipeline.py --quick    # smoke test on a 1,000-message subsample

All outputs are written to ``results/`` (tables) and ``results/figures/``.
With a fixed seed, every file except ``timing.csv`` is byte-for-byte
reproducible across runs on the same library versions.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config, plots
from .data import load_dataset, split
from .evaluate import bootstrap_ci, mcnemar_exact, point_metrics
from .features import FEATURE_SETS, HANDCRAFTED_FEATURE_NAMES, make_tfidf
from .models import (
    FEATURE_LABELS, MODEL_LABELS, MODEL_NAMES, PARAM_GRIDS, make_classifier, make_pipeline,
)

PROBES_PATH = config.DATA_DIR / "probes" / "smishing_probes.tsv"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _grid_search(pipe: Pipeline, model: str, X, y) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    # Random forest already parallelises over trees; avoid nested parallelism.
    n_jobs = 1 if model == "random_forest" else -1
    gs = GridSearchCV(pipe, PARAM_GRIDS[model], scoring=config.CV_SCORING, cv=cv,
                      n_jobs=n_jobs, refit=True)
    return gs.fit(X, y)


def _evaluate(gs: GridSearchCV, X_te, y_te) -> tuple[dict, np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    score = gs.predict_proba(X_te)[:, 1]
    predict_s = time.perf_counter() - t0
    pred = (score >= config.DECISION_THRESHOLD).astype(int)
    m = point_metrics(y_te, pred, score)
    m.update(bootstrap_ci(y_te, pred, score))
    best = gs.best_index_
    m["cv_f1_mean"] = float(gs.cv_results_["mean_test_score"][best])
    m["cv_f1_std"] = float(gs.cv_results_["std_test_score"][best])
    m["best_params"] = json.dumps({k.replace("clf__", ""): v for k, v in gs.best_params_.items()})
    return m, score, pred, predict_s


def _dataset_summary(df: pd.DataFrame, stats: dict, y_tr, y_te) -> dict:
    lengths = df.assign(n_chars=df["text"].str.len(), n_words=df["text"].str.split().str.len())
    by_class = lengths.groupby("label_name")[["n_chars", "n_words"]].agg(["mean", "median"]).round(1)
    summary = dict(stats)
    summary.update({
        "spam_fraction_dedup": round(stats["n_dedup_spam"] / stats["n_dedup"], 4),
        "n_train": int(len(y_tr)), "n_train_spam": int(y_tr.sum()),
        "n_test": int(len(y_te)), "n_test_spam": int(y_te.sum()),
        "length_by_class": {cls: {f"{col}_{agg}": float(by_class.loc[cls, (col, agg)])
                                  for col in ("n_chars", "n_words") for agg in ("mean", "median")}
                            for cls in by_class.index},
    })
    return summary


def _top_linear_terms(pipe: Pipeline, k: int = 20) -> pd.DataFrame:
    """Largest positive (spam) and negative (ham) coefficients of a linear model."""
    names = pipe.named_steps["features"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_.ravel()
    order = np.argsort(coef)
    rows = [(names[i], coef[i], "spam") for i in order[::-1][:k]]
    rows += [(names[i], coef[i], "ham") for i in order[:k]]
    df = pd.DataFrame(rows, columns=["feature", "coefficient", "direction"])
    df["feature"] = df["feature"].str.replace(r"^(tfidf|hand)__", "", regex=True)
    return df


def _nb_top_terms(pipe: Pipeline, k: int = 20) -> pd.DataFrame:
    """Naive Bayes log-likelihood ratio log P(w|spam) - log P(w|ham)."""
    names = pipe.named_steps["features"].get_feature_names_out()
    llr = np.diff(pipe.named_steps["clf"].feature_log_prob_, axis=0).ravel()
    order = np.argsort(llr)
    rows = [(names[i], llr[i], "spam") for i in order[::-1][:k]]
    rows += [(names[i], llr[i], "ham") for i in order[:k]]
    return pd.DataFrame(rows, columns=["feature", "log_likelihood_ratio", "direction"])


def _markdown_table(df: pd.DataFrame) -> str:
    cols = ["features", "model", "accuracy", "precision", "recall", "f1", "f1_ci",
            "roc_auc", "pr_auc", "fpr", "fp", "fn", "cv_f1"]
    t = df.copy()
    t["features"] = t["features"].map(FEATURE_LABELS)
    t["model"] = t["model"].map(MODEL_LABELS)
    t["f1_ci"] = t.apply(lambda r: f"[{r.f1_ci_low:.3f}, {r.f1_ci_high:.3f}]", axis=1)
    t["cv_f1"] = t.apply(lambda r: f"{r.cv_f1_mean:.3f} ± {r.cv_f1_std:.3f}", axis=1)
    t["fpr"] = t["fpr"].map(lambda v: f"{100 * v:.2f}%")
    for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        t[c] = t[c].map(lambda v: f"{v:.3f}")
    t = t[cols]
    header = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return header + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in t.to_numpy()) + "\n"


# --------------------------------------------------------------------------
# Main experiment
# --------------------------------------------------------------------------
def run(quick: bool = False) -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    # Quick (smoke-test) runs write to a separate, git-ignored directory so
    # they can never overwrite the reported results.
    out = config.ROOT_DIR / "results_quick" if quick else config.RESULTS_DIR
    figs = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    plots.set_style()

    # ---- 1. Data ---------------------------------------------------------
    df, stats = load_dataset()
    if quick:  # stratified subsample, smoke test only (results are NOT reported)
        df = df.groupby("label", group_keys=False).sample(frac=1000 / len(df), random_state=config.SEED)
    X_tr, X_te, y_tr, y_te = split(df)
    summary = _dataset_summary(df, stats, y_tr, y_te)
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[data] {stats['n_raw']} raw -> {stats['n_dedup']} unique messages "
          f"({stats['n_dedup_spam']} spam); train={len(y_tr)} test={len(y_te)}")
    plots.length_distribution(df, figs / "length_distribution.png")

    # ---- 2. Main grid: 3 feature sets x 3 models --------------------------
    rows, timing, fitted, scores, preds = [], [], {}, {}, {}
    for feat in FEATURE_SETS:
        for model in MODEL_NAMES:
            t0 = time.perf_counter()
            gs = _grid_search(make_pipeline(model, feat), model, X_tr, y_tr)
            search_s = time.perf_counter() - t0
            m, score, pred, predict_s = _evaluate(gs, X_te, y_te)
            rows.append({"features": feat, "model": model, **m})
            timing.append({"features": feat, "model": model, "grid_search_s": round(search_s, 2),
                           "refit_s": round(gs.refit_time_, 3),
                           "predict_ms_per_1k_msgs": round(1000 * predict_s / len(y_te) * 1000, 2)})
            fitted[(feat, model)], scores[(feat, model)], preds[(feat, model)] = \
                gs.best_estimator_, score, pred
            print(f"[model] {FEATURE_LABELS[feat]:>12} | {MODEL_LABELS[model]:<19} "
                  f"F1={m['f1']:.3f} AUC={m['roc_auc']:.4f} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} FP={m['fp']} FN={m['fn']} cvF1={m['cv_f1_mean']:.3f} "
                  f"({search_s:.0f}s)")
    metrics = pd.DataFrame(rows)
    metrics.round(6).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "timing.csv", index=False)

    # Model selection uses cross-validated F1 on the TRAINING split only, so
    # the test set is never used to pick the "best" system.
    best_row = metrics.loc[metrics["cv_f1_mean"].idxmax()]
    best_key = (best_row["features"], best_row["model"])
    print(f"[select] best by CV F1: {FEATURE_LABELS[best_key[0]]} + {MODEL_LABELS[best_key[1]]}")

    # Paired significance: is the selected system's error pattern different
    # from each alternative's on the same test messages?
    mcnemar = pd.DataFrame([
        {"features": f, "model": m, **mcnemar_exact(y_te, preds[best_key], preds[(f, m)])}
        for (f, m) in preds if (f, m) != best_key
    ])
    mcnemar.round(6).to_csv(out / "mcnemar_vs_selected.csv", index=False)

    # ---- 3. Preprocessing ablation (TF-IDF only) --------------------------
    ablation_rows = []
    variants = {
        "raw (sklearn default tokenizer)": dict(normalise=False),
        "cleaned + entity tokens (default)": dict(normalise=True, remove_stopwords=False),
        "cleaned + entity tokens + stop-words removed": dict(normalise=True, remove_stopwords=True),
    }
    for model in ["naive_bayes", "logistic_regression"]:
        for vname, kw in variants.items():
            pipe = Pipeline([("features", make_tfidf(**kw)), ("clf", make_classifier(model))])
            gs = _grid_search(pipe, model, X_tr, y_tr)
            m, *_ = _evaluate(gs, X_te, y_te)
            vocab = len(gs.best_estimator_.named_steps["features"].vocabulary_)
            ablation_rows.append({"model": model, "preprocessing": vname, "vocab_size": vocab, **m})
    ablation = pd.DataFrame(ablation_rows)
    ablation.round(6).to_csv(out / "ablation_preprocessing.csv", index=False)

    # ---- 4. Interpretability ---------------------------------------------
    top_lr = _top_linear_terms(fitted[("tfidf", "logistic_regression")])
    top_lr.round(6).to_csv(out / "top_features_lr_tfidf.csv", index=False)
    _nb_top_terms(fitted[("tfidf", "naive_bayes")]).round(6).to_csv(
        out / "top_features_nb_tfidf.csv", index=False)
    _top_linear_terms(fitted[("combined", "logistic_regression")]).round(6).to_csv(
        out / "top_features_lr_combined.csv", index=False)
    hand_lr = _top_linear_terms(fitted[("handcrafted", "logistic_regression")], k=len(HANDCRAFTED_FEATURE_NAMES))
    hand_lr = hand_lr.drop_duplicates("feature").sort_values("coefficient", ascending=False)

    # permutation_importance shuffles columns of a 2-D matrix, so we extract
    # the raw hand-crafted matrix once and permute it in front of the
    # (already fitted) log1p -> scale -> forest tail of the pipeline.
    rf_hand = fitted[("handcrafted", "random_forest")]
    hand_steps = rf_hand.named_steps["features"]
    tail = Pipeline([("prep", hand_steps[1:]), ("clf", rf_hand.named_steps["clf"])])
    H_te = hand_steps.named_steps["extract"].transform(X_te)
    perm = permutation_importance(tail, H_te, y_te, scoring="f1",
                                  n_repeats=config.N_PERMUTATION_REPEATS,
                                  random_state=config.SEED, n_jobs=1)
    hand_imp = pd.DataFrame({
        "feature": HANDCRAFTED_FEATURE_NAMES,
        "permutation_importance_f1_mean": perm.importances_mean,
        "permutation_importance_f1_std": perm.importances_std,
        "rf_gini_importance": rf_hand.named_steps["clf"].feature_importances_,
    }).merge(hand_lr[["feature", "coefficient"]].rename(columns={"coefficient": "lr_coefficient"}),
             on="feature").sort_values("permutation_importance_f1_mean", ascending=False)
    hand_imp.round(6).to_csv(out / "handcrafted_feature_importance.csv", index=False)

    # Class-conditional prevalence of each cue (computed on the training split).
    raw_tr = hand_steps.named_steps["extract"].transform(X_tr)
    prev = pd.DataFrame(raw_tr, columns=HANDCRAFTED_FEATURE_NAMES).assign(label=y_tr)
    prevalence = pd.DataFrame({
        "feature": HANDCRAFTED_FEATURE_NAMES,
        "ham_mean": prev[prev.label == 0][HANDCRAFTED_FEATURE_NAMES].mean().to_numpy(),
        "spam_mean": prev[prev.label == 1][HANDCRAFTED_FEATURE_NAMES].mean().to_numpy(),
        "ham_pct_nonzero": 100 * (prev[prev.label == 0][HANDCRAFTED_FEATURE_NAMES] > 0).mean().to_numpy(),
        "spam_pct_nonzero": 100 * (prev[prev.label == 1][HANDCRAFTED_FEATURE_NAMES] > 0).mean().to_numpy(),
    })
    prevalence.round(4).to_csv(out / "handcrafted_feature_by_class.csv", index=False)

    # ---- 5. Error analysis of the selected model -------------------------
    best_score, best_pred = scores[best_key], preds[best_key]
    wrong = np.flatnonzero(best_pred != y_te)
    errors = pd.DataFrame({
        "true_label": [config.LABEL_NAMES[v] for v in y_te[wrong]],
        "predicted": [config.LABEL_NAMES[v] for v in best_pred[wrong]],
        "spam_probability": best_score[wrong].round(4),
        "text": X_te[wrong],
    }).sort_values(["true_label", "spam_probability"])
    errors.to_csv(out / "misclassified_best_model.csv", index=False)

    # ---- 6. Out-of-distribution probe: modern smishing ---------------------
    if PROBES_PATH.exists():
        probes = pd.read_csv(PROBES_PATH, sep="\t")
        probe_rows = []
        for key in [best_key, ("handcrafted", "random_forest"), ("tfidf", "naive_bayes")]:
            p = fitted[key].predict_proba(probes["text"].to_numpy(dtype=object))[:, 1]
            probe_rows.append(pd.DataFrame({
                "system": f"{FEATURE_LABELS[key[0]]} + {MODEL_LABELS[key[1]]}",
                "probe_id": probes["id"], "category": probes["category"],
                "true_label": probes["label"], "spam_probability": p.round(4),
                "predicted": np.where(p >= config.DECISION_THRESHOLD, "spam", "ham"),
            }))
        probe_df = pd.concat(probe_rows, ignore_index=True)
        probe_df.to_csv(out / "smishing_probe_predictions.csv", index=False)
        probe_summary = (probe_df.assign(correct=probe_df.predicted == probe_df.true_label)
                         .groupby(["system", "true_label"])["correct"].agg(["sum", "count"])
                         .reset_index())
        probe_summary.to_csv(out / "smishing_probe_summary.csv", index=False)

    # ---- 7. Figures -------------------------------------------------------
    plots.model_comparison(metrics, figs / "model_comparison_f1.png")
    plots.confusion_matrix_single(best_row, figs / "confusion_matrix_best.png",
                                  title=f"{FEATURE_LABELS[best_key[0]]} + {MODEL_LABELS[best_key[1]]}")
    plots.confusion_matrix_grid(metrics, figs / "confusion_matrices_all.png")
    plots.roc_curves(y_te, scores, metrics, figs / "roc_curves.png")
    plots.pr_curves(y_te, scores, metrics, figs / "pr_curves.png")
    plots.top_words(top_lr, figs / "top_predictive_words.png")
    plots.handcrafted_importance(hand_imp, figs / "handcrafted_feature_importance.png")
    plots.feature_prevalence(prevalence, figs / "handcrafted_feature_prevalence.png")

    # ---- 8. Summary tables & provenance ----------------------------------
    (out / "results_table.md").write_text(_markdown_table(metrics))
    (out / "run_info.json").write_text(json.dumps({
        "quick_mode": quick,
        "seed": config.SEED, "test_size": config.TEST_SIZE, "cv_folds": config.CV_FOLDS,
        "cv_scoring": config.CV_SCORING, "decision_threshold": config.DECISION_THRESHOLD,
        "selected_model": {"features": best_key[0], "model": best_key[1],
                           "params": json.loads(best_row["best_params"])},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
    }, indent=2))
    print(f"[done] results written to {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smoke test on a 1,000-message subsample")
    run(quick=ap.parse_args(argv).quick)


if __name__ == "__main__":
    main()
