"""
train.py
Trains and compares multiple models for 10-year CHD risk prediction,
calibrates the best one, and saves an explainability report.

Modifications vs. the base CardioPredictor paper:
  1. Model comparison: the paper reports 5 models in a table but only
     really commits to Gradient Boosting. This script actually trains
     and evaluates each one against the same metrics and picks the
     best by ROC-AUC, so the choice is data-driven, not assumed.
  2. Calibration: the paper's model outputs a raw probability with no
     calibration check. This script fits Platt (sigmoid) calibration
     via CalibratedClassifierCV and reports Brier score before/after,
     so the "% chance" it produces is actually reliable, not just
     the model's uncalibrated confidence.
  3. Explainability: the paper has none. This script uses SHAP when
     available, and falls back to permutation importance otherwise
     (this happens automatically -- no code changes needed if SHAP is
     later installed in the target environment).

Usage:
    python src/train.py
Outputs:
    models/best_model.joblib          -- calibrated best model
    models/model_comparison.json      -- metrics for every model tried
    models/feature_importance.json    -- explainability report
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, roc_curve,
)
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from preprocess import get_train_test_split, FEATURE_COLUMNS
from explain import make_background

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def get_candidate_models():
    """Models compared against each other. Gradient Boosting stands in
    for the paper's headline model; XGBoost is used instead if the
    xgboost package is available in the environment (it usually will
    be on a normal Windows/venv setup, even where it wasn't available
    when this file was generated)."""
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=25, weights="distance")),
        ]),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )
    except ImportError:
        pass
    return models


def cross_validate_models(candidates, X_train, y_train, folds: int = 10) -> dict:
    """10-fold stratified cross-validation on the TRAINING set only, so the
    held-out test set stays untouched. Reports mean +/- std per metric."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy", "precision": "precision", "recall": "recall",
        "f1": "f1", "roc_auc": "roc_auc",
    }
    out = {}
    for name, model in candidates.items():
        res = cross_validate(clone(model), X_train, y_train, cv=skf, scoring=scoring, n_jobs=1)
        out[name] = {
            m: {
                "mean": round(float(np.mean(res[f"test_{m}"])), 4),
                "std": round(float(np.std(res[f"test_{m}"])), 4),
            }
            for m in scoring
        }
        print(f"[10-fold CV] {name}: AUC {out[name]['roc_auc']['mean']:.3f} +/- {out[name]['roc_auc']['std']:.3f}, "
              f"acc {out[name]['accuracy']['mean']:.3f}")
    return out


def evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "brier_score": round(brier_score_loss(y_test, proba), 4),
    }


def find_best_threshold(model, X_train, y_train) -> float:
    """Youden's J statistic (maximizes sensitivity + specificity - 1),
    chosen via 5-fold cross-validated predictions on the TRAINING set
    only -- the test set is never touched for threshold selection, to
    avoid leaking test information into an operating-point choice.
    A plain 0.5 cutoff on a ~15%-prevalence target is not a meaningful
    operating point after calibration; this is the honest fix."""
    cv_proba = cross_val_predict(
        clone(model), X_train, y_train, cv=5, method="predict_proba"
    )[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, cv_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def compute_feature_importance(model, X_test, y_test) -> dict:
    """SHAP if available, else permutation importance. Either way the
    result is a plain {feature: importance} dict so the API and any
    downstream report can use it the same way regardless of method."""
    try:
        import shap
        explainer = shap.Explainer(model.predict_proba, X_test)
        shap_values = explainer(X_test[:200])  # cap for speed
        mean_abs = np.abs(shap_values.values[..., 1]).mean(axis=0)
        method = "shap"
    except Exception:
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10,
            random_state=RANDOM_STATE, scoring="roc_auc",
        )
        mean_abs = result.importances_mean
        method = "permutation_importance"

    importance = {
        col: round(float(val), 5)
        for col, val in sorted(
            zip(FEATURE_COLUMNS, mean_abs), key=lambda x: -x[1]
        )
    }
    return {"method": method, "importance": importance}


def main():
    X_train, X_test, y_train, y_test = get_train_test_split()

    candidates = get_candidate_models()
    comparison = {}
    fitted = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        comparison[name] = evaluate(model, X_test, y_test)
        print(f"{name}: {comparison[name]}")

    # 10-fold cross-validation on the training data (matches the paper's
    # evaluation protocol). The model is chosen by CROSS-VALIDATED ROC-AUC,
    # not by the test set, so the test set stays an unbiased final check.
    cv_results = cross_validate_models(candidates, X_train, y_train, folds=10)
    best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc"]["mean"])
    print(f"\nBest model by 10-fold CV ROC-AUC: {best_name}")

    # Calibrate the best model (Platt/sigmoid scaling via 5-fold CV).
    best_uncalibrated = clone(candidates[best_name])
    calibrated = CalibratedClassifierCV(best_uncalibrated, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)

    # A naive 0.5 cutoff is not a meaningful operating point once
    # probabilities are calibrated to a ~15%-prevalence target (it
    # collapses recall). Report both, but the threshold-optimal one
    # is what the API actually uses.
    calibrated_metrics_default = evaluate(calibrated, X_test, y_test, threshold=0.5)
    best_threshold = find_best_threshold(calibrated, X_train, y_train)
    calibrated_metrics_optimal = evaluate(calibrated, X_test, y_test, threshold=best_threshold)

    print(f"\nCalibrated {best_name} @ 0.5 threshold: {calibrated_metrics_default}")
    print(f"Calibrated {best_name} @ optimal threshold ({best_threshold:.3f}): {calibrated_metrics_optimal}")
    print(
        f"Brier score improvement: "
        f"{comparison[best_name]['brier_score']} -> {calibrated_metrics_optimal['brier_score']}"
    )

    # Explainability report, computed on the underlying (uncalibrated)
    # fitted model for interpretability of raw feature effects.
    importance_report = compute_feature_importance(fitted[best_name], X_test, y_test)
    print(f"\nTop 5 features ({importance_report['method']}):")
    for feat, val in list(importance_report["importance"].items())[:5]:
        print(f"  {feat}: {val}")

    # Save everything
    joblib.dump(calibrated, MODELS_DIR / "best_model.joblib")
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "feature_columns.joblib")
    joblib.dump(make_background(X_train), MODELS_DIR / "background.joblib")

    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(
            {
                "candidates": comparison,
                "cv_10_fold": cv_results,
                "best_model": best_name,
                "calibrated_metrics_default_threshold": calibrated_metrics_default,
                "calibrated_metrics_optimal_threshold": calibrated_metrics_optimal,
                "decision_threshold": best_threshold,
            },
            f, indent=2,
        )

    with open(MODELS_DIR / "decision_threshold.json", "w") as f:
        json.dump({"threshold": best_threshold}, f, indent=2)

    with open(MODELS_DIR / "feature_importance.json", "w") as f:
        json.dump(importance_report, f, indent=2)

    print(f"\nSaved: {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved: {MODELS_DIR / 'model_comparison.json'}")
    print(f"Saved: {MODELS_DIR / 'feature_importance.json'}")


if __name__ == "__main__":
    main()
