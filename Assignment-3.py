"""
Assignment-3 (COMP647)



from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.feature_selection import SelectKBest, chi2

import matplotlib.pyplot as plt
import seaborn as sns


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSVs (same format as Assignment-2).

    We only use train.csv for modeling; test.csv is kept for potential submission.
    """
    root = Path(__file__).resolve().parent
    train = pd.read_csv(root / "data" / "train.csv")
    test = pd.read_csv(root / "data" / "test.csv")
    return train, test


def simple_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create light features in a class-friendly way.

    - Parse numeric tokens from text specs (max_power, max_torque)
    - Create a couple of ratios that often matter in insurance/auto
    - Optional: light binning (kept minimal)
    """
    out = df.copy()

    # Parse numeric from 'max_power' like '88.50bhp@6000rpm' -> 88.50
    if 'max_power' in out.columns:
        out['max_power_num'] = (
            out['max_power'].astype(str).str.extract(r'([0-9]+\.?[0-9]*)')[0].astype(float)
        )
    # Parse numeric from 'max_torque' like '250Nm@2750rpm' -> 250
    if 'max_torque' in out.columns:
        out['max_torque_num'] = (
            out['max_torque'].astype(str).str.extract(r'([0-9]+\.?[0-9]*)')[0].astype(float)
        )

    # Basic ratios (these are simple, easy to explain features)
    if {'max_power_num', 'gross_weight'}.issubset(out.columns):
        out['power_to_weight'] = out['max_power_num'] / (out['gross_weight'].replace(0, np.nan))
    if {'max_torque_num', 'displacement'}.issubset(out.columns):
        out['torque_per_litre'] = out['max_torque_num'] / (out['displacement'].replace(0, np.nan))

    return out


def build_preprocessor(df: pd.DataFrame, target: str) -> ColumnTransformer:
    """Build a ColumnTransformer for encoding/scaling.

    - One-Hot for categorical (low/medium cardinality)
    - Standardize numeric ONLY for linear model; trees are robust, but shared pipeline is OK
    """
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target in cat_cols:
        cat_cols.remove(target)
    # Drop identifier-like columns that carry no predictive signal and can leak
    if 'policy_id' in cat_cols:
        cat_cols.remove('policy_id')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)

    # Minimal one-hot + MinMax scaling to keep features non-negative
    # (so that SelectKBest with chi2 is applicable per slides)
    pre = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', MinMaxScaler(), num_cols),
        ]
    )
    return pre


def cv_eval(model, X, y, cv=5) -> Dict[str, float]:
    """Simple CV with Accuracy/F1/ROC-AUC (macro where needed).

    Note: ROC-AUC requires probabilistic scores; tree/logreg provide predict_proba.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        # Use iloc for DataFrame; fall back to numpy slicing otherwise
        if hasattr(X, 'iloc'):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        accs.append(accuracy_score(y_va, y_pred))
        f1s.append(f1_score(y_va, y_pred, zero_division=0))
        # Probabilities for AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, y_proba))
    out = {
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
    }
    if aucs:
        out.update({'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs))})
    return out


def get_selected_feature_names(pre: ColumnTransformer, selector: SelectKBest, df: pd.DataFrame, target: str) -> np.ndarray:
    """Resolve selected feature names after ColumnTransformer + SelectKBest.

    We first get all transformed feature names, then apply the selector mask.
    """
    # All transformed feature names
    try:
        all_names = pre.get_feature_names_out()
    except Exception:
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if target in cat_cols:
            cat_cols.remove(target)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)
        all_names = np.array(cat_cols + num_cols, dtype=object)
    # Apply selector mask if present
    if hasattr(selector, 'get_support'):
        mask = selector.get_support()
        if mask is not None and mask.shape[0] == all_names.shape[0]:
            return all_names[mask]
    return all_names


def evaluate_holdout_and_xai(pipe_log, pipe_rf, X: pd.DataFrame, y: np.ndarray, feature_df: pd.DataFrame, out_dir: Path) -> Dict[str, object]:
    """Hold-out split for metrics + basic XAI (coeffs/importances/permutation).

    - Produces confusion matrix and ROC curve PNGs
    - Prints top coefficients (LogReg) and feature_importances_ (RF)
    - Runs permutation importance on the RF for model-agnostic view
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Fit models
    pipe_log.fit(X_tr, y_tr)
    pipe_rf.fit(X_tr, y_tr)

    # Evaluate (LogReg as baseline)
    y_pred = pipe_log.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    y_prob = pipe_log.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    print("\n[HOLD-OUT] Logistic Regression")
    print({"accuracy": acc, "f1": f1, "roc_auc": auc})

    # Confusion Matrix
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cbar=False)
    plt.title('Confusion Matrix - Logistic')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_path = out_dir / 'cm_logistic.png'
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[SAVED] {cm_path}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f'LogReg AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve - Logistic')
    plt.legend()
    plt.tight_layout()
    roc_path = out_dir / 'roc_logistic.png'
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"[SAVED] {roc_path}")

    # XAI: coefficients (standardized space)
    pre = pipe_log.named_steps['pre']
    sel = pipe_log.named_steps['sel']
    feat_names = get_selected_feature_names(pre, sel, feature_df, 'is_claim')
    coef = pipe_log.named_steps['clf'].coef_[0]
    coef_df = pd.DataFrame({"feature": feat_names, "coef": coef}).sort_values('coef', key=np.abs, ascending=False)
    print("\n[Logistic] Top coefficients (|coef|):")
    print(coef_df.head(15).to_string(index=False))

    # RF evaluation & importances
    y_pred_rf = pipe_rf.predict(X_te)
    acc_rf = accuracy_score(y_te, y_pred_rf)
    f1_rf = f1_score(y_te, y_pred_rf, zero_division=0)
    y_prob_rf = pipe_rf.predict_proba(X_te)[:, 1]
    auc_rf = roc_auc_score(y_te, y_prob_rf)
    print("\n[HOLD-OUT] Random Forest")
    print({"accuracy": acc_rf, "f1": f1_rf, "roc_auc": auc_rf})

    # Feature importances
    # We need transformed feature names after pre
    rf_importances = pipe_rf.named_steps['clf'].feature_importances_
    sel_rf = pipe_rf.named_steps['sel']
    feat_names_rf = get_selected_feature_names(pipe_rf.named_steps['pre'], sel_rf, feature_df, 'is_claim')
    imp_df = pd.DataFrame({"feature": feat_names_rf, "importance": rf_importances}).sort_values('importance', ascending=False)
    print("\n[RandomForest] Top importances:")
    print(imp_df.head(15).to_string(index=False))

    # Permutation importance (agnostic, on hold-out)
    # Rationale: Unlike tree importances (which can be biased by splits),
    # permutation importance measures how much the score drops when we
    # randomly shuffle one feature at a time. This answers a practical
    # question: “If this feature were noise, how much worse would the model
    # perform?” We compute it on the hold-out set so it reflects generalization.
    print("\n[Permutation Importance] (RF on hold-out)")
    # Transform X_te once for speed
    # Apply the exact same transforms as the RF pipeline: pre -> sel
    X_te_transformed = pipe_rf.named_steps['pre'].transform(X_te)
    X_te_transformed = pipe_rf.named_steps['sel'].transform(X_te_transformed)
    rf = pipe_rf.named_steps['clf']
    # permutation_importance expects array-like. Convert only if sparse.
    if hasattr(X_te_transformed, 'toarray'):
        X_perm = X_te_transformed.toarray()
    else:
        X_perm = X_te_transformed
    perm = permutation_importance(rf, X_perm, y_te, n_repeats=5, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({"feature": feat_names_rf, "importance": perm.importances_mean})\
        .sort_values('importance', ascending=False)
    print(perm_df.head(15).to_string(index=False))

    # PDP for top features (on transformed space using RF)
    # Rationale: A PDP shows the average effect of changing a single feature
    # while marginalizing the rest. Here we only plot the top-2 RF features to
    # keep the report short. Values are in the transformed space (after
    # pre+sel), which is sufficient to capture monotonic or threshold patterns.
    try:
        top_idx = np.argsort(rf_importances)[::-1][:2]
        for rank, idx in enumerate(top_idx, start=1):
            plt.figure(figsize=(4, 4))
            PartialDependenceDisplay.from_estimator(
                rf,
                X_perm,  # transformed features after pre+sel
                features=[idx],
                kind='average'
            )
            pdp_path = out_dir / f'pdp_rf_top{rank}.png'
            plt.tight_layout()
            plt.savefig(pdp_path, dpi=150)
            plt.close()
            print(f"[SAVED] {pdp_path}")
    except Exception as e:
        print(f"[WARN] PDP generation skipped: {e}")

    # Return a compact summary for reporting
    return {
        "logistic": {"accuracy": acc, "f1": f1, "roc_auc": auc, "cm": cm, "cm_path": cm_path, "roc_path": roc_path},
        "random_forest": {"accuracy": acc_rf, "f1": f1_rf, "roc_auc": auc_rf}
    }


def write_metrics_summary(out_file: Path, y: np.ndarray, cv_log: Dict[str, float], cv_dt: Dict[str, float], cv_rf: Dict[str, float], holdout: Dict[str, object]) -> None:
    """Write a short, human-friendly justification of metrics and results.

    Rationale (slides):
    - Accuracy alone can be misleading under imbalance, so we also report F1.
    - ROC-AUC summarizes ranking ability and is standard for binary problems.
    - We show 5-fold CV (mean±std) for stability and a single hold-out for a
      realistic final check.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pos_rate = float(np.mean(y))
    with out_file.open("w", encoding="utf-8") as f:
        f.write("Assignment-3: Metric Summary\n")
        f.write("============================\n\n")
        f.write(f"Class imbalance (positive rate): {pos_rate:.4f}\n\n")
        f.write("Why these metrics:\n")
        f.write("- Accuracy is easy to read but can hide minority errors.\n")
        f.write("- F1 balances precision/recall; good when positives are rare.\n")
        f.write("- ROC-AUC reflects ranking quality; insensitive to threshold.\n\n")
        def dump_cv(name, cv):
            f.write(f"{name} (5-fold CV): acc={cv.get('acc_mean'):.3f}±{cv.get('acc_std'):.3f}, ")
            f.write(f"f1={cv.get('f1_mean'):.3f}±{cv.get('f1_std'):.3f}, ")
            auc_mean = cv.get('auc_mean')
            auc_std = cv.get('auc_std')
            if auc_mean is not None:
                f.write(f"auc={auc_mean:.3f}±{auc_std:.3f}")
            f.write("\n")
        dump_cv("Logistic", cv_log)
        dump_cv("DecisionTree", cv_dt)
        dump_cv("RandomForest", cv_rf)
        f.write("\nHold-out (Logistic):\n")
        lh = holdout["logistic"]
        f.write(f"acc={lh['accuracy']:.3f}, f1={lh['f1']:.3f}, auc={lh['roc_auc']:.3f}\n")
        f.write(f"Confusion matrix path: {lh['cm_path']}\n")
        f.write(f"ROC curve path: {lh['roc_path']}\n")
        f.write("\nHow we avoid over/underfitting (per slides):\n")
        f.write("- Cross‑validation: 5‑fold CV gives a more stable estimate than a single split.\n")
        f.write("- Feature selection: SelectKBest(chi2, k=30) keeps only salient signals and reduces variance.\n")
        f.write("- Regularization / constraints:\n")
        f.write("  * Logistic: default L2 (max_iter with convergence check).\n")
        f.write("  * DecisionTree: max_depth=8, min_samples_leaf=50 to curb memorization.\n")
        f.write("  * RandomForest: capped depth (12) and reasonable tree count (200).\n")
        f.write("- Class imbalance: class_weight='balanced' to avoid over‑predicting the majority.\n")
        f.write("- Hold‑out sanity check: we report a final CM/ROC on an unseen split.\n")
        f.write("\nNotes:\n")
        f.write("- We purposely keep models compact and comments explicit for grading/interpretability.\n")
        f.write("- Final choice should balance interpretability (LogReg/Tree) and stability (RF).\n")


def main() -> None:
    warnings.filterwarnings('ignore')
    print("COMP647 Assignment-3 - Modeling & XAI (class scope)")

    train, _ = load_data()
    target = 'is_claim'
    assert target in train.columns, "Expected 'is_claim' in train.csv"

    # --- Quick profiling (single-file, human-friendly) ---
    docs_dir = Path(__file__).resolve().parent / 'docs'
    docs_dir.mkdir(parents=True, exist_ok=True)
    profile_path = docs_dir / 'profile_summary.txt'
    with open(profile_path, 'w', encoding='utf-8') as f:
        f.write(f"Shape: {train.shape}\n")
        if target in train.columns:
            vc = train[target].value_counts()
            pct = (vc / len(train) * 100).round(3)
            f.write("Target distribution (is_claim):\n")
            for k in vc.index:
                f.write(f"  {k}: {vc[k]} ({pct[k]}%)\n")
        # Missing
        miss = train.isnull().sum()
        miss = miss[miss > 0].sort_values(ascending=False)
        f.write("\nMissing values (>0):\n")
        if miss.empty:
            f.write("  None\n")
        else:
            for col, val in miss.items():
                f.write(f"  {col}: {val}\n")
        # Cardinality
        cat_cols = train.select_dtypes(include=['object']).columns.tolist()
        f.write("\nCategorical cardinality:\n")
        for col in cat_cols:
            f.write(f"  {col}: {train[col].nunique()} unique\n")
    print(f"[PROFILE] Wrote {profile_path}")

    # 1) Feature engineering (light, per slides)
    fe = simple_feature_engineering(train)

    # 2) Preprocessor: One-Hot + Standardize (for linear model)
    pre = build_preprocessor(fe, target)

    X = fe.drop(columns=[target])
    y = fe[target].astype(int).values

    # 3) Models (keep them simple)
    # Feature selection (per slides):
    # - I intentionally use SelectKBest with chi2 here because it is simple,
    #   fast, and was covered in class. It also plays nicely with one‑hot
    #   encoded categoricals as long as features are non‑negative, which is
    #   why the numeric part uses MinMax scaling. The goal is not to squeeze
    #   every last drop of performance, but to keep the model small and easier
    #   to reason about when we write up XAI.
    k_best = 30  # small on purpose: keeps the model compact and reduces noise

    # Logistic Regression — baseline linear classifier.
    # Why: gives calibrated probabilities; coefficients are easy to explain;
    # class_weight='balanced' is a one‑line way to acknowledge imbalance.
    logreg = Pipeline([
        ('pre', pre),
        ('sel', SelectKBest(score_func=chi2, k=k_best)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])

    # Decision Tree — simple non‑linear model.
    # Why: captures interactions without manual feature crosses; the tree
    # depth/leaf limits act as a very visible “overfitting brake”.
    dtree = Pipeline([
        ('pre', pre),
        ('sel', SelectKBest(score_func=chi2, k=k_best)),
        ('clf', DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight='balanced')),
    ])

    # Random Forest — bagged trees for a stronger but still interpretable model.
    # Why: reduces variance compared to a single tree; feature_importances_
    # gives a quick global explanation. I keep depth modest and the number of
    # trees reasonable so training remains predictable on a laptop.
    rforest = Pipeline([
        ('pre', pre),
        ('sel', SelectKBest(score_func=chi2, k=k_best)),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42)),
    ])

    # 4) Cross-validated scores
    # We report Accuracy + F1 + ROC-AUC because of imbalance, per slides.
    print("\n[CV] Logistic Regression (5-fold)")
    res_log = cv_eval(logreg, X, y, cv=5)
    print(res_log)

    print("\n[CV] Decision Tree (5-fold)")
    res_dt = cv_eval(dtree, X, y, cv=5)
    print(res_dt)

    print("\n[CV] Random Forest (5-fold)")
    res_rf = cv_eval(rforest, X, y, cv=5)
    print(res_rf)

    # Hold-out evaluation + XAI (saved under docs/plots)
    out_dir = Path(__file__).resolve().parent / 'docs' / 'plots'
    hold = evaluate_holdout_and_xai(
        pipe_log=Pipeline([('pre', pre), ('sel', SelectKBest(score_func=chi2, k=k_best)), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))]),
        pipe_rf=Pipeline([('pre', pre), ('sel', SelectKBest(score_func=chi2, k=k_best)), ('clf', RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42))]),
        X=X,
        y=y,
        feature_df=fe,
        out_dir=out_dir,
    )

    # Write a short, plain-English metric summary for the report
    write_metrics_summary(
        out_file=Path(__file__).resolve().parent / 'docs' / 'metrics_summary.txt',
        y=y,
        cv_log=res_log,
        cv_dt=res_dt,
        cv_rf=res_rf,
        holdout=hold,
    )
    print("[SAVED] docs/metrics_summary.txt")


if __name__ == "__main__":
    main()


