"""
Assignment-2 (COMP647) - Single File Submission

Student: Sungmin Lee (1163957)
Repository: Alex-Lee-1163957/COMP647Project_SungminLee

Purpose:
  Using a Kaggle dataset, perform the following tasks in a single Python file:
  1) Data Preprocessing with clear in-code comments
     - Cleaning the data
     - Missing data imputation (methods discussed in class)
     - Dealing with outliers
  2) Exploratory Data Analysis (EDA)
     - Investigate correlations and potential relationships among features
  3) Provide insightful comments related to the features within the code
  4) Briefly discuss potential research questions backed by EDA results

How to run:
  - Ensure `data/train.csv` is present
  - Install dependencies: `pip install -r requirements.txt`
  - Execute: `python Assignment-2.py`
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix



# Utility: plotting destination

def ensure_output_dirs(base_dir: str = "docs/plots") -> Path:
    """Ensure output directory for plots exists so figures can be saved for marking."""
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir



# 1) Data Loading & Basic Info

def load_train_data(data_dir: str = "data", filename: str = "train.csv") -> Optional[pd.DataFrame]:
    """Load training data from `data/train.csv`.

    Returns None if file is missing so caller can handle gracefully.
    """
    file_path = Path(data_dir) / filename
    if not file_path.exists():
        print(f"[ERROR] Data file not found: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        print(f"[INFO] Loaded training data from {file_path} (shape={df.shape})")
        return df
    except Exception as exc:
        print(f"[ERROR] Failed to read CSV: {exc}")
        return None


def basic_overview(data: pd.DataFrame, target_column: Optional[str]) -> None:
    """Print essential dataset information and target distribution summary.

    This mirrors the first EDA steps taught in the lab: info(), describe(),
    and distribution of the target variable.
    """
    print("\n=== BASIC DATA OVERVIEW ===")
    print(f"Shape: {data.shape}")
    print("\nData types and non-null counts:")
    print(data.info())
    print("\nStatistical summary (numerical):")
    print(data.describe())

    if target_column and target_column in data.columns:
        print(f"\nTarget '{target_column}' distribution:")
        print(data[target_column].value_counts())
        print("\nTarget percentages:")
        print((data[target_column].value_counts(normalize=True) * 100).round(2))



# 2) Data Preprocessing (with notes)

def handle_missing_values(data: pd.DataFrame, strategy: str = "auto") -> pd.DataFrame:
    """Impute missing values.

    - For categorical (object) features: use mode (most frequent) as robust default.
    - For numerical features: use median (robust to outliers and skew).
    - If strategy == 'drop', simply drop rows with any missing values (rarely ideal).
    """
    print("\n[Preprocess] Handling missing values (strategy=auto)")
    if data.isnull().sum().sum() == 0:
        print("No missing values detected.")
        return data.copy()

    processed = data.copy()
    if strategy == "drop":
        processed = processed.dropna()
        print(f"Rows remaining after drop: {len(processed)}")
        return processed

    for col in processed.columns:
        if processed[col].isnull().any():
            if processed[col].dtype == "object":
                mode_val = processed[col].mode(dropna=True)
                if len(mode_val) > 0:
                    processed[col] = processed[col].fillna(mode_val.iloc[0])
                print(f"  - {col}: filled missing with mode")
            else:
                median_val = processed[col].median()
                processed[col] = processed[col].fillna(median_val)
                print(f"  - {col}: filled missing with median={median_val}")
    # Insight (comment): Median/mode imputation minimizes information loss and reduces sensitivity to extremes
    # Human-style insight
    after_missing = processed.isnull().sum().sum()
    print(
        f"[Insight] Finished handling missing values (remaining: {after_missing}). "
        "Numerical features were imputed with the median and categorical features with the mode to reduce the impact of extremes and avoid data loss."
    )
    return processed


def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows to prevent biased learning and inflated counts."""
    print("\n[Preprocess] Removing duplicates")
    before = len(data)
    processed = data.drop_duplicates()
    removed = before - len(processed)
    print(f"Removed {removed} duplicate rows")
    # Insight (comment): Removing duplicates prevents overweighting repeated samples in analysis/modeling
    # Human-style insight
    if removed > 0:
        print("[Insight] Removed duplicate rows to prevent the same samples from overweighting the model.")
    else:
        print("[Insight] No duplicate rows found.")
    return processed


def detect_outliers_iqr(data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Detect outliers per numerical column using IQR rule (boxplot method).

    Returns a dict mapping column -> bounds and count for subsequent handling.
    """
    print("\n[Preprocess] Detecting outliers (IQR)")
    outlier_info: Dict[str, Dict[str, float]] = {}
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (data[col] < lower) | (data[col] > upper)
        count = int(mask.sum())
        outlier_info[col] = {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "count": count,
            "percentage": (count / len(data) * 100.0) if len(data) else 0.0,
        }
        print(f"  - {col}: {count} outliers ({outlier_info[col]['percentage']:.1f}%)")
    # Insight (comment): Identify columns with many outliers to decide capping/removal strategies
    # Human-style insight
    flagged = {c: info for c, info in outlier_info.items() if info["count"] > 0}
    if flagged:
        top = sorted(flagged.items(), key=lambda x: x[1]["percentage"], reverse=True)[:3]
        msg = ", ".join([f"{c}({v['percentage']:.1f}%)" for c, v in top])
        print(f"[Insight] Columns with relatively many outliers: {msg}. Applying capping to improve model stability.")
    else:
        print("[Insight] Few statistical outliers detected in numerical columns.")
    return outlier_info


def handle_outliers(data: pd.DataFrame, outlier_info: Dict[str, Dict[str, float]], strategy: str = "cap") -> pd.DataFrame:
    """Handle outliers according to chosen strategy:
    - 'cap': clip values to IQR bounds (keeps all rows; reduces extreme influence)
    - 'remove': remove rows outside bounds (may reduce dataset size)
    - 'keep': do nothing
    """
    print(f"\n[Preprocess] Handling outliers (strategy={strategy})")
    if strategy == "keep":
        return data.copy()

    processed = data.copy()
    for col, info in outlier_info.items():
        low = info["lower_bound"]
        high = info["upper_bound"]
        if info["count"] <= 0:
            continue
        if strategy == "cap":
            processed[col] = processed[col].clip(lower=low, upper=high)
        elif strategy == "remove":
            mask = (processed[col] >= low) & (processed[col] <= high)
            processed = processed.loc[mask]
    if strategy == "remove":
        print(f"  Final shape after removal: {processed.shape}")
    # Insight (comment): Capping at IQR bounds reduces extreme influence without dropping rows
    # Human-style insight
    if strategy == "cap":
        print("[Insight] Capped values at IQR bounds to reduce extreme influence, smoothing tails without losing samples.")
    return processed


def encode_categorical_variables(data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """Simple label encoding per object column (excluding target if provided).

    This preserves all rows while converting string categories to integers, which
    is sufficient for the EDA and simple modeling baselines.
    """
    print("\n[Preprocess] Encoding categorical variables (label mapping per column)")
    processed = data.copy()
    cat_cols = processed.select_dtypes(include=["object"]).columns
    if target_column in cat_cols:
        cat_cols = cat_cols.drop(target_column)

    for col in cat_cols:
        categories = processed[col].astype(str).unique()
        mapping = {cat: i + 1 for i, cat in enumerate(categories)}
        processed[col] = processed[col].map(mapping)
        print(f"  - {col}: {len(mapping)} categories encoded")
    # Insight (comment): Simple label mapping enables correlation and baseline models without heavy encoders
    # Human-style insight
    print("[Insight] Converted categorical variables using label mapping; this simple encoding is sufficient for correlations and baseline modeling.")
    return processed


def normalize_numerical_features(data: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """Standardize numerical columns to zero mean and unit variance.

    Normalization helps compare scales across variables and supports correlation
    analysis without scale-driven bias. Target column (if numeric) is excluded.
    """
    print("\n[Preprocess] Normalizing numerical features (z-score)")
    processed = data.copy()
    num_cols = processed.select_dtypes(include=[np.number]).columns
    if target_column in num_cols:
        num_cols = num_cols.drop(target_column)
    for col in num_cols:
        mean = processed[col].mean()
        std = processed[col].std()
        if std and std > 0:
            processed[col] = (processed[col] - mean) / std
            print(f"  - {col}: mean={mean:.2f}, std={std:.2f}")
        else:
            print(f"  - {col}: skipped (std=0)")
    # Insight (comment): Standardization removes scale effects for fair comparison among features
    # Human-style insight
    print("[Insight] Standardized numeric features so scale differences do not bias correlations or visual interpretation.")
    return processed


#
# 3) EDA - Visual and Quant Analyses

def create_histograms(data: pd.DataFrame, target_column: Optional[str], save_dir: Optional[Path] = None) -> None:
    """Plot histograms for numerical features to inspect distributions."""
    num_cols = data.select_dtypes(include=[np.number]).columns
    if target_column in num_cols:
        num_cols = num_cols.drop(target_column)
    if len(num_cols) == 0:
        print("[EDA] No numerical features for histograms.")
        return
    ax = data[num_cols].hist(bins=30, figsize=(15, 10))
    plt.suptitle("Distribution of Numerical Features", fontsize=16)
    plt.tight_layout()
    if save_dir is not None:
        out = save_dir / "histograms.png"
        plt.savefig(out, dpi=150)
        print(f"[EDA] Saved histograms to {out}")
    plt.show()
    # Insight (comment): Highly skewed distributions may benefit from log-transform before modeling
    # Human-style insight
    skew = data.select_dtypes(include=[np.number]).skew(numeric_only=True).sort_values(ascending=False)
    top_skew = skew.head(3)
    print("[Insight] Distribution summary (top skew):")
    for col, val in top_skew.items():
        print(f"  - {col}: skew={val:.2f}{' (log-transform candidate)' if val>1.0 else ''}")


def analyze_categorical_features(
    data: pd.DataFrame,
    target_column: Optional[str],
    save_dir: Optional[Path] = None,
    max_plots_per_page: int = 8,
) -> None:
    """Summarize top categories and visualize distributions for categorical features.

    To avoid figures being cut off, large numbers of categorical plots are
    paginated across multiple figures with constrained layout.
    """
    cat_cols = data.select_dtypes(include=["object"]).columns
    if target_column in cat_cols:
        cat_cols = cat_cols.drop(target_column)
    if len(cat_cols) == 0:
        print("[EDA] No categorical features.")
        return
    print(f"[EDA] Categorical features: {list(cat_cols)}")

    # Paginate plots
    pages = [cat_cols[i : i + max_plots_per_page] for i in range(0, len(cat_cols), max_plots_per_page)]
    for page_idx, page_cols in enumerate(pages, start=1):
        n = len(page_cols)
        n_cols = 2 if n > 1 else 1
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.6 * n_rows), constrained_layout=True)
        axes_arr = np.array(axes).reshape(-1) if n_rows * n_cols > 1 else np.array([axes])

        for i, col in enumerate(page_cols):
            vc = data[col].value_counts().head(10)
            axes_arr[i].bar(vc.index.astype(str), vc.values)
            axes_arr[i].set_title(f"{col} (Top 10)")
            axes_arr[i].set_ylabel("Count")
            axes_arr[i].tick_params(axis="x", rotation=35, labelsize=8)

        # Remove any unused axes
        for j in range(len(page_cols), len(axes_arr)):
            fig.delaxes(axes_arr[j])

        if save_dir is not None:
            out = save_dir / ("categorical_distributions_page_%02d.png" % page_idx)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"[EDA] Saved categorical distributions (page {page_idx}) to {out}")

        plt.show()
    # Human-style insight: class imbalance & rare categories
    for col in cat_cols:
        vc = data[col].value_counts(normalize=True)
        if not vc.empty:
            top_ratio = vc.iloc[0]
            if top_ratio > 0.9:
                # Insight (comment): Strong dominance of a single category indicates class imbalance risk
                print(f"[Insight] '{col}' is highly concentrated in a single value ({top_ratio*100:.1f}%) → potential class imbalance.")
        if (data[col].value_counts() == 1).sum() > 0:
            # Insight (comment): Rare categories (freq=1) can cause sparsity and leakage during CV splits
            print(f"[Insight] '{col}' contains rare categories (freq=1). Beware of sparsity and leakage across folds.")


def correlation_analysis(data: pd.DataFrame, target_column: Optional[str], save_dir: Optional[Path] = None) -> pd.DataFrame:
    """Compute and visualize correlation matrix for numerical features."""
    num_cols = data.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        print("[EDA] Need at least 2 numerical features for correlation analysis.")
        return pd.DataFrame()

    corr = data[num_cols].corr()
    print("\n[EDA] Correlation matrix (rounded):")
    print(corr.round(3))

    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Correlation Coefficient")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix Heatmap")
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    if save_dir is not None:
        out = save_dir / "correlation_heatmap.png"
        plt.savefig(out, dpi=150)
        print(f"[EDA] Saved correlation heatmap to {out}")
        # Save raw correlations for marker review
        corr_out = save_dir / "correlation_matrix.csv"
        corr.round(6).to_csv(corr_out)
        print(f"[EDA] Saved correlation matrix CSV to {corr_out}")
    plt.show()
    # Insight (comment): Features with highest |corr| to target are promising predictors
    # Human-style insight
    if target_column in corr.columns:
        tgt = corr[target_column].drop(labels=[target_column], errors="ignore").abs().sort_values(ascending=False)
        print("[Insight] Top 5 features most correlated with the target:")
        for feat, val in tgt.head(5).items():
            print(f"  - {feat}: |corr|={val:.3f}")
    # Multicollinearity hint
    high_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if abs(corr.iloc[i, j]) > 0.85:
                high_pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    if high_pairs:
        pair_msg = ", ".join([f"{a}-{b}({v:.2f})" for a,b,v in high_pairs[:3]])
        # Insight (comment): Very high inter-feature correlation suggests potential multicollinearity
        print(f"[Insight] Strongly correlated feature pairs detected → potential multicollinearity: {pair_msg}")
    return corr


def scatter_matrix_analysis(data: pd.DataFrame, target_column: Optional[str], save_dir: Optional[Path] = None) -> None:
    """Create a scatter matrix for up to 5 numerical features (incl. target if numeric)."""
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    features: List[str]
    if target_column in num_cols:
        features = [target_column] + [c for c in num_cols if c != target_column][:4]
    else:
        features = num_cols[:5]
    if len(features) < 2:
        print("[EDA] Not enough numerical features for scatter matrix.")
        return
    print(f"[EDA] Scatter matrix features: {features}")
    scatter_matrix(data[features], figsize=(12, 10), alpha=0.6)
    plt.suptitle("Scatter Matrix of Key Features", fontsize=16)
    plt.tight_layout()
    if save_dir is not None:
        out = save_dir / "scatter_matrix.png"
        plt.savefig(out, dpi=150)
        print(f"[EDA] Saved scatter matrix to {out}")
    plt.show()


def feature_vs_target_analysis(data: pd.DataFrame, target_column: str) -> None:
    """Textual analysis linking features to the target variable.

    - If target is numeric: rank features by correlation magnitude.
    - If target is categorical: summarize numerical features grouped by target.
    """
    if target_column not in data.columns:
        print(f"[EDA] Target column '{target_column}' not found.")
        return
    print(f"\n[EDA] Feature vs Target analysis for '{target_column}'")
    if pd.api.types.is_numeric_dtype(data[target_column]):
        num_cols = data.select_dtypes(include=[np.number]).columns
        num_cols = [c for c in num_cols if c != target_column]
        correlations: List[Tuple[str, float]] = []
        for col in num_cols:
            correlations.append((col, data[col].corr(data[target_column])))
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        print("Top correlations with target:")
        for feat, corr in correlations[:10]:
            print(f"  {feat}: {corr:.3f}")
    else:
        # Group numerical features by categorical target to view mean/std/count
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        for feat in num_cols[:5]:
            grouped = data.groupby(target_column)[feat].agg(["mean", "std", "count"]).round(2)
            print(f"\n{feat} by {target_column}:")
            print(grouped)



# 4) Research Questions (grounded in EDA outcomes)

def print_research_questions_summary(target_column: str, corr: Optional[pd.DataFrame]) -> None:
    """Print concise research questions backed by EDA observations."""
    print("\n=== Research Questions (Backed by EDA) ===")
    print("1) Customer Segmentation: Do vehicle features and demographics form distinct clusters?")
    print("   - Evidence: Distribution patterns and scatter matrix groupings.")
    print("2) Risk Profiling: Which features show strongest association with the target?")
    if corr is not None and not corr.empty and target_column in corr.columns:
        abs_corr = corr[target_column].drop(labels=[target_column], errors="ignore").abs().sort_values(ascending=False)
        print("   - Evidence: Top correlated features => ")
        for feat in abs_corr.index[:5]:
            print(f"     * {feat}: |corr|={abs_corr.loc[feat]:.3f}")
    print("3) Premium/Policy Strategy: How do safety or vehicle specs relate to claim propensity?")
    print("   - Evidence: Correlation heatmap and feature-vs-target summaries.")


# ------------------------------
# Insight markdown writer
# ------------------------------
def write_insights_markdown(
    output_dir: Path,
    dataset_shape: Tuple[int, int],
    target_column: str,
    corr: Optional[pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Assignment-2 Insights\n")
    md.append(f"- Dataset shape: {dataset_shape[0]} rows × {dataset_shape[1]} columns\n")
    md.append(f"- Target: `{target_column}`\n")
    md.append("\n## Key Takeaways\n")
    md.append("- Missing values handled (mode/median); outliers capped via IQR.\n")
    md.append("- Categorical imbalance and rare categories flagged for caution.\n")
    if corr is not None and not corr.empty and target_column in corr.columns:
        top = corr[target_column].drop(labels=[target_column], errors="ignore").abs().sort_values(ascending=False).head(5)
        md.append("- Top correlations with target:\n")
        for feat, val in top.items():
            md.append(f"  - {feat}: |corr|={val:.3f}\n")
    (output_dir / "insights.md").write_text("".join(md), encoding="utf-8")
    print(f"[INSIGHTS] Wrote markdown to {(output_dir / 'insights.md').as_posix()}")



# ------------------------------
# Business insights (integrated)
# ------------------------------
def business_insights(data: pd.DataFrame, target_column: str) -> None:
    """Lightweight, business-oriented summaries inspired by src/business_insights.py.

    Prints high-level, human-readable takeaways useful for a non-technical audience.
    """
    print("\n=== Business Insights (Integrated) ===")

    # Overall target rate (if numeric target like is_claim)
    if target_column in data.columns and pd.api.types.is_numeric_dtype(data[target_column]):
        try:
            overall_rate = float(data[target_column].mean()) * 100.0
            print(f"- Overall target mean: {overall_rate:.2f}%")
        except Exception:
            pass

    # Age group example (only if an 'age' like column exists)
    for age_col in ["age", "policyholder_age", "driver_age"]:
        if age_col in data.columns and pd.api.types.is_numeric_dtype(data[age_col]):
            try:
                bins = [0, 25, 35, 50, 65, 120]
                labels = ["Under25", "25-34", "35-49", "50-64", "65+"]
                age_group = pd.cut(data[age_col], bins=bins, labels=labels)
                summary = pd.DataFrame({
                    "Customers": age_group.value_counts().reindex(labels, fill_value=0)
                })
                if target_column in data.columns and pd.api.types.is_numeric_dtype(data[target_column]):
                    summary["TargetMean"] = data.groupby(age_group)[target_column].mean().reindex(labels)
                print(f"- {age_col} grouped view:")
                print(summary.round(3))
            except Exception:
                pass
            break

    # Simple vehicle-related segmentation example
    veh_cols = [c for c in data.columns if any(k in c.lower() for k in ["vehicle", "car", "model", "make", "segment"])]
    if veh_cols:
        vc = veh_cols[0]
        top = data[vc].value_counts().head(5)
        print(f"- Top categories in '{vc}':")
        print(top)

    print("[Business Insight] Identifying high-risk (or priority) segments early enables proactive guidance and pre-prepared documents, reducing handling time and improving satisfaction.")


# Main entry point
def autodetect_target_column(df: pd.DataFrame) -> str:
    """Heuristic to pick the likely target column by keyword; fallback to last column."""
    keywords = ["is_claim", "claim", "target", "label", "class", "outcome"]
    for col in df.columns:
        lower = col.lower()
        if any(k in lower for k in keywords):
            return col
    return df.columns[-1]


def main() -> Optional[pd.DataFrame]:
    print("COMP647 Assignment-2: Single-File Analysis Pipeline")
    print("Student: Sungmin Lee (1163957)")
    print("=" * 60)

    # Step 0: Load data
    data = load_train_data()
    if data is None:
        return None

    # Identify target column (explicit if known; otherwise heuristic)
    target_column = autodetect_target_column(data)
    print(f"[INFO] Using '{target_column}' as target variable")

    # Step 1: Data Preprocessing
    # - Missing imputation (mode/median)
    # - Remove duplicates
    # - Detect and handle outliers (IQR + capping)
    clean = handle_missing_values(data, strategy="auto")
    clean = remove_duplicates(clean)
    outlier_info = detect_outliers_iqr(clean)
    clean = handle_outliers(clean, outlier_info, strategy="cap")

    # Optional: encode categoricals and normalize numericals for downstream EDA
    # (Keeps target untouched so its semantics are preserved.)
    encoded = encode_categorical_variables(clean, target_column=target_column)
    processed = normalize_numerical_features(encoded, target_column=target_column)

    # Step 2: EDA - Visual and quantitative analyses
    out_dir = ensure_output_dirs()
    basic_overview(processed, target_column)
    create_histograms(processed, target_column, save_dir=out_dir)
    analyze_categorical_features(clean, target_column, save_dir=out_dir)  # use pre-encoding for readability
    corr = correlation_analysis(processed, target_column, save_dir=out_dir)
    scatter_matrix_analysis(processed, target_column, save_dir=out_dir)
    feature_vs_target_analysis(processed, target_column)

    # Step 3: Summary & Research questions (with references to EDA outputs)
    print("\n=== Pipeline Summary ===")
    print(f"Original shape: {data.shape} -> After preprocessing: {processed.shape}")
    num_cols = processed.select_dtypes(include=[np.number]).shape[1]
    cat_cols = processed.select_dtypes(include=["object"]).shape[1]
    print(f"Numerical features: {num_cols}, Categorical features: {cat_cols}")
    print_research_questions_summary(target_column, corr)

    # Write concise insights markdown
    write_insights_markdown(Path("docs"), processed.shape, target_column, corr)

    # Run integrated business insights
    business_insights(processed, target_column)

    print("\nNext Steps: Feature engineering and predictive modeling (beyond scope of A2)")
    return processed


if __name__ == "__main__":
    _ = main()


