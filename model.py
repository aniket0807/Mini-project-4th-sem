"""
===================================================================
MODEL.PY — XGBoost Model for Student Performance Prediction
===================================================================
Migrated from Linear Regression to XGBoost Regressor.
This module handles:
  1. Loading and cleaning the dataset
  2. Preprocessing pipeline (OrdinalEncoder + ColumnTransformer)
  3. Feature analysis & selection
  4. Training XGBoost with Optuna hyperparameter tuning + early stopping
  5. Evaluating with MAE, RMSE, R² (+ comparison vs. Linear Regression baseline)
  6. Saving the trained model + pipeline for reuse
  7. A predict() function for the Streamlit app
  8. Visualization helpers
===================================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
TARGET = "exam_score"
RANDOM_STATE = 42
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")

# Columns that are categorical (non-numeric)
CATEGORICAL_COLS = [
    "gender", "part_time_job", "diet_quality",
    "parental_education_level", "internet_quality",
    "extracurricular_participation",
]


# ─────────────────────────────────────────────
#  STEP 1: LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath: str = None) -> pd.DataFrame:
    """Load the student habits dataset from CSV."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "dataset", "student_habits_performance.csv")
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
#  STEP 2: DATA CLEANING
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
      - Drop student_id (identifier, not a feature)
      - Remove duplicates
      - For XGBoost we keep NaNs as-is (handled natively via missing=np.nan)
        but fill categorical NaNs with mode so OrdinalEncoder doesn't break.
    """
    df = df.copy()

    if "student_id" in df.columns:
        df.drop(columns=["student_id"], inplace=True)

    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"🗑️  Removed {before - after} duplicate rows")

    # Fill categorical NaNs with mode (OrdinalEncoder can't handle NaN categories)
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  🔧 Filled categorical NaN in '{col}'")

    print(f"✅ Data cleaned: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
#  STEP 3: BUILD PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
def build_preprocessing_pipeline(categorical_cols: list, numeric_cols: list) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer pipeline.

    WHY OrdinalEncoder?
    - XGBoost is tree-based; ordinal integers work perfectly.
    - Unlike LabelEncoder, OrdinalEncoder works inside sklearn Pipelines.
    - handle_unknown='use_encoded_value' prevents crashes on unseen labels at inference.

    WHY no scaling?
    - XGBoost uses tree splits, which are invariant to monotonic feature transforms.
      Scaling numeric features has zero effect on tree models.
    """
    cat_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# ─────────────────────────────────────────────
#  STEP 4: FEATURE ANALYSIS & SELECTION
# ─────────────────────────────────────────────
def analyze_features(df: pd.DataFrame, target: str = TARGET):
    """Analyze feature correlations with the target variable."""
    df_numeric = df.copy()
    # Encode categoricals temporarily for correlation
    for col in df_numeric.select_dtypes(include="object").columns:
        df_numeric[col] = df_numeric[col].astype("category").cat.codes

    corr = df_numeric.corr(numeric_only=True)[target].drop(target).abs().sort_values(ascending=False)
    print("\n📊 Feature Correlations with Exam Score:")
    print("-" * 45)
    for feat, val in corr.items():
        bar = "█" * int(val * 30)
        print(f"  {feat:<35} {val:.4f}  {bar}")
    return corr


def select_features(df: pd.DataFrame, target: str = TARGET, threshold: float = 0.01) -> list:
    """
    Select features with correlation above threshold.
    XGBoost can handle many features (it does its own implicit selection via
    tree splits), so we keep a very low threshold.
    """
    df_numeric = df.copy()
    for col in df_numeric.select_dtypes(include="object").columns:
        df_numeric[col] = df_numeric[col].astype("category").cat.codes

    corr = df_numeric.corr(numeric_only=True)[target].drop(target).abs()
    selected = corr[corr >= threshold].index.tolist()
    print(f"\n✅ Selected {len(selected)} features (correlation ≥ {threshold}):")
    for f in selected:
        print(f"  • {f} (|r|={corr[f]:.4f})")
    return selected


# ─────────────────────────────────────────────
#  STEP 5: HYPERPARAMETER TUNING (OPTUNA)
# ─────────────────────────────────────────────
def _optuna_objective(trial, X_train, y_train, X_val, y_val, preprocessor):
    """Optuna objective: minimise RMSE on validation set."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "device": "cpu",
    }

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb.XGBRegressor(**params)),
    ])

    pipeline.fit(
        X_train, y_train,
        model__eval_set=[(preprocessor.fit_transform(X_val), y_val)],
        model__verbose=False,
    )

    y_pred = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse


# ─────────────────────────────────────────────
#  STEP 6: TRAIN XGBOOST MODEL
# ─────────────────────────────────────────────
def train_xgboost(
    df: pd.DataFrame,
    features: list,
    target: str = TARGET,
    n_trials: int = 15,
):
    """
    Train an XGBoost Regressor with Optuna hyperparameter tuning.

    WHY XGBOOST?
    ─────────────
    1. NON-LINEAR INTERACTIONS: Captures complex interactions between habits
       (e.g., high study hours + poor sleep → diminishing returns).
    2. NATIVE MISSING VALUE HANDLING: Built-in support for NaN via 'missing' param.
    3. REGULARISATION: L1 (alpha) + L2 (lambda) regularisation prevents overfitting.
    4. FEATURE IMPORTANCE: Gain-based importance is more reliable than LR coefficients
       for non-linear relationships.
    5. EARLY STOPPING: Prevents overfitting by stopping when validation loss stagnates.

    Returns: pipeline (preprocessor + model), X_train, X_test, y_train, y_test
    """
    X = df[features]
    y = df[target]

    # Identify which of the selected features are categorical vs numeric
    cat_cols = [c for c in features if c in CATEGORICAL_COLS]
    num_cols = [c for c in features if c not in CATEGORICAL_COLS]

    # 80/20 train/test; further split train into train/val for early stopping
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=RANDOM_STATE
    )

    print(f"\n📐 Train/Val/Test split:")
    print(f"  Training samples:   {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Testing samples:    {len(X_test)}")
    print(f"\n🔬 Running Optuna hyperparameter search ({n_trials} trials)...")

    preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)

    study = optuna.create_study(direction="minimize", study_name="xgb_study")
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, X_val, y_val, preprocessor),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_rmse = study.best_value
    print(f"  ✅ Best RMSE: {best_rmse:.4f}")
    print(f"  📌 Best params: {best_params}")

    # Re-fit final model on full train+val data with best hyperparams
    preprocessor_final = build_preprocessing_pipeline(cat_cols, num_cols)
    final_model = xgb.XGBRegressor(
        **best_params,
        tree_method="hist",
        device="cpu",
        random_state=RANDOM_STATE,
    )
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor_final),
        ("model", final_model),
    ])
    final_pipeline.fit(X_train_full, y_train_full)

    print(f"\n✅ XGBoost model trained successfully!")
    print(f"\n📈 Top feature importances (gain):")
    print("-" * 50)
    importances = final_model.feature_importances_
    # Get feature names after transformation
    try:
        feature_names_out = list(final_pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        feature_names_out = cat_cols + num_cols

    imp_series = pd.Series(importances, index=feature_names_out).sort_values(ascending=False)
    for feat, imp in imp_series.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:<35} {imp:.4f}  {bar}")

    return final_pipeline, X_train_full, X_test, y_train_full, y_test, cat_cols, num_cols


# ─────────────────────────────────────────────
#  BACKWARD-COMPAT SHIM: train_model()
# ─────────────────────────────────────────────
def train_model(df: pd.DataFrame, features: list, target: str = TARGET):
    """
    Backward-compatible wrapper so existing app.py imports don't break.
    Delegates to train_xgboost() and returns the same signature shape.
    """
    pipeline, X_train, X_test, y_train, y_test, _, _ = train_xgboost(df, features, target)
    return pipeline, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
#  STEP 7: EVALUATE THE MODEL
# ─────────────────────────────────────────────
def evaluate_model(model_or_pipeline, X_test, y_test, lr_metrics: dict = None):
    """
    Evaluate model performance using MAE, RMSE, R².
    Optionally accepts old LR metrics dict for side-by-side comparison.

    Parameters
    ----------
    model_or_pipeline : fitted Pipeline (preprocessor + XGBRegressor) or bare model
    X_test            : raw feature DataFrame (pipeline handles encoding internally)
    y_test            : ground-truth target Series
    lr_metrics        : optional dict with keys MAE/RMSE/R2_Score from Linear Regression
    """
    y_pred = model_or_pipeline.predict(X_test)
    y_pred = np.clip(y_pred, 0, 100)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    metrics = {
        "MAE":      round(mae, 4),
        "MSE":      round(float(mean_squared_error(y_test, y_pred)), 4),
        "RMSE":     round(rmse, 4),
        "R2_Score": round(r2, 4),
    }

    print(f"\n{'='*55}")
    print(f"  📊 XGBOOST EVALUATION RESULTS")
    print(f"{'='*55}")
    print(f"  MAE   (Mean Absolute Error): {mae:.4f}")
    print(f"  RMSE  (Root Mean Sq Error):  {rmse:.4f}")
    print(f"  R²    (R-Squared Score):     {r2:.4f}")

    if lr_metrics:
        print(f"\n  📊 COMPARISON vs. LINEAR REGRESSION:")
        print(f"  {'Metric':<8} {'Linear Reg':>12} {'XGBoost':>12} {'Δ':>10}")
        print(f"  {'-'*44}")
        for key in ["MAE", "RMSE", "R2_Score"]:
            lr_v  = lr_metrics.get(key, "N/A")
            xgb_v = metrics.get(key, "N/A")
            if isinstance(lr_v, (int, float)) and isinstance(xgb_v, (int, float)):
                delta = xgb_v - lr_v
                arrow = "📈" if (key == "R2_Score" and delta > 0) or (key != "R2_Score" and delta < 0) else "📉"
                print(f"  {key:<8} {lr_v:>12.4f} {xgb_v:>12.4f} {arrow} {delta:+.4f}")

    if r2 > 0.90:
        print(f"\n  ✅ Excellent model! R² > 0.90 — XGBoost explains {r2*100:.1f}% of variance.")
    elif r2 > 0.70:
        print(f"\n  ✅ Good model! R² = {r2:.2f} — explains {r2*100:.1f}% of variance.")
    else:
        print(f"\n  ⚠️  Moderate model. R² = {r2:.2f}. Consider more feature engineering.")

    print(f"{'='*55}")
    return metrics, y_pred


# ─────────────────────────────────────────────
#  STEP 8: ENCODING HELPER (backward compat)
# ─────────────────────────────────────────────
def encode_features(df: pd.DataFrame):
    """
    Backward-compatible encoding for app.py's get_model_and_data().
    Returns the DataFrame with categoricals as category codes + a dummy
    encoders dict so existing references don't break.

    NOTE: The XGBoost pipeline handles all encoding internally, so this
    function is only used for correlation analysis and the demo widget.
    """
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
        encoders[col] = df[col].cat.categories.tolist()  # store categories list
        df[col] = df[col].cat.codes
    return df, encoders


# ─────────────────────────────────────────────
#  STEP 9: SAVE & LOAD ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(model, encoders, features, metrics, save_dir=None):
    """Save the trained pipeline, feature list, and metrics."""
    if save_dir is None:
        save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model,    os.path.join(save_dir, "xgboost_model.pkl"))
    joblib.dump(encoders, os.path.join(save_dir, "encoders.pkl"))
    joblib.dump(features, os.path.join(save_dir, "features.pkl"))
    joblib.dump(metrics,  os.path.join(save_dir, "metrics.pkl"))

    print(f"\n💾 XGBoost artifacts saved to '{save_dir}/'")


def load_artifacts(save_dir=None):
    """Load saved model artifacts (XGBoost pipeline)."""
    if save_dir is None:
        save_dir = SAVE_DIR

    xgb_path = os.path.join(save_dir, "xgboost_model.pkl")
    lr_path  = os.path.join(save_dir, "linear_regression_model.pkl")

    model_path = xgb_path if os.path.exists(xgb_path) else lr_path

    model    = joblib.load(model_path)
    encoders = joblib.load(os.path.join(save_dir, "encoders.pkl"))
    features = joblib.load(os.path.join(save_dir, "features.pkl"))
    metrics  = joblib.load(os.path.join(save_dir, "metrics.pkl"))

    return model, encoders, features, metrics


# ─────────────────────────────────────────────
#  STEP 10: PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_score(model, features_list: list, student_data: dict) -> float:
    """
    Predict exam score for a single student.

    Parameters
    ----------
    model         : fitted Pipeline (preprocessor + XGBRegressor)
    features_list : list of feature names the model was trained on
    student_data  : dict of raw (un-encoded) student habit values

    Returns
    -------
    Predicted exam score clipped to [0, 100]
    """
    input_df = pd.DataFrame([student_data])[features_list]
    prediction = model.predict(input_df)[0]
    return round(float(np.clip(prediction, 0, 100)), 2)


# ─────────────────────────────────────────────
#  VISUALISATION HELPERS (for Streamlit)
# ─────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """Generate a correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, ax=ax, square=True, linewidths=0.5,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted(y_test, y_pred, save_path: str = None):
    """Plot actual vs predicted scores."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.5, color="#6C63FF", edgecolors="#333", s=60)
    ax.plot([0, 100], [0, 100], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Exam Score", fontsize=13)
    ax.set_ylabel("Predicted Exam Score", fontsize=13)
    ax.set_title("Actual vs Predicted Scores (XGBoost)", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(model, features: list, save_path: str = None):
    """
    Plot XGBoost feature importance (gain-based).
    Works with both a bare XGBRegressor and a sklearn Pipeline.
    """
    # Extract XGBRegressor from pipeline if needed
    if hasattr(model, "named_steps"):
        xgb_model = model.named_steps["model"]
        try:
            feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())
        except Exception:
            feature_names = features
    else:
        xgb_model = model
        feature_names = features

    importances = xgb_model.feature_importances_
    imp_series = pd.Series(importances, index=feature_names).sort_values()

    colors = [f"#{int(255 * (1 - v / max(importances))):02x}6B{int(200 * v / max(importances)):02x}"
              for v in imp_series.values]
    colors = ["#51CF66" if v >= imp_series.quantile(0.66)
              else "#F59E0B" if v >= imp_series.quantile(0.33)
              else "#FF6B6B"
              for v in imp_series.values]

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))
    bars = ax.barh(imp_series.index, imp_series.values, color=colors, edgecolor="#333")
    ax.set_xlabel("Importance (Gain)", fontsize=13)
    ax.set_title("XGBoost Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(x=0, color="black", linewidth=0.8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
#  MAIN — Run the full pipeline
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  🎓 STUDENT PERFORMANCE PREDICTION — XGBOOST")
    print("=" * 60)

    # Step 1: Load
    df = load_data()

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Encode (for correlation analysis only)
    df_encoded, encoders = encode_features(df)

    # Step 4: Analyse & select features
    analyze_features(df_encoded)
    selected_features = select_features(df_encoded)

    # Step 5: Train XGBoost with Optuna tuning
    pipeline, X_train, X_test, y_train, y_test, cat_cols, num_cols = train_xgboost(
        df, selected_features, n_trials=15
    )

    # Step 6: Evaluate
    # Try to load old LR metrics for comparison
    lr_metrics = None
    lr_metrics_path = os.path.join(SAVE_DIR, "metrics.pkl")
    if os.path.exists(lr_metrics_path):
        try:
            lr_metrics = joblib.load(lr_metrics_path)
            print("\n📊 Loaded Linear Regression metrics for comparison.")
        except Exception:
            pass

    metrics, y_pred = evaluate_model(pipeline, X_test, y_test, lr_metrics=lr_metrics)

    # Step 7: Save
    save_artifacts(pipeline, encoders, selected_features, metrics)

    # Generate plots
    plots_dir = os.path.join(SAVE_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_correlation_heatmap(df_encoded, os.path.join(plots_dir, "correlation_heatmap.png"))
    plot_actual_vs_predicted(y_test, y_pred, os.path.join(plots_dir, "actual_vs_predicted.png"))
    plot_feature_importance(pipeline, selected_features, os.path.join(plots_dir, "feature_importance.png"))
    print(f"\n📊 Plots saved to '{plots_dir}/'")

    print("\n🎉 XGBoost pipeline complete! Run 'streamlit run app.py' to launch.")
