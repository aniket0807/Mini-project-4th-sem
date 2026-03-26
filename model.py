"""
===================================================================
MODEL.PY — Linear Regression Model for Student Performance Prediction
===================================================================
This module handles:
  1. Loading and cleaning the dataset
  2. Encoding categorical variables
  3. Feature correlation analysis & selection
  4. Training a Linear Regression model
  5. Evaluating with MAE, MSE, RMSE, R² Score
  6. Saving the trained model + encoders for reuse
  7. A predict() function for the Streamlit app
===================================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
      - Drop student_id (not predictive)
      - Handle missing values
      - Remove duplicates
    """
    df = df.copy()

    # Drop student ID — it's just an identifier, not a feature
    if "student_id" in df.columns:
        df.drop(columns=["student_id"], inplace=True)

    # Drop duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"🗑️  Removed {before - after} duplicate rows")

    # Handle missing values — fill numeric with median, categorical with mode
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  🔧 Filled missing values in '{col}'")

    print(f"✅ Data cleaned: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
#  STEP 3: ENCODE CATEGORICAL VARIABLES
# ─────────────────────────────────────────────
def encode_features(df: pd.DataFrame):
    """
    Encode categorical columns using LabelEncoder.
    Returns the encoded DataFrame and a dict of encoders for later use.

    WHY Label Encoding?
    - Simple and effective for ordinal features (diet_quality, internet_quality)
    - Works well with Linear Regression for binary features (gender, part_time_job)
    - Easy to reverse for interpretability
    """
    df = df.copy()
    encoders = {}

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  🏷️  Encoded '{col}': {list(le.classes_)}")

    print(f"✅ Encoded {len(categorical_cols)} categorical columns")
    return df, encoders


# ─────────────────────────────────────────────
#  STEP 4: FEATURE ANALYSIS & SELECTION
# ─────────────────────────────────────────────
def analyze_features(df: pd.DataFrame, target: str = "exam_score"):
    """
    Analyze feature correlations with the target variable.
    Returns a sorted Series of absolute correlations.
    """
    corr = df.corr(numeric_only=True)[target].drop(target).abs().sort_values(ascending=False)
    print("\n📊 Feature Correlations with Exam Score:")
    print("-" * 45)
    for feat, val in corr.items():
        bar = "█" * int(val * 30)
        print(f"  {feat:<35} {val:.4f}  {bar}")
    return corr


def select_features(df: pd.DataFrame, target: str = "exam_score", threshold: float = 0.02):
    """
    Select features with correlation above threshold.
    We use a low threshold to keep most features since Linear Regression
    benefits from having more features for a better fit.
    """
    corr = df.corr(numeric_only=True)[target].drop(target).abs()
    selected = corr[corr >= threshold].index.tolist()
    print(f"\n✅ Selected {len(selected)} features (correlation >= {threshold}):")
    for f in selected:
        print(f"  • {f} (r={corr[f]:.4f})")
    return selected


# ─────────────────────────────────────────────
#  STEP 5: TRAIN THE MODEL
# ─────────────────────────────────────────────
def train_model(df: pd.DataFrame, features: list, target: str = "exam_score"):
    """
    Train a Linear Regression model.

    WHY LINEAR REGRESSION?
    ──────────────────────
    1. INTERPRETABILITY: Each coefficient tells us exactly how much each
       habit affects the exam score. For example, if the coefficient for
       study_hours is +5.2, it means each extra hour of study adds ~5.2
       marks — very useful for generating study plans!

    2. CONTINUOUS TARGET: Exam scores are continuous numbers (0-100).
       Linear Regression is designed for continuous outputs.

    3. LINEAR RELATIONSHIPS: Student habits like study hours and attendance
       tend to have approximately linear relationships with academic scores.

    4. SIMPLICITY: For a BCA mini-project, Linear Regression is perfect
       because it's easy to understand, implement, and explain.

    5. FAST TRAINING: Works instantly even on larger datasets.

    Returns: model, X_train, X_test, y_train, y_test
    """
    X = df[features]
    y = df[target]

    # 80% training, 20% testing — standard split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n📐 Train-Test Split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")

    # Train linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"\n✅ Linear Regression model trained successfully!")

    # Print coefficients (how much each feature affects the score)
    print(f"\n📈 Model Coefficients (feature importance):")
    print("-" * 50)
    coefs = sorted(zip(features, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in coefs:
        direction = "📈" if coef > 0 else "📉"
        print(f"  {direction} {feat:<35} {coef:+.4f}")
    print(f"  📍 Intercept: {model.intercept_:.4f}")

    return model, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
#  STEP 6: EVALUATE THE MODEL
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using multiple metrics:

    - MAE (Mean Absolute Error):  Average error in marks
    - MSE (Mean Squared Error):   Penalizes large errors more
    - RMSE (Root MSE):            Error in the same unit as scores
    - R² Score:                   How much variance the model explains (0 to 1)
                                  Higher R² = better model
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2_Score": round(r2, 4),
    }

    print(f"\n{'='*50}")
    print(f"  📊 MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"  MAE  (Mean Absolute Error):  {mae:.4f}")
    print(f"  MSE  (Mean Squared Error):   {mse:.4f}")
    print(f"  RMSE (Root Mean Sq Error):   {rmse:.4f}")
    print(f"  R²   (R-Squared Score):      {r2:.4f}")
    print(f"{'='*50}")

    if r2 > 0.70:
        print(f"  ✅ Good model! R² > 0.70 means the model explains")
        print(f"     {r2*100:.1f}% of the variance in exam scores.")
    elif r2 > 0.50:
        print(f"  ⚠️  Moderate model. R² = {r2:.2f}. The model explains")
        print(f"     {r2*100:.1f}% of variance. Consider adding more features.")
    else:
        print(f"  ❌ Weak model. R² = {r2:.2f}. Linear Regression may not")
        print(f"     capture the full complexity of this data.")

    return metrics, y_pred


# ─────────────────────────────────────────────
#  STEP 7: SAVE & LOAD MODEL
# ─────────────────────────────────────────────
def save_artifacts(model, encoders, features, metrics, save_dir=None):
    """Save the trained model, encoders, features, and metrics."""
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "saved_model")
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model, os.path.join(save_dir, "linear_regression_model.pkl"))
    joblib.dump(encoders, os.path.join(save_dir, "encoders.pkl"))
    joblib.dump(features, os.path.join(save_dir, "features.pkl"))
    joblib.dump(metrics, os.path.join(save_dir, "metrics.pkl"))

    print(f"\n💾 Model artifacts saved to '{save_dir}/'")


def load_artifacts(save_dir=None):
    """Load saved model artifacts."""
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "saved_model")

    model = joblib.load(os.path.join(save_dir, "linear_regression_model.pkl"))
    encoders = joblib.load(os.path.join(save_dir, "encoders.pkl"))
    features = joblib.load(os.path.join(save_dir, "features.pkl"))
    metrics = joblib.load(os.path.join(save_dir, "metrics.pkl"))

    return model, encoders, features, metrics


# ─────────────────────────────────────────────
#  STEP 8: PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_score(model, features_list, student_data: dict) -> float:
    """
    Predict exam score for a single student.

    Parameters:
        model:          Trained LinearRegression model
        features_list:  List of feature names the model expects
        student_data:   Dict of student's habit values

    Returns:
        Predicted exam score (clipped to 0-100 range)
    """
    input_df = pd.DataFrame([student_data])[features_list]
    prediction = model.predict(input_df)[0]
    # Clip to valid score range
    prediction = np.clip(prediction, 0, 100)
    return round(prediction, 2)


# ─────────────────────────────────────────────
#  VISUALIZATION HELPERS (for Streamlit)
# ─────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """Generate a correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
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
    ax.set_title("Actual vs Predicted Scores", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(model, features: list, save_path: str = None):
    """Plot feature importance based on model coefficients."""
    coefs = pd.Series(model.coef_, index=features).sort_values()
    colors = ["#FF6B6B" if c < 0 else "#51CF66" for c in coefs.values]

    fig, ax = plt.subplots(figsize=(10, 6))
    coefs.plot(kind="barh", color=colors, edgecolor="#333", ax=ax)
    ax.set_xlabel("Coefficient Value", fontsize=13)
    ax.set_title("Feature Importance (Linear Regression Coefficients)", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
#  MAIN — Run the full pipeline
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  🎓 STUDENT PERFORMANCE PREDICTION — LINEAR REGRESSION")
    print("=" * 60)

    # Step 1: Load
    df = load_data()

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Encode
    df_encoded, encoders = encode_features(df)

    # Step 4: Analyze & Select features
    analyze_features(df_encoded)
    selected_features = select_features(df_encoded)

    # Step 5: Train
    model, X_train, X_test, y_train, y_test = train_model(df_encoded, selected_features)

    # Step 6: Evaluate
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # Step 7: Save
    save_artifacts(model, encoders, selected_features, metrics)

    # Generate plots
    plots_dir = os.path.join(os.path.dirname(__file__), "saved_model", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_correlation_heatmap(df_encoded, os.path.join(plots_dir, "correlation_heatmap.png"))
    plot_actual_vs_predicted(y_test, y_pred, os.path.join(plots_dir, "actual_vs_predicted.png"))
    plot_feature_importance(model, selected_features, os.path.join(plots_dir, "feature_importance.png"))
    print(f"\n📊 Plots saved to '{plots_dir}/'")

    print("\n🎉 Pipeline complete! Run 'streamlit run app.py' to launch the web app.")
