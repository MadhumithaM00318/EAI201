"""
FIFA World Cup 2026 Finalists Prediction
Complete script: Tasks 1 - 6
Run in Python 3.8+ environment with required packages installed:
pip install numpy pandas scikit-learn matplotlib joblib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

# ---------------------------
# Setup output directory
# ---------------------------
OUT = "output"
os.makedirs(OUT, exist_ok=True)
PLOTS_DIR = os.path.join(OUT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------
# Task 1: Data collection & preparation (synthetic dataset)
# ---------------------------
np.random.seed(42)

years = np.repeat([2002, 2006, 2010, 2014, 2018, 2022], 80)[:480]
teams = [
    "Brazil","Germany","Argentina","France","Italy","Spain","England","Netherlands",
    "Portugal","Uruguay","Croatia","Belgium","Mexico","USA","Sweden","Denmark",
    "Japan","South Korea","Australia","Switzerland","Poland","Chile","Nigeria",
    "Cameroon","Ghana","Iran","Morocco","Saudi Arabia","Russia","Serbia","Turkey",
    "Costa Rica","Ecuador","Colombia","Paraguay","Senegal","Ukraine","Greece",
    "Czech Republic","Tunisia"
]

n = len(years)
df = pd.DataFrame({
    "Year": years,
    "Team1": np.random.choice(teams, n),
    "Team2": np.random.choice(teams, n),
    "Goals_Team1": np.random.poisson(1.5, n),
    "Goals_Team2": np.random.poisson(1.2, n),
    "FIFA_Ranking_Team1": np.random.randint(1, 101, n),
    "FIFA_Ranking_Team2": np.random.randint(1, 101, n),
    "Avg_Age_Team1": np.random.normal(27, 2, n).round(1),
    "Avg_Age_Team2": np.random.normal(27, 2, n).round(1),
    "Experience_Team1": np.random.randint(1, 11, n),
    "Experience_Team2": np.random.randint(1, 11, n),
    "Possession_Team1": np.random.uniform(40, 60, n).round(1),
    "Possession_Team2": np.random.uniform(40, 60, n).round(1),
    "Shots_On_Target_T1": np.random.randint(0, 11, n),
    "Shots_On_Target_T2": np.random.randint(0, 11, n),
    "Past_WC_Performance_T1": np.random.randint(0, 6, n),
    "Past_WC_Performance_T2": np.random.randint(0, 6, n)
})

# Remove rows where Team1 == Team2
df = df[df["Team1"] != df["Team2"]].reset_index(drop=True)

# Determine Winner and binary target Team1_Win
df["Winner"] = np.where(df["Goals_Team1"] > df["Goals_Team2"], df["Team1"],
                        np.where(df["Goals_Team1"] < df["Goals_Team2"], df["Team2"], "Draw"))
df["Team1_Win"] = (df["Winner"] == df["Team1"]).astype(int)

# Feature engineering: difference features (Team1 minus Team2)
df["Goal_Diff"] = df["Goals_Team1"] - df["Goals_Team2"]
df["Rank_Diff"] = df["FIFA_Ranking_Team2"] - df["FIFA_Ranking_Team1"]  # positive => Team1 better
df["Age_Diff"] = df["Avg_Age_Team1"] - df["Avg_Age_Team2"]
df["Exp_Diff"] = df["Experience_Team1"] - df["Experience_Team2"]
df["Poss_Diff"] = df["Possession_Team1"] - df["Possession_Team2"]
df["ShotsOT_Diff"] = df["Shots_On_Target_T1"] - df["Shots_On_Target_T2"]
df["PastPerf_Diff"] = df["Past_WC_Performance_T1"] - df["Past_WC_Performance_T2"]

# Save cleaned dataset and a short columns description CSV
dataset_path = os.path.join(OUT, "FIFA_WorldCup_Compact.csv")
df.to_csv(dataset_path, index=False)

columns_desc = {
    "Year": "Tournament year",
    "Team1/Team2": "Teams playing",
    "Goals_Team1/Goals_Team2": "Goals scored",
    "Winner": "Match winner or Draw",
    "FIFA_Ranking_Team1/Team2": "Pre-tournament FIFA ranking",
    "Avg_Age_Team1/Team2": "Average age of squad",
    "Experience_Team1/Team2": "Average previous WC experience",
    "Possession_Team1/Team2": "Percent possession",
    "Shots_On_Target_T1/T2": "Shots on target",
    "Past_WC_Performance_T1/T2": "Historical success index (0-5)",
    "Goal_Diff,Rank_Diff,...": "Engineered diff features (Team1 minus Team2)",
    "Team1_Win": "Target: 1 if Team1 won, else 0"
}
pd.Series(columns_desc).to_csv(os.path.join(OUT, "data_columns_description.csv"))

print(f"[Task 1] Dataset saved: {dataset_path}")

# ---------------------------
# Task 2: Model building & training
# ---------------------------
features = ["Goal_Diff", "Rank_Diff", "Age_Diff", "Exp_Diff", "Poss_Diff", "ShotsOT_Diff", "PastPerf_Diff"]
X = df[features]
y = df["Team1_Win"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale (only needed for LR)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with grid search
lr = LogisticRegression(max_iter=2000, random_state=42)
lr_grid = GridSearchCV(lr, param_grid={"C": [0.01, 0.1, 1, 10]}, cv=3, scoring="f1")
lr_grid.fit(X_train_scaled, y_train)
best_lr = lr_grid.best_estimator_

# Random Forest with grid search
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, param_grid={"n_estimators": [50, 100], "max_depth": [3, 5, 10]}, cv=3, scoring="f1")
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# Save models & scaler
joblib.dump(best_lr, os.path.join(OUT, "best_logistic_regression.joblib"))
joblib.dump(best_rf, os.path.join(OUT, "best_random_forest.joblib"))
joblib.dump(scaler, os.path.join(OUT, "scaler.joblib"))

print("[Task 2] Models trained and saved (joblib)")

# ---------------------------
# Task 3: Model evaluation
# ---------------------------
def evaluate_print_and_save(y_true, y_pred, y_proba, model_name, save_prefix):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1s = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    metrics = {"model": model_name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1s, "roc_auc": roc_auc}
    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Not Team1 Win","Team1 Win"]); ax.set_yticklabels(["Not Team1 Win","Team1 Win"])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, f"{save_prefix}_confusion.png")
    fig.savefig(cm_path); plt.close(fig)

    # Save ROC plot
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(fpr, tpr, label=f"{model_name} (AUC = {auc(fpr,tpr):.2f})")
    ax2.plot([0,1], [0,1], "--", color="gray")
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate"); ax2.set_title(f"ROC - {model_name}")
    ax2.legend(loc="lower right")
    roc_path = os.path.join(PLOTS_DIR, f"{save_prefix}_roc.png")
    fig2.savefig(roc_path); plt.close(fig2)

    return metrics, cm_path, roc_path

# Predictions & evaluation
y_pred_lr = best_lr.predict(X_test_scaled)
y_proba_lr = best_lr.predict_proba(X_test_scaled)[:, 1]
lr_metrics, lr_cm_path, lr_roc_path = evaluate_print_and_save(y_test, y_pred_lr, y_proba_lr, "LogisticRegression", "lr")

y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
rf_metrics, rf_cm_path, rf_roc_path = evaluate_print_and_save(y_test, y_pred_rf, y_proba_rf, "RandomForest", "rf")

# Save evaluation summary CSV
eval_df = pd.DataFrame([lr_metrics, rf_metrics])
eval_df.to_csv(os.path.join(OUT, "model_evaluation_summary.csv"), index=False)

print("[Task 3] Evaluation complete. Metrics saved to model_evaluation_summary.csv")
print("  - LR ROC:", lr_roc_path)
print("  - RF ROC:", rf_roc_path)

# ---------------------------
# Task 4: Feature importance & interpretation
# ---------------------------
# Random Forest importances
rf_importances = pd.Series(best_rf.feature_importances_, index=features).sort_values(ascending=False)
rf_importances.to_csv(os.path.join(OUT, "feature_importance_rf.csv"))

# Logistic Regression coefficients
lr_coefs = pd.Series(best_lr.coef_[0], index=features).sort_values(key=abs, ascending=False)
lr_coefs.to_csv(os.path.join(OUT, "feature_coefficients_lr.csv"))

# Plot and save feature importance
plt.figure(figsize=(7,5))
rf_importances.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
fi_path = os.path.join(PLOTS_DIR, "feature_importance_rf.png")
plt.savefig(fi_path); plt.close()
print("[Task 4] Feature importance saved:", fi_path)

# Interpretation note saved
interpretation = [
    f"Feature importances (Random Forest):\n{rf_importances.to_string()}",
    "\nLogistic Regression coefficients (sorted by absolute value):\n" + lr_coefs.to_string(),
    "\nInterpretation: Rank_Diff and Goal_Diff are expected to be strong predictors; positive Rank_Diff indicates Team1 has better ranking. Shots and Past performance tend to support predictive power. Note: dataset is synthetic — real scraped data will change importances."
]
with open(os.path.join(OUT, "feature_importance_interpretation.txt"), "w") as f:
    f.write("\n\n".join(interpretation))

# ---------------------------
# Task 5: Final prediction for 2026 (simulated)
# ---------------------------
# List of likely teams (example)
likely_teams = ["Brazil","Argentina","France","Germany","England","Portugal","Spain","Netherlands",
                "Belgium","Croatia","Uruguay","Mexico","USA","Japan","Senegal","Morocco"]

# Simulate pairwise matchups and use models to compute win probabilities for each team
sim_rows = []
for team in likely_teams:
    for opp in likely_teams:
        if team == opp:
            continue
        # Create a simulated matchup using small random perturbations around 0 for difference features
        row = {
            "Team": team,
            "Opponent": opp,
            "Goal_Diff": np.random.normal(0.2, 0.8),
            "Rank_Diff": np.random.normal(5, 10),
            "Age_Diff": np.random.normal(0, 1.5),
            "Exp_Diff": np.random.normal(0.5, 2),
            "Poss_Diff": np.random.normal(0, 5),
            "ShotsOT_Diff": np.random.normal(0.5, 2),
            "PastPerf_Diff": np.random.normal(0.1, 1)
        }
        sim_rows.append(row)

sim_df = pd.DataFrame(sim_rows)
X_sim = sim_df[features]
X_sim_scaled = scaler.transform(X_sim)
sim_df["proba_lr"] = best_lr.predict_proba(X_sim_scaled)[:,1]
sim_df["proba_rf"] = best_rf.predict_proba(X_sim)[:,1]
# average of models as ensemble score
sim_df["proba_avg"] = sim_df[["proba_lr", "proba_rf"]].mean(axis=1)

team_strength = sim_df.groupby("Team")["proba_avg"].mean().sort_values(ascending=False).reset_index()
team_strength.columns = ["Team", "Avg_Win_Prob"]
team_strength.to_csv(os.path.join(OUT, "simulated_2026_team_strength.csv"), index=False)

predicted_finalists = team_strength.head(2)
with open(os.path.join(OUT, "simulated_prediction_2026.txt"), "w") as f:
    f.write("Top predicted finalists (simulated):\n")
    f.write(predicted_finalists.to_string(index=False))

print("[Task 5] Simulated 2026 team strength saved: simulated_2026_team_strength.csv")
print("[Task 5] Predicted finalists (top 2 teams):")
print(predicted_finalists.to_string(index=False))

# ---------------------------
# Task 6: Simple application functions + final report summary
# ---------------------------
# Small API-like helper functions (can be imported)
def load_models_and_scaler(base_dir="output"):
    lr_m = joblib.load(os.path.join(base_dir, "best_logistic_regression.joblib"))
    rf_m = joblib.load(os.path.join(base_dir, "best_random_forest.joblib"))
    sc = joblib.load(os.path.join(base_dir, "scaler.joblib"))
    return lr_m, rf_m, sc

def predict_match(team_diff_features, model="ensemble"):
    """
    team_diff_features: dict with keys matching `features` in order
    model: "lr", "rf", or "ensemble"
    returns probability Team1 wins
    """
    lr_m, rf_m, sc = load_models_and_scaler(OUT)
    X = np.array([[team_diff_features[f] for f in features]])
    Xs = sc.transform(X)
    p_lr = lr_m.predict_proba(Xs)[:,1][0]
    p_rf = rf_m.predict_proba(X)[:,1][0]
    if model == "lr":
        return p_lr
    elif model == "rf":
        return p_rf
    else:
        return (p_lr + p_rf) / 2

# Save a short final report summary
report_text = f"""
Final outputs summary (generated on {datetime.utcnow().isoformat()} UTC)

Files produced in: {os.path.abspath(OUT)}

Key files:
- Cleaned dataset: FIFA_WorldCup_Compact.csv
- Model artifacts: best_logistic_regression.joblib, best_random_forest.joblib, scaler.joblib
- Evaluation summary: model_evaluation_summary.csv
- Feature importance: feature_importance_rf.csv, feature_coefficients_lr.csv
- Plots: {PLOTS_DIR} (confusion matrices, rocs, feature importance)
- Simulated 2026 predictions: simulated_2026_team_strength.csv, simulated_prediction_2026.txt

Note:
- Dataset is synthetic for assignment/testing purposes. Replace with real scraped data in Task 1 for production quality.
- Interpret model outputs carefully; sports outcomes are stochastic and influenced by many unmeasured factors.
"""
with open(os.path.join(OUT, "final_report_summary.txt"), "w") as f:
    f.write(report_text)

print("[Task 6] Application helpers saved. Final summary written.")
print("All tasks 1-6 completed. Check the `output/` folder for CSVs, models and plots.")
