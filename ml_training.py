import warnings
warnings.filterwarnings("ignore")

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow import MlflowClient

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed - skipping")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed - skipping")


# ================================================================
# CONFIG
# ================================================================
BASE          = Path(__file__).parent
CUST_CSV      = BASE / "customer.csv"
ACQ_CSV       = BASE / "customer_acquisition_data.csv"
ART_DIR       = BASE / "artifacts"
ART_DIR.mkdir(exist_ok=True)

MLFLOW_URI    = (BASE / "mlruns").resolve().as_uri()
EXP_NAME      = "CLV_Prediction"
REGISTRY_NAME = "CLV_Best_Model"
SEED          = 42


# ================================================================
# 1. DATA LOADING & PREPROCESSING
# ================================================================
print("\nLoading data...")
cust = pd.read_csv(CUST_CSV)
acq  = pd.read_csv(ACQ_CSV)

cust["_id"] = cust["CustomerID"].astype(str).str.extract(r"(\d+)").astype(int)
df = cust.merge(acq, left_on="_id", right_on="customer_id", how="left")
df.drop(columns=["_id", "customer_id"], inplace=True, errors="ignore")

if "cost"            in df.columns: df["cost"]            = df["cost"].fillna(df["cost"].median())
if "conversion_rate" in df.columns: df["conversion_rate"] = df["conversion_rate"].fillna(df["conversion_rate"].median())
if "channel"         in df.columns: df["channel"]         = df["channel"].fillna("unknown")

NUM_COLS = [
    "Age","Annual_Income","Household_Size","Quantity","Unit_Price",
    "Discount_Applied","Revenue","Profit_Margin","Loyalty_Score",
    "Customer_Satisfaction","Purchase_Frequency","Days_Since_Last_Purchase",
    "Newsletter_Subscribed","Support_Tickets","Email_Open_Rate",
    "Social_Media_Engagement","Churn_Risk_Score","Return_Rate",
    "Payment_Delay_Days","Customer_Acquisition_Cost","Net_Revenue",
    "CLV_to_CAC_Ratio","Engagement_Score","Profitability_Score",
    "Seasonal_Factor","Cohort_Index","cost","conversion_rate",
]
CAT_COLS = [
    "Customer_Segment","Gender","Job_Category","Education_Level",
    "Marital_Status","Payment_Method","Marketing_Channel",
    "First_Purchase_Channel","Season","Age_Group","Revenue_Tier",
    "Loyalty_Tier","channel",
]

NUM_COLS = [c for c in NUM_COLS if c in df.columns]
CAT_COLS = [c for c in CAT_COLS if c in df.columns]

le_map = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_map[col] = le

FEAT_COLS = NUM_COLS + CAT_COLS
TARGET    = "Customer_Lifetime_Value"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found")

X = df[FEAT_COLS].copy()
y = df[TARGET].copy()

mask = y <= y.quantile(0.995)
X, y = X[mask], y[mask]

pickle.dump(le_map,  open(ART_DIR / "encoders.pkl",      "wb"))
json.dump(FEAT_COLS, open(ART_DIR / "feature_cols.json", "w"))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
pickle.dump(scaler, open(ART_DIR / "scaler.pkl", "wb"))


# ================================================================
# 2. MODEL DEFINITIONS
# ================================================================
MODELS = {
    "Ridge":      Ridge(alpha=10.0),
    "Lasso":      Lasso(alpha=1.0, max_iter=5000),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
}
if HAS_XGB:
    MODELS["XGBoost"] = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=SEED, verbosity=0,
    )
if HAS_LGB:
    MODELS["LightGBM"] = lgb.LGBMRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=SEED, verbose=-1,
    )


# ================================================================
# 3. MLFLOW SETUP  (no deprecated APIs)
# ================================================================
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)

# Create or reuse experiment
existing = client.search_experiments(filter_string=f"name = '{EXP_NAME}'")
if existing:
    experiment_id = existing[0].experiment_id
else:
    experiment_id = client.create_experiment(
        name=EXP_NAME,
        tags={"project": "CLV Prediction", "version": "2.0"},
    )
mlflow.set_experiment(EXP_NAME)

# Ensure registered model exists in registry
try:
    client.create_registered_model(
        name=REGISTRY_NAME,
        description="Best CLV prediction model — auto-promoted Champion each run",
        tags={"task": "regression", "target": "Customer_Lifetime_Value"},
    )
    print(f"Created registered model '{REGISTRY_NAME}'")
except Exception:
    print(f"Registered model '{REGISTRY_NAME}' already exists — reusing")

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
lb = []

print(f"\nTraining {len(MODELS)} models")
print(f"  MLflow URI  -> {MLFLOW_URI}")
print(f"  Experiment  -> {EXP_NAME}")
print(f"  Registry    -> {REGISTRY_NAME}")
print("-" * 65)


# ================================================================
# 4. TRAINING LOOP  — full MLflow tracing per model
# ================================================================
for name, model in MODELS.items():
    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id
        print(f"\n> {name:<20}  run={run_id[:8]}...")

        # -- Tags -------------------------------------------------
        mlflow.set_tags({
            "model_name": name,
            "project":    "CLV Prediction",
            "stage":      "training",
            "n_features": str(len(FEAT_COLS)),
            "train_rows": str(len(X_tr)),
            "test_rows":  str(len(X_te)),
        })

        # -- Hyperparameters --------------------------------------
        hp = {k: str(v) for k, v in model.get_params().items()}
        hp.update({
            "model_type": name,
            "seed":       str(SEED),
            "n_features": str(len(FEAT_COLS)),
            "train_size": str(len(X_tr)),
            "test_size":  str(len(X_te)),
        })
        mlflow.log_params(hp)

        # -- Per-fold CV  (step = fold index) ---------------------
        print(f"  |- CV 5-fold...", end=" ", flush=True)
        cv_r2s   = cross_val_score(model, X_tr_s, y_tr, cv=kf, scoring="r2", n_jobs=1)
        cv_maes  = cross_val_score(model, X_tr_s, y_tr, cv=kf,
                                   scoring="neg_mean_absolute_error", n_jobs=1)
        cv_rmses = cross_val_score(model, X_tr_s, y_tr, cv=kf,
                                   scoring="neg_root_mean_squared_error", n_jobs=1)

        for fold, (r2f, maef, rmsef) in enumerate(zip(cv_r2s, -cv_maes, -cv_rmses), 1):
            mlflow.log_metrics({
                "cv_fold_r2":   round(float(r2f),  6),
                "cv_fold_mae":  round(float(maef),  4),
                "cv_fold_rmse": round(float(rmsef), 4),
            }, step=fold)

        cv_r2_mean   = float(cv_r2s.mean())
        cv_r2_std    = float(cv_r2s.std())
        cv_mae_mean  = float((-cv_maes).mean())
        cv_rmse_mean = float((-cv_rmses).mean())
        print(f"R2={cv_r2_mean:.4f} +/- {cv_r2_std:.4f}")

        # -- Train ------------------------------------------------
        t0 = time.time()
        model.fit(X_tr_s, y_tr)
        train_time = round(time.time() - t0, 3)

        # -- Holdout metrics --------------------------------------
        preds     = model.predict(X_te_s)
        r2        = float(r2_score(y_te, preds))
        rmse      = float(np.sqrt(mean_squared_error(y_te, preds)))
        mae       = float(mean_absolute_error(y_te, preds))
        mape      = float(mean_absolute_percentage_error(y_te, preds) * 100)
        max_err   = float(np.max(np.abs(y_te - preds)))
        median_ae = float(np.median(np.abs(y_te - preds)))

        # -- Train metrics (overfit proxy) ------------------------
        tr_preds = model.predict(X_tr_s)
        tr_r2    = float(r2_score(y_tr, tr_preds))
        tr_mae   = float(mean_absolute_error(y_tr, tr_preds))
        overfit  = round(tr_r2 - r2, 6)

        # -- Log all summary metrics ------------------------------
        mlflow.log_metrics({
            # Holdout
            "R2":           round(r2,        6),
            "RMSE":         round(rmse,      4),
            "MAE":          round(mae,       4),
            "MAPE":         round(mape,      4),
            "Max_Error":    round(max_err,   4),
            "Median_AE":    round(median_ae, 4),
            # CV summary
            "CV_R2":        round(cv_r2_mean,   6),
            "CV_R2_std":    round(cv_r2_std,    6),
            "CV_MAE":       round(cv_mae_mean,  4),
            "CV_RMSE":      round(cv_rmse_mean, 4),
            # Train (overfit check)
            "Train_R2":     round(tr_r2,    6),
            "Train_MAE":    round(tr_mae,   4),
            "Overfit_Gap":  overfit,
            # Efficiency
            "Train_Time_s": train_time,
        })

        # -- Feature importance artifact + top-5 metrics ----------
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=FEAT_COLS)
            top5 = fi.nlargest(5)
            mlflow.log_metrics({f"fi_{k}": round(float(v), 6) for k, v in top5.items()})

            fi_path = ART_DIR / f"fi_{name}.json"
            fi.sort_values(ascending=False).to_json(fi_path)
            mlflow.log_artifact(str(fi_path), artifact_path="feature_importance")

        # -- Register model in MLflow Model Registry --------------
        sig        = infer_signature(X_tr, model.predict(X_tr_s[:5]))
        mlflow.sklearn.log_model(
            sk_model              = model,
            name                  = "model",
            signature             = sig,
            input_example         = X_tr.iloc[:3],
            registered_model_name = REGISTRY_NAME,  # auto-creates new version
        )

        # Tag the new registry version
        reg_versions = client.search_model_versions(
            f"name='{REGISTRY_NAME}' and run_id='{run_id}'"
        )
        if reg_versions:
            ver = reg_versions[0].version
            client.set_model_version_tag(REGISTRY_NAME, ver, "model_type",  name)
            client.set_model_version_tag(REGISTRY_NAME, ver, "R2",          str(round(r2,   6)))
            client.set_model_version_tag(REGISTRY_NAME, ver, "RMSE",        str(round(rmse, 4)))
            client.set_model_version_tag(REGISTRY_NAME, ver, "Overfit_Gap", str(overfit))
            print(f"  |- Registered -> '{REGISTRY_NAME}' v{ver}")

        lb.append({
            "Model":        name,
            "R2":           r2,
            "CV_R2":        cv_r2_mean,
            "CV_R2_std":    cv_r2_std,
            "RMSE":         rmse,
            "MAE":          mae,
            "MAPE":         mape,
            "Overfit_Gap":  overfit,
            "Train_Time_s": train_time,
            "run_id":       run_id,
        })

        print(f"  |- R2={r2:.4f}  CV-R2={cv_r2_mean:.4f}  "
              f"RMSE=${rmse:,.0f}  MAE=${mae:,.0f}  Overfit={overfit:+.4f}")


# ================================================================
# 5. LEADERBOARD
# ================================================================
lb_df = (
    pd.DataFrame(lb)
    .sort_values("R2", ascending=False)
    .reset_index(drop=True)
)
lb_df.to_csv(ART_DIR / "leaderboard.csv", index=False)

best_row   = lb_df.iloc[0]
best_name  = best_row["Model"]
best_run   = best_row["run_id"]
best_model = MODELS[best_name]

pickle.dump(best_model, open(ART_DIR / "best_model.pkl",      "wb"))
open(ART_DIR / "best_model_name.txt", "w").write(best_name)


# ================================================================
# 6. PROMOTE BEST MODEL -> "Champion" alias in Registry
# ================================================================
best_rv = client.search_model_versions(
    f"name='{REGISTRY_NAME}' and run_id='{best_run}'"
)
if best_rv:
    champion_ver = best_rv[0].version
    try:
        client.set_registered_model_alias(REGISTRY_NAME, "Champion", champion_ver)
        print(f"\nPromoted '{REGISTRY_NAME}' v{champion_ver} -> alias 'Champion'")
    except Exception as e:
        print(f"[WARN] Could not set Champion alias: {e}")


# ================================================================
# 7. SUMMARY
# ================================================================
print("\n" + "=" * 65)
print(f"Best model    : {best_name}")
print(f"  R2          : {best_row['R2']:.4f}")
print(f"  CV R2       : {best_row['CV_R2']:.4f} +/- {best_row['CV_R2_std']:.4f}")
print(f"  RMSE        : ${best_row['RMSE']:,.0f}")
print(f"  MAE         : ${best_row['MAE']:,.0f}")
print(f"  Overfit Gap : {best_row['Overfit_Gap']:+.4f}")
print(f"\nArtifacts     : {ART_DIR}")
print(f"\nLaunch MLflow UI:")
print(f"  mlflow ui --backend-store-uri {MLFLOW_URI} --port 5000")
print(f"  -> Experiments tab : all runs + per-fold CV step metrics")
print(f"  -> Models tab      : '{REGISTRY_NAME}' versions + 'Champion' alias")
print("=" * 65)
print(lb_df[["Model","R2","CV_R2","RMSE","MAE","Overfit_Gap","Train_Time_s"]].to_string(index=False))