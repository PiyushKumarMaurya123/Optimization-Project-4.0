"""Train multi-output composition model (C1-C6) with LOOCV comparison."""
import pandas as pd, numpy as np, joblib
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error

IMPUTE_SMALL = 0.005  # below-detection-limit fill
FEATURES = ["V", "T", "P1", "P2", "RM1", "YM23", "M3"]  # M1 constant -> excluded
TARGETS = ["C1", "C2", "C3", "C4", "C5", "C6"]

def load():
    df = pd.read_csv("data.csv")
    df["YM23"] = df["YM23"].astype(str).str.replace("%", "").astype(float)
    df = df.dropna(subset=TARGETS, how="all").reset_index(drop=True)
    df[TARGETS] = df[TARGETS].fillna(IMPUTE_SMALL)
    return df

def loocv_eval(model_fn, X, Y):
    loo = LeaveOneOut()
    preds = np.zeros_like(Y)
    for tr, te in loo.split(X):
        m = model_fn()
        m.fit(X[tr], Y[tr])
        preds[te] = m.predict(X[te])
    preds = np.clip(preds, 0, None)
    r2 = [r2_score(Y[:, i], preds[:, i]) for i in range(Y.shape[1])]
    mae = [mean_absolute_error(Y[:, i], preds[:, i]) for i in range(Y.shape[1])]
    return np.array(r2), np.array(mae)

if __name__ == "__main__":
    df = load()
    X, Y = df[FEATURES].values, df[TARGETS].values
    print(f"n = {len(df)} runs, {len(FEATURES)} features")

    candidates = {
        "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, random_state=42, n_jobs=-1),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=1, random_state=42, n_jobs=-1),
        "GradBoost": lambda: MultiOutputRegressor(GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)),
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }

    results = {}
    for name, fn in candidates.items():
        r2, mae = loocv_eval(fn, X, Y)
        results[name] = (r2, mae)
        print(f"\n{name}: mean R2 = {r2.mean():.3f}")
        for t, r, m in zip(TARGETS, r2, mae):
            print(f"  {t}: R2={r:6.3f}  MAE={m:6.2f}")

    best = max(results, key=lambda k: results[k][0].mean())
    print(f"\nBEST: {best}")

    final = candidates[best]()
    final.fit(X, Y)
    if hasattr(final, "feature_importances_"):
        fi = final.feature_importances_
    else:
        fi = np.mean([abs(np.corrcoef(X[:, j], Y[:, i])[0, 1]) for i in range(Y.shape[1]) for j in range(len(FEATURES))], axis=0)
    bundle = {
        "model": final,
        "model_name": best,
        "features": FEATURES,
        "targets": TARGETS,
        "feature_importance": dict(zip(FEATURES, final.feature_importances_)) if hasattr(final, "feature_importances_") else None,
        "safe_ranges": {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES},
        "loocv_r2": dict(zip(TARGETS, results[best][0])),
        "loocv_mae": dict(zip(TARGETS, results[best][1])),
        "impute_small": IMPUTE_SMALL,
        "M1_fixed": 0.5,
    }
    joblib.dump(bundle, "model_bundle.joblib")
    print("Saved model_bundle.joblib")
    print("Safe ranges:", bundle["safe_ranges"])
    if bundle["feature_importance"]:
        print("Feature importance:", {k: round(v, 3) for k, v in bundle["feature_importance"].items()})
