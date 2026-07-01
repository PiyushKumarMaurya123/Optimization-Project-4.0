"""
Product Composition Predictor - C1-C6
Streamlit app with self-healing model bundle.
Run:  streamlit run app.py --server.address 0.0.0.0
"""
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Composition Predictor", layout="wide", page_icon=":test_tube:")

# ---------------------------------------------------------------- model load
BUNDLE_PATH = Path(__file__).parent / "model_bundle.joblib"
DATA_PATH = Path(__file__).parent / "data.csv"

@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    """Self-healing: load the saved bundle; if it fails (missing file,
    sklearn version mismatch, corruption), retrain from data.csv."""
    import joblib
    try:
        b = load_bundle_raw = joblib.load(BUNDLE_PATH)
        _ = b["model"].predict(np.zeros((1, len(b["features"]))))  # sanity ping
        return b
    except Exception:
        return retrain_bundle()

def retrain_bundle():
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor
    FEATURES = ["V", "T", "P1", "P2", "RM1", "YM23", "M3"]
    TARGETS = ["C1", "C2", "C3", "C4", "C5", "C6"]
    df = pd.read_csv(DATA_PATH)
    df["YM23"] = df["YM23"].astype(str).str.replace("%", "").astype(float)
    df = df.dropna(subset=TARGETS, how="all").reset_index(drop=True)
    df[TARGETS] = df[TARGETS].fillna(0.005)
    model = ExtraTreesRegressor(n_estimators=600, min_samples_leaf=1,
                                max_features="sqrt", random_state=42, n_jobs=-1)
    model.fit(df[FEATURES].values, df[TARGETS].values)
    b = {
        "model": model, "model_name": "ExtraTrees (600 trees, sqrt features)",
        "features": FEATURES, "targets": TARGETS,
        "feature_importance": dict(zip(FEATURES, model.feature_importances_)),
        "safe_ranges": {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES},
        "loocv_r2": None, "loocv_mae": None, "impute_small": 0.005, "M1_fixed": 0.5,
    }
    try:
        joblib.dump(b, BUNDLE_PATH)
    except Exception:
        pass
    return b

bundle = load_bundle()
MODEL, FEATURES, TARGETS = bundle["model"], bundle["features"], bundle["targets"]
SAFE = bundle["safe_ranges"]

# ------------------------------------------------------- derived quantities
def derived_quantities(M1, M3, YM23, RM1, V):
    """Solve for M2, RM2, X, YRM12 from the calibrated process relations.
    (Internal only - the UI never exposes the equations.)"""
    if YM23 <= 0 or YM23 >= 100 or RM1 <= 0 or M3 <= 0 or V <= 0:
        return None
    KSM = RM1 / (1 + 1.04898 * M1)
    k = (100.0 / YM23 - 1.0) / 0.1836          # = X * M3 / M2
    if k <= 0:
        return None
    a = KSM * 1.059818
    b = RM1 + KSM * 0.19082 * M3
    c = -60.0 * V * M3 / k
    disc = b * b - 4 * a * c
    if disc < 0 or a == 0:
        return None
    M2 = (-b + math.sqrt(disc)) / (2 * a)
    if M2 <= 0:
        return None
    RM2 = KSM * (1.059818 * M2 + 0.19082 * M3)
    X = (V * 60.0) / (RM1 + RM2)
    YRM12 = RM1 / RM2 if RM2 > 0 else float("nan")
    return {"M2": M2, "RM2": RM2, "X": X, "YRM12": YRM12}

# ------------------------------------------------------------------- header
st.title("Product Composition Predictor")
st.caption(
    f"Multi-output machine-learning model ({bundle['model_name']}) trained on plant "
    "trial data. Objective: **maximize C2** while suppressing the remaining components."
)

# ------------------------------------------------------------------ sliders
# Extended exploration ranges (wider than the trusted data window on purpose)
EXT = {
    "V":    dict(lo=8,    hi=80,   step=8,    fmt="%d",   label="V - feed rate"),
    "T":    dict(lo=20.0, hi=200.0, step=1.0,  fmt="%.0f", label="T - temperature ( deg C)"),
    "P1":   dict(lo=1.0,  hi=15.0, step=0.1,  fmt="%.1f", label="P1 - pressure 1"),
    "P2":   dict(lo=0.0,  hi=10.0, step=0.1,  fmt="%.1f", label="P2 - pressure 2"),
    "RM1":  dict(lo=1.0,  hi=30.0, step=0.1,  fmt="%.1f", label="RM1 - raw material 1 flow"),
    "YM23": dict(lo=5.0,  hi=95.0, step=0.1,  fmt="%.1f", label="YM23 - ratio index (%)"),
    "M3":   dict(lo=5.0,  hi=100.0, step=0.1, fmt="%.1f", label="M3 - moles of component 3"),
}
DEFAULTS = {"V": 40, "T": 75, "P1": 5.4, "P2": 5.0, "RM1": 10.4, "YM23": 68.5, "M3": 10.0}

st.sidebar.header(" Process inputs")
st.sidebar.caption("Sliders extend beyond the validated operating window for exploration.")

vals, out_of_range = {}, []
for f in FEATURES:
    cfg = EXT[f]
    if f == "V":
        vals[f] = st.sidebar.slider(cfg["label"], min_value=int(cfg["lo"]),
                                    max_value=int(cfg["hi"]), value=int(DEFAULTS[f]),
                                    step=int(cfg["step"]),
                                    help="Constrained to multiples of 8")
    else:
        vals[f] = st.sidebar.slider(cfg["label"], min_value=float(cfg["lo"]),
                                    max_value=float(cfg["hi"]),
                                    value=float(DEFAULTS[f]), step=float(cfg["step"]))
    lo, hi = SAFE[f]
    if not (lo <= vals[f] <= hi):
        out_of_range.append((f, vals[f], lo, hi))

M1 = st.sidebar.number_input("M1 - moles of component 1", value=float(bundle["M1_fixed"]),
                             min_value=0.05, max_value=5.0, step=0.05,
                             help="All trials to date used M1 = 0.5; the model assumes this value.")

if out_of_range:
    msg = " | ".join(f"**{f}** = {v:g} (validated window: {lo:g}-{hi:g})"
                     for f, v, lo, hi in out_of_range)
    st.warning(
        f" **Extrapolation mode** - you are operating outside the validated data "
        f"window for: {msg}. Predictions here are directional estimates, not "
        f"interpolations; treat them as hypotheses to verify with a plant trial.",
        icon="",
    )
if abs(M1 - bundle["M1_fixed"]) > 1e-9:
    st.info(" M1 differs from the trial value (0.5). Derived quantities update "
            "accordingly, but the ML prediction is calibrated at M1 = 0.5.")

# ------------------------------------------- trend compass (Ridge, cached)
@st.cache_resource(show_spinner=False)
def get_trend_model():
    """Linear Ridge model used ONLY as a directional 'trend compass' when
    inputs leave the validated window. Never shown in-window."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(DATA_PATH)
    df["YM23"] = df["YM23"].astype(str).str.replace("%", "").astype(float)
    df = df.dropna(subset=TARGETS, how="all").reset_index(drop=True)
    df[TARGETS] = df[TARGETS].fillna(bundle.get("impute_small", 0.005))
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    m.fit(df[FEATURES].values, df[TARGETS].values)
    return m

def norm100(p):
    p = np.clip(p, 0, None)
    return p / p.sum() * 100 if p.sum() > 0 else p

# ---------------------------------------------------------------- predict
x = np.array([[vals[f] for f in FEATURES]])
pred = norm100(MODEL.predict(x)[0])

pred_trend, trend_delta = None, None
if out_of_range:
    ridge = get_trend_model()
    # inputs clipped back to the validated window = "edge point"
    x_edge = np.array([[min(max(vals[f], SAFE[f][0]), SAFE[f][1]) for f in FEATURES]])
    pred_trend = norm100(ridge.predict(x)[0])          # linear estimate out here
    ridge_edge = norm100(ridge.predict(x_edge)[0])     # linear estimate at edge
    trend_delta = pred_trend - ridge_edge              # trend's estimated change

order = np.argsort(pred)  # ascending, like the reference chart
comp = [(TARGETS[i], pred[i]) for i in order]

# --------------------------------------------------------- derived metrics
dq = derived_quantities(M1, vals["M3"], vals["YM23"], vals["RM1"], vals["V"])
st.subheader("Derived process quantities")
c1, c2, c3, c4 = st.columns(4)
if dq:
    c1.metric("M2", f"{dq['M2']:.3f}", help="Function of YM23, M3, RM1, M1 and V")
    c2.metric("RM2", f"{dq['RM2']:.3f}", help="Function of RM1, M1, M2 and M3")
    c3.metric("X", f"{dq['X']:.2f}", help="Function of V, RM1 and RM2")
    c4.metric("YRM12", f"{dq['YRM12']:.3f}", help="Function of RM1 and RM2")
else:
    st.info("Derived quantities are undefined for this input combination "
            "(check YM23, RM1, M3 and V values).")

# ----------------------------------------------------------------- chart
st.subheader("Predicted composition")
delta_map = dict(zip(TARGETS, trend_delta)) if trend_delta is not None else {}
hl_cols = st.columns(len(comp))
for col, (name, v) in zip(hl_cols, comp):
    trend_html = ""
    if name in delta_map:
        d = delta_map[name]
        color = "#5aa89b" if d > 0 else ("#d97b7b" if d < 0 else "#888")
        arrow = "&#9650;" if d > 0 else ("&#9660;" if d < 0 else "&#8212;")
        trend_html = (f"<div style='font-size:0.8rem;color:{color}'>"
                      f"{arrow} trend {d:+.1f}</div>")
    col.markdown(
        f"<div style='font-size:0.85rem;opacity:0.7'>{name} %</div>"
        f"<div style='font-size:2.1rem;font-weight:800'>{v:.1f}%</div>"
        f"{trend_html}",
        unsafe_allow_html=True,
    )

palette = ["#2e4372", "#4a7a9d", "#5aa89b", "#c9924a", "#a35d6a", "#7d5ba6"]
if pred_trend is None:
    fig = go.Figure(go.Bar(
        x=[v for _, v in comp],
        y=[n + " %" for n, _ in comp],
        orientation="h",
        marker_color=palette[: len(comp)],
        text=[f"{v:.1f}" for _, v in comp],
        textposition="outside",
    ))
    xmax = max(40, pred.max() * 1.15)
else:
    names = [n for n, _ in comp]
    et_vals = [v for _, v in comp]
    tr_vals = [pred_trend[TARGETS.index(n)] for n in names]
    fig = go.Figure([
        go.Bar(name="Nearest known behavior (Extra Trees)",
               x=et_vals, y=[n + " %" for n in names], orientation="h",
               marker_color="#5aa89b",
               text=[f"{v:.1f}" for v in et_vals], textposition="outside"),
        go.Bar(name="Linear trend estimate (Ridge)",
               x=tr_vals, y=[n + " %" for n in names], orientation="h",
               marker_color="#7d5ba6",
               text=[f"{v:.1f}" for v in tr_vals], textposition="outside"),
    ])
    fig.update_layout(barmode="group",
                      legend=dict(orientation="h", y=1.08, x=0))
    xmax = max(40, float(pred.max()), float(pred_trend.max())) * 1.15
fig.update_layout(
    template="plotly_dark", height=460 if pred_trend is not None else 420,
    margin=dict(l=10, r=30, t=10, b=10),
    xaxis=dict(range=[0, xmax], dtick=2,
               gridcolor="rgba(255,255,255,0.08)"),
    yaxis=dict(autorange="reversed"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)

if pred_trend is not None:
    gaps = np.abs(pred - pred_trend)
    worst = TARGETS[int(np.argmax(gaps))]
    c2_dir = delta_map.get("C2", 0.0)
    dir_txt = ("the linear trend suggests C2 would INCREASE further out here"
               if c2_dir > 0.5 else
               ("the linear trend suggests C2 would DECREASE out here"
                if c2_dir < -0.5 else
                "the linear trend suggests little change in C2 out here"))
    st.caption(
        f"Model agreement check: {dir_txt} ({c2_dir:+.1f} points vs the window "
        f"edge). Largest disagreement between the two estimates: "
        f"{gaps.max():.1f} points on {worst}. Large disagreement means this "
        f"region is genuinely unknown - the strongest signal that a plant "
        f"trial here would be informative."
    )

# ------------------------------------------------------ feature importance
st.subheader("What drives the prediction")
fi = bundle.get("feature_importance") or {}
if fi:
    fi_sorted = dict(sorted(fi.items(), key=lambda kv: kv[1]))
    fig2 = go.Figure(go.Bar(
        x=list(fi_sorted.values()), y=list(fi_sorted.keys()),
        orientation="h", marker_color="#5aa89b",
        text=[f"{v:.2f}" for v in fi_sorted.values()], textposition="outside",
    ))
    fig2.update_layout(template="plotly_dark", height=320,
                       margin=dict(l=10, r=30, t=10, b=10),
                       xaxis_title="Relative importance",
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)
    top = max(fi, key=fi.get)
    st.caption(f"**{top}** is currently the strongest lever on the predicted composition.")

# --------------------------------------------------------------- footer
with st.expander("Model details"):
    st.write(f"Algorithm: {bundle['model_name']}")
    if bundle.get("loocv_r2"):
        dfm = pd.DataFrame({
            "Target": TARGETS,
            "LOOCV R^2": [round(bundle["loocv_r2"][t], 3) for t in TARGETS],
            "LOOCV MAE": [round(bundle["loocv_mae"][t], 2) for t in TARGETS],
        })
        st.dataframe(dfm, hide_index=True, use_container_width=True)
    st.caption("Trained on 38 plant trials. C1 and C2 predictions are the most "
               "reliable; minor components (C4-C6) carry higher relative uncertainty.")
