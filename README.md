# 🧪 Product Composition Predictor (C1–C6)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ExtraTrees-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An interactive machine-learning tool that predicts the output composition
(**C1–C6**) of a continuous reactor from process conditions, with the goal of
**maximizing C2** while suppressing the remaining components.

Built as a multi-output **Extra Trees** model trained on plant trial data and
deployed as a **Streamlit** web app for live what-if exploration.

---

## ✨ Features

- **Live prediction** — six-component composition updates instantly as sliders move, normalized to 100 %
- **Dynamic composition chart** — headline percentages + dark horizontal bar chart
- **Extended exploration ranges** — sliders reach well beyond the validated data window; an *Extrapolation mode* banner warns when inputs leave the trusted zone
- **Derived process quantities** — M2, RM2, X and YRM12 computed live from calibrated process relations (implementation details intentionally not exposed in the UI)
- **Feature importance** — shows which inputs drive the prediction most
- **Constraint handling** — V restricted to multiples of 8
- **Self-healing model bundle** — if `model_bundle.joblib` fails to load (e.g. scikit-learn version mismatch), the app silently retrains from `data.csv` and re-saves

## 📊 Model

| | |
|---|---|
| Algorithm | Extra Trees (600 trees, √-feature sampling), multi-output |
| Training data | 38 plant trials |
| Validation | Leave-one-out cross-validation (LOOCV) |
| Key scores | C2 R² = 0.72 · C1 R² = 0.68 |

Minor components (C4–C6) sit near the detection limit and carry higher
relative uncertainty — this is surfaced honestly in the app's *Model details*
panel.

## 🚀 Quick start

```bash
git clone <your-repo-url>
cd composition-predictor
pip install -r requirements.txt
streamlit run app.py
```

### Share on a local / corporate network

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then share `http://<your-ip>:8501` with colleagues (allow the port through
the firewall if needed).

### Retrain from scratch

```bash
python train.py
```

Runs the full LOOCV comparison (Extra Trees vs. Random Forest vs. Gradient
Boosting vs. Ridge) and saves the best bundle to `model_bundle.joblib`.

## 📁 Repository structure

```
├── app.py                  # Streamlit UI
├── train.py                # LOOCV model comparison + retraining
├── data.csv                # plant trial dataset
├── model_bundle.joblib     # trained model + metadata (safe ranges, importances)
├── requirements.txt
├── .streamlit/
│   └── config.toml         # dark theme
├── .gitignore
├── LICENSE
└── README.md
```

## ⚠️ Notes

- Predictions outside the validated operating window are **directional
  estimates**, not interpolations — treat them as hypotheses to verify with a
  plant trial.
- If the dataset is proprietary, keep this repository **private** (or remove
  `data.csv` and `model_bundle.joblib` before publishing).

## 📄 License

MIT — see [LICENSE](LICENSE).
