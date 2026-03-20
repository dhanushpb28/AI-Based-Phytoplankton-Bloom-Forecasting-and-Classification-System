import os
import numpy as np
import shap
import joblib

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "hab_xgboost_model.pkl"
)

FEATURE_NAMES = [
    "chl",
    "nppv",
    "no3",
    "po4",
    "nutrients",
    "sst_anomaly",
    "current_speed"
]

FEATURE_LABELS = {
    "chl": "Chlorophyll Concentration",
    "nppv": "Primary Productivity",
    "no3": "Nitrate",
    "po4": "Phosphate",
    "nutrients": "Total Nutrients",
    "sst_anomaly": "Sea Surface Temperature Anomaly",
    "current_speed": "Ocean Current Speed"
}


def get_high_risk_importance(ds_day):

    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)

    nutrients = ds_day.no3 + ds_day.po4
    current_speed = np.sqrt(ds_day.uo**2 + ds_day.vo**2)

    mask = ~np.isnan(ds_day["chl"].values)

    X = np.stack([
        ds_day["chl"].values[mask],
        ds_day["nppv"].values[mask],
        ds_day["no3"].values[mask],
        ds_day["po4"].values[mask],
        nutrients.values[mask],
        ds_day["sea_surface_temperature_anomaly"].values[mask],
        current_speed.values[mask]
    ], axis=1)

    shap_values = explainer.shap_values(X)

    # Class 2 = High HAB Risk
    shap_class2 = shap_values[2]

    mean_abs = np.mean(np.abs(shap_class2), axis=0)

    importance = sorted(
        zip(FEATURE_NAMES, mean_abs),
        key=lambda x: x[1],
        reverse=True
    )

    ranked_features = [
        FEATURE_LABELS.get(feat, feat)
        for feat, _ in importance
    ]

    return ranked_features, shap_class2, X