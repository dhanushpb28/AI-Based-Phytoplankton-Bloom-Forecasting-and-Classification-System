import os
import numpy as np
import joblib

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "hab_xgboost_model.pkl"
)

FEATURES = [
    "chl",
    "nppv",
    "no3",
    "po4",
    "nutrients",
    "sea_surface_temperature_anomaly",
    "current_speed"
]


def classify_hab(ds_day, bloom_threshold):

    model = joblib.load(MODEL_PATH)

    # Derived variables
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

    preds = model.predict(X)

    risk_map = np.full(ds_day["chl"].shape, np.nan)
    risk_map[mask] = preds

    # -----------------------------
    # BLOOM GATING (important)
    # -----------------------------
    chl = ds_day["chl"].values
    bloom_mask = chl >= bloom_threshold

    risk_map = np.where(bloom_mask, risk_map, 0)

    return risk_map