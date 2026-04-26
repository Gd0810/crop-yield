"""
app.py  –  Flask backend for Crop Yield Prediction
Run: python app.py   (after training with train_model.py)
"""
import os, pickle, json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from pathlib import Path

app = Flask(__name__)
MODEL_DIR = Path("model")

# ── Load once at startup ───────────────────────────────────────────────────────
with open(MODEL_DIR / "encoders_scaler.pkl", "rb") as f:
    (le_state, le_district, le_season, le_crop,
     scalerX, CROP_YIELD_RANGES, state_district_map, district_crop_map) = pickle.load(f)

DEFAULT_RANGE = (0, 50)

MODELS = {}
def get_model(name):
    if name not in MODELS:
        with open(MODEL_DIR / f"{name}.json") as jf:
            m = tf.keras.models.model_from_json(jf.read())
        m.load_weights(str(MODEL_DIR / f"{name}.weights.h5"))
        MODELS[name] = m
    return MODELS[name]

# Pre-load all three
for _m in ("ffmodel", "rnnmodel", "lstmmodel"):
    try: get_model(_m)
    except Exception as e: print(f"⚠️  Could not pre-load {_m}: {e}")

# ── Helpers ────────────────────────────────────────────────────────────────────
def state_list():
    return sorted(le_state.classes_.tolist())

def districts_for(state_name):
    sc = int(le_state.transform([state_name])[0])
    codes = state_district_map.get(sc, [])
    return sorted(le_district.inverse_transform(codes).tolist())

def crops_for(district_name):
    dc = int(le_district.transform([district_name])[0])
    return sorted(district_crop_map.get(dc, le_crop.classes_.tolist()))

def season_list():
    return sorted(le_season.classes_.tolist())

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           states=state_list(),
                           seasons=season_list(),
                           models=["ffmodel", "rnnmodel", "lstmmodel"])

@app.route("/api/districts")
def api_districts():
    state = request.args.get("state", "")
    try:
        return jsonify(districts_for(state))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/crops")
def api_crops():
    district = request.args.get("district", "")
    try:
        return jsonify(crops_for(district))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        state    = data["state"]
        district = data["district"]
        season   = data["season"]
        crop     = data["crop"]
        year     = float(data["year"])
        area     = float(data["area"])
        model_id = data.get("model", "ffmodel")

        # Encode
        state_enc    = int(le_state.transform([state])[0])
        district_enc = int(le_district.transform([district])[0])
        season_enc   = int(le_season.transform([season])[0])
        crop_enc     = int(le_crop.transform([crop])[0])

        X_in = np.array([[state_enc, district_enc, year, season_enc, crop_enc, area]], dtype=np.float32)
        X_sc = scalerX.transform(X_in)

        model = get_model(model_id)
        if len(model.input_shape) == 3:          # LSTM
            X_sc = X_sc.reshape(1, X_sc.shape[1], 1)

        y_norm = float(model.predict(X_sc, verbose=0)[0][0])
        y_norm = np.clip(y_norm, 0, 1)

        lo, hi = CROP_YIELD_RANGES.get(crop, DEFAULT_RANGE)
        y_yield      = y_norm * (hi - lo) + lo
        y_yield      = float(np.clip(y_yield, lo, hi))
        y_production = y_yield * area

        # Comparison across all three models
        comparison = {}
        for mid in ("ffmodel", "rnnmodel", "lstmmodel"):
            try:
                m = get_model(mid)
                xi = scalerX.transform(X_in)
                if len(m.input_shape) == 3:
                    xi = xi.reshape(1, xi.shape[1], 1)
                yn = float(np.clip(m.predict(xi, verbose=0)[0][0], 0, 1))
                comparison[mid] = round(yn * (hi - lo) + lo, 3)
            except:
                comparison[mid] = None

        # Monthly seasonal yield distribution (illustrative based on crop range)
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        np.random.seed(int(crop_enc + season_enc))
        seasonal_weights = np.random.dirichlet(np.ones(12) * 2) * 12
        seasonal_yields  = [round(float(y_yield * w / 12), 3) for w in seasonal_weights]

        return jsonify({
            "yield"      : round(y_yield, 3),
            "production" : round(y_production, 3),
            "area"       : area,
            "crop"       : crop,
            "state"      : state,
            "district"   : district,
            "season"     : season,
            "year"       : int(year),
            "model"      : model_id,
            "lo"         : lo,
            "hi"         : hi,
            "comparison" : comparison,
            "months"     : months,
            "seasonal_yields": seasonal_yields,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)