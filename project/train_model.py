"""
train_model.py
Run this FIRST to train all three models before launching the Flask app.
Usage: python train_model.py
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
CROP_PATH    = "dataset/Agriculture In India.csv"
MODEL_DIR    = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── CROP YIELD RANGES (tons/ha)  ───────────────────────────────────────────────
CROP_YIELD_RANGES = {
    "Rice"          : (0, 10),
    "Wheat"         : (0, 10),
    "Maize"         : (0, 15),
    "Sugarcane"     : (0, 100),
    "Cotton(lint)"  : (0, 5),
    "Groundnut"     : (0, 6),
    "Bajra"         : (0, 6),
    "Jowar"         : (0, 6),
    "Coconut"       : (0, 30),
    "Moong(Green Gram)": (0, 4),
    "Urad"          : (0, 4),
}
DEFAULT_RANGE = (0, 50)

# ── Load & Clean ───────────────────────────────────────────────────────────────
print("📂  Loading dataset …")
df = pd.read_csv(CROP_PATH)
df.fillna(0, inplace=True)
df['Production'] = pd.to_numeric(df['Production'], errors='coerce').fillna(0).astype(np.float32)
df['Area']       = pd.to_numeric(df['Area'],       errors='coerce').fillna(0).astype(np.float32)
df['Area']       = df['Area'].replace(0, 0.0001)
df['Yield']      = df['Production'] / df['Area']

# Drop extreme outliers (yield > 200 t/ha — clearly bad data)
df = df[df['Yield'] <= 200].reset_index(drop=True)

# ── Encode ─────────────────────────────────────────────────────────────────────
le_state    = LabelEncoder().fit(df['State_Name'])
le_district = LabelEncoder().fit(df['District_Name'])
le_season   = LabelEncoder().fit(df['Season'])
le_crop     = LabelEncoder().fit(df['Crop'])

df['State_Enc']    = le_state.transform(df['State_Name'])
df['District_Enc'] = le_district.transform(df['District_Name'])
df['Season_Enc']   = le_season.transform(df['Season'])
df['Crop_Enc']     = le_crop.transform(df['Crop'])

# ── Build lookup maps (for app dropdowns) ──────────────────────────────────────
state_district_map  = df.groupby('State_Enc')['District_Enc'].unique().to_dict()
district_crop_map   = {}
for dc, grp in df.groupby('District_Enc'):
    district_crop_map[dc] = list(le_crop.inverse_transform(grp['Crop_Enc'].unique()))

# ── Features & Target ─────────────────────────────────────────────────────────
FEATURES = ['State_Enc', 'District_Enc', 'Crop_Year', 'Season_Enc', 'Crop_Enc', 'Area']
X = df[FEATURES].values.astype(np.float32)
Y_raw = df['Yield'].values.astype(np.float32)

# Per-crop normalised target  [0, 1]
def get_range(crop_name):
    return CROP_YIELD_RANGES.get(crop_name, DEFAULT_RANGE)

crop_names_arr = le_crop.inverse_transform(df['Crop_Enc'].values)
lo_arr = np.array([get_range(c)[0] for c in crop_names_arr], dtype=np.float32)
hi_arr = np.array([get_range(c)[1] for c in crop_names_arr], dtype=np.float32)
Y_norm = np.clip((Y_raw - lo_arr) / (hi_arr - lo_arr + 1e-8), 0, 1).reshape(-1, 1)

# ── Scale X ───────────────────────────────────────────────────────────────────
scalerX = StandardScaler()
X_scaled = scalerX.fit_transform(X)

# ── Train / Val split ─────────────────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, Y_norm, test_size=0.15, random_state=42)

# ── Save encoders & scalers ───────────────────────────────────────────────────
with open(f"{MODEL_DIR}/encoders_scaler.pkl", "wb") as f:
    pickle.dump(
        (le_state, le_district, le_season, le_crop,
         scalerX, CROP_YIELD_RANGES, state_district_map, district_crop_map),
        f
    )
print("✅  Encoders & mappings saved.")

# ── Common callbacks ──────────────────────────────────────────────────────────
def get_callbacks():
    return [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)
    ]

def save_model(model, name):
    with open(f"{MODEL_DIR}/{name}.json", "w") as f:
        f.write(model.to_json())
    model.save_weights(f"{MODEL_DIR}/{name}.weights.h5")
    print(f"✅  {name} saved.")

# ── 1. Feedforward NN (improved) ──────────────────────────────────────────────
print("\n🔄  Training FeedForward NN …")
ffmodel = Sequential([
    Dense(512, input_dim=X_tr.shape[1], activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(64,  activation='relu'),
    Dense(1,   activation='sigmoid')       # output in [0,1] for normalised yield
])
ffmodel.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
ffmodel.fit(X_tr, y_tr, epochs=50, batch_size=256,
            validation_data=(X_val, y_val), callbacks=get_callbacks(), verbose=1)
save_model(ffmodel, "ffmodel")

# ── 2. RNN-style Deep Dense (improved) ───────────────────────────────────────
print("\n🔄  Training RNN (deep dense) …")
rnnmodel = Sequential([
    Dense(512, input_dim=X_tr.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64,  activation='relu'),
    Dense(1,   activation='sigmoid')
])
rnnmodel.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss='mse', metrics=['mae'])
rnnmodel.fit(X_tr, y_tr, epochs=50, batch_size=256,
             validation_data=(X_val, y_val), callbacks=get_callbacks(), verbose=1)
save_model(rnnmodel, "rnnmodel")

# ── 3. LSTM (improved) ────────────────────────────────────────────────────────
print("\n🔄  Training LSTM …")
X_tr_lstm  = X_tr.reshape(X_tr.shape[0], X_tr.shape[1], 1)
X_val_lstm = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

lstmmodel = Sequential([
    LSTM(256, input_shape=(X_tr.shape[1], 1), return_sequences=True),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1,  activation='sigmoid')
])
lstmmodel.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss='mse', metrics=['mae'])
lstmmodel.fit(X_tr_lstm, y_tr, epochs=50, batch_size=256,
              validation_data=(X_val_lstm, y_val), callbacks=get_callbacks(), verbose=1)
save_model(lstmmodel, "lstmmodel")

print("\n🎉  All models trained successfully! You can now run: python app.py")