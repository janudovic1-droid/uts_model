import os
import pickle
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------
# 0) Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "Mechanical_properties_edited.xlsx")
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

# ---------------------------------------------------------
# 1) Load and clean data
# ---------------------------------------------------------
df = pd.read_excel(data_path)

df = df.rename(columns={
    "Structure": "structure",
    "Infill %": "infill",
    "št kontur": "contours",
    "Material": "material",
    "Debelina layer mm": "layer_thickness",
    "Natezna trdnost_max": "UTS"
})

df = df[["structure", "infill", "contours", "material", "layer_thickness", "UTS"]]

# poenoti ime materiala
df["material"] = df["material"].replace({"PLA CF": "PLA+CF"})

df = df[df["material"].isin(["PLA", "PLA+CF"])].reset_index(drop=True)

# ---------------------------------------------------------
# 2) One-hot encoding za structure
# ---------------------------------------------------------
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["structure"]])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["structure"])
)

df_encoded = pd.concat([df.drop(columns=["structure"]), encoded_df], axis=1)

# ---------------------------------------------------------
# 3) Split PLA / PLA+CF
# ---------------------------------------------------------
df_pla = df_encoded[df_encoded["material"] == "PLA"].copy()
df_cf = df_encoded[df_encoded["material"] == "PLA+CF"].copy()

df_pla = df_pla.drop(columns=["material"])
df_cf = df_cf.drop(columns=["material"])

def prepare_xy(df_local):
    X = df_local.drop(columns=["UTS"]).copy()
    y = df_local["UTS"].copy()
    return X, y

X_pla, y_pla = prepare_xy(df_pla)
X_cf, y_cf = prepare_xy(df_cf)

# ---------------------------------------------------------
# 4) MODELI — ENSEMBLE
# ---------------------------------------------------------

def train_ensemble(X, y):
    models = []

    # CatBoost
    cb = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        n_estimators=800,
        random_seed=42,
        verbose=False
    )
    cb.fit(X, y)
    models.append(cb)

    # Random Forest (bolj občutljiv)
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_split=2,
        random_state=42
    )
    rf.fit(X, y)
    models.append(rf)

    # Gradient Boosting (nelinearen)
    gb = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    gb.fit(X, y)
    models.append(gb)

    return models

# treniraj PLA
models_pla = train_ensemble(X_pla, y_pla)
with open(os.path.join(model_dir, "model_pla.pkl"), "wb") as f:
    pickle.dump({"models": models_pla, "encoder": encoder}, f)

print(" PLA ensemble OK")

# treniraj PLA+CF
models_cf = train_ensemble(X_cf, y_cf)
with open(os.path.join(model_dir, "model_pla_cf.pkl"), "wb") as f:
    pickle.dump({"models": models_cf, "encoder": encoder}, f)

print(" PLA+CF ensemble OK")

