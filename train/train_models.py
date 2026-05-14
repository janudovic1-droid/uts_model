import os
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "Mechanical_properties_edited.xlsx")
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

# ---------------------------------------------------------
# Load + clean
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

# normalize structure
df["structure"] = df["structure"].astype(str).str.strip().str.capitalize()

# unify material names
df["material"] = df["material"].replace({"PLA CF": "PLA+CF"})

df = df[df["material"].isin(["PLA", "PLA+CF"])].reset_index(drop=True)

# ---------------------------------------------------------
# One-hot encode structure
# ---------------------------------------------------------
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["structure"]])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["structure"])
)

df_encoded = pd.concat([df.drop(columns=["structure"]), encoded_df], axis=1)

# ---------------------------------------------------------
# Split PLA / PLA+CF
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
# FINAL CatBoost model (smooth, responsive)
# ---------------------------------------------------------
def train_catboost(X, y):
    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.05,
        n_estimators=1200,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )
    model.fit(X, y)
    return [model]  # app expects list

# PLA
models_pla = train_catboost(X_pla, y_pla)
with open(os.path.join(model_dir, "model_pla.pkl"), "wb") as f:
    pickle.dump({"models": models_pla, "encoder": encoder}, f)

print(" FINAL PLA model OK")

# PLA+CF
models_cf = train_catboost(X_cf, y_cf)
with open(os.path.join(model_dir, "model_pla_cf.pkl"), "wb") as f:
    pickle.dump({"models": models_cf, "encoder": encoder}, f)

print(" FINAL PLA+CF model OK")
