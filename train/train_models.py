import os
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder
from statsmodels.nonparametric.smoothers_lowess import lowess

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
# Smooth UTS using LOWESS (realistic + smooth)
# ---------------------------------------------------------
def smooth_material(df_local):
    df_local = df_local.copy()
    df_local = df_local.sort_values("infill")

    # LOWESS smoothing
    smoothed = lowess(df_local["UTS"], df_local["infill"], frac=0.4)
    df_local["UTS_smooth"] = smoothed[:, 1]

    # interpolate missing infill values (1% resolution)
    infill_new = np.arange(df_local["infill"].min(), df_local["infill"].max() + 1, 1)
    uts_new = np.interp(infill_new, df_local["infill"], df_local["UTS_smooth"])

    # rebuild dataframe
    df_new = pd.DataFrame({
        "infill": infill_new,
        "UTS": uts_new,
        "contours": df_local["contours"].iloc[0],
        "layer_thickness": df_local["layer_thickness"].iloc[0]
    })

    # add structure one-hot columns
    for col in df_local.columns:
        if col.startswith("structure_"):
            df_new[col] = df_local[col].iloc[0]

    return df_new

# split by material
df_pla = df_encoded[df_encoded["material"] == "PLA"].drop(columns=["material"])
df_cf = df_encoded[df_encoded["material"] == "PLA+CF"].drop(columns=["material"])

df_pla_smooth = smooth_material(df_pla)
df_cf_smooth = smooth_material(df_cf)

# ---------------------------------------------------------
# Train CatBoost (smooth + real)
# ---------------------------------------------------------
def train_catboost(df_local):
    X = df_local.drop(columns=["UTS"])
    y = df_local["UTS"]

    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.03,
        n_estimators=1500,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )
    model.fit(X, y)
    return [model]

# PLA
models_pla = train_catboost(df_pla_smooth)
with open(os.path.join(model_dir, "model_pla.pkl"), "wb") as f:
    pickle.dump({"models": models_pla, "encoder": encoder}, f)

print(" FINAL SMOOTH REAL PLA model OK")

# PLA+CF
models_cf = train_catboost(df_cf_smooth)
with open(os.path.join(model_dir, "model_pla_cf.pkl"), "wb") as f:
    pickle.dump({"models": models_cf, "encoder": encoder}, f)

print(" FINAL SMOOTH REAL PLA+CF model OK")
