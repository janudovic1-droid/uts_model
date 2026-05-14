import os
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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

# normalizacija struktur (da se ujema z UI: Hex, Tri, Lin)
df["structure"] = df["structure"].astype(str).str.strip().str.capitalize()

# poenoti ime materiala
df["material"] = df["material"].replace({"PLA CF": "PLA+CF"})

# obdrži samo PLA in PLA+CF
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
# 4) Data augmentation (da model reagira tudi na manjše spremembe)
# ---------------------------------------------------------
def augment(X, y, n_times=10, noise_infill=1.0, noise_layer=0.02):
    X_aug = [X]
    y_aug = [y]
    for _ in range(n_times):
        X_noisy = X.copy()
        if "infill" in X_noisy.columns:
            X_noisy["infill"] = X_noisy["infill"] + np.random.normal(0, noise_infill, size=len(X_noisy))
        if "layer_thickness" in X_noisy.columns:
            X_noisy["layer_thickness"] = X_noisy["layer_thickness"] + np.random.normal(0, noise_layer, size=len(X_noisy))
        y_noisy = y + np.random.normal(0, 0.1, size=len(y))
        X_aug.append(X_noisy)
        y_aug.append(y_noisy)
    X_all = pd.concat(X_aug, ignore_index=True)
    y_all = pd.concat(y_aug, ignore_index=True)
    return X_all, y_all

X_pla_aug, y_pla_aug = augment(X_pla, y_pla)
X_cf_aug, y_cf_aug = augment(X_cf, y_cf)

# ---------------------------------------------------------
# 5) RandomForest modeli (preprost, odziven, stabilen)
# ---------------------------------------------------------
def train_rf(X, y):
    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    rf.fit(X, y)
    # app pričakuje "models" (seznam), zato ga damo v list
    return [rf]

models_pla = train_rf(X_pla_aug, y_pla_aug)
with open(os.path.join(model_dir, "model_pla.pkl"), "wb") as f:
    pickle.dump({"models": models_pla, "encoder": encoder}, f)

print(" FINAL PLA model OK")

models_cf = train_rf(X_cf_aug, y_cf_aug)
with open(os.path.join(model_dir, "model_pla_cf.pkl"), "wb") as f:
    pickle.dump({"models": models_cf, "encoder": encoder}, f)

print(" FINAL PLA+CF model OK")

