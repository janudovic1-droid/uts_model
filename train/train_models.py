import os
import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder

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

# obdrži samo to, kar UI uporablja
df = df[["structure", "infill", "contours", "material", "layer_thickness", "UTS"]]

# samo PLA in PLA CF
df = df[df["material"].isin(["PLA", "PLA CF"])].reset_index(drop=True)

# ---------------------------------------------------------
# 2) One-hot encoding za structure (da ostane kompatibilno z UI)
# ---------------------------------------------------------
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["structure"]])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["structure"])
)

df_encoded = pd.concat([df.drop(columns=["structure"]), encoded_df], axis=1)

# ---------------------------------------------------------
# 3) Split PLA / PLA CF
# ---------------------------------------------------------
df_pla = df_encoded[df_encoded["material"] == "PLA"].copy()
df_cf = df_encoded[df_encoded["material"] == "PLA CF"].copy()

df_pla = df_pla.drop(columns=["material"])
df_cf = df_cf.drop(columns=["material"])

def prepare_xy(df_local):
    X = df_local.drop(columns=["UTS"]).copy()
    y = df_local["UTS"].copy()

    # jitter: malo šuma na numeričnih feature-ih (ne na one-hot)
    num_cols = ["infill", "contours", "layer_thickness"]
    for c in num_cols:
        if c in X.columns:
            X[c] = X[c] + np.random.normal(0, 0.2, size=len(X))  # zelo majhen šum

    return X, y

X_pla, y_pla = prepare_xy(df_pla)
X_cf, y_cf = prepare_xy(df_cf)

# ---------------------------------------------------------
# 4) CatBoost modeli
# ---------------------------------------------------------
params = dict(
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    n_estimators=800,
    random_seed=42,
    verbose=False,
    l2_leaf_reg=5.0
)

model_pla = CatBoostRegressor(**params)
model_pla.fit(X_pla, y_pla)

with open(os.path.join(model_dir, "model_pla.pkl"), "wb") as f:
    pickle.dump({"model": model_pla, "encoder": encoder}, f)

print("✅ PLA CatBoost model OK")

model_cf = CatBoostRegressor(**params)
model_cf.fit(X_cf, y_cf)

with open(os.path.join(model_dir, "model_pla_cf.pkl"), "wb") as f:
    pickle.dump({"model": model_cf, "encoder": encoder}, f)

print("✅ PLA+CF CatBoost model OK")
