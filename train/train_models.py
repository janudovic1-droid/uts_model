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
# Load + clean Excel
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

df["structure"] = df["structure"].astype(str).str.strip().str.capitalize()
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
# Synthetic feature generator (ADVANCED)
# ---------------------------------------------------------
def add_synthetic_features(df_local: pd.DataFrame) -> pd.DataFrame:
    df_local = df_local.copy()

    # Polinomi
    df_local["infill2"] = df_local["infill"] ** 2
    df_local["infill3"] = df_local["infill"] ** 3
    df_local["infill4"] = df_local["infill"] ** 4

    df_local["contours2"] = df_local["contours"] ** 2
    df_local["contours3"] = df_local["contours"] ** 3

    df_local["layer2"] = df_local["layer_thickness"] ** 2

    # Interakcije
    df_local["infill_x_contours"] = df_local["infill"] * df_local["contours"]
    df_local["infill_x_layer"] = df_local["infill"] * df_local["layer_thickness"]
    df_local["contours_x_layer"] = df_local["contours"] * df_local["layer_thickness"]

    # Log transformacije
    df_local["log_infill"] = np.log(df_local["infill"] + 1)

    # Koren
    df_local["sqrt_infill"] = np.sqrt(df_local["infill"])

    # Eksponent
    df_local["exp_layer"] = np.exp(-df_local["layer_thickness"])

    # Sinus za mikro-gladkost
    df_local["sin_infill"] = np.sin(df_local["infill"] / 10)

    return df_local

# ---------------------------------------------------------
# Wrapper model, ki sam generira dummies pri predict()
# ---------------------------------------------------------
class UTSModel:
    def __init__(self, cat_model, feature_columns):
        self.cat_model = cat_model
        self.feature_columns = feature_columns  # osnovni featurji, ki jih dobi iz appa

    def predict(self, X: pd.DataFrame):
        # X pride iz app.py: infill, contours, layer_thickness + one-hot structure
        X = X.copy()
        # poskrbimo, da so stolpci v pravem vrstnem redu / prisotni
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]
        X_ext = add_synthetic_features(X)
        return self.cat_model.predict(X_ext)

# ---------------------------------------------------------
# Train CatBoost na razširjenih featurjih
# ---------------------------------------------------------
def train_catboost(df_local: pd.DataFrame):
    X_base = df_local.drop(columns=["UTS"])
    y = df_local["UTS"]

    # osnovni featurji, ki jih bo app poslal:
    base_feature_cols = list(X_base.columns)

    # razširjeni featurji za trening
    X_ext = add_synthetic_features(X_base)

    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.03,
        n_estimators=2000,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )
    model.fit(X_ext, y)

    wrapped = UTSModel(model, base_feature_cols)
    return [wrapped]

# ---------------------------------------------------------
# Train PLA
# ---------------------------------------------------------
df_pla = df_encoded[df_encoded["material"] == "PLA"].drop(columns=["material"])
models_pla = train_catboost(df_pla)

with open(os.path.join(model_dir, "model_pla.pkl"), "wb") as f:
    pickle.dump({"models": models_pla, "encoder": encoder}, f)

print(" NAJMOČNEJŠI PLA model OK")

# ---------------------------------------------------------
# Train PLA+CF
# ---------------------------------------------------------
df_cf = df_encoded[df_encoded["material"] == "PLA+CF"].drop(columns=["material"])
models_cf = train_catboost(df_cf)

with open(os.path.join(model_dir, "model_pla_cf.pkl"), "wb") as f:
    pickle.dump({"models": models_cf, "encoder": encoder}, f)

print(" NAJMOČNEJŠI PLA+CF model OK")



