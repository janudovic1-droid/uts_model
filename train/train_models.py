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
# Synthetic feature generator (ULTRA+)
# ---------------------------------------------------------
def add_synthetic_features(df_local: pd.DataFrame) -> pd.DataFrame:
    df_local = df_local.copy()

    df_local["infill2"] = df_local["infill"] ** 2
    df_local["infill3"] = df_local["infill"] ** 3
    df_local["infill4"] = df_local["infill"] ** 4
    df_local["infill5"] = df_local["infill"] ** 5

    df_local["contours2"] = df_local["contours"] ** 2
    df_local["contours3"] = df_local["contours"] ** 3

    df_local["layer2"] = df_local["layer_thickness"] ** 2
    df_local["layer3"] = df_local["layer_thickness"] ** 3

    df_local["infill_x_contours"] = df_local["infill"] * df_local["contours"]
    df_local["infill_x_layer"] = df_local["infill"] * df_local["layer_thickness"]
    df_local["contours_x_layer"] = df_local["contours"] * df_local["layer_thickness"]

    df_local["infill2_x_layer"] = (df_local["infill"] ** 2) * df_local["layer_thickness"]
    df_local["infill_x_contours2"] = df_local["infill"] * (df_local["contours"] ** 2)

    df_local["log_infill"] = np.log(df_local["infill"] + 1)
    df_local["log_contours"] = np.log(df_local["contours"] + 1)

    df_local["sqrt_infill"] = np.sqrt(df_local["infill"])
    df_local["sqrt_layer"] = np.sqrt(df_local["layer_thickness"])

    df_local["exp_layer"] = np.exp(-df_local["layer_thickness"])
    df_local["exp_infill"] = np.exp(-df_local["infill"] / 100.0)

    df_local["sin_infill"] = np.sin(df_local["infill"] / 10)
    df_local["cos_infill"] = np.cos(df_local["infill"] / 10)

    return df_local
# ---------------------------------------------------------
# Train base model (na realnih Excel podatkih)
# ---------------------------------------------------------
def train_base_model(df_local: pd.DataFrame):
    X_base = df_local.drop(columns=["UTS"])
    y = df_local["UTS"]

    X_ext = add_synthetic_features(X_base)

    model = CatBoostRegressor(
        depth=8,
        learning_rate=0.03,
        n_estimators=1500,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )
    model.fit(X_ext, y)
    return model, list(X_base.columns)

# ---------------------------------------------------------
# Generate dense synthetic grid (infill, contours, layer)
# ---------------------------------------------------------
def generate_dense_grid(material_name: str, encoder: OneHotEncoder, df_full: pd.DataFrame):
    df_mat = df_full[df_full["material"] == material_name]

    inf_min, inf_max = int(df_mat["infill"].min()), int(df_mat["infill"].max())
    cont_min, cont_max = int(df_mat["contours"].min()), int(df_mat["contours"].max())
    layer_min, layer_max = df_mat["layer_thickness"].min(), df_mat["layer_thickness"].max()

    inf_range = np.arange(inf_min, inf_max + 1, 1)          # vsak % infilla
    cont_range = np.arange(cont_min, cont_max + 1, 1)       # vsaka kontura
    layer_range = np.arange(layer_min, layer_max + 0.0001, 0.02)  # vsakih 0.02 mm

    structures = list(encoder.categories_[0])

    rows = []
    for s in structures:
        struct_ohe = encoder.transform(pd.DataFrame([[s]], columns=["structure"]))[0]
        struct_cols = encoder.get_feature_names_out(["structure"])

        for inf in inf_range:
            for c in cont_range:
                for lay in layer_range:
                    row = {
                        "material": material_name,
                        "infill": float(inf),
                        "contours": float(c),
                        "layer_thickness": float(lay),
                    }
                    for col, val in zip(struct_cols, struct_ohe):
                        row[col] = float(val)
                    rows.append(row)

    return pd.DataFrame(rows)

# ---------------------------------------------------------
# Train ULTRA+ model for one material (Excel weighted)
# ---------------------------------------------------------
def train_ultra_plus(material_name: str):
    df_mat = df_encoded[df_encoded["material"] == material_name].drop(columns=["material"])

    # 1) Base model samo na Excelu
    base_model, feature_cols = train_base_model(df_mat)

    # 2) Synthetic grid
    grid_df = generate_dense_grid(material_name, encoder, df)

    grid_encoded = grid_df.drop(columns=["material"])
    X_grid_ext = add_synthetic_features(grid_encoded)

    uts_grid = base_model.predict(X_grid_ext)
    grid_encoded["UTS"] = uts_grid

    # 3) Združimo Excel + synthetic
    df_full = pd.concat([df_mat, grid_encoded], axis=0).reset_index(drop=True)

    # 4) Uteži (Excel = 10, synthetic = 1)
    weights = np.where(df_full.index < len(df_mat), 10, 1)

    X_base_full = df_full.drop(columns=["UTS"])
    y_full = df_full["UTS"]

    X_ext_full = add_synthetic_features(X_base_full)

    # 5) Final ULTRA+ model
    final_model = CatBoostRegressor(
        depth=8,
        learning_rate=0.03,
        n_estimators=2000,
        loss_function="RMSE",
        random_seed=123,
        verbose=False
    )
    final_model.fit(X_ext_full, y_full, sample_weight=weights)

    return final_model, feature_cols

# ---------------------------------------------------------
# Train PLA
# ---------------------------------------------------------
model_pla, feature_cols_pla = train_ultra_plus("PLA")

with open(os.path.join(model_dir, "model_pla.pkl"), "wb") as f:
    pickle.dump(
        {"model": model_pla, "encoder": encoder, "feature_columns": feature_cols_pla},
        f,
    )

print(" ULTRA+ PLA model OK")

# ---------------------------------------------------------
# Train PLA+CF
# ---------------------------------------------------------
model_cf, feature_cols_cf = train_ultra_plus("PLA+CF")

with open(os.path.join(model_dir, "model_pla_cf.pkl"), "wb") as f:
    pickle.dump(
        {"model": model_cf, "encoder": encoder, "feature_columns": feature_cols_cf},
        f,
    )

print(" ULTRA+ PLA+CF model OK")
