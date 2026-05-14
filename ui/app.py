import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# ------------------------------------------------------------
# Synthetic feature generator – ISTO kot v train_models.py
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Load model + encoder + feature_columns
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False, ttl=1)
def load_model(material):
    if material == "PLA":
        model_path = os.path.join(MODEL_DIR, "model_pla.pkl")
    else:
        model_path = os.path.join(MODEL_DIR, "model_pla_cf.pkl")

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    return obj["model"], obj["encoder"], obj["feature_columns"]

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("Napovedni model UTS")

structure = st.selectbox("Izberi strukturo:", ["Hex", "Tri", "Lin"])
material = st.selectbox("Izberi material:", ["PLA", "PLA+CF"])
infill = st.number_input("Infill (%)", min_value=0, max_value=100, value=40, step=5)
contours = st.number_input("Število kontur", min_value=0, max_value=100, value=1)
layer = st.number_input("Debelina layerja (mm)", min_value=0.05, max_value=1.0, value=0.20, step=0.01)

# ------------------------------------------------------------
# History init (OUTSIDE button → no duplicates)
# ------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------------------------------------------
# Prediction button
# ------------------------------------------------------------
if st.button("Napovej UTS", key="predict"):
    model, encoder, feature_columns = load_model(material)

    # osnovni input
    input_df = pd.DataFrame([{
        "structure": structure,
        "infill": infill,
        "contours": contours,
        "layer_thickness": layer
    }])

    # one-hot encoding
    encoded = encoder.transform(input_df[["structure"]])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(["structure"])
    )

    X_base = pd.concat(
        [input_df.drop(columns=["structure"]), encoded_df],
        axis=1
    )

    # poskrbimo za isti vrstni red stolpcev kot pri treningu
    for col in feature_columns:
        if col not in X_base.columns:
            X_base[col] = 0
    X_base = X_base[feature_columns]

    # synthetic dummies
    X_ext = add_synthetic_features(X_base)

    uts_pred = float(model.predict(X_ext)[0])

    st.success(f"Napovedana natezna trdnost (UTS): {uts_pred:.2f} MPa")

    # --------------------------------------------------------
    # Add to history (NO DUPLICATES)
    # --------------------------------------------------------
    entry = {
        "Structure": structure,
        "Material": material,
        "Infill": infill,
        "Contours": contours,
        "Layer": layer,
        "UTS napoved": round(uts_pred, 2)
    }

    if len(st.session_state.history) == 0 or st.session_state.history[-1] != entry:
        st.session_state.history.append(entry)

# ------------------------------------------------------------
# Show history
# ------------------------------------------------------------
st.subheader("Zgodovina napovedi")
st.dataframe(pd.DataFrame(st.session_state.history))
