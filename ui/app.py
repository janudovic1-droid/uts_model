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
# Load model + encoder (always reload)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False, ttl=1)
def load_model(material):
    if material == "PLA":
        model_path = os.path.join(MODEL_DIR, "model_pla.pkl")
    else:
        model_path = os.path.join(MODEL_DIR, "model_pla_cf.pkl")

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    models = obj["models"]
    encoder = obj["encoder"]

    return models, encoder

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("Napovedni model UTS")

structure = st.selectbox("Izberi strukturo:", ["Hex", "Gyroid", "Grid"])
material = st.selectbox("Izberi material:", ["PLA", "PLA+CF"])
infill = st.number_input("Infill (%)", min_value=0, max_value=100, value=40)
contours = st.number_input("Število kontur", min_value=0, max_value=100, value=1)
layer = st.number_input("Debelina layerja (mm)", min_value=0.05, max_value=1.0, value=0.20, step=0.01)

if st.button("Napovej UTS"):
    models, encoder = load_model(material)

    # Prepare input
    input_df = pd.DataFrame([{
        "structure": structure,
        "infill": infill,
        "contours": contours,
        "layer_thickness": layer
    }])

    # One-hot encode structure
    encoded = encoder.transform(input_df[["structure"]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["structure"]))

    X = pd.concat([input_df.drop(columns=["structure"]), encoded_df], axis=1)

    # Predict (single model)
    model = models[0]
    uts_pred = float(model.predict(X)[0])

    st.success(f"Napovedana natezna trdnost (UTS): {uts_pred:.2f} MPa")

    # History
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Structure": structure,
        "Material": material,
        "Infill": infill,
        "Contours": contours,
        "Layer": layer,
        "UTS napoved": round(uts_pred, 2)
    })

    st.subheader("Zgodovina napovedi")
    st.dataframe(pd.DataFrame(st.session_state.history))
