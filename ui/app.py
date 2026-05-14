import os
import pickle
import streamlit as st
import pandas as pd

# ---------------------------------------------------------
# 1) Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# ---------------------------------------------------------
# 2) Load model based on material
# ---------------------------------------------------------
@st.cache_resource
def load_model(material):
    if material == "PLA":
        model_path = os.path.join(MODEL_DIR, "model_pla.pkl")
    else:
        model_path = os.path.join(MODEL_DIR, "model_pla_cf.pkl")

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    return obj["model"], obj["encoder"]

# ---------------------------------------------------------
# 3) Init history
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------------------------------------------------
# 4) UI
# ---------------------------------------------------------
st.title("Napovedni model UTS")
st.write("Vnesi parametre 3D tiska za napoved natezne trdnosti (UTS).")

structure = st.selectbox("Structure", ["Hex", "Tri", "Lin"])
material = st.selectbox("Material", ["PLA", "PLA+CF"])

infill = st.number_input("Infill (%)", min_value=0, max_value=100, value=40, step=1)
contours = st.number_input("Število kontur", min_value=1, max_value=10, value=1, step=1)
layer = st.number_input("Debelina layerja (mm)", min_value=0.1, max_value=1.0, value=0.2, step=0.05)

# ---------------------------------------------------------
# 5) Prediction
# ---------------------------------------------------------
if st.button("Napovej UTS"):
    model, encoder = load_model(material)

    # Prepare input
    input_df = pd.DataFrame({
        "structure": [structure],
        "infill": [infill],
        "contours": [contours],
        "layer_thickness": [layer]
    })

    # One-hot encoding for structure
    encoded = encoder.transform(input_df[["structure"]])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(["structure"])
    )

    # Final model input
    model_input = pd.concat(
        [input_df.drop(columns=["structure"]), encoded_df],
        axis=1
    )

    # Predict
    prediction = model.predict(model_input)[0]

    # Save to history
    st.session_state["history"].append({
        "Structure": structure,
        "Material": material,
        "Infill": infill,
        "Contours": contours,
        "Layer": layer,
        "UTS napoved": round(prediction, 2)
    })

    st.success(f"Napovedana natezna trdnost (UTS): {prediction:.2f} MPa")

# ---------------------------------------------------------
# 6) Show history
# ---------------------------------------------------------
st.subheader("Zgodovina napovedi")

if len(st.session_state["history"]) > 0:
    st.dataframe(pd.DataFrame(st.session_state["history"]))
else:
    st.info("Zaenkrat še ni napovedi.")
