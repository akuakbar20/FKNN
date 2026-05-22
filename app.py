import streamlit as st
import numpy as np
import pickle

# LOAD MODEL
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

st.write(type(model))
st.write(model)

# =========================
# LABEL MAP (Covertype)
# =========================
label_map = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="Covertype Classifier", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌲 Covertype Forest Classification")
st.write("Masukkan nilai fitur untuk memprediksi jenis hutan")

# =========================
# INPUT FEATURES
# =========================
col1, col2 = st.columns(2)

with col1:
    elevation = st.number_input("Elevation", value=0.0)
    aspect = st.number_input("Aspect", value=0.0)
    slope = st.number_input("Slope", value=0.0)
    hydro_dist = st.number_input("Horizontal Distance to Hydrology", value=0.0)
    vert_hydro = st.number_input("Vertical Distance to Hydrology", value=0.0)

with col2:
    road_dist = st.number_input("Horizontal Distance to Roadways", value=0.0)
    hill_9 = st.number_input("Hillshade 9am", value=0.0)
    hill_noon = st.number_input("Hillshade Noon", value=0.0)
    hill_3 = st.number_input("Hillshade 3pm", value=0.0)

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi"):
    input_data = np.array([[
        elevation, aspect, slope,
        hydro_dist, vert_hydro,
        road_dist,
        hill_9, hill_noon, hill_3
    ]])

    prediction = model.predict(input_data)[0]

    hasil = label_map.get(prediction, "Unknown")

    st.success(f"🌲 Hasil Prediksi: {hasil} (Class {prediction})")
