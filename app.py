import streamlit as st
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances

# =========================
# LOAD MODEL FKNN
# =========================
with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)

k = data["k"]
m = data["m"]

classes = data["classes"]

scaler = data["scaler"]

X_train = data["X_train"]
U_train = data["U_train"]

# =========================
# LABEL CLASS
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
# PAGE
# =========================
st.set_page_config(
    page_title="Covertype Classifier",
    layout="centered"
)

st.title(
    "🌲 Covertype Forest Classification"
)

st.write(
    "Masukkan nilai fitur untuk memprediksi jenis hutan"
)

# =========================
# INPUT
# =========================
col1, col2 = st.columns(2)

with col1:

    elevation = st.number_input(
        "Elevation",
        value=0.0
    )

    aspect = st.number_input(
        "Aspect",
        value=0.0
    )

    slope = st.number_input(
        "Slope",
        value=0.0
    )

    hydro_dist = st.number_input(
        "Horizontal Distance to Hydrology",
        value=0.0
    )

    vert_hydro = st.number_input(
        "Vertical Distance to Hydrology",
        value=0.0
    )

with col2:

    road_dist = st.number_input(
        "Horizontal Distance to Roadways",
        value=0.0
    )

    hill_9 = st.number_input(
        "Hillshade 9am",
        value=0.0
    )

    hill_noon = st.number_input(
        "Hillshade Noon",
        value=0.0
    )

    hill_3 = st.number_input(
        "Hillshade 3pm",
        value=0.0
    )

    fire_dist = st.number_input(
        "Horizontal Distance to Fire Points",
        value=0.0
    )

# =========================
# FKNN PREDICT
# =========================
def fknn_predict(x):

    x_scaled = scaler.transform(x)

    dist = pairwise_distances(
        x_scaled,
        X_train,
        metric="euclidean"
    )[0]

    idx = np.argsort(dist)[:k]

    nearest_dist = dist[idx]

    nearest_u = U_train[idx]

    nearest_dist = np.maximum(
        nearest_dist,
        1e-10
    )

    weight = 1 / (
        nearest_dist ** (
            2/(m-1)
        )
    )

    membership = np.zeros(
        len(classes)
    )

    for j in range(
        len(classes)
    ):

        atas = np.sum(
            nearest_u[:, j] *
            weight
        )

        bawah = np.sum(
            weight
        )

        membership[j] = (
            atas / bawah
        )

    pred = classes[
        np.argmax(
            membership
        )
    ]

    return pred, membership

# =========================
# PREDICTION
# =========================
if st.button(
    "🔍 Prediksi"
):

    input_data = np.array([[
        elevation,
        aspect,
        slope,
        hydro_dist,
        vert_hydro,
        road_dist,
        hill_9,
        hill_noon,
        hill_3,
        fire_dist
    ]])

    pred, membership = fknn_predict(
        input_data
    )

    pred = int(pred)

    hasil = label_map.get(
        pred,
        "Unknown"
    )

    st.success(
        f"🌲 Hasil Prediksi: {hasil} (Class {pred})"
    )

    st.subheader(
        "Nilai Keanggotaan"
    )

    for i, cls in enumerate(
        classes
    ):

        st.write(
            f"Class {cls} : {membership[i]:.4f}"
        )
