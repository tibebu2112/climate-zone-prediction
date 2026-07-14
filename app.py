import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ethiopia Climate Zone Predictor",
    page_icon="🌾",
    layout="wide",
)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model/xgb_climate_model.pkl")
    features = joblib.load("model/features.pkl")
    return model, features

model, FEATURES = load_model()

# ── Crop Recommendations ──────────────────────────────────────────────────────
CROP_RECOMMENDATIONS = {
    "Dega": {
        "description": "Highland zone (>2300 m). Cool temperatures (10–16°C), high rainfall. Afromontane vegetation.",
        "crops": [
            ("🌾 Barley", "Primary highland cereal; altitude-tolerant"),
            ("🌾 Wheat", "Cool-season grain; thrives above 2000 m"),
            ("🥔 Irish Potato", "High-yield root crop for cool climates"),
            ("🫘 Faba Bean", "Cool-season protein legume"),
            ("🫘 Field Pea", "Nitrogen-fixing; 2000–3000 m"),
            ("🌿 Linseed/Flax", "Oil and fibre crop for highlands"),
            ("🌿 Rapeseed", "Oil crop; tolerates cool moist conditions"),
            ("🥗 Cabbage & Kale", "Cool-weather vegetables"),
        ],
        "color": "#1f77b4",
        "emoji": "🏔️",
    },
    "Weina Dega": {
        "description": "Mid-altitude zone (1500–2300 m). Moderate temperatures (16–24°C). Most productive belt of Ethiopia.",
        "crops": [
            ("🌽 Maize", "High-yielding staple; best at 1500–2200 m"),
            ("🌾 Teff", "Ethiopia's iconic staple grain"),
            ("🌾 Sorghum", "Drought-tolerant cereal"),
            ("☕ Coffee (Arabica)", "Premium coffee belt; 1500–2200 m"),
            ("🫘 Haricot Bean", "Major export legume"),
            ("🌻 Sunflower", "Oil crop for mid-altitudes"),
            ("🍌 Enset", "Ethiopian 'false banana'; food security crop"),
            ("🥑 Avocado", "Fruit tree; 1500–2200 m"),
        ],
        "color": "#2ca02c",
        "emoji": "🌄",
    },
    "Kolla": {
        "description": "Lowland zone (<1500 m). Hot temperatures (24–36°C), low rainfall. Semi-arid to arid conditions.",
        "crops": [
            ("🌾 Sorghum", "Primary Kolla cereal; heat & drought tolerant"),
            ("🌿 Sesame", "Major export oil crop for dry lowlands"),
            ("🫘 Cowpea", "Heat-tolerant legume"),
            ("🌿 Cotton", "Fibre cash crop for hot lowland valleys"),
            ("🍉 Watermelon & Melons", "Warm-season irrigated fruits"),
            ("🌴 Date Palm", "Dryland fruit for arid lowlands"),
            ("🌿 Groundnut", "Warm sandy lowland soils"),
            ("🍌 Banana", "Irrigated lowland fruit"),
        ],
        "color": "#ff7f0e",
        "emoji": "🏜️",
    },
}

ZONE_NAMES = {0: "Dega", 1: "Kolla", 2: "Weina Dega"}
ZONE_COLORS = {"Dega": "#1f77b4", "Weina Dega": "#2ca02c", "Kolla": "#ff7f0e"}


def predict_zone(inputs):
    X = pd.DataFrame([inputs])[FEATURES]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return ZONE_NAMES[int(pred)], proba


def confidence_chart(proba, predicted_zone):
    zones = ["Dega", "Kolla", "Weina Dega"]
    colors = [ZONE_COLORS[z] if z == predicted_zone else "#d3d3d3" for z in zones]
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(zones, proba * 100, color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_title("Prediction Confidence by Zone", fontsize=13, fontweight="bold")
    for bar, p in zip(bars, proba):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌾 Ethiopia Climate Adaptability Zone Predictor")
st.markdown(
    "Enter site-level climate and soil parameters to predict the "
    "**Ethiopian agro-ecological zone** and get **crop recommendations**."
)
st.divider()

with st.sidebar:
    st.header("📍 Site Parameters")
    altitude = st.slider("Altitude (m)", 500, 4000, 2000, 10)
    ph = st.slider("Soil pH (0–60 cm)", 4.0, 9.0, 6.5, 0.1)
    bio1 = st.slider("Mean Annual Temperature (°C)", 8.0, 30.0, 18.0, 0.1)
    bio4 = st.slider("Temperature Seasonality (BIO4)", 400, 2800, 1200, 10)
    bio12 = st.slider("Annual Precipitation (mm)", 400, 2800, 1200, 10)
    bio15 = st.slider("Precipitation Seasonality (CV)", 40.0, 160.0, 100.0, 0.5)
    predict_btn = st.button("🔍 Predict Zone", use_container_width=True, type="primary")

if predict_btn:
    inputs = {
        "Altitude": altitude,
        "ph_ho_60cm": ph,
        "bio1_1970_2000": bio1,
        "bio4_1970_2000": bio4,
        "bio12_1970_2000": bio12,
        "bio15_1970_2000": bio15,
    }

    zone, proba = predict_zone(inputs)
    info = CROP_RECOMMENDATIONS[zone]

    st.markdown(
        f"""<div style="background:{info['color']}22; border-left:6px solid {info['color']};
                padding:18px 24px; border-radius:8px; margin-bottom:20px">
            <h2 style='margin:0; color:{info['color']}'>{info['emoji']} Predicted Zone: {zone}</h2>
            <p style='margin:6px 0 0 0; font-size:15px'>{info['description']}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Prediction Confidence")
        fig = confidence_chart(proba, zone)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🎯 Zone Probabilities")
        prob_df = pd.DataFrame({
            "Zone": ["Dega", "Kolla", "Weina Dega"],
            "Confidence": [f"{p*100:.2f}%" for p in proba],
        })
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(f"🌱 Recommended Crops for {zone} Zone")
    cols = st.columns(3)
    for i, (crop_name, crop_desc) in enumerate(info["crops"]):
        with cols[i % 3]:
            st.markdown(
                f"""<div style="border:1px solid {info['color']}55; border-radius:8px;
                        padding:12px 14px; margin-bottom:10px; background:#fafafa">
                    <strong>{crop_name}</strong><br>
                    <span style='color:#555; font-size:13px'>{crop_desc}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    with st.expander("📚 View Crops for All Zones"):
        tabs = st.tabs(["🏔️ Dega", "🌄 Weina Dega", "🏜️ Kolla"])
        for tab, z in zip(tabs, ["Dega", "Weina Dega", "Kolla"]):
            with tab:
                z_info = CROP_RECOMMENDATIONS[z]
                st.markdown(f"**{z_info['description']}**")
                for c_name, c_desc in z_info["crops"]:
                    st.markdown(f"- **{c_name}** — {c_desc}")
else:
    st.info("👈 Adjust the parameters in the sidebar and click **Predict Zone**.")
    cols = st.columns(3)
    for col, (zone_name, info) in zip(cols, CROP_RECOMMENDATIONS.items()):
        with col:
            st.markdown(
                f"""<div style="border-top:5px solid {info['color']}; padding:14px;
                        border-radius:6px; background:#fafafa">
                    <h3 style='color:{info['color']}; margin-top:0'>{info['emoji']} {zone_name}</h3>
                    <p style='font-size:13px'>{info['description']}</p>
                </div>""",
                unsafe_allow_html=True,
            )

st.markdown("---")
st.caption("XGBoost Classifier · Features: Altitude, Soil pH, BIO1, BIO4, BIO12, BIO15 · Ethiopian MoA/EIAR zone standards")
