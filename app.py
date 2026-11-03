import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

# --- 1. UI Configuration & Styling ---
st.set_page_config(page_title="FreshShelf AI", page_icon="🥗", layout="wide", initial_sidebar_state="expanded")

# Custom CSS using Streamlit's native variables for perfect visibility
st.markdown("""
<style>
    /* Style the metric containers to look like elevated cards */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--faded-text-10);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Style headers to be crisp */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join("models", "freshshelf_pipeline.joblib")

@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        import train_model
        train_model.build_and_train()
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
pipeline = bundle["pipeline"]

# --- 2. Main Header ---
col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.markdown("<h1 style='text-align: center; margin-top: 0px;'>🥗</h1>", unsafe_allow_html=True)
with col_title:
    st.title("FreshShelf AI Dashboard")
    
st.markdown("##### *Automated shelf-life estimation, biochemical decay analysis, and dynamic markdown triage.*")
st.divider()

# --- 3. SIDEBAR: Grouped & Structured Inputs ---
with st.sidebar:
    st.header("⚙️ Batch Intake Settings")
    st.markdown("Adjust the parameters below to simulate different storage scenarios.")
    
    with st.expander("🍎 Produce Profile", expanded=True):
        produce = st.selectbox("Produce Type", ["Leafy Greens", "Tomato", "Banana", "Apple", "Strawberry", "Potato"])
        maturity = st.slider("Initial Maturity", 1, 5, 2, help="1: Unripe, 5: Fully Ripe")
    
    with st.expander("🌡️ Storage Conditions", expanded=True):
        storage_temp = st.slider("Storage Temp (°C)", 0.0, 30.0, 5.0, 0.5)
        humidity = st.slider("Relative Humidity (%)", 40.0, 100.0, 85.0, 1.0)
        ethylene = st.selectbox("Ethylene Exposure", ["Low", "Moderate", "High"])
        
    with st.expander("📦 Logistics", expanded=True):
        transit_days = st.number_input("Transit Days", min_value=1, max_value=10, value=2)
        packaging = st.selectbox("Packaging Method", ["Corrugated Box", "Perforated Poly", "Open Crate"])

# --- 4. DATA PROCESSING ---
svp = 0.61078 * np.exp((17.27 * storage_temp) / (storage_temp + 237.3))
avp = svp * (humidity / 100.0)
vpd = round(svp - avp, 3)
thermal_stress = max(0.0, storage_temp - 8.0)

input_df = pd.DataFrame([{
    "produce_type": produce,
    "initial_maturity": maturity,
    "storage_temp": storage_temp,
    "relative_humidity": humidity,
    "ethylene_exposure": ethylene,
    "transit_days": transit_days,
    "packaging_type": packaging,
    "vpd_kpa": vpd,
    "thermal_stress": thermal_stress
}])

pred_days = max(0.1, float(pipeline.predict(input_df)[0]))

# --- 5. STATUS LOGIC & DYNAMIC RECOMMENDATIONS ---
if pred_days >= 6.0:
    status_color, status_text = "🟢", "Optimal (Standard Inventory)"
    action, alert_type = "Full Price Shelf Retention", st.success
    progress_val = min(100, int((pred_days / 14.0) * 100))
    recs = [
        "**Display:** Keep in standard aisles at full retail price.",
        "**Rotation:** Follow standard First-In, First-Out (FIFO) practices.",
        "**Storage:** Current temperature and humidity are preserving the batch well."
    ]
elif 3.0 <= pred_days < 6.0:
    status_color, status_text = "🟡", "Moderate (Accelerate Rotation)"
    action, alert_type = "Front-Row Display / Promo", st.info
    progress_val = int((pred_days / 14.0) * 100)
    recs = [
        "**Display:** Move to high-visibility areas (endcaps or entrance displays).",
        "**Promotion:** Consider a 'Buy 1 Get 1' or minor bundling offer to accelerate sales.",
        "**Monitoring:** Inspect daily for early signs of bruising or softening."
    ]
elif 1.5 <= pred_days < 3.0:
    status_color, status_text = "🟠", "Urgent (Discount Clearance)"
    action, alert_type = "Apply 30%-50% Markdown", st.warning
    progress_val = int((pred_days / 14.0) * 100)
    recs = [
        "**Pricing:** Immediately apply clearance stickers (30% to 50% off).",
        "**Display:** Move to a dedicated 'Flash Sale' or 'Eat Today' markdown bin.",
        "**Logistics:** Do not stack heavily; produce is likely soft and prone to crushing."
    ]
else:
    status_color, status_text = "🔴", "Critical (Imminent Spoilage)"
    action, alert_type = "Divert to Kitchen / Food Rescue", st.error
    progress_val = max(5, int((pred_days / 14.0) * 100))
    recs = [
        "**Removal:** Pull from the retail floor immediately to protect brand image.",
        "**Salvage:** Send to in-store deli/kitchen for soups, smoothies, or prepared foods.",
        "**Donation:** If salvage is not possible, divert to local food bank/rescue partners today."
    ]

# --- 6. MAIN DASHBOARD LAYOUT ---

# Top Metrics Row
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric(label="⏳ Estimated Shelf Life", value=f"{pred_days:.1f} Days")
with col_m2:
    st.metric(label="🏷️ Inventory Status", value=f"{status_color} {status_text.split(' ')[0]}")
with col_m3:
    st.metric(label="⚡ Recommended Action", value=action.split(' ')[0], help=action)

st.markdown("<br>", unsafe_allow_html=True)

# Visual Progress Bar
st.caption(f"**Shelf Life Timeline Indicator:** ({pred_days:.1f} days remaining)")
st.progress(progress_val)

# Contextual Alert Banner
alert_type(f"**Current Status:** {status_text} &nbsp; | &nbsp; **Primary Action:** {action}")

# Action Plan & Recommendations Panel
st.subheader("📋 Action Plan & Recommendations")
with st.container(border=True):
    for rec in recs:
        st.markdown(f"- {rec}")

st.markdown("<br>", unsafe_allow_html=True)

# Detailed Analysis Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Economic Salvage Estimator")
    with st.container(border=True):
        batch_col1, batch_col2 = st.columns(2)
        with batch_col1:
            batch_kg = st.number_input("Batch Size (kg)", 50, 5000, 200, step=50)
        with batch_col2:
            cost_per_kg = st.number_input("Retail Value/kg (₹)", 10, 500, 60, step=5)
        
        total_val = batch_kg * cost_per_kg
        st.write(f"**Total Batch Value:** `₹{total_val:,.2f}`")
        
        if pred_days < 3.0:
            salvage_val = total_val * 0.55
            st.error(f"📉 **Potential Waste Loss:** ₹{total_val:,.2f}")
            st.success(f"📈 **Recoverable via Markdown:** ₹{salvage_val:,.2f}")
        else:
            st.info("✅ **Waste Risk:** Low (< 5%)")

with col2:
    st.subheader("🌡️ Physical Stress Indicators")
    with st.container(border=True):
        st.metric(
            "💧 Vapor Pressure Deficit (VPD)", 
            f"{vpd} kPa", 
            delta="High Moisture Loss Risk" if vpd > 1.0 else "Stable", 
            delta_color="inverse"
        )
        st.metric(
            "🔥 Thermal Overexposure", 
            f"+{thermal_stress:.1f} °C", 
            delta="Above safe zone" if thermal_stress > 0 else "Optimal", 
            delta_color="inverse"
        )

# --- 7. EXPLAINABILITY SECTION ---
st.divider()
with st.expander("🧠 View AI Decision Drivers (SHAP Analysis)", expanded=False):
    st.markdown("This chart breaks down the exact environmental and logistical factors contributing to the predicted shelf life. Factors pushing to the **right** extend life; factors pushing to the **left** reduce it.")
    try:
        regressor = pipeline.named_steps["model"]
        transformer = pipeline.named_steps["preprocessor"]
        transformed_input = transformer.transform(input_df)
        explainer = shap.Explainer(regressor)
        shap_values = explainer(transformed_input)
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        # Making the chart background transparent to match the UI
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        # Adjust text color for dark/light mode compatibility in matplotlib
        ax.tick_params(colors='gray')
        
        shap.plots.waterfall(shap_values[0], max_display=6, show=False)
        st.pyplot(fig)
    except Exception:
        st.info("💡 The SHAP waterfall plot will dynamically generate based on model inputs to show which physical and logistical factors are reducing or extending the shelf life.")