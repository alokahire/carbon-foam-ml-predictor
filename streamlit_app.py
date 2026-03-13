import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Carbon Foam ML Predictor",
    page_icon="🔥",
    layout="wide"
)

# -------------------------------
# LOAD TRAINED MODELS
# -------------------------------

rf = joblib.load("rf_model.pkl")
xgb = joblib.load("xgb_model.pkl")

w_rf = 0.503
w_xgb = 0.497

# -------------------------------
# FEATURE TEMPLATE
# -------------------------------

feature_columns = [
    "Bulk density (g/cm³)",
    "Porosity (%)",
    "Carbonization temperature (˚C)",
    "Heating rate (˚C/min)",
    "Holding time (hr)",
    "Precursor_Category_Composite-based",
    "Precursor_Category_Other",
    "Precursor_Category_Pitch-based",
    "Precursor_Category_Polymer-based",
    "Precursor_Category_Resin-based"
]

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------

st.sidebar.title("Input Parameters")

bulk_density = st.sidebar.slider("Bulk Density (g/cm³)", 0.05, 1.5, 0.45)
porosity = st.sidebar.slider("Porosity (%)", 15.0, 99.0, 80.0)
carbon_temp = st.sidebar.slider("Carbonization Temperature (°C)", 150, 1300, 900)
heating_rate = st.sidebar.slider("Heating Rate (°C/min)", 0.2, 14.0, 5.0)
holding_time = st.sidebar.slider("Holding Time (hr)", 0.5, 25.0, 2.0)

precursor = st.sidebar.selectbox(
    "Precursor Category",
    [
        "Composite-based",
        "Other",
        "Pitch-based",
        "Polymer-based",
        "Resin-based"
    ]
)

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------

def predict():

    input_dict = {col: 0 for col in feature_columns}

    input_dict["Bulk density (g/cm³)"] = bulk_density
    input_dict["Porosity (%)"] = porosity
    input_dict["Carbonization temperature (˚C)"] = carbon_temp
    input_dict["Heating rate (˚C/min)"] = heating_rate
    input_dict["Holding time (hr)"] = holding_time

    category_col = "Precursor_Category_" + precursor
    if category_col in input_dict:
        input_dict[category_col] = 1

    input_df = pd.DataFrame([input_dict])

    rf_val = rf.predict(input_df)[0]
    xgb_val = xgb.predict(input_df)[0]

    ensemble_val = (w_rf * rf_val) + (w_xgb * xgb_val)

    strength_mpa = 10 ** ensemble_val

    return round(float(strength_mpa), 3)

predicted_strength = predict()

# -------------------------------
# HEADER
# -------------------------------

st.title("🔥 Carbon Foam Compressive Strength Predictor")
st.caption("Machine Learning Based Material Property Prediction System")

st.markdown("---")

# -------------------------------
# PREDICTION DISPLAY
# -------------------------------

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Predicted Compressive Strength",
        value=f"{predicted_strength} MPa"
    )

with col2:
    st.info(
        "Prediction generated using a weighted ensemble of "
        "Random Forest and XGBoost models."
    )

st.markdown("---")

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_excel("Final_dataset.xlsx")

y = df["Log_Compressive_Strength"]

X = df.drop(columns=[
    "Compressive strength (Mpa)",
    "Log_Compressive_Strength",
    "Precursor"
])

X = pd.get_dummies(X, columns=["Precursor_Category"], drop_first=True)

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL PREDICTIONS
# -------------------------------

rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

ensemble_pred = (w_rf * rf_pred) + (w_xgb * xgb_pred)

# -------------------------------
# METRICS
# -------------------------------

r2 = r2_score(y_test, ensemble_pred)
rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
mae = mean_absolute_error(y_test, ensemble_pred)

actual_mpa = 10 ** y_test
predicted_mpa = 10 ** ensemble_pred

# -------------------------------
# MODEL PERFORMANCE DISPLAY
# -------------------------------

st.subheader("Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("R² Score", f"{r2:.3f}")
c2.metric("RMSE (log scale)", f"{rmse:.3f}")
c3.metric("MAE (log scale)", f"{mae:.3f}")

st.markdown("---")

# -------------------------------
# ACTUAL VS PREDICTED PLOT
# -------------------------------

fig1 = px.scatter(
    x=actual_mpa,
    y=predicted_mpa,
    labels={"x": "Actual Strength (MPa)", "y": "Predicted Strength (MPa)"},
    title="Actual vs Predicted Strength",
    template="plotly_white"
)

fig1.add_trace(
    go.Scatter(
        x=[actual_mpa.min(), actual_mpa.max()],
        y=[actual_mpa.min(), actual_mpa.max()],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="red", dash="dash")
    )
)

fig1.update_layout(hovermode=False)

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------

importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig2 = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance (Random Forest)",
    template="plotly_white"
)

fig2.update_layout(hovermode=False)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# RESIDUAL DISTRIBUTION
# -------------------------------

residuals = actual_mpa - predicted_mpa

fig3 = px.histogram(
    residuals,
    nbins=30,
    title="Residual Distribution (MPa)",
    template="plotly_white"
)

fig3.update_layout(hovermode=False)

st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

st.success("Robust Ensemble Learning Framework using Random Forest + XGBoost")