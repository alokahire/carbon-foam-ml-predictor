import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

PRIMARY_BLUE = "#1E3A8A"
STEEL_BLUE = "#3B82F6"
LIGHT_BLUE = "#60A5FA"
ORANGE_HIGHLIGHT = "#F97316"
EMERALD = "#10B981"
SLATE_GRAY = "#334155"
SOFT_WHITE = "#F8FAFC"
CARD_WHITE = "#FFFFFF"
BORDER_GRAY = "#E2E8F0"
CHART_FONT = "Arial, sans-serif"

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Carbon Foam ML Predictor",
    page_icon="🔥",
    layout="wide"
)

# -------------------------------
# GLOBAL STYLE (UI ONLY)
# -------------------------------

st.markdown("""
<style>

:root {
    --primary-blue: #1E3A8A;
    --steel-blue: #3B82F6;
    --light-blue: #60A5FA;
    --orange-highlight: #F97316;
    --emerald: #10B981;
    --slate-gray: #334155;
    --soft-white: #F8FAFC;
    --card-white: #FFFFFF;
    --border-gray: #E2E8F0;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(96, 165, 250, 0.18), transparent 28%),
        linear-gradient(180deg, #F8FAFC 0%, #EFF6FF 100%);
}

.main > div {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    gap: 1.2rem;
}

.block-container {
    padding-left: 2rem;
    padding-right: 2rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(30, 58, 138, 0.98) 0%, rgba(51, 65, 85, 0.98) 100%);
    border-right: 1px solid rgba(226, 232, 240, 0.25);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stSlider > div[data-baseweb="slider"] {
    background: transparent;
}

.dashboard-header {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.98) 0%, rgba(59, 130, 246, 0.94) 60%, rgba(96, 165, 250, 0.9) 100%);
    color: white;
    padding: 1.9rem 2rem;
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.22);
    box-shadow: 0 22px 45px rgba(30, 58, 138, 0.18);
    margin-bottom: 1.1rem;
}

.dashboard-title {
    font-size: 2.25rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}

.dashboard-subtitle {
    font-size: 1rem;
    opacity: 0.9;
    letter-spacing: 0.02em;
}

.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--slate-gray);
    letter-spacing: 0.02em;
    margin: 1.6rem 0 0.9rem 0;
    padding-left: 0.1rem;
}

.panel-card {
    background: var(--card-white);
    border: 1px solid var(--border-gray);
    border-radius: 22px;
    padding: 1.2rem 1.25rem;
    box-shadow: 0 14px 36px rgba(15, 23, 42, 0.06);
    margin-bottom: 1rem;
}

.prediction-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.96) 0%, rgba(239,246,255,0.98) 100%);
    padding: 2rem;
    border-radius: 24px;
    text-align: center;
    border: 1px solid var(--border-gray);
    box-shadow: 0 18px 40px rgba(30, 58, 138, 0.08);
    margin-bottom: 1rem;
}

.eyebrow {
    color: var(--steel-blue);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.76rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.prediction-label {
    color: var(--slate-gray);
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.prediction-value {
    font-size: 3.4rem;
    font-weight: 800;
    color: var(--primary-blue);
    line-height: 1;
    margin-bottom: 0.75rem;
}

.prediction-note {
    color: #475569;
    font-size: 0.98rem;
    margin-top: 0.45rem;
}

.info-card {
    background: var(--card-white);
    border: 1px solid var(--border-gray);
    border-radius: 20px;
    padding: 1.15rem 1.2rem;
    min-height: 130px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}

.info-label {
    color: #64748B;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
}

.info-value {
    color: var(--slate-gray);
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: 0.35rem;
}

.info-caption {
    color: #64748B;
    font-size: 0.92rem;
    line-height: 1.45;
}

.chart-card {
    background: var(--card-white);
    border: 1px solid var(--border-gray);
    border-radius: 22px;
    padding: 1rem 1rem 0.4rem 1rem;
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.05);
    margin-bottom: 1rem;
}

.footer {
    margin-top: 1.4rem;
    padding: 1rem 0 0.5rem 0;
    text-align: center;
    color: #64748B;
    font-size: 0.9rem;
}

div[data-testid="stPlotlyChart"] {
    border-radius: 18px;
    overflow: hidden;
}

@media (max-width: 900px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .dashboard-header,
    .prediction-card,
    .panel-card,
    .chart-card {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .dashboard-title {
        font-size: 1.8rem;
    }

    .prediction-value {
        font-size: 2.7rem;
    }
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD TRAINED MODELS (UNCHANGED)
# -------------------------------

rf = joblib.load("rf_model.pkl")
xgb = joblib.load("xgb_model.pkl")

w_rf = 0.503
w_xgb = 0.497

# -------------------------------
# FEATURE TEMPLATE (UNCHANGED)
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
# SIDEBAR (INPUTS PRESERVED)
# -------------------------------

st.sidebar.title("Input Parameters")

st.sidebar.markdown("### Material Properties")
bulk_density = st.sidebar.slider("Bulk Density (g/cm³)", 0.05, 1.5, 0.45)
porosity = st.sidebar.slider("Porosity (%)", 15.0, 99.0, 80.0)

st.sidebar.markdown("### Process Parameters")
carbon_temp = st.sidebar.slider("Carbonization Temperature (°C)", 150, 1300, 900)
heating_rate = st.sidebar.slider("Heating Rate (°C/min)", 0.2, 14.0, 5.0)
holding_time = st.sidebar.slider("Holding Time (hr)", 0.5, 25.0, 2.0)

st.sidebar.markdown("### Material Type")
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
# PREDICTION FUNCTION (UNCHANGED)
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


def info_card(label, value, caption):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
            <div class="info-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_style(fig, title, xaxis_title, yaxis_title, height=500):
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, font=dict(size=18, color=SLATE_GRAY), x=0.02, xanchor="left"),
        font=dict(family=CHART_FONT, size=13, color=SLATE_GRAY),
        margin=dict(t=70, r=30, b=70, l=70),
        hovermode="closest",
        hoverdistance=12,
        plot_bgcolor=CARD_WHITE,
        paper_bgcolor=CARD_WHITE,
    )
    fig.update_xaxes(
        title_text=xaxis_title,
        showgrid=True,
        gridcolor="rgba(226, 232, 240, 0.8)",
        zeroline=False,
        title_font=dict(size=14, color=SLATE_GRAY),
        tickfont=dict(size=12, color=SLATE_GRAY),
    )
    fig.update_yaxes(
        title_text=yaxis_title,
        showgrid=True,
        gridcolor="rgba(226, 232, 240, 0.8)",
        zeroline=False,
        title_font=dict(size=14, color=SLATE_GRAY),
        tickfont=dict(size=12, color=SLATE_GRAY),
    )

# -------------------------------
# HEADER
# -------------------------------

st.markdown(
    """
    <div class="dashboard-header">
        <div class="dashboard-title">Carbon Foam Compressive Strength Predictor</div>
        <div class="dashboard-subtitle">Engineering analytics dashboard for material property prediction and model validation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# PREDICTION CARD
# -------------------------------

st.markdown(
    f"""
    <div class="prediction-card">
        <div class="eyebrow">Prediction Card</div>
        <div class="prediction-label">Predicted Compressive Strength</div>
        <div class="prediction-value">{predicted_strength} MPa</div>
        <div class="prediction-note">Prediction generated using a weighted ensemble of Random Forest and XGBoost models.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# LOAD DATASET (UNCHANGED)
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
# TRAIN TEST SPLIT (UNCHANGED)
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL PREDICTIONS (UNCHANGED)
# -------------------------------

rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

ensemble_pred = (w_rf * rf_pred) + (w_xgb * xgb_pred)

# -------------------------------
# METRICS (UNCHANGED)
# -------------------------------

r2 = r2_score(y_test, ensemble_pred)
rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
mae = mean_absolute_error(y_test, ensemble_pred)

actual_mpa = 10 ** y_test
predicted_mpa = 10 ** ensemble_pred

# -------------------------------
# DATASET SUMMARY
# -------------------------------

st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    info_card("Dataset Size", f"{len(df)}", "Samples available for model training and evaluation")
with c2:
    info_card("Number of Features", f"{X.shape[1]}", "Encoded predictors used in the ensemble pipeline")
with c3:
    info_card("Train/Test Split", "80 / 20", "Consistent validation split used for performance reporting")

# -------------------------------
# MODEL PERFORMANCE
# -------------------------------

st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    info_card("R² Score", f"{r2:.3f}", "Explained variance captured by the blended model")
with m2:
    info_card("RMSE", f"{rmse:.3f}", "Root mean squared error measured on log-scale targets")
with m3:
    info_card("MAE", f"{mae:.3f}", "Mean absolute error measured on log-scale targets")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------

st.markdown('<div class="section-header">Model Interpretation</div>', unsafe_allow_html=True)

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
    color="Importance",
    color_continuous_scale=[PRIMARY_BLUE, STEEL_BLUE, LIGHT_BLUE],
    text=feature_importance["Importance"].map(lambda value: f"{value:.3f}")
)

fig2.update_traces(
    marker_line_color=PRIMARY_BLUE,
    marker_line_width=0,
    width=0.72,
    textposition="outside",
    cliponaxis=False,
    hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
)

apply_chart_style(
    fig2,
    "Feature Importance",
    "Importance Score",
    "Model Features",
    height=500,
)
fig2.update_layout(coloraxis_showscale=False)
fig2.update_yaxes(autorange="reversed")

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig2, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# ACTUAL VS PREDICTED
# -------------------------------

st.markdown('<div class="section-header">Model Validation</div>', unsafe_allow_html=True)

fig1 = px.scatter(
    x=actual_mpa,
    y=predicted_mpa,
    labels={"x": "Actual Strength (MPa)", "y": "Predicted Strength (MPa)"},
)

fig1.update_traces(
    mode="markers",
    marker=dict(
        size=9,
        color=STEEL_BLUE,
        opacity=0.75,
        line=dict(width=0.5, color="rgba(255,255,255,0.85)"),
    ),
    hovertemplate="Actual: %{x:.2f} MPa<br>Predicted: %{y:.2f} MPa<extra></extra>",
)

fig1.add_trace(
    go.Scatter(
        x=[actual_mpa.min(), actual_mpa.max()],
        y=[actual_mpa.min(), actual_mpa.max()],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color=ORANGE_HIGHLIGHT, width=3.5),
        hoverinfo="skip"
    )
)

apply_chart_style(
    fig1,
    "Actual vs Predicted",
    "Actual Strength (MPa)",
    "Predicted Strength (MPa)",
    height=500,
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig1, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# RESIDUAL DISTRIBUTION
# -------------------------------

residuals = actual_mpa - predicted_mpa

fig3 = px.histogram(
    residuals,
    nbins=30,
    labels={"value": "Residual (MPa)", "count": "Frequency"}
)

mean_residual = float(np.mean(residuals))

fig3.update_traces(
    marker=dict(color=SLATE_GRAY, line=dict(color=BORDER_GRAY, width=0.8)),
    opacity=0.88,
    hovertemplate="Residual: %{x:.2f} MPa<br>Count: %{y}<extra></extra>",
)

fig3.add_vline(
    x=mean_residual,
    line_color=ORANGE_HIGHLIGHT,
    line_width=3,
    line_dash="dash",
    annotation_text=f"Mean: {mean_residual:.2f} MPa",
    annotation_position="top right",
)

apply_chart_style(
    fig3,
    "Residual Distribution",
    "Residual (MPa)",
    "Frequency",
    height=500,
)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(fig3, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------

st.markdown(
    '<div class="footer">Carbon Foam Strength Predictor – Developed by Alokraj Ahire</div>',
    unsafe_allow_html=True
)