# carbon-foam-ml-predictor
This is a machine learning framework for predicting compressive strength of carbon foams using ensemble regression (Random Forest + XGBoost) with interactive Streamlit deployment.

# Carbon Foam Compressive Strength Predictor

## 📌 Overview

This project presents an industry-grade machine learning framework for predicting the compressive strength of carbon foams using structural and processing parameters.

The model significantly reduces experimental workload by providing rapid and reliable strength estimation for untested configurations.

---

## 🎯 Problem Statement

Carbon foams are lightweight cellular materials widely used in structural and multifunctional applications. Experimental compressive strength testing is:

- Time-consuming
- Resource-intensive
- Expensive

This project develops a robust ensemble machine learning model to predict compressive strength from material parameters.

---

## 🧠 Machine Learning Approach

We implemented:

- Linear Regression (Baseline)
- Random Forest Regressor
- XGBoost Regressor
- Weighted Ensemble Model (Final)

### Final Model Performance:

- **R² Score ≈ 0.74**
- **RMSE ≈ 3.4 MPa**
- **MAE ≈ 1.4 MPa**

The ensemble approach improves robustness and nonlinear modeling capability.

---

## 📊 Input Parameters

The prediction system uses:

- Bulk Density (g/cm³)
- Porosity (%)
- Carbonization Temperature (°C)
- Heating Rate (°C/min)
- Holding Time (hr)
- Precursor Category

---

## 🚀 Deployment

This project is deployed using Streamlit.

Live App:
👉 [Your Streamlit Link Here]

---

## 📈 Visual Analytics Included

- Actual vs Predicted Strength
- Feature Importance Analysis
- Residual Distribution
- Interactive Parameter Controls

---

## 🛠 Tech Stack

- Python
- Scikit-Learn
- XGBoost
- Pandas
- Plotly
- Streamlit

---

## 🏗 Project Structure
