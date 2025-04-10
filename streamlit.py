import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Page config
st.set_page_config(page_title="UCLA Admission Predictor", layout="wide")
st.title("🎓 UCLA Admission Predictor")
st.markdown("Predict your admission chance using a trained neural network model.")

# Load model
with open("models/MLP.pkl", "rb") as model_file:
    MLP_model = pickle.load(model_file)

# Sidebar for user input
st.sidebar.header("📋 Enter Your Academic Profile")

greScore = st.sidebar.number_input("GRE Score", min_value=260, max_value=340, value=300)
tofelScore = st.sidebar.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
univRating = st.sidebar.selectbox("University Rating", options=["1", "2", "3", "4", "5"])
sop = st.sidebar.number_input("SOP Strength", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
lor = st.sidebar.number_input("LOR Strength", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
cgpa = st.sidebar.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, step=0.1, value=8.0)
research = st.sidebar.selectbox("Research Experience", options=["Yes", "No"])

# Predict button
if st.sidebar.button("🚀 Predict My Admission Chance"):

    # One-hot encoding
    University_Rating_1 = 1 if univRating == "1" else 0
    University_Rating_2 = 1 if univRating == "2" else 0
    University_Rating_3 = 1 if univRating == "3" else 0
    University_Rating_4 = 1 if univRating == "4" else 0
    University_Rating_5 = 1 if univRating == "5" else 0
    Research_0 = 1 if research == "No" else 0
    Research_1 = 1 if research == "Yes" else 0

    input_data = [[greScore, tofelScore, sop, lor, cgpa,
                   University_Rating_1, University_Rating_2, University_Rating_3,
                   University_Rating_4, University_Rating_5, Research_0, Research_1]]

    input_df = pd.DataFrame(input_data, columns=[
        'GRE_Score', 'TOEFL_Score', 'SOP', 'LOR', 'CGPA',
        'University_Rating_1', 'University_Rating_2', 'University_Rating_3',
        'University_Rating_4', 'University_Rating_5', 'Research_0', 'Research_1'])

    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(pd.DataFrame({
        'GRE_Score': [260, 340],
        'TOEFL_Score': [0, 120],
        'SOP': [1.0, 5.0],
        'LOR': [1.0, 5.0],
        'CGPA': [0.0, 10.0],
        'University_Rating_1': [0, 1],
        'University_Rating_2': [0, 1],
        'University_Rating_3': [0, 1],
        'University_Rating_4': [0, 1],
        'University_Rating_5': [0, 1],
        'Research_0': [0, 1],
        'Research_1': [0, 1]
    }))
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = MLP_model.predict(input_scaled)[0]
    probability = MLP_model.predict_proba(input_scaled)[0][1]

    # Result section
    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.success(f"✅ High chance of admission! (Probability: {probability:.2f})")
    else:
        st.warning(f"❌ Low chance of admission. (Probability: {probability:.2f})")

    with st.expander("📉 Show Model Loss Curve"):
        loss_values = MLP_model.loss_curve_
        fig, ax = plt.subplots()
        ax.plot(loss_values, label='Loss', color='blue')
        ax.set_title("Loss Curve")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

# Display saved graphs
st.subheader("📊 Admission Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**GRE vs TOEFL Scores by Admission Chance**")
    try:
        st.image("reports/figures/gre_vs_toefl.png", use_column_width=True)
    except FileNotFoundError:
        st.warning("Missing: gre_vs_toefl.png")

with col2:
    st.markdown("**CGPA Distribution by Admission Chance**")
    try:
        st.image("reports/figures/cgpa_hist.png", use_column_width=True)
    except FileNotFoundError:
        st.warning("Missing: cgpa_hist.png")

# Full width for pairplot
st.markdown("**Relationships Between GRE, TOEFL, and CGPA**")
try:
    st.image("reports/figures/pairplot.png", use_column_width=True)
except FileNotFoundError:
    st.warning("Missing: pairplot.png")

# Footer
st.caption("Made with ❤️ using a trained MLPClassifier.")
