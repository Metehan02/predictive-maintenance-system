import streamlit as st
import pandas as pd

from src.load_model import load_model
from src.system import predict_and_recommend


st.title("🚗 Predictive Maintenance System")

model = load_model()

st.write("Enter machine parameters:")

type_val = st.selectbox("Type", ["L", "M", "H"])
air_temp = st.number_input("Air temperature [K]", value=300.0)
process_temp = st.number_input("Process temperature [K]", value=310.0)
rpm = st.number_input("Rotational speed [rpm]", value=1500)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool wear [min]", value=50)

# feature engineering
temp_diff = process_temp - air_temp

type_map = {"L": 0, "M": 1, "H": 2}

input_data = pd.Series({
    "Type": type_map[type_val],
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rpm,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "temp_diff": temp_diff
})

if st.button("Predict"):
    result = predict_and_recommend(model, input_data)

    st.subheader("Prediction")
    st.write(result["prediction"])

    st.subheader("Failure Probability")
    st.write(round(result["probability"], 3))

    st.subheader("Recommendations")
    for rec in result["recommendations"]:
        st.write("- " + rec)