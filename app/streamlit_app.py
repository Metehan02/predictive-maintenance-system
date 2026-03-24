import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from src.load_model import load_model
from src.system import predict_and_recommend

st.set_page_config(page_title="Predictive Maintenance System", page_icon="🚗", layout="centered")

st.title("🚗 Predictive Maintenance System")
st.markdown("Predict machine failure risk and get maintenance recommendations.")

model_choice = st.selectbox(
    "Choose Model",
    ["random_forest", "xgboost"]
)

model = load_model(model_choice)

feature_names = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff"
]

feature_importances = None

if model_choice == "random_forest":
    feature_importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

with st.container():
    st.subheader("Machine Parameters")

    col1, col2 = st.columns(2)

    with col1:
        type_val = st.selectbox("Type", ["L", "M", "H"])
        air_temp = st.number_input("Air temperature [K]", value=300.0)
        process_temp = st.number_input("Process temperature [K]", value=310.0)

    with col2:
        rpm = st.number_input("Rotational speed [rpm]", value=1500)
        torque = st.number_input("Torque [Nm]", value=40.0)
        tool_wear = st.number_input("Tool wear [min]", value=50)

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

st.markdown("---")
st.subheader("Derived Feature")
st.info(f"Temperature difference (process - air): **{temp_diff:.2f} K**")

if st.button("Predict", use_container_width=True):
    result = predict_and_recommend(model, input_data)

    st.markdown("---")
    st.subheader("Prediction Result")

    probability = float(result["probability"])

    if result["prediction"] == "Failure likely":
        st.error(f"⚠️ {result['prediction']}")
    else:
        st.success(f"✅ {result['prediction']}")

    st.metric("Failure Probability", f"{probability:.2%}")

    st.subheader("Maintenance Recommendations")
    for rec in result["recommendations"]:
        st.warning(rec)

st.markdown("---")
st.subheader("Feature Importance")

st.markdown("---")
st.subheader("Feature Importance")

if feature_importances is not None:
    fig, ax = plt.subplots()
    feature_importances.plot(kind="bar", ax=ax)
    ax.set_title("Most Important Features")
    ax.set_ylabel("Importance Score")
    st.pyplot(fig)
else:
    st.info("Feature importance chart is currently shown for the Random Forest model.")