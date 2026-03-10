import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Medical Insurance Predictor", page_icon="🏥", layout="wide"
)

model = joblib.load("insurance_health_Regression.pkl")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.title("Project Dashboard")
    st.info(
        "System: Medical Cost Analytics\nModel: Gradient Boosting Regressor\nAccuracy: 84.4%"
    )
    st.markdown("---")
    st.markdown("### Developed By")
    st.write("**Hafsa Ibrahim**")
    st.write("AI & NLP Engineer")
    st.markdown(
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hafsa-ibrahim-ml-cs/)"
    )
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/HafsaIbrahim5)"
    )

st.title("🏥 Intelligent Health Insurance Predictor")
st.markdown(
    "Estimate annual medical charges based on advanced Machine Learning analysis."
)
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("👤 Personal Details")
    age = st.slider("Select Age", 18, 100, 25)
    sex = st.selectbox("Biological Sex", ["female", "male"])
    children = st.number_input("Number of Dependents (Children)", 0, 10, 0)
    region = st.selectbox(
        "Residential Region", ["northeast", "northwest", "southeast", "southwest"]
    )

with col2:
    st.subheader("⚖️ Health Indicators")
    bmi = st.number_input(
        "Body Mass Index (BMI)", 10.0, 60.0, 25.0, help="Healthy range is 18.5 - 24.9"
    )
    smoker = st.radio("Do you smoke?", ["no", "yes"], horizontal=True)
    st.markdown("###")
    predict_btn = st.button(
        "Generate Cost Analysis", use_container_width=True, type="primary"
    )

if predict_btn:
    data = {
        "age": age,
        "sex": 1 if sex == "male" else 0,
        "bmi": bmi,
        "children": children,
        "smoker": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }

    input_df = pd.DataFrame([data])

    try:
        prediction = model.predict(input_df)
        cost = float(prediction[0])
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(label="Predicted Annual Charge", value=f"${cost:,.2f}")
        with res_col2:
            if smoker == "yes":
                st.error(
                    "⚠️ High Risk Factor: Smoking increases your estimated premium significantly."
                )
            elif bmi > 30:
                st.warning(
                    "ℹ️ Notice: BMI over 30 is considered a high-cost factor in this model."
                )
            else:
                st.success(
                    "✅ Standard Risk: Your profile indicates a stable insurance rate."
                )
    except Exception as e:
        st.error(f"Analysis Error: {e}")

st.markdown("---")
st.caption("© 2026 | Developed for Medical Data Analytics Portfolio")
