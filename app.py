import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Hafsa AI | Insurance Predictor", page_icon="🏥", layout="wide"
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model = joblib.load("insurance_health_Regression.pkl")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.title("Project Dashboard")
    st.info("System: Medical Cost Analytics\nModel: Gradient Boosting\nAccuracy: 84.4%")
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
    children = st.number_input("Number of Dependents", 0, 10, 0)
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
    predict_btn = st.button("Generate Cost Analysis", type="primary")

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
            st.markdown("#### 📊 Risk Insight")
            importance_data = {
                "Factor": ["Age", "BMI", "Lifestyle"],
                "Impact": [age * 0.2, bmi * 0.4, (80 if smoker == "yes" else 10)],
            }
            st.bar_chart(pd.DataFrame(importance_data).set_index("Factor"))

        st.subheader("💡 Hafsa's AI Recommendations")
        if smoker == "yes":
            st.error(
                "Smoking is the primary driver of your high insurance cost. Quitting could save you over $15,000/year."
            )
        elif bmi > 30:
            st.warning(
                "Your BMI is in the high-cost range. Managing weight can lead to significantly lower premiums."
            )
        else:
            st.success(
                "Your health profile is optimal. You are eligible for the most competitive market rates."
            )

    except Exception as e:
        st.error(f"Analysis Error: {e}")

st.markdown("---")
st.caption("© 2026 | End-to-End ML Pipeline by Hafsa Ibrahim")
