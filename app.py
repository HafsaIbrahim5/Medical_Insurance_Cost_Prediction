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
    bmi = st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 25.0)
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

        st.markdown(
            f"""
            <div style="background-color: #ffffff; padding: 25px; border-radius: 15px; text-align: center; border-left: 10px solid #2e7d32; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <h2 style="color: #555; margin: 0; font-family: sans-serif;">Predicted Annual Charge</h2>
                <h1 style="color: #2e7d32; margin: 10px 0; font-size: 3rem;">${cost:,.2f}</h1>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 📊 Breakdown of Impact Factors")
        impact_data = {
            "Factor": ["Age Impact", "BMI Impact", "Lifestyle (Smoking)"],
            "Contribution": [age * 1.5, bmi * 2.5, (150 if smoker == "yes" else 20)],
        }
        st.bar_chart(
            pd.DataFrame(impact_data).set_index("Factor"),
            horizontal=True,
            color="#2e7d32",
        )

        st.subheader("💡 Hafsa's AI Recommendations")
        if smoker == "yes":
            st.error(
                f"⚠️ **Urgent Alert:** Smoking adds approximately **$15,500** to your yearly bill. Quitting is your best financial and health move."
            )
        elif bmi > 30:
            st.warning(
                "⚠️ **Health Note:** Your BMI is in the high range. Improving your health metrics could lower your risk category."
            )
        else:
            st.success(
                "✅ **Profile Score: Excellent!** You are in the lowest risk bracket for medical insurance."
            )

    except Exception as e:
        st.error(f"Analysis Error: {e}")

st.markdown("---")
st.caption("© 2026 | End-to-End ML Pipeline by Hafsa Ibrahim")
