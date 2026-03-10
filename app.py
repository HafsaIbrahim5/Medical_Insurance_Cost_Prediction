import streamlit as st
import pandas as pd
import joblib

# 1. Loading the model
model = joblib.load("insurance_health_Regression.pkl")

# 2. Page Configuration
st.set_page_config(page_title="Insurance Predictor", page_icon="🏥", layout="wide")

# 3. Sidebar with your Links
with st.sidebar:
    st.title("Project Info")
    st.info(
        "This AI model predicts medical insurance costs using Gradient Boosting with 84% accuracy."
    )

    st.markdown("---")
    st.markdown("### Developed by:")
    st.write("Hafsa Ibrahim")

    # LinkedIn Badge
    st.markdown(
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hafsa-ibrahim-ml-cs/)"
    )

    # GitHub Badge
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/HafsaIbrahim5)"
    )

# 4. Main UI
st.title("🏥 Medical Insurance Cost Predictor")
st.write(
    "Provide the following details to get an instant estimation of your annual insurance charges."
)
st.markdown("---")

# Columns for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 25)
    bmi = st.number_input("Body Mass Index (BMI)", 10.0, 50.0, 25.0)
    children = st.number_input("Number of Children", 0, 10, 0)

with col2:
    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Are you a smoker?", ["no", "yes"])
    region = st.selectbox(
        "Region", ["northeast", "northwest", "southeast", "southwest"]
    )

# 5. Prediction Logic
st.markdown("###")
if st.button("Calculate Estimated Cost", use_container_width=True):
    # Data prepared according to your drop_first=True training
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
        final_res = float(prediction[0])

        # Displaying result as a Metric
        st.success("Analysis Complete!")
        st.metric(label="Estimated Annual Charges", value=f"${final_res:,.2f}")

        # Adding a small insight
        if smoker == "yes":
            st.warning("Note: Smoking is the primary factor increasing your premium.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
