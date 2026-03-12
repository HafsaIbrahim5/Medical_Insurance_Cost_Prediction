# 🏥 Intelligent Medical Insurance Cost Predictor

An end-to-end Machine Learning web application designed to estimate annual medical insurance charges with high precision. This project leverages a **Gradient Boosting Regressor** to analyze demographic and lifestyle factors, providing real-time cost analysis through an interactive UI.

## 🚀 Live Demo
[🔗 View Live App](https://medicalinsurancecostprediction-bd4njuxhrkulsx56usw566.streamlit.app/)

## 🛠️ Tech Stack
- **Engine:** Python 3.10+
- **Machine Learning:** Scikit-Learn (Gradient Boosting)
- **Web Framework:** Streamlit
- **Data Handling:** Pandas & NumPy
- **Serialization:** Joblib

## 📊 Model Performance
The underlying model was trained on a comprehensive healthcare dataset, achieving a solid balance between bias and variance:
- **Accuracy (R² Score):** ~84.4%
- **Key Features:** Smoking Status, BMI, Age, and Regional Data.



## 💡 Features
- **Real-time Prediction:** Instant cost estimation based on user inputs.
- **Risk Analysis:** Built-in logic to flag high-risk factors like smoking or high BMI.
- **Responsive Design:** A modern, wide-layout dashboard with a professional sidebar.
- **Professional Integration:** Direct links to developer's LinkedIn and GitHub profiles.

## 📂 Project Structure
```text
├── app.py                     # Main Streamlit application code
├── insurance_health_Regression.pkl # Trained serialized model
├── requirements.txt           # Required Python libraries
└── README.md                  # Project documentation
