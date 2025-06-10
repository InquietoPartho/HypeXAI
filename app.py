import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ----------------- Load Model & Scaler -----------------
model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')

# ----------------- Page Setup -----------------
st.set_page_config(
    page_title="🫀 HypeXAI: Hypertension Risk Prediction",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("<h1 style='text-align: center;'>🫀 HypeXAI: Hypertension Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 16px;'>
Estimate <b>risk of hypertension</b> based on common clinical parameters.<br>
Enter patient data and click <b>Predict</b> to see the results.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ----------------- User Inputs -----------------
st.header("📋 Patient Information")

col1, col2, col3 = st.columns(3)
age = col1.slider('Age (years)', 11, 98, 55)
sex = col2.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
cp = col3.slider('Chest Pain Type (0-3)', 0, 3, 1)

col4, col5, col6 = st.columns(3)
trestbps = col4.slider('Resting Blood Pressure (mmHg)', 94, 200, 130)
chol = col5.slider('Serum Cholesterol (mg/dl)', 126, 564, 245)
fbs = col6.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)', [1, 0])

col7, col8, col9 = st.columns(3)
restecg = col7.slider('Resting ECG (0 = Normal, 1 = ST-T abnormality, 2 = LVH)', 0, 2, 1)
thalach = col8.slider('Max Heart Rate Achieved', 71, 202, 150)
exang = col9.selectbox('Exercise-Induced Angina (1 = Yes, 0 = No)', [1, 0])

col10, col11, col12 = st.columns(3)
oldpeak = col10.slider('ST Depression (Oldpeak)', 0.0, 6.2, 1.0, step=0.1)
slope = col11.slider('Slope of Peak Exercise ST (0-2)', 0, 2, 1)
ca = col12.slider('Number of Major Vessels (0-4)', 0, 4, 0)

thal = st.slider('Thalassemia (0 = Unknown, 1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', 0, 3, 2)



# ----------------- Prepare Input -----------------
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)

# ----------------- Prediction -----------------
st.markdown("---")
if st.button("🔮 Predict Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("### 🎯 Prediction Result")
    risk_tag = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"
    risk_color = "danger" if prediction == 1 else "success"

    st.metric("Risk Probability", f"{probability * 100:.2f}%", label_visibility="visible")
    
    if prediction == 1:
        st.error(f"{risk_tag}: Medical consultation is strongly recommended.")
    else:
        st.success(f"{risk_tag}: Keep maintaining a healthy lifestyle.")

    st.markdown("---")
    st.info("📌 This is a predictive tool. Always consult a certified cardiologist for real clinical advice.")

    # ---------------- SHAP Force Plot + Explanation ----------------
    st.markdown("## 🔬 Model Explanation (SHAP Force Plot)")

    try:
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        cat_model = model.named_estimators_['cat']
        explainer = shap.TreeExplainer(cat_model)
        shap_values = explainer.shap_values(input_scaled)

        shap.initjs()
        force_plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[0],
            features=input_scaled[0],
            feature_names=feature_names,
            matplotlib=False
        )
        components.html(
            f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
            height=130
        )

        # AI-generated Explanation
        st.markdown("## 🧠 AI-Generated Risk Explanation")
        shap_dict = {
            name: (value, shap_val)
            for name, value, shap_val in zip(feature_names, input_scaled[0], shap_values[0])
        }

        sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1][1]), reverse=True)

        st.markdown("**Top influencing features:**")
        for feature, (value, shap_val) in sorted_features[:5]:
            direction = "⬆️ increased" if shap_val > 0 else "⬇️ decreased"
            color = "#e74c3c" if shap_val > 0 else "#27ae60"
            st.markdown(f"<span style='color:{color}'>→ <b>{feature}</b> = {value:.2f} ({direction} risk)</span>", unsafe_allow_html=True)

    except Exception as e:
        st.error("⚠️ SHAP explanation failed.")
        st.exception(e)

    
    

# ----------------- Footer -----------------
st.markdown("""
---
<div style='text-align: center; font-size: 15px;'>
🧠 Developed by <b>Pijush Kanti Roy Partho</b>  
Hajee Mohammad Danesh Science and Technology University  
© 2025 All Rights Reserved.
</div>
""", unsafe_allow_html=True)

