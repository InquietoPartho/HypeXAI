# import streamlit as st
# import numpy as np
# import joblib
# import shap
# import matplotlib.pyplot as plt
# import streamlit.components.v1 as components

# # ----------------- Load Model & Scaler -----------------
# model = joblib.load('ensemble_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # ----------------- Page Setup -----------------
# st.set_page_config(
#     page_title="🫀 Heart Disease Risk Predictor",
#     page_icon="🫀",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )

# st.markdown("<h1 style='text-align: center;'>🫀 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
# st.markdown("""
# <div style='text-align: center; font-size: 16px;'>
# Estimate <b>risk of heart disease</b> based on common clinical parameters.<br>
# Enter patient data and click <b>Predict</b> to see the results.
# </div>
# """, unsafe_allow_html=True)
# st.markdown("---")

# # ----------------- User Inputs -----------------
# st.header("📋 Patient Information")

# col1, col2, col3 = st.columns(3)
# age = col1.slider('Age (years)', 11, 98, 55)
# sex = col2.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
# cp = col3.slider('Chest Pain Type (0-3)', 0, 3, 1)

# col4, col5, col6 = st.columns(3)
# trestbps = col4.slider('Resting Blood Pressure (mmHg)', 94, 200, 130)
# chol = col5.slider('Serum Cholesterol (mg/dl)', 126, 564, 245)
# fbs = col6.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)', [1, 0])

# col7, col8, col9 = st.columns(3)
# restecg = col7.slider('Resting ECG (0 = Normal, 1 = ST-T abnormality, 2 = LVH)', 0, 2, 1)
# thalach = col8.slider('Max Heart Rate Achieved', 71, 202, 150)
# exang = col9.selectbox('Exercise-Induced Angina (1 = Yes, 0 = No)', [1, 0])

# col10, col11, col12 = st.columns(3)
# oldpeak = col10.slider('ST Depression (Oldpeak)', 0.0, 6.2, 1.0, step=0.1)
# slope = col11.slider('Slope of Peak Exercise ST (0-2)', 0, 2, 1)
# ca = col12.slider('Number of Major Vessels (0-4)', 0, 4, 0)

# thal = st.slider('Thalassemia (0 = Unknown, 1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', 0, 3, 2)



# # ----------------- Prepare Input -----------------
# input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
#                         thalach, exang, oldpeak, slope, ca, thal]])
# input_scaled = scaler.transform(input_data)

# # ----------------- Prediction -----------------
# st.markdown("---")
# if st.button("🔮 Predict Risk"):
#     prediction = model.predict(input_scaled)[0]
#     probability = model.predict_proba(input_scaled)[0][1]

#     st.markdown("### 🎯 Prediction Result")
#     risk_tag = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"
#     risk_color = "danger" if prediction == 1 else "success"

#     st.metric("Risk Probability", f"{probability * 100:.2f}%", label_visibility="visible")
    
#     if prediction == 1:
#         st.error(f"{risk_tag}: Medical consultation is strongly recommended.")
#     else:
#         st.success(f"{risk_tag}: Keep maintaining a healthy lifestyle.")

#     st.markdown("---")
#     st.info("📌 This is a predictive tool. Always consult a certified cardiologist for real clinical advice.")

#     # ---------------- SHAP Force Plot + Explanation ----------------
#     st.markdown("## 🔬 Model Explanation (SHAP Force Plot)")

#     try:
#         feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

#         cat_model = model.named_estimators_['cat']
#         explainer = shap.TreeExplainer(cat_model)
#         shap_values = explainer.shap_values(input_scaled)

#         shap.initjs()
#         force_plot = shap.force_plot(
#             base_value=explainer.expected_value,
#             shap_values=shap_values[0],
#             features=input_scaled[0],
#             feature_names=feature_names,
#             matplotlib=False
#         )
#         components.html(
#             f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
#             height=130
#         )

#         # AI-generated Explanation
#         st.markdown("## 🧠 AI-Generated Risk Explanation")
#         shap_dict = {
#             name: (value, shap_val)
#             for name, value, shap_val in zip(feature_names, input_scaled[0], shap_values[0])
#         }

#         sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1][1]), reverse=True)

#         st.markdown("**Top influencing features:**")
#         for feature, (value, shap_val) in sorted_features[:5]:
#             direction = "⬆️ increased" if shap_val > 0 else "⬇️ decreased"
#             color = "#e74c3c" if shap_val > 0 else "#27ae60"
#             st.markdown(f"<span style='color:{color}'>→ <b>{feature}</b> = {value:.2f} ({direction} risk)</span>", unsafe_allow_html=True)

#     except Exception as e:
#         st.error("⚠️ SHAP explanation failed.")
#         st.exception(e)

    
    

# # ----------------- Footer -----------------
# st.markdown("""
# ---
# <div style='text-align: center; font-size: 15px;'>
# 🧠 Developed by <b>Pijush Kanti Roy Partho</b>  
# Hajee Mohammad Danesh Science and Technology University  
# © 2025 All Rights Reserved.
# </div>
# """, unsafe_allow_html=True)


# import streamlit as st
# import numpy as np
# import joblib
# import shap
# import matplotlib.pyplot as plt
# import streamlit.components.v1 as components
# import google.generativeai as genai
# import os

# # ----------------- Gemini API Config -----------------
# genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Replace with your API key

# # ----------------- Load Model & Scaler -----------------
# model = joblib.load('ensemble_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # ----------------- Page Setup -----------------
# st.set_page_config(
#     page_title="🫀 Heart Disease Risk Predictor",
#     page_icon="🫀",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )

# st.markdown("<h1 style='text-align: center;'>🫀 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
# st.markdown("""
# <div style='text-align: center; font-size: 16px;'>
# Estimate <b>risk of heart disease</b> based on common clinical parameters.<br>
# Enter patient data and click <b>Predict</b> to see the results.
# </div>
# """, unsafe_allow_html=True)
# st.markdown("---")

# # ----------------- User Inputs -----------------
# st.header("📋 Patient Information")

# col1, col2, col3 = st.columns(3)
# age = col1.slider('Age (years)', 11, 98, 55)
# sex = col2.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
# cp = col3.slider('Chest Pain Type (0-3)', 0, 3, 1)

# col4, col5, col6 = st.columns(3)
# trestbps = col4.slider('Resting Blood Pressure (mmHg)', 94, 200, 130)
# chol = col5.slider('Serum Cholesterol (mg/dl)', 126, 564, 245)
# fbs = col6.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)', [1, 0])

# col7, col8, col9 = st.columns(3)
# restecg = col7.slider('Resting ECG (0 = Normal, 1 = ST-T abnormality, 2 = LVH)', 0, 2, 1)
# thalach = col8.slider('Max Heart Rate Achieved', 71, 202, 150)
# exang = col9.selectbox('Exercise-Induced Angina (1 = Yes, 0 = No)', [1, 0])

# col10, col11, col12 = st.columns(3)
# oldpeak = col10.slider('ST Depression (Oldpeak)', 0.0, 6.2, 1.0, step=0.1)
# slope = col11.slider('Slope of Peak Exercise ST (0-2)', 0, 2, 1)
# ca = col12.slider('Number of Major Vessels (0-4)', 0, 4, 0)

# thal = st.slider('Thalassemia (0 = Unknown, 1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', 0, 3, 2)

# # ----------------- Prepare Input -----------------
# input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
#                         thalach, exang, oldpeak, slope, ca, thal]])
# input_scaled = scaler.transform(input_data)

# # ----------------- Prediction -----------------
# st.markdown("---")
# if st.button("🔮 Predict Risk"):
#     prediction = model.predict(input_scaled)[0]
#     probability = model.predict_proba(input_scaled)[0][1]

#     st.markdown("### 🎯 Prediction Result")
#     risk_tag = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"
#     risk_color = "danger" if prediction == 1 else "success"

#     st.metric("Risk Probability", f"{probability * 100:.2f}%", label_visibility="visible")

#     if prediction == 1:
#         st.error(f"{risk_tag}: Medical consultation is strongly recommended.")
#     else:
#         st.success(f"{risk_tag}: Keep maintaining a healthy lifestyle.")

#     st.markdown("---")
#     st.info("📌 This is a predictive tool. Always consult a certified cardiologist for real clinical advice.")

#     # ---------------- SHAP Force Plot + Explanation ----------------
#     st.markdown("## 🔬 Model Explanation (SHAP Force Plot)")

#     try:
#         feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

#         cat_model = model.named_estimators_['cat']
#         explainer = shap.TreeExplainer(cat_model)
#         shap_values = explainer.shap_values(input_scaled)

#         shap.initjs()
#         force_plot = shap.force_plot(
#             base_value=explainer.expected_value,
#             shap_values=shap_values[0],
#             features=input_scaled[0],
#             feature_names=feature_names,
#             matplotlib=False
#         )
#         components.html(
#             f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
#             height=130
#         )

#         # Top Features Explanation
#         st.markdown("## 🧠 AI-Generated Risk Explanation")
#         shap_dict = {
#             name: (value, shap_val)
#             for name, value, shap_val in zip(feature_names, input_scaled[0], shap_values[0])
#         }

#         sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1][1]), reverse=True)

#         st.markdown("**Top influencing features:**")
#         for feature, (value, shap_val) in sorted_features[:5]:
#             direction = "⬆️ increased" if shap_val > 0 else "⬇️ decreased"
#             color = "#e74c3c" if shap_val > 0 else "#27ae60"
#             st.markdown(f"<span style='color:{color}'>→ <b>{feature}</b> = {value:.2f} ({direction} risk)</span>", unsafe_allow_html=True)

#         # ---------------- Gemini Consultation ----------------
#         if st.button("💬 Ask Gemini for Consultation"):
#             try:
#                 prompt = (
#                     "Given a patient with the following parameters:\n"
#                     f"Age: {age}, Sex: {'Male' if sex == 1 else 'Female'}, Chest Pain Type: {cp}, "
#                     f"Resting BP: {trestbps}, Cholesterol: {chol}, Fasting Sugar: {fbs}, "
#                     f"Resting ECG: {restecg}, Max HR: {thalach}, Exercise Angina: {exang}, "
#                     f"Oldpeak: {oldpeak}, Slope: {slope}, Major Vessels: {ca}, Thalassemia: {thal}.\n\n"
#                     f"The model predicted a {'HIGH' if prediction == 1 else 'LOW'} risk of heart disease with a probability of {probability*100:.2f}%.\n"
#                     "Please explain the result like a cardiologist and suggest next steps in a medical context."
#                 )
#                 model_gemini = genai.GenerativeModel('gemini-pro')
#                 response = model_gemini.generate_content(prompt)
#                 st.markdown("### 🩺 Gemini Medical Advice")
#                 st.info(response.text)

#             except Exception as e:
#                 st.error("⚠️ Gemini API failed.")
#                 st.exception(e)

#     except Exception as e:
#         st.error("⚠️ SHAP explanation failed.")
#         st.exception(e)

# # ----------------- Footer -----------------
# st.markdown("""
# ---
# <div style='text-align: center; font-size: 15px;'>
# 🧠 Developed by <b>Pijush Kanti Roy Partho</b>  
# Hajee Mohammad Danesh Science and Technology University  
# © 2025 All Rights Reserved.
# </div>
# """, unsafe_allow_html=True)


# import streamlit as st
# import numpy as np
# import joblib
# import shap
# import matplotlib.pyplot as plt
# import streamlit.components.v1 as components

# # ----------------- Load Model & Scaler -----------------
# model = joblib.load('ensemble_model.pkl')  # Must include a CatBoost model in ensemble
# scaler = joblib.load('scaler.pkl')

# # ----------------- Page Setup -----------------
# st.set_page_config(
#     page_title="🫀 Heart Disease Risk Predictor",
#     page_icon="🫀",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )

# st.markdown("<h1 style='text-align: center;'>🫀 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
# st.markdown("""
# <div style='text-align: center; font-size: 16px;'>
# Estimate <b>risk of heart disease</b> based on clinical parameters.<br>
# Enter patient data and click <b>Predict</b> to see results and explanations.
# </div>
# """, unsafe_allow_html=True)
# st.markdown("---")

# # ----------------- User Inputs -----------------
# st.header("📋 Patient Information")

# col1, col2, col3 = st.columns(3)
# age = col1.slider('Age (years)', 11, 98, 55)
# sex = col2.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
# cp = col3.slider('Chest Pain Type (0-3)', 0, 3, 1)

# col4, col5, col6 = st.columns(3)
# trestbps = col4.slider('Resting Blood Pressure (mmHg)', 94, 200, 130)
# chol = col5.slider('Serum Cholesterol (mg/dl)', 126, 564, 245)
# fbs = col6.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)', [1, 0])

# col7, col8, col9 = st.columns(3)
# restecg = col7.slider('Resting ECG (0 = Normal, 1 = ST-T abnormality, 2 = LVH)', 0, 2, 1)
# thalach = col8.slider('Max Heart Rate Achieved', 71, 202, 150)
# exang = col9.selectbox('Exercise-Induced Angina (1 = Yes, 0 = No)', [1, 0])

# col10, col11, col12 = st.columns(3)
# oldpeak = col10.slider('ST Depression (Oldpeak)', 0.0, 6.2, 1.0, step=0.1)
# slope = col11.slider('Slope of Peak Exercise ST (0-2)', 0, 2, 1)
# ca = col12.slider('Number of Major Vessels (0-4)', 0, 4, 0)

# thal = st.slider('Thalassemia (0 = Unknown, 1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', 0, 3, 2)

# # ----------------- Prepare Input -----------------
# input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
#                         thalach, exang, oldpeak, slope, ca, thal]])
# input_scaled = scaler.transform(input_data)

# # ----------------- Prediction -----------------
# st.markdown("---")
# if st.button("🔮 Predict Risk"):
#     prediction = model.predict(input_scaled)[0]
#     probability = model.predict_proba(input_scaled)[0][1]

#     st.markdown("### 🎯 Prediction Result")
#     risk_tag = "🔴 High Risk" if prediction == 1 else "🟢 Low Risk"
#     st.metric("Risk Probability", f"{probability * 100:.2f}%")

#     if prediction == 1:
#         st.error(f"{risk_tag}: Medical consultation is strongly recommended.")
#     else:
#         st.success(f"{risk_tag}: Keep maintaining a healthy lifestyle.")

#     st.markdown("---")
#     st.info("📌 This is a predictive tool. Always consult a certified cardiologist for real clinical advice.")

#     # ---------------- SHAP Force Plot + Explanation ----------------
#     st.markdown("## 🔬 SHAP Explanation (Force Plot)")

#     try:
#         # Extract CatBoost model from ensemble
#         cat_model = model.named_estimators_['cat']
#         explainer = shap.TreeExplainer(cat_model)
#         shap_values = explainer.shap_values(input_scaled)

#         feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
#                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

#         shap.initjs()
#         html = shap.force_plot(
#             base_value=explainer.expected_value,
#             shap_values=shap_values[0],
#             features=input_scaled[0],
#             feature_names=feature_names,
#             matplotlib=False
#         ).html()

#         components.html(f"<head>{shap.getjs()}</head><body>{html}</body>", height=130)

#         # ---------------- GEMINI - AI Explanation ----------------
#         st.markdown("## 🧠 GEMINI: AI-Generated Explanation")
#         shap_dict = {
#             name: (value, shap_val)
#             for name, value, shap_val in zip(feature_names, input_scaled[0], shap_values[0])
#         }

#         sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1][1]), reverse=True)

#         st.markdown("**Top 5 contributing features to the prediction:**")
#         for feature, (value, shap_val) in sorted_features[:5]:
#             direction = "increased" if shap_val > 0 else "decreased"
#             risk_type = "higher" if shap_val > 0 else "lower"
#             color = "#e74c3c" if shap_val > 0 else "#2ecc71"

#             st.markdown(
#                 f"<div style='padding:6px; border-left: 6px solid {color}; background-color:#f9f9f9;'>"
#                 f"<b>{feature.capitalize()}</b> = <code>{value:.2f}</code> → "
#                 f"{direction} the risk of heart disease ({risk_type} contribution)"
#                 f"</div>", unsafe_allow_html=True
#             )

#     except Exception as e:
#         st.error("⚠️ SHAP explanation failed.")
#         st.exception(e)

# # ----------------- Footer -----------------
# st.markdown("""
# ---
# <div style='text-align: center; font-size: 15px;'>
# 🧠 Developed by <b>Pijush Kanti Roy Partho</b><br>
# Hajee Mohammad Danesh Science and Technology University<br>
# © 2025 All Rights Reserved.
# </div>
# """, unsafe_allow_html=True)
import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from dotenv import load_dotenv
import google.generativeai as genai
import os

# ----------------- Load API Key -----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("api_key")
if not GOOGLE_API_KEY:
    raise RuntimeError("Google API Key not found! Please add it to your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-pro")

# ----------------- Load Model -----------------
model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')

# ----------------- Session State Init -----------------
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'probability' not in st.session_state:
    st.session_state.probability = None
if 'shap_summary' not in st.session_state:
    st.session_state.shap_summary = ""
if 'input_scaled' not in st.session_state:
    st.session_state.input_scaled = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = gemini_model.start_chat(history=[])

# ----------------- Page Layout -----------------
st.set_page_config(page_title="🫀 Heart Risk + Gemini AI Doctor", layout="centered")
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Enter patient details to predict risk and consult Gemini AI doctor.")

# ----------------- Inputs -----------------
st.header("📋 Patient Information")
col1, col2, col3 = st.columns(3)
age = col1.slider('Age (years)', 11, 98, 55)
sex = col2.selectbox('Sex (1 = Male, 0 = Female)', [1, 0])
cp = col3.slider('Chest Pain Type (0-3)', 0, 3, 1)

col4, col5, col6 = st.columns(3)
trestbps = col4.slider('Resting Blood Pressure', 94, 200, 130)
chol = col5.slider('Cholesterol (mg/dl)', 126, 564, 245)
fbs = col6.selectbox('Fasting Sugar > 120? (1 = Yes, 0 = No)', [1, 0])

col7, col8, col9 = st.columns(3)
restecg = col7.slider('ECG (0-2)', 0, 2, 1)
thalach = col8.slider('Max Heart Rate', 71, 202, 150)
exang = col9.selectbox('Exercise Angina (1 = Yes, 0 = No)', [1, 0])

col10, col11, col12 = st.columns(3)
oldpeak = col10.slider('ST Depression (oldpeak)', 0.0, 6.2, 1.0, step=0.1)
slope = col11.slider('Slope (0-2)', 0, 2, 1)
ca = col12.slider('Major Vessels (0-4)', 0, 4, 0)

thal = st.slider('Thalassemia (0-3)', 0, 3, 2)

# ----------------- Predict -----------------
feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)

if st.button("🔮 Predict Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    st.session_state.predicted = True
    st.session_state.probability = probability
    st.session_state.input_scaled = input_scaled

    # SHAP
    cat_model = model.named_estimators_['cat']
    explainer = shap.TreeExplainer(cat_model)
    shap_values = explainer.shap_values(input_scaled)
    shap_dict = {
        name: (value, shap_val)
        for name, value, shap_val in zip(feature_names, input_scaled[0], shap_values[0])
    }
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1][1]), reverse=True)
    summary = "Patient SHAP Summary:\n"
    for feature, (val, impact) in sorted_features[:5]:
        direction = "increased" if impact > 0 else "decreased"
        summary += f"- {feature}: {val:.2f} → {direction} risk\n"
    st.session_state.shap_summary = summary

# ----------------- Show Results -----------------
if st.session_state.predicted:
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    prob = st.session_state.probability
    pred_tag = "🔴 High Risk" if prob > 0.5 else "🟢 Low Risk"
    st.metric("Risk Probability", f"{prob * 100:.2f}%")
    st.success(pred_tag if prob <= 0.5 else pred_tag)

    # SHAP Force Plot
    try:
        st.subheader("🔬 SHAP Explanation")
        cat_model = model.named_estimators_['cat']
        explainer = shap.TreeExplainer(cat_model)
        shap_values = explainer.shap_values(st.session_state.input_scaled)

        shap.initjs()
        force_plot = shap.force_plot(
            base_value=explainer.expected_value,
            shap_values=shap_values[0],
            features=st.session_state.input_scaled[0],
            feature_names=feature_names,
            matplotlib=False
        )
        components.html(f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>", height=150)
    except:
        st.warning("Could not generate SHAP force plot.")

    # ----------------- Gemini Chat Assistant -----------------
    st.subheader("🧠 Gemini AI Doctor Chat")

    st.markdown("### 🧬 SHAP-Based Risk Summary")
    st.text(st.session_state.shap_summary)

    # Show previous chat
    for msg in st.session_state.chat_session.history:
        with st.chat_message("assistant" if msg.role == "model" else msg.role):
            st.markdown(msg.parts[0].text)

    # Chat input
    chat_input = st.chat_input("Ask Gemini about this patient...")
    if chat_input:
        st.chat_message("user").markdown(chat_input)

        # Append SHAP summary only in first message
        if len(st.session_state.chat_session.history) == 0:
            chat_input = f"Patient SHAP Summary:\n{st.session_state.shap_summary}\n\nUser: {chat_input}"

        response = st.session_state.chat_session.send_message(chat_input)
        with st.chat_message("assistant"):
            st.markdown(response.text)

# ----------------- Footer -----------------
st.markdown("""
---
<div style='text-align: center; font-size: 14px;'>
🧠 Built by <b>Pijush Kanti Roy Partho</b> — HSTU  
© 2025 All Rights Reserved.
</div>
""", unsafe_allow_html=True)
