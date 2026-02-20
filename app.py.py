import streamlit as st
import math

st.title("  mindSignal - Mental Health Risk Predictor")

work = st.slider("Work hours per week", 0, 100, 40)
sleep = st.slider("Sleep hours", 0.0, 12.0, 6.5)
wlb = st.slider("Work-life balance (1-10)", 1, 10, 6)
sad = st.slider("Sadness (1-10)", 1, 10, 6)
fatigue = st.slider("Fatigue (1-10)", 1, 10, 5)
social = st.slider("Social support (1-10)", 1, 10, 6)

def predict(work, sleep, wlb, sad, fatigue, social):
    logit = (-0.8 +
             (work - 40) * 0.06 +
             (7 - sleep) * 0.35 +
             (5 - wlb) * 0.25 +
             (sad - 5) * 0.2 +
             (fatigue - 5) * 0.18 +
             (5 - social) * 0.2)

    prob = 1 / (1 + math.exp(-logit))
    return max(0.12, min(prob, 0.92))

if st.button("Assess Risk"):
    prob = predict(work, sleep, wlb, sad, fatigue, social)
    st.subheader(f"Risk Probability: {round(prob*100,2)}%")
