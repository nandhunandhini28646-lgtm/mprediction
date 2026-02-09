import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Student Mental Health Predictor", layout="centered")

st.title("🧠 Student Mental Health Prediction System")
st.write("Predict student mental health status using Machine Learning")

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("Student Mental health.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# DATA PREPROCESSING
# -----------------------------
encoder = LabelEncoder()

for col in data.columns:
    data[col] = encoder.fit_transform(data[col])

X = data.drop(columns=[data.columns[-1]])
y = data[data.columns[-1]]

# -----------------------------
# TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

st.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# USER INPUT FORM
# -----------------------------
st.subheader("📝 Enter Student Details")

inputs = []
for col in X.columns:
    value = st.slider(col, 0, int(data[col].max()))
    inputs.append(value)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Mental Health"):
    prediction = model.predict([inputs])

    if prediction[0] == 1:
        st.error("⚠️ Student may have Mental Health Issues")
    else:
        st.success("😊 Student Mental Health is Normal")
