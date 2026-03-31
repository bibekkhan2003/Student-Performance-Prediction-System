import streamlit as st
import numpy as np
import pickle

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Student Dashboard", layout="wide")

# -------------------------
# DARK CSS 
# -------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}

.title {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
    color: #00C9A7;
}

.card {
    background-color: #1c1f26;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.6);
    text-align: center;
}

.metric {
    font-size: 32px;
    font-weight: bold;
}

.input-box {
    background-color: #1c1f26;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# -------------------------
# TITLE
# -------------------------
st.markdown('<div class="title">Student Performance Predictor</div>', unsafe_allow_html=True)
st.write("")

# -------------------------
# INPUT SECTION (GRID STYLE)
# -------------------------
st.subheader("Enter Student Details")

cols = st.columns(3)
user_input = []

for i, feature in enumerate(features):
    with cols[i % 3]:

        #  Gender Dropdown
        if feature == "Gender":
            gender = st.selectbox("Gender", ["Male", "Female"])
            val = 1 if gender == "Male" else 0

        #Tutoring Example (if exists)
        elif feature == "Tutoring":
            tutoring = st.selectbox("Tutoring", ["No", "Yes"])
            val = 1 if tutoring == "Yes" else 0

        #  Parental Support Example
        elif feature == "ParentalSupport":
            support = st.selectbox("Parental Support", ["Low", "Medium", "High"])
            val = {"Low": 0, "Medium": 1, "High": 2}[support]

        #  Default numeric input
        else:
            val = st.number_input(f"{feature}", value=0.0)

        user_input.append(val)

# Convert input
input_array = np.array(user_input).reshape(1, -1)
input_scaled = scaler.transform(input_array)

# -------------------------
# BUTTON
# -------------------------
st.write("")
if st.button(" Predict Performance"):

    prediction = model.predict(input_scaled)[0]

    # -------------------------
    # RESULT CARDS
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>Predicted Score</h4>
            <div class="metric">{prediction:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if prediction >= 3.7:
            status = "Excellent "
            color = "#00FFAA"
        elif prediction >= 3.0:
            status = "Good "
            color = "#00C9A7"
        elif prediction >= 2.5:
            status = "Average "
            color = "#FFA500"
        else:
            status = "Poor "
            color = "#FF4B4B"

        st.markdown(f"""
        <div class="card">
            <h4>Performance</h4>
            <div class="metric" style="color:{color}">{status}</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------
    # PROGRESS BAR
    # -------------------------
    st.write("")
    st.progress(min(int(prediction), 100))

    # -------------------------
    # INSIGHT
    # -------------------------
    st.subheader(" Insight")
    if prediction >= 3.7:
        st.success("Excellent performance.")
    elif prediction >= 3.0:
        st.info("Good performance, but can improve further.")
    else:
        st.warning("Performance needs improvement.")