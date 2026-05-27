import streamlit as st
import pandas as pd
import re
import joblib

# -------------------------------
# LOAD FILES
# -------------------------------

model = joblib.load("saved_models/best_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
encoders = joblib.load("saved_models/encoders.pkl")
feature_columns = joblib.load("saved_models/feature_columns.pkl")

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="MediScan AI",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #fff8f4;
    color: #1a0a00;
}

.stApp {
    background:
        radial-gradient(ellipse 70% 50% at 95% 5%,  #ffd4b8 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 5% 90%,  #ffe0cc 0%, transparent 55%),
        radial-gradient(ellipse 80% 60% at 50% 50%, #fff4ee 0%, transparent 80%),
        #fff8f4;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 3rem 5rem; max-width: 1180px; }

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #ff6b35 0%, #f7931e 40%, #ff4e50 100%);
    border-radius: 24px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 2.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(255,107,53,0.35), 0 4px 16px rgba(255,107,53,0.2);
}
.hero-wrap::before {
    content: '🩺';
    position: absolute;
    right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    opacity: 0.13;
    line-height: 1;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    top: -50px; right: 28%;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.22);
    border: 1px solid rgba(255,255,255,0.38);
    border-radius: 100px;
    padding: 0.25rem 0.95rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #fff;
    margin-bottom: 0.8rem;
}
.hero-wrap h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: #fff;
    margin: 0 0 0.45rem;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.hero-wrap p {
    color: rgba(255,255,255,0.84);
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
    max-width: 55%;
}

/* ── Section labels ── */
.sec-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #e05a1e;
    margin: 0 0 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.sec-label .num {
    background: linear-gradient(135deg, #ff6b35, #f7931e);
    color: white;
    font-size: 0.68rem;
    font-weight: 800;
    width: 22px; height: 22px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1.5px;
    background: linear-gradient(90deg, #f7931e40, transparent);
    border-radius: 2px;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1.5px solid #ffe0cc;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 24px rgba(255,107,53,0.08), 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.3s, border-color 0.3s;
}
.card:hover {
    border-color: #ffb899;
    box-shadow: 0 8px 40px rgba(255,107,53,0.14);
}

/* ── Labels ── */
label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label {
    color: #7a3a1a !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Inputs ── */
.stTextInput input,
.stNumberInput input {
    background: #fff8f4 !important;
    border: 1.5px solid #ffd4b8 !important;
    border-radius: 12px !important;
    color: #1a0a00 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,0.15) !important;
    outline: none !important;
    background: #fff !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background: #fff8f4 !important;
    border: 1.5px solid #ffd4b8 !important;
    border-radius: 12px !important;
    color: #1a0a00 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: #ff6b35 !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,0.15) !important;
}
[data-baseweb="popover"] {
    background: #fff !important;
    border: 1.5px solid #ffd4b8 !important;
    border-radius: 14px !important;
    box-shadow: 0 12px 40px rgba(255,107,53,0.15) !important;
}

/* ── Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #ff6b35 0%, #f7931e 50%, #ff4e50 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    box-shadow: 0 8px 28px rgba(255,107,53,0.4) !important;
    transition: all 0.25s ease !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 14px 40px rgba(255,107,53,0.55) !important;
    filter: brightness(1.06) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result cards ── */
.res-diagnosis {
    background: linear-gradient(135deg, #fff4ee, #ffe8d6);
    border: 2px solid #ffb880;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    height: 100%;
}
.res-diagnosis .rlabel {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #c0521a; margin-bottom: 0.5rem;
}
.res-diagnosis .rvalue {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem; font-weight: 800;
    color: #c43d00;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.res-diagnosis .rsub {
    font-size: 0.8rem; color: #b06040;
    margin-top: 0.4rem; font-weight: 400;
}

.res-confidence {
    background: linear-gradient(135deg, #fff9f0, #ffecd9);
    border: 2px solid #ffc077;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    height: 100%;
}
.res-confidence .rlabel {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #b06010; margin-bottom: 0.5rem;
}
.res-confidence .rvalue {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem; font-weight: 800;
    color: #d97706;
    letter-spacing: -0.02em;
}
.conf-track {
    background: #ffe0b2;
    border-radius: 100px;
    height: 10px;
    margin-top: 1rem;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ff6b35, #f7931e);
    box-shadow: 0 0 10px rgba(255,107,53,0.45);
}

/* ── Vital pills ── */
.pills-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1.2rem;
}
.pill {
    background: #fff4ee;
    border: 1.5px solid #ffd4b8;
    border-radius: 100px;
    padding: 0.35rem 1rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #c04020;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.pill-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: linear-gradient(135deg, #ff6b35, #f7931e);
    box-shadow: 0 0 5px #ff6b3570;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1.5px solid #ffd4b8 !important;
}
.stDataFrame thead th {
    background: #fff4ee !important;
    color: #c04020 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.stDataFrame tbody tr:hover { background: #fff8f4 !important; }

/* ── Disclaimer ── */
.disclaimer {
    margin-top: 1.5rem;
    padding: 1rem 1.4rem;
    background: #fffbf0;
    border: 1.5px solid #fcd34d;
    border-radius: 14px;
    font-size: 0.82rem;
    color: #92610a;
    line-height: 1.65;
}

.stAlert { border-radius: 14px !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# HERO
# -----------------------------------------------

st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🔬 AI-Powered · Random Forest Model</div>
    <h1>MediScan AI</h1>
    <p>Enter patient vitals and symptoms below to receive an instant AI-generated diagnostic prediction.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------
# SECTION 1 — Patient Profile
# -----------------------------------------------

st.markdown('<div class="sec-label"><span class="num">1</span> Patient Profile</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    age = st.number_input("Age (years)", min_value=1, max_value=100, value=25)
with c2:
    gender = st.selectbox("Biological Sex", encoders['Gender'].classes_)
with c3:
    bp = st.text_input("Blood Pressure (mmHg)", placeholder="120/80")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------
# SECTION 2 — Symptoms
# -----------------------------------------------

st.markdown('<div class="sec-label"><span class="num">2</span> Reported Symptoms</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    symptom1 = st.selectbox("Primary Symptom", encoders['Symptom_1'].classes_)
with c5:
    symptom2 = st.selectbox("Secondary Symptom", encoders['Symptom_2'].classes_)
with c6:
    symptom3 = st.selectbox("Tertiary Symptom", encoders['Symptom_3'].classes_)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------
# SECTION 3 — Vitals
# -----------------------------------------------

st.markdown('<div class="sec-label"><span class="num">3</span> Vital Signs</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
c7, c8, c9 = st.columns(3)
with c7:
    temperature = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
with c8:
    oxygen = st.number_input("SpO₂ — Oxygen Saturation (%)", min_value=70, max_value=100, value=98)
with c9:
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=180, value=72)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------
# PREDICT BUTTON
# -----------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)
clicked = st.button("⚡  Run Diagnostic Analysis")

# -----------------------------------------------
# RESULTS
# -----------------------------------------------

if clicked:
    if not bp or not re.match(r'^\d{2,3}/\d{2,3}$', bp):
        st.error("⚠️  Please enter Blood Pressure in the format **120/80**")
        st.stop()

    systolic, diastolic = map(int, bp.split('/'))

    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Symptom_1': symptom1,
        'Symptom_2': symptom2,
        'Symptom_3': symptom3,
        'Body_Temperature_C': temperature,
        'Oxygen_Saturation_%': oxygen,
        'Heart_Rate_bpm': heart_rate,
        'Systolic_BP': systolic,
        'Diastolic_BP': diastolic
    }])

    for col in ['Gender', 'Symptom_1', 'Symptom_2', 'Symptom_3']:
        input_data[col] = encoders[col].transform(input_data[col])

    input_data = input_data[feature_columns]
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    result = encoders['Diagnosis'].inverse_transform(prediction)[0]

    probs = model.predict_proba(input_scaled)
    confidence = max(probs[0]) * 100

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label"><span class="num">4</span> Diagnostic Results</div>', unsafe_allow_html=True)

    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""
        <div class="res-diagnosis">
            <div class="rlabel">🧬 Predicted Diagnosis</div>
            <div class="rvalue">{result}</div>
            <div class="rsub">Based on provided symptoms &amp; vitals</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="res-confidence">
            <div class="rlabel">📊 Model Confidence</div>
            <div class="rvalue">{confidence:.1f}%</div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{int(confidence)}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pills-wrap">
        <div class="pill"><span class="pill-dot"></span> 🌡 {temperature}°C</div>
        <div class="pill"><span class="pill-dot"></span> 💧 SpO₂ {oxygen}%</div>
        <div class="pill"><span class="pill-dot"></span> ❤️ {heart_rate} bpm</div>
        <div class="pill"><span class="pill-dot"></span> 🩸 {systolic}/{diastolic} mmHg</div>
        <div class="pill"><span class="pill-dot"></span> 👤 {age} yrs · {gender}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label"><span class="num">5</span> Full Probability Distribution</div>', unsafe_allow_html=True)

    prob_df = pd.DataFrame({
        "Disease": encoders['Diagnosis'].classes_,
        "Probability (%)": (probs[0] * 100).round(2)
    }).sort_values(by="Probability (%)", ascending=False).reset_index(drop=True)
    prob_df.index += 1

    st.dataframe(prob_df, use_container_width=True, height=min(40 + len(prob_df) * 35, 420))

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Clinical Disclaimer:</strong> This AI-generated result is for decision-support
        purposes only and must <em>not</em> replace a licensed physician's evaluation, clinical
        judgement, or formal diagnosis. Always consult a qualified medical professional.
    </div>
    """, unsafe_allow_html=True)
