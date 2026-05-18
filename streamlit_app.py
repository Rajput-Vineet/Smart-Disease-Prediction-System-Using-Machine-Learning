import streamlit as st
import pandas as pd
import pickle
import re

# -------------------------------
# LOAD FILES
# -------------------------------

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="MediScan AI · Diagnosis System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --card:      #161e2e;
    --border:    #1e2d45;
    --accent:    #00d4aa;
    --accent2:   #3b82f6;
    --danger:    #f87171;
    --warn:      #fbbf24;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --radius:    14px;
    --glow:      0 0 30px rgba(0,212,170,0.15);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 0%, rgba(0,212,170,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(59,130,246,0.06) 0%, transparent 60%);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text); }

/* ── Hero title ── */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    line-height: 1.1;
    background: linear-gradient(135deg, #00d4aa 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* ── Section headers ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 2rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Card panels ── */
.stForm, div[data-testid="stVerticalBlock"] > div {
    border-radius: var(--radius);
}

/* ── Override inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,212,170,0.12) !important;
}

/* ── Labels ── */
label, [data-testid="stWidgetLabel"] {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0ea5e9) !important;
    color: #0a0e1a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.85rem 3rem !important;
    width: 100% !important;
    margin-top: 1rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 20px rgba(0,212,170,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,212,170,0.45) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Alert boxes ── */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(59,130,246,0.06));
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: var(--radius);
    padding: 1.75rem 2rem;
    margin: 1.5rem 0;
    box-shadow: var(--glow);
    animation: fadeUp 0.4s ease-out;
}
.result-diagnosis {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--accent);
    margin-bottom: 0.25rem;
}
.result-meta {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 1rem;
}
.confidence-bar-track {
    background: var(--border);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #00d4aa, #3b82f6);
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}

/* ── Vitals grid badges ── */
.vitals-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0 1.5rem;
}
.vital-badge {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.vital-badge .vb-val {
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--accent2);
    line-height: 1;
}
.vital-badge .vb-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-top: 0.25rem;
}

/* ── Sidebar badge ── */
.sb-badge {
    background: linear-gradient(135deg, rgba(0,212,170,0.12), rgba(59,130,246,0.08));
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.78rem;
    color: var(--accent);
    font-weight: 500;
    margin: 0.4rem 0;
    display: inline-block;
}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity:0; transform: translateY(12px); }
    to   { opacity:1; transform: translateY(0); }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(0,212,170,0.2); }
    50%       { box-shadow: 0 0 40px rgba(0,212,170,0.4); }
}

/* ── Metric override ── */
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 2rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SIDEBAR
# -------------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem;">
        <div style="font-family:'DM Serif Display',serif; font-size:1.5rem; color:#00d4aa;">MediScan AI</div>
        <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#64748b; margin-top:2px;">Diagnostic Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#64748b; margin-bottom:0.75rem;">Model Engine</div>
    <div class="sb-badge">🌲 Random Forest Classifier</div>
    <div style="font-size:0.8rem; color:#94a3b8; margin:0.75rem 0 1.25rem; line-height:1.6;">
        Ensemble learning model trained on clinical patient data. Handles complex symptom interactions with high accuracy.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.15em; color:#64748b; margin-bottom:0.75rem;">Input Features</div>
    """, unsafe_allow_html=True)

    features = [
        ("👤", "Demographics", "Age & Gender"),
        ("🤒", "Symptoms", "3 primary symptoms"),
        ("🌡️", "Temperature", "Body temp in °C"),
        ("💧", "Oxygen Sat.", "SpO₂ percentage"),
        ("💓", "Heart Rate", "BPM reading"),
        ("🩸", "Blood Pressure", "Systolic / Diastolic"),
        ("💊", "Treatment Plan", "Prescribed regimen"),
    ]

    for icon, label, detail in features:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.6rem; padding:0.4rem 0; border-bottom:1px solid #1e2d45;">
            <span style="font-size:1rem;">{icon}</span>
            <div>
                <div style="font-size:0.78rem; color:#e2e8f0; font-weight:500;">{label}</div>
                <div style="font-size:0.68rem; color:#64748b;">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.2); border-radius:8px; padding:0.75rem 1rem;">
        <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#fbbf24; margin-bottom:0.3rem;">⚠ Clinical Disclaimer</div>
        <div style="font-size:0.72rem; color:#94a3b8; line-height:1.5;">Predictions are model-based estimates. Always consult a licensed physician for diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# MAIN CONTENT
# -------------------------------

st.markdown('<div class="hero-title">Disease Diagnosis System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-Powered Clinical Decision Support · Random Forest Model</div>', unsafe_allow_html=True)

# ── PATIENT INFO ──────────────────────────────────────────
st.markdown('<div class="section-label">👤 Patient Demographics</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    age = st.number_input("Age (years)", 1, 100, 25)
with col2:
    gender = st.selectbox("Biological Sex", encoders['Gender'].classes_)
with col3:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:#161e2e; border:1px solid #1e2d45; border-radius:8px; padding:0.85rem 1rem; margin-top:0.5rem;'>
        <div style='font-size:0.65rem; text-transform:uppercase; letter-spacing:0.12em; color:#64748b;'>Patient ID</div>
        <div style='font-size:0.95rem; color:#e2e8f0; font-weight:500; margin-top:0.2rem;'>MS-{age:02d}{str(gender)[:1].upper()}-AUTO</div>
    </div>
    """, unsafe_allow_html=True)

# ── SYMPTOMS ─────────────────────────────────────────────
st.markdown('<div class="section-label">🤒 Primary Symptoms</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)
with col4:
    symptom1 = st.selectbox("Symptom 1", encoders['Symptom_1'].classes_)
with col5:
    symptom2 = st.selectbox("Symptom 2", encoders['Symptom_2'].classes_)
with col6:
    symptom3 = st.selectbox("Symptom 3", encoders['Symptom_3'].classes_)

# ── VITALS ────────────────────────────────────────────────
st.markdown('<div class="section-label">❤️ Vital Signs</div>', unsafe_allow_html=True)

col7, col8, col9, col10 = st.columns(4)
with col7:
    temperature = st.number_input("Body Temp (°C)", 35.0, 42.0, 37.0, step=0.1, format="%.1f")
with col8:
    oxygen = st.number_input("SpO₂ (%)", 70, 100, 98)
with col9:
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 180, 72)
with col10:
    bp = st.text_input("Blood Pressure", placeholder="120/80")

# ── TREATMENT ─────────────────────────────────────────────
st.markdown('<div class="section-label">💊 Clinical Plan</div>', unsafe_allow_html=True)

treatment = st.selectbox("Treatment Plan", encoders['Treatment_Plan'].classes_)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ── LIVE VITALS PREVIEW ───────────────────────────────────
# Determine vitals status
temp_color = "#f87171" if temperature > 38.5 else "#fbbf24" if temperature > 37.5 else "#00d4aa"
oxy_color = "#f87171" if oxygen < 90 else "#fbbf24" if oxygen < 95 else "#00d4aa"
hr_color = "#f87171" if heart_rate > 120 or heart_rate < 50 else "#fbbf24" if heart_rate > 100 else "#00d4aa"

st.markdown(f"""
<div style="margin: 0.5rem 0 1.5rem;">
    <div style="font-size:0.65rem; text-transform:uppercase; letter-spacing:0.15em; color:#64748b; margin-bottom:0.6rem;">Live Vitals Monitor</div>
    <div class="vitals-grid">
        <div class="vital-badge">
            <div class="vb-val" style="color:{temp_color};">{temperature:.1f}°</div>
            <div class="vb-label">Temperature</div>
        </div>
        <div class="vital-badge">
            <div class="vb-val" style="color:{oxy_color};">{oxygen}%</div>
            <div class="vb-label">Oxygen Sat.</div>
        </div>
        <div class="vital-badge">
            <div class="vb-val" style="color:{hr_color};">{heart_rate}</div>
            <div class="vb-label">Heart Rate</div>
        </div>
        <div class="vital-badge">
            <div class="vb-val" style="color:#3b82f6;">{bp if bp else "—"}</div>
            <div class="vb-label">Blood Pressure</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── PREDICT BUTTON ────────────────────────────────────────
predict_clicked = st.button("⚡ Run Diagnostic Analysis")

# -------------------------------
# PREDICTION LOGIC
# -------------------------------

if predict_clicked:

    if not bp or not re.match(r'^\d{2,3}/\d{2,3}$', bp.strip()):
        st.markdown("""
        <div style="background:rgba(248,113,113,0.08); border:1px solid rgba(248,113,113,0.3); border-radius:10px; padding:1rem 1.25rem; margin-top:1rem;">
            <div style="color:#f87171; font-weight:600; margin-bottom:0.25rem;">⚠ Invalid Blood Pressure Format</div>
            <div style="color:#94a3b8; font-size:0.85rem;">Please enter in format <code style="background:#1e2d45; padding:2px 6px; border-radius:4px;">120/80</code></div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    try:
        systolic, diastolic = bp.strip().split('/')
        systolic, diastolic = int(systolic), int(diastolic)

        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Symptom_1': symptom1,
            'Symptom_2': symptom2,
            'Symptom_3': symptom3,
            'Body_Temperature_C': temperature,
            'Oxygen_Saturation_%': oxygen,
            'Heart_Rate_bpm': heart_rate,
            'Treatment_Plan': treatment,
            'Systolic_BP': systolic,
            'Diastolic_BP': diastolic
        }])

        for col in ['Gender', 'Symptom_1', 'Symptom_2', 'Symptom_3', 'Treatment_Plan']:
            input_data[col] = encoders[col].transform(input_data[col])

        input_data = input_data[feature_columns]
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        result = encoders['Diagnosis'].inverse_transform(prediction)[0]

        proba = model.predict_proba(input_scaled)
        confidence = max(proba[0])
        conf_pct = round(confidence * 100, 1)

        # Confidence level label + color
        if confidence > 0.9:
            conf_label = "High Confidence"
            conf_color = "#00d4aa"
            conf_icon = "✅"
        elif confidence > 0.7:
            conf_label = "Moderate Confidence"
            conf_color = "#fbbf24"
            conf_icon = "⚠️"
        else:
            conf_label = "Low Confidence"
            conf_color = "#f87171"
            conf_icon = "❌"

        # ── Result Card ──
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:0.65rem; text-transform:uppercase; letter-spacing:0.15em; color:#64748b; margin-bottom:0.4rem;">Predicted Diagnosis</div>
            <div class="result-diagnosis">{result}</div>
            <div class="result-meta">Patient: {age}y {gender} · {symptom1}, {symptom2}, {symptom3}</div>
            <div style="display:flex; align-items:center; gap:0.75rem; margin-top:1rem;">
                <div style="flex:1;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                        <span style="font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em;">Confidence Score</span>
                        <span style="font-size:0.9rem; font-weight:700; color:{conf_color};">{conf_pct}%</span>
                    </div>
                    <div class="confidence-bar-track">
                        <div class="confidence-bar-fill" style="width:{conf_pct}%; background:linear-gradient(90deg, {conf_color}, #3b82f6);"></div>
                    </div>
                </div>
            </div>
            <div style="margin-top:1rem; background:rgba(0,0,0,0.2); border-radius:8px; padding:0.6rem 1rem; display:inline-block;">
                <span style="color:{conf_color}; font-weight:600; font-size:0.85rem;">{conf_icon} {conf_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics row ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Diagnosis", result[:12] + ("…" if len(result) > 12 else ""))
        m2.metric("Confidence", f"{conf_pct}%")
        m3.metric("Systolic BP", f"{systolic} mmHg")
        m4.metric("Diastolic BP", f"{diastolic} mmHg")

        # ── Input Summary (collapsible) ──
        with st.expander("📋 View Full Input Summary", expanded=False):
            st.dataframe(
                input_data.style.set_properties(**{
                    'background-color': '#161e2e',
                    'color': '#e2e8f0',
                    'border': '1px solid #1e2d45'
                }),
                use_container_width=True
            )

    except Exception as e:
        st.markdown(f"""
        <div style="background:rgba(248,113,113,0.08); border:1px solid rgba(248,113,113,0.3); border-radius:10px; padding:1rem 1.25rem; margin-top:1rem;">
            <div style="color:#f87171; font-weight:600;">⚠ Processing Error</div>
            <div style="color:#94a3b8; font-size:0.82rem; margin-top:0.3rem; font-family:monospace;">{str(e)}</div>
        </div>
        """, unsafe_allow_html=True)