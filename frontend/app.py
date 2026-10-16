"""
KRISHI RAKSHAK AI - Web Application
Team: Codex Verse
Description: AI-powered crop leaf disease and pest assistance system for Indian farmers.
             Features MobileNetV2 inference, bilingual (EN/HI) UI, agro-weather advisories,
             structured remedies, and a local SQLite farmer analytics dashboard.
"""

import json
import sys
from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure backend directory is in Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import config
from src.preprocessing import load_and_validate_image, ImageValidationError
from src.predictor import KrishiPredictor
from src.recommendations import RemedyEngine
from src.weather import WeatherAdvisor
from src.database import ScanHistoryDB

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Krishi Rakshak AI | Team Codex Verse",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Agriculture Themed CSS
st.markdown("""
<style>
    /* Main Theme Variables */
    :root {
        --primary-green: #2E7D32;
        --accent-green: #4CAF50;
        --soft-green: #E8F5E9;
        --border-green: #C8E6C9;
        --earth-tan: #8D6E63;
        --text-dark: #1B2E1B;
    }
    
    /* Global Container Adjustments */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    /* Header Card */
    .krishi-header {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #43a047 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
        margin-bottom: 24px;
    }
    .krishi-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #ffffff;
    }
    .krishi-header p {
        margin: 6px 0 0 0;
        font-size: 1.05rem;
        color: #e8f5e9;
        font-weight: 400;
    }
    .krishi-badge {
        display: inline-block;
        background-color: #81c784;
        color: #1b5e20;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-top: 8px;
    }
    
    /* Card Styles */
    .metric-card {
        background: #ffffff;
        border: 1px solid var(--border-green);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .metric-number {
        font-size: 2rem;
        font-weight: 800;
        color: var(--primary-green);
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555555;
        font-weight: 600;
    }
    
    /* Prediction Banner */
    .prediction-box {
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid;
    }
    .prediction-healthy {
        background-color: #f1f8e9;
        border-color: #689f38;
        color: #2e7d32;
    }
    .prediction-diseased {
        background-color: #fff8e1;
        border-color: #ffa000;
        color: #b78103;
    }
    
    /* Helpline Box */
    .helpline-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 15px 0;
        color: #0d47a1;
        font-weight: 600;
    }
    
    /* Disclaimer Card */
    .disclaimer-card {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        padding: 14px;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #616161;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_app_components():
    """Initializes and caches model predictor, recommendation engine, weather, and DB."""
    predictor = KrishiPredictor()
    remedy_engine = RemedyEngine()
    weather_advisor = WeatherAdvisor()
    db = ScanHistoryDB()
    
    # Load translations
    with open(config.TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        translations = json.load(f)
        
    return predictor, remedy_engine, weather_advisor, db, translations


# Initialize Components
predictor, remedy_engine, weather_advisor, history_db, translations = get_app_components()

# Session State for Language & Image
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = None
if "current_scan" not in st.session_state:
    st.session_state.current_scan = None

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/plant-under-sun.png", width=64)
    st.title("🌾 Krishi Rakshak AI")
    st.caption("Team Codex Verse")
    
    # Language Switcher
    lang_choice = st.radio(
        translations[st.session_state.lang]["select_language"],
        options=["English", "हिंदी (Hindi)"],
        index=0 if st.session_state.lang == "en" else 1,
        horizontal=True
    )
    st.session_state.lang = "en" if "English" in lang_choice else "hi"
    T = translations[st.session_state.lang]
    
    st.markdown("---")
    
    # Navigation Radio
    nav_option = st.radio(
        "Navigation",
        options=[
            T["nav_scanner"],
            T["nav_symptoms"],
            T["nav_remedies"],
            T["nav_weather"],
            T["nav_dashboard"],
            T["nav_about"]
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    # Quick Offline & Model Status Indicator
    is_demo = predictor.is_demo_mode()
    if is_demo:
        st.warning("⚠️ **Mode**: Demo Simulation (Run `python train_model.py` to activate trained weights)")
    else:
        st.success("🟢 **AI Engine**: MobileNetV2 (Active & Ready)")
        
    st.caption(f"📞 Kisan Helpline: {config.KISAN_CALL_CENTRE}")
    st.caption(T["offline_notice"])

# Header Banner
st.markdown(f"""
<div class="krishi-header">
    <h1>{T["app_title"]}</h1>
    <p>{T["app_subtitle"]}</p>
    <span class="krishi-badge">{T["team_credit"]} | MobileNetV2 ML</span>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# TAB 1: AI CROP SCANNER
# ==============================================================================
if nav_option == T["nav_scanner"]:
    st.subheader(T["scanner_title"])
    st.write(T["scanner_subtitle"])
    
    col_input, col_preview = st.columns([1.1, 0.9])
    
    uploaded_file = None
    camera_photo = None
    
    with col_input:
        tab_upload, tab_camera, tab_samples = st.tabs([
            T["upload_tab"],
            T["camera_tab"],
            "💡 " + ("नमूना पत्तियां (Samples)" if st.session_state.lang == "hi" else "Sample Leaves")
        ])
        
        with tab_upload:
            uploaded_file = st.file_uploader(
                T["drag_drop"],
                type=["jpg", "jpeg", "png"],
                help="Upload a leaf picture under 15MB"
            )
            
        with tab_camera:
            camera_photo = st.camera_input(
                "Capture live leaf from field" if st.session_state.lang == "en" else "खेत से सीधे पत्ती की फोटो लें"
            )
            
        with tab_samples:
            st.caption("Click any real test leaf to test instant diagnosis:" if st.session_state.lang == "en" else "त्वरित जांच के लिए किसी भी नमूना पत्ती पर क्लिक करें:")
            sample_files = list(config.SAMPLES_DIR.glob("*_sample.jpg"))
            if sample_files:
                sample_cols = st.columns(2)
                for i, s_path in enumerate(sample_files[:6]):
                    cls_display = s_path.name.replace("_sample.jpg", "").replace("___", " - ")
                    with sample_cols[i % 2]:
                        if st.button(f"🍃 {cls_display}", key=f"btn_sample_{i}", use_container_width=True):
                            st.session_state.selected_sample = s_path
                            uploaded_file = None
                            camera_photo = None
            else:
                st.info("No sample leaves found in data/samples/. Run `python download_dataset.py` to populate samples.")

    # Determine Active Image Source
    active_image = None
    image_name = "Image"
    
    if camera_photo is not None:
        active_image = camera_photo
        image_name = "Live Camera Capture"
    elif uploaded_file is not None:
        active_image = uploaded_file
        image_name = uploaded_file.name
    elif st.session_state.selected_sample is not None and st.session_state.selected_sample.exists():
        active_image = st.session_state.selected_sample
        image_name = st.session_state.selected_sample.name

    with col_preview:
        if active_image is not None:
            try:
                pil_img = load_and_validate_image(active_image if not isinstance(active_image, Path) else str(active_image))
                st.image(pil_img, caption=f"Selected Leaf: {image_name}", use_container_width=True)
                
                scan_col1, scan_col2 = st.columns([1, 1])
                with scan_col1:
                    scan_clicked = st.button(f"🚀 {T['scan_now_btn']}", type="primary", use_container_width=True)
                with scan_col2:
                    if st.button(f"❌ {T['scan_another_btn']}", use_container_width=True):
                        st.session_state.selected_sample = None
                        st.session_state.current_scan = None
                        st.rerun()

                if scan_clicked:
                    with st.spinner(T["scanning_progress"]):
                        # 1. Run Real Preprocessing & Inference
                        prediction = predictor.predict(pil_img)
                        diagnosis = remedy_engine.get_diagnosis(prediction.class_name, lang=st.session_state.lang)
                        
                        # 2. Get local default weather for Bhopal hub as context
                        hub_info = config.AGRICULTURAL_HUBS["Bhopal (Malwa Plateau)"]
                        w_data = weather_advisor.get_weather(hub_info["lat"], hub_info["lon"])
                        curr_t = w_data["current"]["temperature"] if w_data.get("current") else None
                        curr_h = w_data["current"]["humidity"] if w_data.get("current") else None

                        # 3. Record Scan to SQLite DB
                        scan_id = history_db.record_scan(
                            crop_name=diagnosis["crop_name"],
                            disease_name=diagnosis["disease_name"],
                            is_healthy=diagnosis["is_healthy"],
                            confidence=prediction.confidence,
                            confidence_level=prediction.confidence_level,
                            weather_temp=curr_t,
                            weather_humidity=curr_h,
                            notes=f"Scanned via Web UI - {prediction.model_source}"
                        )
                        
                        st.session_state.current_scan = {
                            "scan_id": scan_id,
                            "prediction": prediction,
                            "diagnosis": diagnosis,
                            "pil_img": pil_img
                        }
            except ImageValidationError as e:
                st.error(f"⚠️ Image Validation Error: {str(e)}")
            except Exception as e:
                st.error(f"⚠️ Inference Error: {str(e)}")
        else:
            st.info("👈 Please upload an image, capture a photo, or select a sample leaf on the left.")

    # Display Diagnostic Assessment Results
    if st.session_state.current_scan is not None:
        scan_data = st.session_state.current_scan
        pred = scan_data["prediction"]
        diag = scan_data["diagnosis"]
        
        st.markdown("---")
        st.markdown(f"### 📋 {T['diagnosis_result_title']}")
        
        res_col1, res_col2 = st.columns([1, 1.2])
        
        with res_col1:
            box_class = "prediction-healthy" if diag["is_healthy"] else "prediction-diseased"
            status_text = T["status_healthy"] if diag["is_healthy"] else T["status_diseased"]
            
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h3 style="margin: 0;">{status_text}</h3>
                <h2 style="margin: 8px 0; color: inherit;">{diag["crop_name"]} : {diag["disease_name"]}</h2>
                <p style="margin: 0; font-size: 1rem;"><strong>Pathogen:</strong> {diag["pathogen"]}</p>
                <p style="margin: 4px 0 0 0; font-size: 0.9rem;"><strong>Severity:</strong> {diag["severity"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence Metrics & Progress Bar
            conf_val = pred.confidence
            st.markdown(f"**{T['confidence_score']}:** `{pred.confidence_percentage}` ({pred.confidence_level})")
            st.progress(conf_val)
            
            # Confidence Tier Advisory Box
            if pred.confidence_level == "High":
                st.success(f"**{T['confidence_high_title']}**: {T['confidence_high_desc']}")
            elif pred.confidence_level == "Medium":
                st.warning(f"**{T['confidence_medium_title']}**: {T['confidence_medium_desc']}")
                if pred.warning_hi if st.session_state.lang == "hi" else pred.warning_en:
                    st.caption(pred.warning_hi if st.session_state.lang == "hi" else pred.warning_en)
            else:
                st.error(f"**{T['confidence_low_title']}**: {T['confidence_low_desc']}")
                st.markdown(f"""
                <div class="helpline-box">
                    🚨 {T['kvk_advisory']}
                </div>
                """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"**Top Alternative Model Predictions:**")
            # Top Predictions Plotly Horizontal Bar
            top_df = pd.DataFrame(pred.top_predictions)
            top_df["Clean Name"] = top_df["class_name"].apply(lambda x: x.replace("___", " - "))
            fig_top = px.bar(
                top_df,
                x="confidence",
                y="Clean Name",
                orientation="h",
                text="percentage",
                color="confidence",
                color_continuous_scale="Greens",
                range_x=[0, 1.0]
            )
            fig_top.update_layout(
                margin=dict(l=0, r=0, t=20, b=0),
                height=180,
                xaxis_title="Confidence Probability",
                yaxis_title="",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_top, use_container_width=True)
            
            st.caption(f"Inference Engine: `{pred.model_source}`")


# ==============================================================================
# TAB 2: DISEASE DIAGNOSIS & SYMPTOMS
# ==============================================================================
elif nav_option == T["nav_symptoms"]:
    st.subheader(T["symptoms_header"])
    
    # Select disease class to explore
    selected_cls = st.selectbox(
        "Choose Crop Disease to View Detailed Pathology Profile:" if st.session_state.lang == "en" else "फसल रोग का विस्तृत विवरण देखने के लिए चुनें:",
        options=config.SUPPORTED_CLASSES,
        index=0 if not st.session_state.current_scan else config.SUPPORTED_CLASSES.index(st.session_state.current_scan["prediction"].class_name) if st.session_state.current_scan["prediction"].class_name in config.SUPPORTED_CLASSES else 0
    )
    
    diag_profile = remedy_engine.get_diagnosis(selected_cls, lang=st.session_state.lang)
    
    st.markdown(f"### 🌿 {diag_profile['crop_name']} — {diag_profile['disease_name']}")
    
    col_sym1, col_sym2 = st.columns([1, 1])
    with col_sym1:
        st.markdown(f"#### 🔍 {T['symptoms_title']}")
        for s in diag_profile["symptoms"]:
            st.markdown(f"- {s}")
            
        st.markdown(f"#### 🧬 {T['pathogen_title']}")
        st.info(f"**{diag_profile['pathogen']}** (Severity: {diag_profile['severity']})")

    with col_sym2:
        st.markdown(f"#### 🌧️ {T['causes_title']}")
        for c in diag_profile["causes"]:
            st.markdown(f"- {c}")
            
        if diag_profile.get("weather_advisory"):
            st.warning(f"**Microclimate Trigger:** {diag_profile['weather_advisory']}")


# ==============================================================================
# TAB 3: REMEDIES & TREATMENT
# ==============================================================================
elif nav_option == T["nav_remedies"]:
    st.subheader(T["remedies_header"])
    
    selected_cls = st.selectbox(
        "Select Crop Condition for Structured Treatment Plan:" if st.session_state.lang == "en" else "उपचार योजना देखने के लिए फसल स्थिति चुनें:",
        options=config.SUPPORTED_CLASSES,
        index=0 if not st.session_state.current_scan else config.SUPPORTED_CLASSES.index(st.session_state.current_scan["prediction"].class_name) if st.session_state.current_scan["prediction"].class_name in config.SUPPORTED_CLASSES else 0
    )
    
    diag = remedy_engine.get_diagnosis(selected_cls, lang=st.session_state.lang)
    treatment = diag["treatment"]
    
    st.markdown(f"### 📋 {diag['crop_name']} : {diag['disease_name']}")
    
    r_tab1, r_tab2, r_tab3 = st.tabs([
        T["preventive_title"],
        T["organic_title"],
        T["chemical_title"]
    ])
    
    with r_tab1:
        st.markdown("#### Cultural & Sanitation Practices")
        for p in diag["preventive_measures"]:
            st.markdown(f"- 🛡️ {p}")
        for cult in treatment.get("cultural", []):
            st.markdown(f"- 🧹 {cult}")
            
    with r_tab2:
        st.markdown("#### Bio-fungicides & Organic Options")
        for org in treatment.get("organic_biological", []):
            st.markdown(f"- 🌱 {org}")
            
    with r_tab3:
        st.markdown("#### Verified Agrochemical Treatments")
        for chem in treatment.get("chemical_guidance", []):
            st.markdown(f"- 🧪 {chem}")
            
        st.error(f"⚠️ {T['chemical_disclaimer']}")
        st.info(f"📞 {T['kvk_advisory']}")


# ==============================================================================
# TAB 4: WEATHER ALERTS & RISK ADVISORY
# ==============================================================================
elif nav_option == T["nav_weather"]:
    st.subheader(T["weather_title"])
    st.write("Agro-meteorological risk evaluation powered by Open-Meteo public API.")
    
    selected_hub = st.selectbox(
        T["weather_select_hub"],
        options=list(config.AGRICULTURAL_HUBS.keys()),
        index=1  # Default Bhopal
    )
    
    hub_coords = config.AGRICULTURAL_HUBS[selected_hub]
    weather_res = weather_advisor.get_weather(hub_coords["lat"], hub_coords["lon"])
    
    if weather_res["offline"]:
        st.warning(f"⚠️ Weather Service Notice: Working in offline mode ({weather_res.get('error', 'Network timeout')}).")
        
    curr = weather_res["current"]
    
    # 4 Weather Metric Cards
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{curr['icon']} {curr['temperature']}°C</div>
            <div class="metric-label">{T['weather_temp']} ({curr['condition_hi' if st.session_state.lang == 'hi' else 'condition_en']})</div>
        </div>
        """, unsafe_allow_html=True)
        
    with w_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">💧 {curr['humidity']}%</div>
            <div class="metric-label">{T['weather_humidity']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with w_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">🌧️ {curr['precipitation']} mm</div>
            <div class="metric-label">{T['weather_rain']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with w_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">💨 {curr['wind_speed']} km/h</div>
            <div class="metric-label">{T['weather_wind']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Disease Susceptibility Rule-based Alerts
    st.markdown(f"### ⚠️ {T['weather_risk_assessment']}")
    alerts = weather_advisor.evaluate_disease_risks(weather_res, remedy_engine.get_all_supported_diseases())
    
    if alerts:
        for al in alerts:
            crop_name = al["crop_hi"] if st.session_state.lang == "hi" else al["crop_en"]
            dis_name = al["disease_hi"] if st.session_state.lang == "hi" else al["disease_en"]
            adv = al["advisory_hi"] if st.session_state.lang == "hi" else al["advisory_en"]
            
            if al["risk_level"] == "High":
                st.error(f"🔴 **HIGH RISK ALERT: {crop_name} - {dis_name}** | {al['matched_criteria']}\n\n👉 {adv}")
            else:
                st.warning(f"🟡 **MODERATE RISK: {crop_name} - {dis_name}** | {al['matched_criteria']}\n\n👉 {adv}")
    else:
        st.success("✅ Current temperature and humidity levels are within safe ranges for supported crops. No immediate epidemic alert.")
        
    st.caption(f"ℹ️ {T['weather_disclaimer']}")

    # 3-Day Forecast Cards
    if weather_res.get("forecast"):
        st.markdown(f"#### 📅 {T['weather_forecast_3day']}")
        f_cols = st.columns(len(weather_res["forecast"]))
        for i, f_day in enumerate(weather_res["forecast"]):
            with f_cols[i]:
                c_name = f_day["condition_hi"] if st.session_state.lang == "hi" else f_day["condition_en"]
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{f_day['date']}</strong>
                    <div style="font-size: 1.5rem; margin: 4px 0;">{f_day['icon']}</div>
                    <div style="font-size: 0.9rem;"><strong>{f_day['max_temp']}° / {f_day['min_temp']}°C</strong></div>
                    <div style="font-size: 0.8rem; color: #666;">Rain: {f_day['precipitation']}mm</div>
                    <div style="font-size: 0.8rem; color: #2e7d32;">{c_name}</div>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# TAB 5: FARMER DASHBOARD & HISTORY
# ==============================================================================
elif nav_option == T["nav_dashboard"]:
    st.subheader(T["dashboard_title"])
    
    analytics = history_db.get_analytics()
    
    # 4 Metric Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{analytics['total_scans']}</div>
            <div class="metric-label">{T['metric_total_scans']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color: #2e7d32;">{analytics['healthy_count']}</div>
            <div class="metric-label">{T['metric_healthy']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color: #d32f2f;">{analytics['diseased_count']}</div>
            <div class="metric-label">{T['metric_diseased']}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number" style="color: #1976d2;">{int(analytics['avg_confidence'] * 100)}%</div>
            <div class="metric-label">{T['metric_avg_conf']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    if analytics["total_scans"] > 0:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown(f"**{T['chart_health_ratio']}**")
            ratio_df = pd.DataFrame([
                {"Status": "Healthy" if st.session_state.lang == "en" else "स्वस्थ", "Count": analytics["healthy_count"]},
                {"Status": "Diseased" if st.session_state.lang == "en" else "रोगग्रस्त", "Count": analytics["diseased_count"]}
            ])
            fig_pie = px.pie(
                ratio_df,
                values="Count",
                names="Status",
                color="Status",
                color_discrete_map={
                    "Healthy": "#4caf50", "स्वस्थ": "#4caf50",
                    "Diseased": "#ef5350", "रोगग्रस्त": "#ef5350"
                },
                hole=0.45
            )
            fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=250)
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            st.markdown(f"**{T['chart_disease_freq']}**")
            if analytics["top_diseases"]:
                dis_df = pd.DataFrame(analytics["top_diseases"])
                fig_bar = px.bar(
                    dis_df,
                    x="count",
                    y="name",
                    orientation="h",
                    color="count",
                    color_continuous_scale="Reds"
                )
                fig_bar.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=250,
                    xaxis_title="Frequency",
                    yaxis_title="",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No diseases detected in current records.")

        # Recent Scans Table
        st.markdown(f"#### 📜 {T['recent_scans_table']}")
        recent = history_db.get_recent_scans(limit=20)
        table_df = pd.DataFrame(recent)
        if not table_df.empty:
            table_df["is_healthy"] = table_df["is_healthy"].apply(lambda x: "✅ Healthy" if x == 1 else "⚠️ Diseased")
            table_df["confidence"] = table_df["confidence"].apply(lambda x: f"{x * 100:.1f}%")
            st.dataframe(
                table_df[["timestamp", "crop_name", "disease_name", "is_healthy", "confidence", "confidence_level", "notes"]],
                use_container_width=True,
                hide_index=True
            )

        # Clear History Option
        st.markdown("---")
        if st.button(T["clear_history_btn"]):
            history_db.clear_history()
            st.success(T["history_cleared_msg"])
            st.rerun()
    else:
        st.info(T["no_history_yet"])


# ==============================================================================
# TAB 6: ABOUT & AI ADVISORY
# ==============================================================================
elif nav_option == T["nav_about"]:
    st.subheader(f"ℹ️ {T['nav_about']}")
    
    st.markdown(f"""
    ### 🛡️ About Krishi Rakshak AI
    **Krishi Rakshak AI** is an advanced crop diagnostics and decision support system developed by **Team Codex Verse** for Indian farmers.
    
    #### ⚙️ Technical Highlights:
    - **Backbone**: MobileNetV2 with Transfer Learning on the PlantVillage pathology repository.
    - **Resolution**: 224 × 224 pixels with RGB color normalization `[-1, 1]` and training data augmentation.
    - **Multilingual Support**: Fully localized English and Hindi experience for grassroots accessibility.
    - **Agrometeorological Risk**: Live integration with Open-Meteo weather API for pathogen microclimate advisories.
    - **Confidence Thresholds**:
      - 🟢 **High (≥ 75%)**: Clear diagnosis and full treatment guidelines.
      - 🟡 **Medium (45%–74%)**: Tentative diagnosis with recommended rescan.
      - 🔴 **Low (< 45%)**: Uncertain classification with mandatory KVK escalation.
    - **Privacy First**: Farmer leaf photos are preprocessed in-memory and not permanently saved to disk.
    
    #### 📚 Evaluation & Accuracy on Held-Out Test Data:
    - **Test Accuracy**: 84.00%
    - **Weighted Precision**: 86.89%
    - **Weighted Recall**: 84.00%
    - **Weighted F1-Score**: 81.66%
    """)
    
    # Display Confusion Matrix if exists
    if config.CONFUSION_MATRIX_PATH.exists():
        st.image(str(config.CONFUSION_MATRIX_PATH), caption="Held-Out Test Set Confusion Matrix (10 Classes)", width=500)

    st.markdown(f"""
    <div class="disclaimer-card">
        <strong>{T['general_disclaimer_title']}:</strong><br>
        {T['general_disclaimer_text']}<br><br>
        <strong>Government of India Kisan Call Centre:</strong> Toll-Free 1800-180-1551
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption(f"Krishi Rakshak AI v{config.VERSION} | {config.TEAM_NAME} | Built for Indian Agricultural Empowerment")
