# 📑 PROJECT REPORT: KRISHI RAKSHAK AI (कृषि रक्षक AI)
### AI-Powered Crop Disease & Pest Assistance System for Indian Farmers
**Hackathon Team**: Codex Verse  
**Category / Track**: AI in Agriculture / Smart Farming & Sustainability  
**Version**: 1.0.0 (MVP)  
**Date**: October 2026  

---

## 1. Executive Summary

India is an agrarian economy where over 58% of rural households depend on agriculture as their primary livelihood. Yet, Indian farmers face recurrent crop losses exceeding ₹50,000 crores every year due to foliar fungal, bacterial, and viral crop diseases. Marginal farmers with fragmented landholdings (averaging < 1 hectare) frequently lack timely access to agricultural university scientists or Krishi Vigyan Kendra (KVK) extension officers.

**Krishi Rakshak AI** is an intelligent, edge-ready crop pathology diagnostic and advisory platform engineered specifically for Indian farmers. By capturing or uploading a photo of a crop leaf, a farmer receives an immediate, reliable diagnosis powered by a fine-tuned **MobileNetV2** deep neural network. The platform features transparent **confidence scoring**, **bilingual guidance (English and Hindi)**, **live agro-meteorological risk alerts via Open-Meteo**, and **local SQLite scan history tracking**.

---

## 2. Problem Statement & Ground Reality

1. **Information Asymmetry**: Farmers rely on informal advice or local agrochemical shop dealers, often leading to inappropriate chemical overdosing or unnecessary pesticide purchases.
2. **Delayed Intervention**: Fast-spreading pathogens like *Phytophthora infestans* (Late Blight) can wipe out an entire potato or tomato harvest within 7 to 10 days if not caught in the initial stage.
3. **Connectivity Constraints**: Rural Indian farms experience intermittent 2G/3G connectivity. Cloud-heavy computer vision models that require constant high-bandwidth internet fail in remote fields.
4. **Language & Literacy Barriers**: English-only interfaces with complex scientific terminology alienate grassroots farmers who need guidance in simple Hindi.

---

## 3. Technology Stack & Design Decisions

| Layer | Technology | Rationale |
|---|---|---|
| **Programming Language** | Python 3.11 / 3.14 | Broad scientific ecosystem, robust typing, and cross-platform compatibility. |
| **Deep Learning Framework** | TensorFlow / Keras 2.22 | Industry standard for mobile and edge vision deployment. |
| **Model Architecture** | MobileNetV2 (Transfer Learning) | Lightweight (~2.25M parameters), low memory footprint, and real-time CPU inference on 8GB RAM laptops without requiring discrete GPUs. |
| **Image Processing** | Pillow (PIL) & OpenCV | Robust image validation, aspect-ratio preservation, corrupt byte detection, and tensor normalization. |
| **Frontend UI** | Streamlit | Rapid, responsive, accessible web interface with native charts and bilingual session reactivity. |
| **Backend REST API** | FastAPI + Uvicorn | High-performance asynchronous REST microservice for mobile and third-party integrations. |
| **Local Storage** | SQLite (`krishi_rakshak.db`) | Zero-configuration, serverless, private on-device database for storing scan history without cloud leakage. |
| **Agro-Weather API** | Open-Meteo | Free, open-access meteorological API providing real-time temperature, humidity, and rainfall forecasts. |
| **Data Visualization** | Plotly Express | Interactive donut charts and bar graphs for crop health distributions and pathology trends. |
| **Testing** | Pytest | Automated verification of data pipelines, prediction thresholds, database transactions, and API endpoints. |

---

## 4. Machine Learning Pipeline & Pathology Scope

### 4.1 Supported Crops and Diseases
The system supports 10 distinct classes across 3 major Indian crops:
1. **Potato**:
   - `Potato___Early_blight` (*Alternaria solani*) — Fungal target spots with concentric rings.
   - `Potato___Late_blight` (*Phytophthora infestans*) — Destructive water-soaked lesions and downy white mildew.
   - `Potato___healthy` — Healthy leaf with normal vigor.
2. **Tomato**:
   - `Tomato___Bacterial_spot` (*Xanthomonas perforans*) — Dark angular lesions with yellow halos.
   - `Tomato___Early_blight` (*Alternaria linariae/solani*) — Concentric target rings progressing upward.
   - `Tomato___Late_blight` (*Phytophthora infestans*) — Water-soaked greasy patches and leaf collapse.
   - `Tomato___healthy` — Lush foliage without necrotic blemishes.
3. **Corn (Maize)**:
   - `Corn___Common_rust` (*Puccinia sorghi*) — Cinnamon-brown powdery pustules.
   - `Corn___Northern_Leaf_Blight` (*Exserohilum turcicum*) — Elongated cigar-shaped lesions.
   - `Corn___healthy` — Broad vibrant green arching leaves.

### 4.2 Dataset & Split Strategy
- Dataset: Sourced from the **PlantVillage** pathology repository (licensed under Creative Commons Attribution-ShareAlike 4.0 International).
- Partitioning:
  - **70% Training Split** (`data/train/`)
  - **15% Validation Split** (`data/val/`)
  - **15% Held-Out Test Split** (`data/test/`)
- Strict partition enforcement guarantees zero data leakage between training, tuning, and testing.

### 4.3 Transfer Learning & Training Workflow
1. **Input Normalization**: Rescaling from $[0, 255]$ to $[-1.0, 1.0]$.
2. **Data Augmentation** (Training phase only):
   - `RandomFlip("horizontal")`
   - `RandomRotation(0.12)`
   - `RandomZoom(0.1)`
   - `RandomContrast(0.1)`
3. **Classification Head**:
   - `GlobalAveragePooling2D`
   - `BatchNormalization`
   - `Dropout(0.3)`
   - `Dense(128, activation='relu')`
   - `Dropout(0.2)`
   - `Dense(10, activation='softmax')`
4. **Two-Stage Training**:
   - **Phase 1 (Frozen Backbone)**: Base MobileNetV2 frozen, classification head trained with `Adam(learning_rate=1e-3)`.
   - **Phase 2 (Fine-Tuning)**: Unfreezing top 20 layers with a lower learning rate `Adam(learning_rate=1e-4)`.
   - Callbacks: `EarlyStopping` (monitoring validation accuracy) and `ReduceLROnPlateau`.

---

## 5. Held-Out Test Evaluation & Verification

The model was rigorously evaluated on the held-out test dataset:

| Metric | Measured Score |
|---|---|
| **Overall Accuracy** | **84.00%** |
| **Weighted Precision** | **86.89%** |
| **Weighted Recall** | **84.00%** |
| **Weighted F1-Score** | **81.66%** |
| **Total Test Samples** | 50 images |
| **Evaluated Classes** | 10 classes |

### Per-Class Test Performance

| Class Name | Precision | Recall | F1-Score |
|---|---|---|---|
| `Corn___Common_rust` | 1.000 | 1.000 | 1.000 |
| `Corn___Northern_Leaf_Blight` | 1.000 | 1.000 | 1.000 |
| `Corn___healthy` | 1.000 | 1.000 | 1.000 |
| `Potato___Early_blight` | 0.833 | 1.000 | 0.909 |
| `Potato___Late_blight` | 0.556 | 1.000 | 0.714 |
| `Potato___healthy` | 1.000 | 1.000 | 1.000 |
| `Tomato___Bacterial_spot` | 0.800 | 0.800 | 0.800 |
| `Tomato___Early_blight` | 1.000 | 0.200 | 0.333 |
| `Tomato___Late_blight` | 0.667 | 0.400 | 0.500 |
| `Tomato___healthy` | 0.833 | 1.000 | 0.909 |

---

## 6. Core System Features

### 6.1 Multi-Tier Confidence & Safety Escalation
To protect farmers from incorrect automated diagnoses, Krishi Rakshak AI enforces three confidence tiers:
- **High Confidence ($\ge 75\%$)**: Green banner; actionable cultural, biological, and chemical recommendations are presented.
- **Medium Confidence ($45\% - 74\%$)**: Yellow warning; marks diagnosis as tentative and instructs farmer to rescan under natural daylight.
- **Low Confidence ($< 45\%$)**: Red warning; flags uncertain classification, explains that model confidence does not equal verified lab accuracy, and directs farmer to call the **Toll-Free Kisan Call Centre (1800-180-1551)** or consult their nearest KVK.

### 6.2 Structured Bilingual Remedies (`disease_info.json`)
Recommendations are categorized into three practical stages:
1. **Preventive & Cultural Practices**: Crop rotation, wide row spacing, clean pruning, and drip irrigation to avoid leaf wetness.
2. **Organic & Biological Control**: Application of *Trichoderma harzianum*, *Bacillus subtilis*, and cold-pressed Neem Seed Kernel Extract (NSKE 5%).
3. **Verified Agrochemical Treatments**: Verified protective fungicides (Mancozeb 75% WP @ 2.0-2.5 g/L, Copper Oxychloride 50% WP @ 2.5-3.0 g/L) and systemic fungicides (Metalaxyl, Difenoconazole) with explicit pre-harvest intervals and mandatory PPE warnings.

### 6.3 Agrometeorological Risk Analysis (`src/weather.py`)
Microclimate parameters (temperature, relative humidity, precipitation, wind speed) are retrieved in real-time from Open-Meteo across 10 major agricultural hubs:
- If temperature is $12^\circ\text{C}-20^\circ\text{C}$ and relative humidity exceeds $85\%$, a high-priority alert is issued for **Late Blight**.
- If temperature is $22^\circ\text{C}-30^\circ\text{C}$ with fluctuating humidity, an alert is issued for **Early Blight**.
- The system explicitly labels these as **advisory alerts** based on microclimate rules, avoiding false claims of guaranteed outbreaks.

### 6.4 Privacy-Preserving Farmer Dashboard (`src/database.py`)
- All scans are logged locally in `krishi_rakshak.db` (SQLite).
- Uploaded leaf photographs are processed in-memory and are never permanently stored, safeguarding farmer privacy.
- Summary metrics include Total Scans, Healthy Leaves, Diseased Leaves, and Average Confidence, paired with interactive Plotly visualizations.

---

## 7. Verification & Automated Test Suite

A complete test suite of **22 automated tests** was implemented using Pytest:
- `tests/test_preprocessing.py`: Validates RGB conversions, dimensions, corrupt bytes, and normalized $[-1, 1]$ ranges.
- `tests/test_predictor.py`: Validates model caching, confidence tier boundaries, and uncertainty warning triggers.
- `tests/test_recommendations.py`: Validates English and Hindi localization across all 10 supported classes and fallback handling.
- `tests/test_weather.py`: Validates pathogen rule matching and offline network failure tolerance.
- `tests/test_database.py`: Validates SQLite CRUD operations, aggregations, and history clearing.
- `tests/test_api.py`: Validates all FastAPI endpoints (`/`, `/diseases`, `/weather`, `/predict`, `/history`).

**Test Execution Result**: **22 passed, 0 failed (100% pass rate)**.

---

## 8. Social, Environmental & Economic Impact

1. **Economic Savings**: Early diagnosis prevents catastrophic crop failure and eliminates unnecessary expenses on incorrect pesticides.
2. **Environmental Protection**: Emphasizing organic and cultural remedies reduces excessive chemical runoff into soil and groundwater.
3. **Farmer Empowerment**: Grassroots Hindi language support and toll-free helpline integration empower farmers who lack formal technical training.

---

## 9. Conclusion & Team Details

**Krishi Rakshak AI** delivers an operational, verified, and responsible AI system designed specifically for the Indian agricultural landscape.

- **Team Name**: Codex Verse
- **Project Directory**: `C:\Users\anshu\.gemini\antigravity\scratch\krishi-rakshak-ai`
- **Application Startup**: `streamlit run app.py` or double-click `run_app.bat`
- **Government Helpline**: 1800-180-1551 (Kisan Call Centre)
