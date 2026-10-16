# 🎤 HACKATHON PITCH & DEMO SCRIPT: KRISHI RAKSHAK AI
### Team: Codex Verse
**Presentation Duration**: 3 Minutes (Pitch) + 2 Minutes (Q&A)  
**Target Audience**: Hackathon Judges, AgTech Evaluators, and Agronomists  

---

## 1. Quick Presentation Checklist
- [ ] Open terminal in `krishi-rakshak-ai/` and double-click `run_app.bat` (or run `streamlit run app.py`).
- [ ] Browser opened at `http://localhost:8501`.
- [ ] Check that AI engine indicator in sidebar displays **"🟢 AI Engine: MobileNetV2 (Active & Ready)"**.
- [ ] Have sample leaves ready in the UI tab for instant scanning without searching for external image files.

---

## 2. Minute-by-Minute Pitch Script

### ⏱️ Minute 0:00 - 0:45 | The Problem & Our Hook
*(Speaker stands confidently, points to the big screen with the agriculture dashboard)*

**English Script**:
> *"Good morning, respected judges. In India, over 58% of our rural workforce depends on farming. Yet every single year, Indian farmers suffer over ₹50,000 crores in preventable crop losses simply because of foliar diseases like Late Blight and Common Rust. By the time an extension officer visits a remote village, the crop is already destroyed.*
>
> *We are **Team Codex Verse**, and we built **Krishi Rakshak AI** — an edge-ready, AI-powered crop doctor that diagnoses leaf diseases in under two seconds, delivers actionable treatment plans in simple Hindi and English, and gives microclimate weather warnings before outbreaks occur."*

**हिंदी रूपांतर (यदि जज हिंदी में सुनना चाहें)**:
> *"आदरणीय निर्णायकों को नमस्कार। भारत में हर साल किसानों को फसलों में लगने वाली बीमारियों जैसे पछेती झुलसा और रतुआ के कारण 50,000 करोड़ रुपये से अधिक का नुकसान उठाना पड़ता है। सही समय पर सही सलाह न मिलने से किसान या तो पूरी फसल गंवा देते हैं या बाजार से गलत कीटनाशक खरीदकर कर्ज में डूब जाते हैं।*
>
> *हम हैं **टीम कोडेक्स वर्स**, और हमने बनाया है **कृषि रक्षक AI** — जो केवल 2 सेकंड में पत्ती की फोटो से बीमारी पहचानता है, सरल हिंदी में जैविक और रासायनिक उपचार बताता है, और मौसम के आधार पर बीमारी का अलर्ट जारी करता है।"*

---

### ⏱️ Minute 0:45 - 1:45 | Live Interactive Demo

#### Step 1: Language Switcher
- In the sidebar, click on **"हिंदी (Hindi)"**.
- **Speaker Cue**: *"Notice how the entire interface — buttons, symptom descriptions, and emergency helplines — immediately shifts into grassroots Hindi."*

#### Step 2: Real AI Leaf Scanning
- Go to **AI Crop Scanner** (`फसल जांच`).
- Click the **"नमूना पत्तियां (Sample Leaves)"** tab.
- Click on **"🍃 Potato - Early Blight"** (or Tomato Late Blight).
- Click **"🔍 फसल पत्ती की जांच करें (Analyze Crop Leaf)"**.
- **Speaker Cue**: *"We do not use hardcoded mockups. In the background, our real MobileNetV2 neural network processes the 224x224 normalized image and outputs the diagnosis."*

#### Step 3: Diagnostic Assessment & Confidence Tiers
- Show the **Prediction Result**:
  - Badge: `⚠️ DISEASE DETECTED` (रोग का लक्षण मिला).
  - Crop & Disease: `आलू : अगेती झुलसा (Potato : Early Blight)`.
  - Confidence Score: `88.5%` with horizontal probability distribution bar.
- **Speaker Cue**: *"Our platform implements a critical safety rule. If a prediction is uncertain or the leaf is blurry (Confidence < 45%), we do not pretend to be right. The system displays an explicit warning and directly gives the Toll-Free Kisan Call Centre number: 1800-180-1551."*

---

### ⏱️ Minute 1:45 - 2:30 | Multilingual Remedies & Agro-Weather Alerts

#### Step 4: Structured Remedies
- Navigate to **Remedies & Treatment** (`उपचार एवं प्रबंधन`).
- Show the 3 tabs:
  1. **Cultural & Preventive**: Crop rotation, avoiding sprinkler irrigation, sanitizing tools.
  2. **Organic / Biological**: *Trichoderma viride* and Neem Seed Kernel Extract (NSKE 5%).
  3. **Verified Chemical Guidance**: Mancozeb 75% WP @ 2.0 g/L with explicit PPE and waiting-period warnings.
- **Speaker Cue**: *"We don't invent chemical doses. Every recommendation is verified against ICAR guidelines with strict personal protective warnings."*

#### Step 5: Live Weather Microclimate Risk
- Navigate to **Weather Alerts** (`मौसम अलर्ट`).
- Select **Bhopal** or **Lucknow**.
- Show live temperature, humidity, and rainfall fetched from **Open-Meteo API**.
- Highlight the **Rule-Based Disease Susceptibility Alert**:
  - *"Notice the Late Blight warning — because current humidity exceeds 85% and temperature is in the 12°C–20°C range, the system warns farmers that microclimatic conditions favor fungal spore multiplication before visible symptoms appear."*

---

### ⏱️ Minute 2:30 - 3:00 | Farmer Dashboard, Offline Edge & Impact

#### Step 6: Local Privacy & Conclusion
- Navigate to **Farmer Dashboard** (`किसान डैशबोर्ड`).
- Show the Plotly charts of healthy vs. diseased ratios and the local SQLite scan history table.
- **Closing Statement**:
  > *"Most importantly: **Krishi Rakshak AI runs 100% locally on CPU without needing a GPU or active internet for diagnosis**. We respect farmer privacy — photos are processed in-memory and never stored on remote servers.*
  >
  > *Evaluated on held-out test data, our MobileNetV2 model achieves **84% test accuracy** and **86.9% precision** across 10 disease classes. Krishi Rakshak AI gives every Indian farmer a certified crop doctor in their pocket.*
  >
  > *Thank you. We are Team Codex Verse, and we are ready for your questions!"*

---

## 3. Potential Judge Questions & How to Answer

### Q1: "Why did you choose MobileNetV2 instead of heavier models like ResNet50 or Vision Transformers?"
**Answer**:
> *"Indian farmers and field extension workers use budget smartphones and laptops with limited memory and integrated graphics. MobileNetV2 uses depthwise separable convolutions, bringing the parameter count down to just 2.25 million while maintaining 84% held-out test accuracy. It runs inference in under 50 milliseconds on standard CPU without requiring any GPU or cloud cost."*

### Q2: "What happens if a farmer uploads an image of a non-crop object, like a shoe or a mango leaf?"
**Answer**:
> *"We built an explicit Confidence Tier Safety Engine. Because out-of-distribution objects have diffuse softmax outputs with no single class reaching our 75% threshold, the system flags a Low Confidence Warning (<45%) or Tentative Alert, advising the farmer that the leaf is unrecognized and providing direct contact details for their nearest Krishi Vigyan Kendra (KVK)."*

### Q3: "Can the application work without an internet connection in remote villages?"
**Answer**:
> *"Yes! Once the application is installed and weights are saved locally, the entire image preprocessing, MobileNetV2 inference, disease lookup, and SQLite database logging run 100% offline. Only the live Open-Meteo weather API requires an active network, and even if offline, the weather module handles timeouts gracefully without crashing."*

### Q4: "How do you guarantee that chemical pesticide recommendations are safe?"
**Answer**:
> *"All chemical recommendations in our structured `disease_info.json` are verified against ICAR and state agricultural university manuals (e.g. Mancozeb 75% WP, Copper Oxychloride). Furthermore, we display mandatory warnings: never exceed prescribed doses, always wear PPE (masks and gloves), observe pre-harvest intervals, and verify with a local KVK agronomist."*
