# A-Multimodal-Deep-Learning-Framework-for-Socioeconomic-Verification
This project aims to develop a dual-phase supervised machine learning framework to accurately classify households' Below Poverty Line (BPL) status and detect potential welfare misrepresentation. To achieve this, the system architecture is designed to evaluate applicants across two sequential phases.
# 🕵️‍♂️ BPL Misrepresentation Detection: A Multimodal Machine Learning Framework

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green.svg)
![EfficientNet](https://img.shields.io/badge/EfficientNet-B7-red.svg)
![Gemini API](https://img.shields.io/badge/Gemini-API-blueviolet.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)

> **Tags/Topics:** `#MachineLearning` `#ComputerVision` `#FraudDetection` `#MultimodalAI` `#E-Governance` `#PredictiveModeling` `#PublicPolicy`

## 📖 Project Objective & Methodology Overview

This project develops a dual-phase supervised machine learning framework to accurately classify households' Below Poverty Line (BPL) status and detect potential welfare misrepresentation. Here one will not find the data in the excel files as they are not to be shared!! 

* **Baseline Classification (Tabular-Only):** Trains an initial predictive model relying exclusively on self-reported demographic and financial measures.
* **Multimodal Classification (Tabular + Visual):** Integrates the baseline tabular data with physical wealth features extracted via an image processing pipeline.
* **The Comparative Approach:** Demonstrates the critical importance of visual verification in socio-economic auditing to mitigate the concealment of economic assets.
* **Triage & Isolation:** The system isolates discrepancies where households are classified as 'BPL' on paper but 'Non-BPL' visually, marking them for physical verification to solve administrative bottlenecks.
* **Methodology Origin:** Draws from an IIT Kanpur thesis, utilizing Logistic Regression as an initial baseline before expanding into advanced classification algorithms.

---

## 📑 Table of Contents
1. [Chapter 1: 1st Classification (Baseline)](#-chapter-1-1st-classification-baseline)
2. [Chapter 2: 1st Image Recognition (Pilot)](#-chapter-2-1st-image-recognition-pilot)
3. [Chapter 3: 2nd Classification (Statewide Scale)](#-chapter-3-2nd-classification-statewide-scale)
4. [Chapter 4: 2nd Image Recognition (Data Curation)](#-chapter-4-2nd-image-recognition-data-curation)
5. [Chapter 5: 3rd Image Recognition (Master Training)](#-chapter-5-3rd-image-recognition-master-training)
6. [Chapter 6: 3rd Classification (Multimodal Integration)](#-chapter-6-3rd-classification-multimodal-integration)
7. [Chapter 7: Production Script & Real-World ROI](#-chapter-7-production-script--real-world-roi)

---

## 📊 Chapter 1: 1st Classification (Baseline)

### 1.1 Tabular Data Preprocessing (`01_Families`)
* **Pilot Dataset:** Utilized a representative dataset of 100 families to establish the preprocessing pipeline.
* **Raw Features:** Included demographic indicators (Family ID, Age, Gender, District) and socio-economic metrics (Engagement, Verified Range, Income Tax Payee, Vehicle ownership).
* **Pipeline Operations:** Standardizes data, one-hot encodes engagement and income categories, and groups member-level records into aggregated household-level features using a sum function.
* **Noise Reduction:** Explicit income columns and non-predictive metadata are dropped to remove noise and prevent target leakage.

### 1.2 Ground-Truth Labeling & Feature Analysis (`02_Labelling`)
* **Exclusion Heuristics:** Households are flagged for exclusion (Non-BPL) based on financial thresholds, four-wheeler ownership, urban property ownership, or government employment.
* **Analysis:** A statistical correlation matrix (`correlation_matrix.png`) is generated to analyze relational dependencies.

### 1.3 & 1.4 Model Benchmarking & Final Training
* **Benchmarking:** Four algorithms were benchmarked using 5-Fold Stratified Cross-Validation on a final dataset of 12 predictive features.
* **Champion Model:** **Logistic Regression (Penalized)** emerged as the top-performing algorithm, achieving a **90% accuracy rate** on the unseen holdout set with minimal false positives and false negatives.
* **Interpretability:** Feature Importances and SHAP (SHapley Additive exPlanations) analysis were deployed to mathematically diagnose the model's predictions and ensure administrative transparency.

---

## 📸 Chapter 2: 1st Image Recognition (Pilot)

### 2.1 - 2.3 Visual Feature Identification & Extraction
* **Objective:** Visually audit BPL households in Sector 10 to quantify welfare discrepancies.
* **Zero-Shot Attempt:** An initial zero-shot classification attempt using the pre-trained CLIP model struggled to accurately classify nuanced architectural materials.
* **LLM Pivot:** The pipeline successfully pivoted to a multimodal Large Language Model (Gemini API) acting as a socio-economic surveyor.
* **Data Extraction:** The Gemini API accurately extracted JSON data regarding the number of stories, roof type, wall type, structural condition, and a normalized structural score.

---

## 📈 Chapter 3: 2nd Classification (Statewide Scale)

### 3.1 & 3.2 Statewide Preprocessing & Ground-Truth Labeling
* **Scope Expansion:** Expanded to a granular, member-level dataset sampling 100 representative families from every district across Haryana.
* **Systematic Transformation:** The preprocessing engine transformed statewide member-level records into structured, aggregated household data while enforcing strict domain-specific business rules.
* **Strict Heuristics:** Nine strict socioeconomic exclusion heuristics (e.g., high-wage labour, massive electricity bills, large rural property) were applied to generate absolute ground-truth labels.

### 3.3 - 3.5 Statewide Model Optimization & Deployment
* **Feature Pruning:** The dataset was pruned to 29 core predictive features, removing administrative metadata and explicit income columns.
* **Champion Model:** Advanced non-linear ensemble models were benchmarked; **XGBoost** emerged as the definitive leading algorithm with an **88.07% holdout accuracy** and near-perfect recall.
* **Deployment:** The finalized XGBoost model (`2nd_pipeline.joblib`) was saved, and localized SHAP analysis was deployed to visualize decision boundaries.

---

## 🖼️ Chapter 4: 2nd Image Recognition (Data Curation)

### 4.1 - 4.10 High-Fidelity Visual Data Curation
* **Scaling Up:** The visual dataset was heavily scaled utilizing administrative PDFs, web scraping, and targeted Google Street View captures.
* **Semantic Filtering:** A heavily iterative zero-shot semantic filtering pipeline using the CLIP model (ranging from binary to penta-class filtering) was engineered to automatically discard document scans, wide street views, human portraits, and micro-textures.
* **Consolidation:** Curated datasets of house exteriors and rural structures from Roboflow were flattened and consolidated.
* **Synthetic Integration:** Proprietary interior imagery and Gemini-generated synthetic interior collages/standalone images were extracted and integrated to provide crucial visibility into internal material wealth.

---

## 🧠 Chapter 5: 3rd Image Recognition (Master Training)

### 5.1 & 5.2 Master Training Repository & Generative Extraction
* **Consolidation:** Nine distinct image directories were consolidated into a unified `IMAGES` repository, explicitly excluding low-fidelity/grayscale images to preserve colorimetric data and edge detection capabilities.
* **Annotation:** The Gemini 2.5 Flash vision model extracted detailed structural JSON labels using a state-managed API script to handle rate limits.
* **Verification:** Every single generated row was subjected to **100% manual human-in-the-loop verification** to guarantee absolute data integrity.

### 5.3 - 5.6 Dataset Analytics & Standardization
* **1-to-1 Parity:** A strict synchronization protocol ensured perfect parity between 3,676 pristine records and 3,676 architectural training images.
* **Class Analytics:** Analytics revealed a severe socioeconomic class imbalance, heavily skewed toward Pucca/Premium structures.
* **Standardization:** Images were mapped into ordinal encodings, binary wealth vectors, and standardized via aspect-ratio preserving letterboxing to uniform **600×600 RGB padded tensors**.

### 5.7 High-Resolution Neural Network Training
* **Architecture:** An **EfficientNet-B7** architecture was trained on a Google Colab T4 GPU to handle the large spatial resolutions.
* **Performance:** The model showed exceptional accuracy in distinguishing exterior/interior textures. The continuous `Overall_Structural_Score` yielded a highly stable Mean Absolute Error (MAE) of **0.0935**.
* **Grad-CAM Analysis:** Revealed the model struggled with spatial counting (e.g., tallying distinct AC units or stories), necessitating a future shift to dedicated Object Detection (YOLO).

---

## 🔗 Chapter 6: 3rd Classification (Multimodal Integration)

### 6.1 - 6.3 Relational Binding via Property ID
* **The Bridge:** A synthesized `Property_ID` was engineered to act as the definitive relational bridge, mapping a family's tabular administrative records directly to their physical structural audit.
* **Pipeline Logic:** A strict 37-step logic pipeline converted member-level demographic/financial data into an aggregated family-level schema.
* **Anchoring:** The dataset was anchored into an immutable source of truth, aligning administrative features side-by-side with ground-truth target labels.

### 6.4 - 6.7 Multimodal Heuristic Matching & AI Integration
* **Digital Twin Generation:** A heuristic matching engine generated a "digital twin" of a family based on their financial data and probabilistically mapped them to the most realistic physical image.
* **Forward Pass:** The EfficientNet-B7 model performed a forward pass on the production schema to append purely AI-generated structural audits to the master records.
* **Data Integrity:** To prevent data contamination, final integration exclusively merged master tabular data with human-verified ground-truth visual features.

### 6.8 - 6.11 The Dual-Filter Exclusion Engine & Final Modeling
* **Dual-Filter Logic:** Validated paper claims against AI physical filters, instantly overriding impoverished claims if premium physical assets (e.g., tile floors, AC units) were detected.
* **Champion Model:** Linear models dominated this space; **Logistic Regression (Penalized)** and **Linear SVM** proved incredibly robust to class imbalance. Logistic Regression (Penalized) was locked in as the champion model (`3rd_pipeline.joblib`).
* **Zero Fraud Policy:** The final model achieved **0 False Negatives** on the test data, enforcing a strict "Zero Fraud" policy.

---

## 🚀 Chapter 7: Production Script & Real-World ROI

### 7.1 - 7.3 The Dual-Inference Workflow
* **Comparative Engine:** The production script (`Production_Script`) operates as a comparative engine running two parallel inference pipelines:
    * **Pipeline A:** Evaluates administrative tabular data.
    * **Pipeline B:** Utilizes the multimodal anchor model.
* **Live Data Sync:** Live data is forced through the exact 37-step preprocessing logic to guarantee feature parity and prevent dimensionality crashes in production.

### 7.4 & 7.5 The Triage Matrix & Ground Truth ROI
* **Automated Triage:** The system cross-references predictions and automatically routes discrepancies into a Triage Matrix for physical verification.

> 🚨 **Crucial Finding:** Out of 2,301 processed families, **1,104 families (48%)** claimed BPL status on paper but were flagged by the visual AI for exhibiting structural wealth. Ground truth validation proved that **nearly 65% of the approved paper-only BPL claims were actively fraudulent**, proving the massive financial impact and absolute necessity of this multimodal validation architecture.
