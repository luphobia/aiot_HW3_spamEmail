# Spam Email Classifier

## DEMOSITE
- https://2025aiot-hw3-spamemail.streamlit.app/

## Project Overview

This project implements a complete machine learning pipeline for classifying SMS messages as spam or ham (not spam). It is designed for educational and practical use, demonstrating best practices in data science, ML engineering, and reproducible research.

### Objectives
- Detect spam messages using supervised machine learning.
- Provide a reproducible pipeline from raw data to deployment.
- Enable interactive exploration and live inference via a Streamlit app.
- Support OpenSpec workflow for transparent change management and validation.

### Features
- **Data Preprocessing:** Cleans and normalizes SMS text, removes noise, and prepares features for modeling.
- **Model Training:** Trains a classifier (Logistic Regression, SVM, or SGD) with threshold tuning and calibration.
- **Evaluation & Visualization:** Generates metrics (F1, Precision, Recall), confusion matrix, ROC/PR curves, and threshold sweep plots.
- **Token Analysis:** Visualizes top tokens and token frequency by class (ham/spam).
- **Streamlit App:** Interactive UI with tabs for overview, visualization, threshold tuning, live inference, and batch upload. Includes bilingual support (English/Traditional Chinese).
- **Batch & Single Prediction:** Scripts for batch CSV prediction and single text prediction.
- **Validation & CI:** Automated validation and continuous integration using OpenSpec and GitHub Actions.

### Workflow
1. **Download Dataset:** Fetch SMS spam data from a public source.
2. **Preprocess Data:** Clean and tokenize messages, save processed CSV.
3. **Train Model:** Fit classifier, calibrate, and save artifacts.
4. **Visualize Results:** Generate plots for class distribution, metrics, and token analysis.
5. **Run App:** Launch Streamlit app for interactive demo and live inference.
6. **Validate Pipeline:** Run automated checks to ensure metrics and outputs meet acceptance criteria.
7. **Deploy & Archive:** Push changes to GitHub, archive proposal and tasks in OpenSpec.

### Technology Stack
- Python 3.x
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- streamlit
- PyYAML, pickle
- GitHub Actions (CI)
- OpenSpec workflow

---

## Quick Start

```bash
# Download dataset
mkdir -p datasets/raw
curl -L -o datasets/raw/sms_spam_no_header.csv "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"

# Preprocess
python scripts/preprocess_emails.py --input datasets/raw/sms_spam_no_header.csv --output datasets/processed/sms_spam_clean.csv --no-header --label-col-index 0 --text-col-index 1

# Train
python scripts/train_spam_classifier.py --seed 42 --eval-threshold 0.5

# Visualize
python scripts/visualize_spam.py --class-dist --confusion-matrix --roc --pr --threshold-sweep

# Run Streamlit app
streamlit run app/streamlit_app.py

# Predict single text
python scripts/predict_spam.py --text "Win a free iPhone now!"

# Validate
python scripts/openspec_validate.py --change add-spam-email-classifier
```

## Dataset Source

- Download with:
  ```bash
  mkdir -p datasets/raw
  curl -L -o datasets/raw/sms_spam_no_header.csv "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
  ```
- Source: [PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity](https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity)

## Project Structure

```
configs/
  default.yaml
scripts/
  preprocess_emails.py
  train_spam_classifier.py
  predict_spam.py
  visualize_spam.py
  openspec_validate.py
app/
  streamlit_app.py
models/
  vectorizer.pkl
  model.pkl
  label_mapping.json
  metadata.json
  pr_table.csv
reports/
  visualizations/
openspec/
  changes/
  specs/
  project.md
  AGENTS.md
```

## CRISP-DM Pipeline
- Business Understanding: Spam detection for email security
- Data Understanding: SMS spam dataset
- Data Preparation: Preprocessing script
- Modeling: ML classifier (LogisticRegression, SVM, SGD)
- Evaluation: Metrics, threshold tuning, visualizations
- Deployment: Streamlit app, batch/single prediction

## Commands

```bash
# Preprocess
python scripts/preprocess_emails.py --input datasets/raw/sms_spam_no_header.csv --output datasets/processed/sms_spam_clean.csv --no-header --label-col-index 0 --text-col-index 1

# Train
python scripts/train_spam_classifier.py --seed 42 --eval-threshold 0.5

# Visualize
python scripts/visualize_spam.py --class-dist --confusion-matrix --roc --pr --threshold-sweep

# Run Streamlit app
streamlit run app/streamlit_app.py

# Predict single text
python scripts/predict_spam.py --text "Win a free iPhone now!"

# Validate
python scripts/openspec_validate.py --change add-spam-email-classifier
```

## OpenSpec Flow
- Propose change in `openspec/changes/`
- Implement tasks in `tasks.md`
- Validate with `openspec_validate.py`
- Archive after deployment

## How to Use

1. **Download the dataset:**
   ```sh
   curl -L -o datasets/raw/sms_spam_no_header.csv "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
   ```
2. **Preprocess the data:**
   ```sh
   python scripts/preprocess_emails.py --input datasets/raw/sms_spam_no_header.csv --output datasets/processed/sms_spam_clean.csv --no-header --label-col-index 0 --text-col-index 1
   ```
3. **Train the spam classifier:**
   ```sh
   python scripts/train_spam_classifier.py --seed 42 --eval-threshold 0.5
   ```
4. **Visualize results:**
   ```sh
   python scripts/visualize_spam.py --class-dist --confusion-matrix --roc --pr --threshold-sweep
   python scripts/visualize_top_tokens_by_class.py
   ```
5. **Run the Streamlit app:**
   ```sh
   streamlit run app/streamlit_app.py
   ```
6. **Validate the pipeline:**
   ```sh
   python scripts/openspec_validate.py --change add-spam-email-classifier
   ```

- The Streamlit app provides interactive tabs for overview, visualization, threshold tuning, live inference, and batch upload.
- Token frequency plots and top tokens tables are shown in the Overview tab.
- Use the Live Inference tab to test predictions with sample or custom messages.

## Troubleshooting
- Check dataset path and format
- Ensure all model artifacts exist in `models/`
- Use `--allow-tiny-sample` for unit tests
- Review logs and metrics in `models/metadata.json`
- For CI failures, check `.github/workflows/ci.yml` and dependencies
