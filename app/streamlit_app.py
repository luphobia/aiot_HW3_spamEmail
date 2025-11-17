import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import json
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# --- Config and Defaults ---
CONFIG_PATH = "configs/default.yaml"
DATASET_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"

def load_config(path=CONFIG_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_artifact(path, mode="rb"):
    if not os.path.exists(path):
        return None
    with open(path, mode) as f:
        if path.endswith(".pkl"):
            return pickle.load(f)
        elif path.endswith(".json"):
            return json.load(f)
        elif path.endswith(".csv"):
            return pd.read_csv(f)
        else:
            return f.read()

def bilingual(label_en, label_zh):
    # Use Traditional Chinese for label_zh
    return f"{label_en} / {label_zh}"

# --- Page Config ---
st.set_page_config(page_title="Spam/Ham Classifier")
st.title("Spam/Ham Classifier")

# --- Sidebar ---
config = load_config()
default_data_path = config["data"]["processed_path"]
default_label_col = "0"
default_text_col = config["data"]["text_clean_col"]

st.sidebar.header(bilingual("Dataset", "資料集"))
dataset_path = st.sidebar.text_input(
    bilingual("CSV Path", "CSV路徑"), value=default_data_path
)
label_col = st.sidebar.text_input(
    bilingual("Label Column", "標籤欄"), value=default_label_col
)
text_col = st.sidebar.text_input(
    bilingual("Text Column", "文字欄"), value=default_text_col
)

st.sidebar.markdown(
    f"""
    <div style="background-color:#e3f2fd;padding:8px;border-radius:6px;margin-bottom:10px;">
        <b>Dataset Source:</b> <a href="{DATASET_URL}">{DATASET_URL}</a><br>
        <span style="font-size:0.9em;">Downloaded to <code>datasets/raw/sms_spam_no_header.csv</code></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Load Data ---
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
else:
    st.error(bilingual("Dataset not found.", "未找到資料集"))
    df = pd.DataFrame()

# --- Load Model Artifacts ---
vectorizer = load_artifact("models/vectorizer.pkl")
model = load_artifact("models/model.pkl")
label_mapping = load_artifact("models/label_mapping.json", "r")
metadata = load_artifact("models/metadata.json", "r")
pr_table = load_artifact("models/pr_table.csv", "r")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    bilingual("Overview", "總覽"),
    bilingual("Visualize", "視覺化"),
    bilingual("Threshold & Metrics", "門檻與指標"),
    bilingual("Live Inference", "即時預測"),
    bilingual("Batch Upload", "批次上傳"),
])

# --- Tab 1: Overview ---
with tab1:
    st.subheader(bilingual("Class Distribution", "類別分布"))
    if not df.empty:
        st.bar_chart(df[label_col].value_counts())
    else:
        st.info(bilingual("No data to show.", "沒有可顯示的資料"))

    st.subheader(bilingual("Top Tokens by Class", "各類別高頻詞彙"))
    token_fig_path = "reports/visualizations/top_tokens_by_class.png"
    if os.path.exists(token_fig_path):
        st.image(token_fig_path, caption=bilingual("Top 20 tokens for ham and spam", "正常與垃圾郵件的前20高頻詞"), use_container_width=True)
    else:
        st.info(bilingual("Token frequency plot not found. Please run the visualization script.", "未找到詞頻圖，請先執行視覺化腳本。"))

    st.subheader(bilingual("Top Tokens", "高頻詞彙"))
    if vectorizer and not df.empty:
        X = df[text_col].astype(str)
        X_vec = vectorizer.transform(X)
        feature_names = np.array(vectorizer.get_feature_names_out())
        token_counts = np.asarray(X_vec.sum(axis=0)).flatten()
        topn = 20
        top_idx = np.argsort(token_counts)[::-1][:topn]
        top_tokens = pd.DataFrame({
            bilingual("Token", "詞"): feature_names[top_idx],
            bilingual("Count", "計數"): token_counts[top_idx]
        })
        st.table(top_tokens)
    else:
        st.info(bilingual("Vectorizer or data missing.", "缺少向量器或資料"))

# --- Tab 2: Visualize ---
with tab2:
    st.subheader(bilingual("Confusion Matrix", "混淆矩陣"))
    if model and vectorizer and not df.empty:
        y_true = df[label_col]
        y_pred = model.predict(vectorizer.transform(df[text_col].astype(str)))
        # Map y_pred (int) back to string labels
        inv_label_mapping = {v: k for k, v in label_mapping.items()} if label_mapping else {0: "ham", 1: "spam"}
        y_pred_str = [inv_label_mapping.get(p, p) for p in y_pred]
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred_str, labels=sorted(label_mapping.keys()))
        st.write(cm)
    else:
        st.info(bilingual("Model, vectorizer, or data missing.", "缺少模型、向量器或資料"))

    st.subheader(bilingual("ROC Curve", "ROC曲線"))
    if model and vectorizer and hasattr(model, "predict_proba") and not df.empty:
        y_true = df[label_col]
        y_proba = model.predict_proba(vectorizer.transform(df[text_col].astype(str)))[:, 1]
        # Convert y_true to binary using label_mapping
        y_bin = pd.Series(y_true).map(label_mapping)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_bin, y_proba, pos_label=label_mapping.get('spam', 1))
        st.line_chart(pd.DataFrame({"FPR": fpr, "TPR": tpr}))
    else:
        st.info(bilingual("Model does not support probability or data missing.", "模型不支援機率或缺少資料"))

    st.subheader(bilingual("Precision-Recall Curve", "精確率-召回率曲線"))
    if model and vectorizer and hasattr(model, "predict_proba") and not df.empty:
        y_true = df[label_col]
        y_proba = model.predict_proba(vectorizer.transform(df[text_col].astype(str)))[:, 1]
        y_bin = pd.Series(y_true).map(label_mapping)
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_bin, y_proba, pos_label=label_mapping.get('spam', 1))
        st.line_chart(pd.DataFrame({"Precision": precision, "Recall": recall}))
    else:
        st.info(bilingual("Model does not support probability or data missing.", "模型不支援機率或缺少資料"))

# --- Tab 3: Threshold & Metrics ---
with tab3:
    st.subheader(bilingual("Threshold & Metrics", "門檻與指標"))
    if pr_table is not None:
        threshold = st.slider(
            bilingual("Eval Threshold", "評估門檻"),
            min_value=float(pr_table["threshold"].min()),
            max_value=float(pr_table["threshold"].max()),
            value=float(metadata.get("eval_threshold", 0.5)) if metadata else 0.5,
            step=0.01
        )
        row = pr_table.loc[(pr_table["threshold"] - threshold).abs().idxmin()]
        st.metric(bilingual("Precision", "精確率"), f"{row['precision']:.3f}")
        st.metric(bilingual("Recall", "召回率"), f"{row['recall']:.3f}")
        st.metric(bilingual("F1", "F1分數"), f"{row['f1']:.3f}" if "f1" in row else "N/A")
        st.progress(row['precision'])
        st.progress(row['recall'])
        st.progress(row['f1'] if "f1" in row else 0)
        st.write(bilingual("Probability Bar (with threshold marker)", "機率條（含門檻標記）"))
        st.write(f"Threshold: {threshold:.2f}")
    else:
        st.info(bilingual("PR table missing.", "缺少PR表"))

# --- Tab 4: Live Inference ---
with tab4:
    st.markdown("## Live Inference")
    st.markdown("""
    <style>
    .result-box {
        padding: 1em;
        border-radius: 8px;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 1em;
    }
    .bar-label {
        font-size: 1.5em;
        font-weight: bold;
        color: #222;
    }
    </style>
    """, unsafe_allow_html=True)

    spam_example = "Free entry in 2 a wkly comp to win cash now! Call +44 906-170-1461 to claim prize"
    ham_example = "Hey, are we still meeting later today?"
    threshold = 0.5
    tmp_dir = "reports/visualizations/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_plot_path = os.path.join(tmp_dir, "live_inference_bar.png")

    col1, col2 = st.columns(2)
    example_clicked = False
    with col1:
        if st.button("Use spam example"):
            st.session_state["live_inference_text"] = spam_example
            example_clicked = True
    with col2:
        if st.button("Use ham example"):
            st.session_state["live_inference_text"] = ham_example
            example_clicked = True

    text = st.text_area(
        "Enter text to classify",
        value=st.session_state.get("live_inference_text", ""),
        height=100
    )
    st.session_state["live_inference_text"] = text

    if example_clicked:
        st.rerun()

    predict_clicked = st.button("Predict")
    result = None
    norm_text = None
    spam_prob = None
    pred_label = None
    error_msg = None

    if predict_clicked and text.strip():
        try:
            if not vectorizer or not model:
                raise ValueError("Model or vectorizer not loaded.")
            # Preprocess (normalize) text: match training pipeline
            norm_text = text.lower().strip()
            X = vectorizer.transform([norm_text])
            if hasattr(model, "predict_proba"):
                spam_prob = float(model.predict_proba(X)[0][1])
            else:
                spam_prob = float(model.predict(X)[0])
            # Use label mapping if available
            if label_mapping:
                inv_label_mapping = {v: k for k, v in label_mapping.items()}
                pred_label_idx = int(spam_prob >= threshold)
                pred_label = inv_label_mapping.get(pred_label_idx, "spam" if pred_label_idx else "ham")
            else:
                pred_label = "spam" if spam_prob >= threshold else "ham"
        except Exception as e:
            error_msg = f"Error: {e}" # Show error in UI

    if error_msg:
        st.error(error_msg)
    elif spam_prob is not None:
        # Result box
        box_color = "#ffcccc" if spam_prob >= threshold else "#ccffcc"
        text_color = "#900" if spam_prob >= threshold else "#060"
        st.markdown(
            f'<div class="result-box" style="background:{box_color};color:{text_color};">'
            f'Prediction: <b>{pred_label}</b> | spam-prob = {spam_prob:.4f} (threshold = {threshold:.2f})'
            f'</div>', unsafe_allow_html=True
        )
        # Probability bar plot
        fig, ax = plt.subplots(figsize=(7, 1.2))
        ax.barh([0], [spam_prob], color="#d32f2f" if spam_prob >= threshold else "#388e3c", height=0.5)
        ax.axvline(threshold, color="black", linestyle="dashed", linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("spam probability", fontsize=14)
        ax.set_facecolor("#f7f7f7")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(spam_prob, 0, f"{spam_prob:.2f}", va="center", ha="center", fontsize=18, color="#222", fontweight="bold")
        plt.tight_layout()
        fig.patch.set_facecolor('#222')
        plt.savefig(tmp_plot_path, bbox_inches="tight", dpi=120)
        st.image(tmp_plot_path)
        # Expander for normalized text
        with st.expander("Show normalized text"):
            st.code(norm_text)

# --- Tab 5: Batch Upload ---
with tab5:
    st.subheader(bilingual("Batch Upload", "批次上傳"))
    uploaded_file = st.file_uploader(
        bilingual("Upload CSV for batch prediction", "上傳CSV進行批次預測"),
        type=["csv"]
    )
    batch_text_col = st.text_input(
        bilingual("Text Column for Prediction", "用於預測的文字欄"),
        value=default_text_col
    )
    if uploaded_file and model and vectorizer:
        df_up = pd.read_csv(uploaded_file)
        texts = df_up[batch_text_col].astype(str).tolist()
        X = vectorizer.transform(texts)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = [None] * len(texts)
        preds = model.predict(X)
        inv_label_mapping = {v: k for k, v in label_mapping.items()} if label_mapping else {0: "ham", 1: "spam"}
        labels = [inv_label_mapping.get(p, p) for p in preds]
        df_out = pd.DataFrame({
            bilingual("Input Text", "輸入文字"): texts,
            bilingual("Predicted Label", "預測標籤"): labels,
            bilingual("Spam Probability", "垃圾郵件機率"): probs
        })
        st.download_button(
            label=bilingual("Download Predictions CSV", "下載預測結果CSV"),
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv"
        )
        st.write(df_out)
    elif uploaded_file:
        st.info(bilingual("Model or vectorizer missing.", "缺少模型或向量器"))
