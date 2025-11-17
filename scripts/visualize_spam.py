"""
Spam classifier visualization script.

Usage Example:
  # Download dataset
  curl -L -o datasets/raw/sms_spam_no_header.csv \
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"

  # Preprocess
  python scripts/preprocess_emails.py

  # Train
  python scripts/train_spam_classifier.py

  # Visualize
  python scripts/visualize_spam.py --class-dist --token-freq --topn 20 --confusion-matrix --roc --pr --threshold-sweep

Arguments:
  --class-dist         Plot class distribution
  --token-freq         Plot token frequency (requires --topn)
  --topn               Top N tokens for frequency plot
  --confusion-matrix   Plot confusion matrix
  --roc                Plot ROC curve
  --pr                 Plot Precision-Recall curve
  --threshold-sweep    Plot PR at different thresholds
  --input              Input CSV (default: configs/default.yaml → data.processed_path)
"""

import argparse
import os
import yaml
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, precision_recall_fscore_support

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_artifact(path, name):
    if not os.path.exists(path):
        print(f"Error: {name} not found at {path}")
        return None
    with open(path, "rb" if name.endswith(".pkl") else "r") as f:
        return pickle.load(f) if name.endswith(".pkl") else json.load(f)

def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Spam classifier visualization script.")
    parser.add_argument("--class-dist", action="store_true")
    parser.add_argument("--token-freq", action="store_true")
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--confusion-matrix", action="store_true")
    parser.add_argument("--roc", action="store_true")
    parser.add_argument("--pr", action="store_true")
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--input", type=str, default=config["data"]["processed_path"])
    args = parser.parse_args()

    os.makedirs("reports/visualizations/", exist_ok=True)
    df = pd.read_csv(args.input)
    label_col = config["data"]["label_col_index"]
    text_col = config["data"]["text_clean_col"]
    y = df.iloc[:, label_col]
    X = df[text_col]

    # Load model artifacts
    vectorizer = load_artifact("models/vectorizer.pkl", "vectorizer.pkl")
    model = load_artifact("models/model.pkl", "model.pkl")
    label_mapping = load_artifact("models/label_mapping.json", "label_mapping.json")
    metadata = load_artifact("models/metadata.json", "metadata.json")
    if None in [vectorizer, model, label_mapping, metadata]:
        print("Error: Missing required model artifacts.")
        return

    # Class distribution
    if args.class_dist:
        plt.figure()
        sns.countplot(x=y)
        plt.title("Class Distribution")
        out_path = "reports/visualizations/class_dist.png"
        plt.savefig(out_path)
        print(f"Class distribution plot saved to {out_path}")

    # Token frequency
    if args.token_freq:
        X_vec = vectorizer.transform(X)
        feature_names = np.array(vectorizer.get_feature_names_out())
        token_counts = np.asarray(X_vec.sum(axis=0)).flatten()
        top_idx = np.argsort(token_counts)[::-1][:args.topn]
        plt.figure(figsize=(10, 5))
        sns.barplot(x=feature_names[top_idx], y=token_counts[top_idx])
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Top {args.topn} Token Frequencies")
        out_path = "reports/visualizations/token_freq.png"
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Token frequency plot saved to {out_path}")

    # Confusion matrix
    if args.confusion_matrix:
        y_pred = model.predict(vectorizer.transform(X))
        # Map y_pred (int) back to string labels
        inv_label_mapping = {v: k for k, v in label_mapping.items()}
        y_pred_str = [inv_label_mapping.get(p, p) for p in y_pred]
        cm = confusion_matrix(y, y_pred_str, labels=sorted(label_mapping.keys()))
        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(label_mapping.keys()), yticklabels=sorted(label_mapping.keys()))
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        out_path = "reports/visualizations/confusion_matrix.png"
        plt.savefig(out_path)
        print(f"Confusion matrix plot saved to {out_path}")

    # ROC curve
    if args.roc and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(vectorizer.transform(X))[:, 1]
        # Convert y to binary using label_mapping
        y_bin = pd.Series(y).map(label_mapping)
        fpr, tpr, _ = roc_curve(y_bin, y_proba, pos_label=label_mapping.get('spam', 1))
        plt.figure()
        plt.plot(fpr, tpr, label="ROC Curve")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        out_path = "reports/visualizations/roc_curve.png"
        plt.savefig(out_path)
        print(f"ROC curve plot saved to {out_path}")

    # Precision-Recall curve
    if args.pr and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(vectorizer.transform(X))[:, 1]
        y_bin = pd.Series(y).map(label_mapping)
        precision, recall, _ = precision_recall_curve(y_bin, y_proba, pos_label=label_mapping.get('spam', 1))
        plt.figure()
        plt.plot(recall, precision, label="PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        out_path = "reports/visualizations/pr_curve.png"
        plt.savefig(out_path)
        print(f"PR curve plot saved to {out_path}")

    # Threshold sweep
    if args.threshold_sweep and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(vectorizer.transform(X))[:, 1]
        y_bin = pd.Series(y).map(label_mapping)
        thresholds = np.linspace(0, 1, 101)
        pr_table = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(y_bin, y_pred, average="binary")
            pr_table.append((t, p, r, f1))
        pr_df = pd.DataFrame(pr_table, columns=["threshold", "precision", "recall", "f1"])
        plt.figure()
        plt.plot(pr_df["threshold"], pr_df["precision"], label="Precision")
        plt.plot(pr_df["threshold"], pr_df["recall"], label="Recall")
        plt.plot(pr_df["threshold"], pr_df["f1"], label="F1")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold Sweep")
        plt.legend()
        out_path = "reports/visualizations/threshold_sweep.png"
        plt.savefig(out_path)
        pr_df.to_csv("reports/visualizations/threshold_sweep.csv", index=False)
        print(f"Threshold sweep plot saved to {out_path}")
        print(f"Threshold sweep table saved to reports/visualizations/threshold_sweep.csv")

if __name__ == "__main__":
    main()
