"""
Spam prediction script.

Usage:
  # Single text prediction
  python scripts/predict_spam.py --text "Win a free iPhone now!"

  # Batch prediction
  python scripts/predict_spam.py --input datasets/processed/sms_spam_clean.csv --text-col text_clean --output datasets/processed/predictions.csv

Artifacts required in models/: vectorizer.pkl, model.pkl, label_mapping.json

Arguments:
  --text         Text string to classify (prints label and spam probability)
  --input        Input CSV for batch prediction
  --text-col     Column name containing text in input CSV
  --output       Output CSV for batch prediction
"""

import argparse
import os
import sys
import pickle
import json
import pandas as pd
import numpy as np

def load_artifact(path, name):
    if not os.path.exists(path):
        print(f"Error: {name} not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "rb" if name.endswith(".pkl") else "r") as f:
        return pickle.load(f) if name.endswith(".pkl") else json.load(f)

def predict(texts, vectorizer, model, label_mapping, threshold=0.5):
    X = vectorizer.transform(texts)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
    else:
        preds = model.predict(X)
        probs = None
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    labels = [inv_label_mapping[p] for p in preds]
    return labels, probs

def main():
    parser = argparse.ArgumentParser(description="Spam prediction script.")
    parser.add_argument("--text", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--text-col", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    # Load artifacts
    vectorizer = load_artifact("models/vectorizer.pkl", "vectorizer.pkl")
    model = load_artifact("models/model.pkl", "model.pkl")
    label_mapping = load_artifact("models/label_mapping.json", "label_mapping.json")
    metadata_path = "models/metadata.json"
    threshold = args.threshold
    if threshold is None and os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            threshold = metadata.get("eval_threshold", 0.5)
    if threshold is None:
        threshold = 0.5

    if args.text:
        labels, probs = predict([args.text], vectorizer, model, label_mapping, threshold)
        print(f"Predicted label: {labels[0]}")
        print(f"Spam probability: {probs[0] if probs is not None else 'N/A'}")
    elif args.input and args.text_col and args.output:
        df = pd.read_csv(args.input)
        texts = df[args.text_col].astype(str).tolist()
        labels, probs = predict(texts, vectorizer, model, label_mapping, threshold)
        df_out = pd.DataFrame({
            "input_text": texts,
            "predicted_label": labels,
            "spam_probability": probs if probs is not None else [None]*len(labels)
        })
        df_out.to_csv(args.output, index=False)
        print(f"Batch predictions saved to {args.output}")
    else:
        print("Error: Provide either --text or --input, --text-col, --output for batch mode.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
