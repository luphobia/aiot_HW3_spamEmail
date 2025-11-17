"""
Train a spam email classifier using processed data.

Usage Example:
  # Download dataset
  curl -L -o datasets/raw/sms_spam_no_header.csv \
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"

  # Preprocess
  python scripts/preprocess_emails.py

  # Train classifier
  python scripts/train_spam_classifier.py

Arguments:
  --config            Path to config YAML (default: configs/default.yaml)
  --input             Path to processed CSV (default: config → data.processed_path)
  --ngram-range       ngram range for TfidfVectorizer (default: config → vectorizer.ngram_range)
  --min-df            min_df for TfidfVectorizer (default: config → vectorizer.min_df)
  --sublinear-tf      Use sublinear_tf in TfidfVectorizer (default: config → vectorizer.sublinear_tf)
  --model             Classifier type: logreg|linearsvc|sgd (default: config → model.type)
  --C                 Regularization strength (default: config → model.C)
  --class-weight      Class weight (default: config → model.class_weight)
  --seed              Random seed (default: config → split.seed)
  --test-size         Test size fraction (default: config → split.test_size)
  --calibrate         Calibration: none|platt|isotonic (default: config → calibration)
  --eval-threshold    Decision threshold for positive class (default: config → eval_threshold)
  --output-dir        Directory to save artifacts (default: models/)
"""

import argparse
import os
import yaml
import pandas as pd
import numpy as np
import pickle
import json
import hashlib
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Train spam email classifier.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--input", type=str, default=config["data"]["processed_path"])
    parser.add_argument("--ngram-range", type=str, default=config["vectorizer"]["ngram_range"])
    parser.add_argument("--min-df", type=int, default=config["vectorizer"]["min_df"])
    parser.add_argument("--sublinear-tf", action="store_true", default=config["vectorizer"]["sublinear_tf"])
    parser.add_argument("--model", type=str, choices=["logreg", "linearsvc", "sgd"], default=config["model"]["type"])
    parser.add_argument("--C", type=float, default=config["model"]["C"])
    parser.add_argument("--class-weight", type=str, default=config["model"]["class_weight"])
    parser.add_argument("--seed", type=int, default=config["split"]["seed"])
    parser.add_argument("--test-size", type=float, default=config["split"]["test_size"])
    parser.add_argument("--calibrate", type=str, choices=["none", "platt", "isotonic"], default=config.get("calibration", "none"))
    parser.add_argument("--eval-threshold", type=float, default=config.get("eval_threshold", 0.5))
    parser.add_argument("--output-dir", type=str, default="models/")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    label_col = config["data"]["label_col_index"]
    text_col = config["data"]["text_clean_col"]
    y = df.iloc[:, label_col]
    X = df[text_col]

    # Label mapping
    classes = sorted(y.unique())
    label_mapping = {str(c): i for i, c in enumerate(classes)}
    y_enc = y.map(label_mapping)

    # Vectorizer
    ngram_range = tuple(map(int, args.ngram_range.split(",")))
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=args.min_df,
        sublinear_tf=args.sublinear_tf
    )
    X_vec = vectorizer.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_enc, test_size=args.test_size, random_state=args.seed, stratify=y_enc
    )

    # Classifier
    if args.model == "logreg":
        clf = LogisticRegression(C=args.C, class_weight=args.class_weight, max_iter=1000, random_state=args.seed)
    elif args.model == "linearsvc":
        clf = LinearSVC(C=args.C, class_weight=args.class_weight, max_iter=1000, random_state=args.seed)
    elif args.model == "sgd":
        clf = SGDClassifier(class_weight=args.class_weight, random_state=args.seed)
    else:
        raise ValueError("Unsupported model type")

    # Calibration
    if args.calibrate != "none" and args.model in ["logreg", "sgd"]:
        clf = CalibratedClassifierCV(clf, method=args.calibrate, cv=3)
    elif args.calibrate != "none" and args.model == "linearsvc":
        clf = CalibratedClassifierCV(clf, method=args.calibrate, cv=3)

    clf.fit(X_train, y_train)

    # Predict
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= args.eval_threshold).astype(int)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_curve = precision_recall_curve(y_test, y_proba)
    else:
        y_pred = clf.predict(X_test)
        y_proba = None
        roc_auc = None
        pr_curve = None

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    train_scores = clf.score(X_train, y_train)
    test_scores = clf.score(X_test, y_test)

    # Save artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(args.output_dir, "model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(args.output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=2)

    # Metadata
    metadata = {
        "flags": vars(args),
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "class_names": classes,
        "scores": {
            "train": float(train_scores),
            "test": float(test_scores),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if roc_auc is not None else None
        },
        "eval_threshold": args.eval_threshold,
        "input_hash": hash_file(args.input)
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Per-threshold PR table
        if y_proba is not None and pr_curve is not None:
            precision_arr, recall_arr, thresholds_arr = pr_curve
            # thresholds_arr is length n-1, precision/recall are length n
            # For table, drop last precision/recall to match thresholds, or pad thresholds
            pr_table = pd.DataFrame({
                "threshold": np.concatenate([thresholds_arr, [args.eval_threshold]]),
                "precision": np.concatenate([precision_arr[:-1], [precision]]),
                "recall": np.concatenate([recall_arr[:-1], [recall]])
            })
            pr_table.to_csv(os.path.join(args.output_dir, "pr_table.csv"), index=False)

    # Print concise metrics summary
    print(f"Test Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.3f}")

if __name__ == "__main__":
    main()
