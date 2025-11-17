"""
Preprocess SMS spam dataset for spam classification.

Usage:
  # Download dataset
  curl -L -o datasets/raw/sms_spam_no_header.csv \
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"

  # Run preprocessing with defaults
  python scripts/preprocess_emails.py

Arguments:
  --input                Path to input CSV (default: configs/default.yaml → data.raw_path)
  --output               Path to output CSV (default: configs/default.yaml → data.processed_path)
  --no-header            Treat input as headerless CSV (default: configs/default.yaml → data.no_header)
  --label-col-index      Index of label column (default: configs/default.yaml → data.label_col_index)
  --text-col-index       Index of text column (default: configs/default.yaml → data.text_col_index)
  --output-text-col      Name for cleaned text column (default: configs/default.yaml → data.text_clean_col)
  --save-step-columns    Save intermediate CSVs for each step
  --steps-out-dir        Directory for intermediate step outputs
  --lemmatize            Optional: apply lemmatization

Outputs:
  - Cleaned CSV with columns: [col_0, text_clean]
  - Summary stats printed and saved to datasets/processed/report_preprocess.json
  - Optional intermediate CSVs if --save-step-columns is set
"""

import argparse
import os
import yaml
import pandas as pd
import re
import json
from string import punctuation

def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def clean_text(text, lemmatize=False):
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = text.translate(str.maketrans("", "", punctuation))
    text = re.sub(r"\\s+", " ", text).strip()
    # Optionally add lemmatization here
    if lemmatize:
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            nltk.download("wordnet", quiet=True)
            lemmatizer = WordNetLemmatizer()
            text = " ".join([lemmatizer.lemmatize(w) for w in text.split()])
        except ImportError:
            pass
    return text

def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Preprocess SMS spam dataset.")
    parser.add_argument("--input", type=str, default=config["data"]["raw_path"])
    parser.add_argument("--output", type=str, default=config["data"]["processed_path"])
    parser.add_argument("--no-header", action="store_true", default=config["data"]["no_header"])
    parser.add_argument("--label-col-index", type=int, default=config["data"]["label_col_index"])
    parser.add_argument("--text-col-index", type=int, default=config["data"]["text_col_index"])
    parser.add_argument("--output-text-col", type=str, default=config["data"]["text_clean_col"])
    parser.add_argument("--save-step-columns", action="store_true")
    parser.add_argument("--steps-out-dir", type=str, default="datasets/processed/steps")
    parser.add_argument("--lemmatize", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input, header=None if args.no_header else 0)
    label_col = args.label_col_index
    text_col = args.text_col_index

    # Step 1: Lowercase
    df[args.output_text_col] = df[text_col].astype(str).str.lower()
    if args.save_step_columns:
        os.makedirs(args.steps_out_dir, exist_ok=True)
        df[[label_col, args.output_text_col]].to_csv(
            os.path.join(args.steps_out_dir, "step1_lowercase.csv"), index=False
        )

    # Step 2: Strip URLs
    df[args.output_text_col] = df[args.output_text_col].apply(lambda x: re.sub(r"http\\S+|www\\S+", "", x))
    if args.save_step_columns:
        df[[label_col, args.output_text_col]].to_csv(
            os.path.join(args.steps_out_dir, "step2_strip_urls.csv"), index=False
        )

    # Step 3: Remove punctuation
    df[args.output_text_col] = df[args.output_text_col].apply(lambda x: x.translate(str.maketrans("", "", punctuation)))
    if args.save_step_columns:
        df[[label_col, args.output_text_col]].to_csv(
            os.path.join(args.steps_out_dir, "step3_remove_punct.csv"), index=False
        )

    # Step 4: Collapse spaces
    df[args.output_text_col] = df[args.output_text_col].apply(lambda x: re.sub(r"\\s+", " ", x).strip())
    if args.save_step_columns:
        df[[label_col, args.output_text_col]].to_csv(
            os.path.join(args.steps_out_dir, "step4_collapse_spaces.csv"), index=False
        )

    # Optional: Lemmatization
    if args.lemmatize:
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            nltk.download("wordnet", quiet=True)
            lemmatizer = WordNetLemmatizer()
            df[args.output_text_col] = df[args.output_text_col].apply(
                lambda x: " ".join([lemmatizer.lemmatize(w) for w in x.split()])
            )
            if args.save_step_columns:
                df[[label_col, args.output_text_col]].to_csv(
                    os.path.join(args.steps_out_dir, "step5_lemmatize.csv"), index=False
                )
        except ImportError:
            print("nltk not installed, skipping lemmatization.")

    # Save final output
    df[[label_col, args.output_text_col]].to_csv(args.output, index=False)

    # Summary stats
    stats = {
        "input_rows": len(df),
        "output_rows": len(df),
        "label_counts": df[label_col].value_counts().to_dict(),
        "text_clean_nulls": int(df[args.output_text_col].isnull().sum()),
        "output_path": args.output,
    }
    print(json.dumps(stats, indent=2))
    with open("datasets/processed/report_preprocess.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()
