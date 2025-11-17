import os
import pandas as pd
import subprocess
import tempfile
import shutil

def test_smoke_pipeline():
    tmpdir = tempfile.mkdtemp()
    try:
        # Create tiny balanced CSV
        csv_path = os.path.join(tmpdir, "tiny.csv")
        df = pd.DataFrame({
            "col_0": ["spam"]*10 + ["ham"]*10,
            "text_clean": ["Win money now!"]*10 + ["Hello friend!"]*10
        })
        df.to_csv(csv_path, index=False)

        # Preprocess (simulate: just copy)
        processed_path = os.path.join(tmpdir, "tiny_clean.csv")
        shutil.copy(csv_path, processed_path)

        # Train
        subprocess.run([
            "python", "scripts/train_spam_classifier.py",
            "--input", processed_path,
            "--output-dir", os.path.join(tmpdir, "models"),
            "--seed", "1", "--test-size", "0.5", "--eval-threshold", "0.5"
        ], check=True)

        # Predict
        subprocess.run([
            "python", "scripts/predict_spam.py",
            "--input", processed_path,
            "--text-col", "text_clean",
            "--output", os.path.join(tmpdir, "predictions.csv")
        ], check=True)

        # Validate
        subprocess.run([
            "python", "scripts/openspec_validate.py",
            "--change", "add-spam-email-classifier",
            "--allow-tiny-sample"
        ], check=True)

        # Check metadata
        meta_path = os.path.join(tmpdir, "models", "metadata.json")
        assert os.path.exists(meta_path)
        import json
        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert meta["scores"]["f1"] >= 0.5
    finally:
        shutil.rmtree(tmpdir)
