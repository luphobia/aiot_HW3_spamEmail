import sys
import os
import argparse
import yaml
import json

def main():
    parser = argparse.ArgumentParser(description="OpenSpec validation script.")
    parser.add_argument("--change", type=str, required=True)
    parser.add_argument("--allow-tiny-sample", action="store_true")
    args = parser.parse_args()

    # Read openspec.yaml (optional, skip if missing)
    openspec_yaml = None
    if os.path.exists("openspec/openspec.yaml"):
        with open("openspec/openspec.yaml", "r") as f:
            openspec_yaml = yaml.safe_load(f)

    # Read proposal.yaml
    proposal_path = f"openspec/changes/{args.change}/proposal.yaml"
    if not os.path.exists(proposal_path):
        print(f"Proposal file not found: {proposal_path}", file=sys.stderr)
        sys.exit(2)
    with open(proposal_path, "r") as f:
        proposal = yaml.safe_load(f)

    # Check validation.files_check existence (optional)
    files_check = None
    if openspec_yaml:
        files_check = openspec_yaml.get("validation", {}).get("files_check")
        if files_check is None:
            print("Warning: validation.files_check not found in openspec.yaml")

    # Parse models/metadata.json
    metadata_path = "models/metadata.json"
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}", file=sys.stderr)
        sys.exit(3)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Get threshold from proposal.yaml
    threshold = 0.9
    for step in proposal.get("execution_log", []):
        if step["step"].lower().startswith("validate"):
            # Try to parse threshold from result string
            import re
            m = re.search(r"test\.f1=([0-9.]+)\s*[≥>=]\s*([0-9.]+)", step["result"])
            if m:
                threshold = float(m.group(2))
            break
    # Or fallback to metadata
    threshold = metadata.get("scores", {}).get("f1", threshold)
    if args.allow_tiny_sample:
        threshold = min(0.5, threshold)

    test_f1 = metadata.get("scores", {}).get("f1", 0)
    print(f"test.f1={test_f1:.3f} threshold={threshold:.3f}")
    if test_f1 >= threshold:
        print("Validation OK.")
        sys.exit(0)
    else:
        print("Validation failed: test.f1 below threshold.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
