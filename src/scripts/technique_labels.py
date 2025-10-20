import pandas as pd
import re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from project_paths import (
        MAPPING_CSV,EXCEL_ATTACK_TECHS,
)

# Extract techniques from Excel and save to CSV
def extract_techniques(excel_path: Path, output_csv: Path):
    df = pd.read_excel(excel_path, sheet_name="techniques", engine="openpyxl")
    df = df.iloc[:, [0, 2]]  # Column A (ID), Column C (name)
    df.columns = ["id", "name"]
    
    # Filter to only valid technique IDs 
    pattern = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE) # Valid technique ID pattern
    df = df[df["id"].astype(str).str.match(pattern)] # Keep only valid technique IDs
    df = df.drop_duplicates(subset=["id"]).dropna()

    # Create label column
    df["label"] = df["id"] + " (" + df["name"] + ")"
    df.to_csv(output_csv, index=False, encoding="utf-8")

    # Log the extraction
    print(f"Extracted {len(df)} techniques -> {output_csv}")
    return output_csv

if __name__ == "__main__":
    excel_path = EXCEL_ATTACK_TECHS
    output_csv = MAPPING_CSV
    extract_techniques(excel_path, output_csv)