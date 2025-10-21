import pandas as pd
import re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from project_paths import (
        MAPPING_CSV,EXCEL_ATTACK_TECHS,
    
)

def extract_techniques(excel_path: Path, output_csv: Path):
    df = pd.read_excel(excel_path, sheet_name="techniques", engine="openpyxl")
    df = df.iloc[:, [0, 2]]  
    df.columns = ["id", "name"]

    pattern = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)
    df = df[df["id"].astype(str).str.match(pattern)]
    df = df.drop_duplicates(subset=["id"]).dropna()

    df["label"] = df["id"] + " (" + df["name"] + ")"
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Extracted {len(df)} techniques -> {output_csv}")
    return output_csv

if __name__ == "__main__":
    excel_path = EXCEL_ATTACK_TECHS
    output_csv = MAPPING_CSV
    extract_techniques(excel_path, output_csv)