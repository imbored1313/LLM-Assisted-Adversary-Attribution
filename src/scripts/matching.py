from __future__ import annotations
import logging
from pathlib import Path
import re
import sys
from typing import Iterable, Set, Tuple
import pandas as pd
# ============================================
# Paths
# ============================================
ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from project_paths import (
    PROJECT_ROOT, MAPPED_DIR, GROUP_TTPS_DETAIL_CSV,GROUP_TTPS_DETAIL_CSV,RANKED_GROUPS_CSV,
)

# Regex definitions
TTP_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$") # Matches MITRE Technique ID format
TTP_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE) # Pattern to find TTPs in text

# Validation of input TTPs
def validate_ttps(ttps: Iterable[str]) -> Tuple[str, ...]: # Validate input TTPs
    ttps = tuple(t.strip().upper() for t in ttps if t.strip()) # Normalize and filter input
    if not ttps: 
        raise ValueError("No TTPs entered.") # Ensure at least one TTP is provided
    if len(ttps) > 5:
        raise ValueError("Maximum of 5 TTPs allowed.") # Limit to 5 TTPs
    for t in ttps:
        if not TTP_RE.match(t):
            raise ValueError(f"Invalid TTP format: {t}")
    return ttps

# ============================================
# Dataset handling — now merges both CSVs
# ============================================
# Dataset handling 
def load_combined_dataset(MAPPED_DIR: Path) -> pd.DataFrame: # Load and merge datasets
    
    # Load and merge both 'group_ttps_detail.csv' and 'ranked_groups.csv'
    group_path = GROUP_TTPS_DETAIL_CSV # Path to group_ttps_detail.csv
    ranked_path = RANKED_GROUPS_CSV # Path to ranked_groups.csv

    dfs = [] # List to hold dataframes
    
    # Load group_ttps_detail.csv if it exists
    if group_path.exists():
        dfs.append(pd.read_csv(group_path)) # Load CSV into DataFrame
        logging.info(f"Loaded {group_path.name} ({len(dfs[-1])} rows)")
    if ranked_path.exists(): # Load ranked_groups.csv if it exists
        dfs.append(pd.read_csv(ranked_path))
        logging.info(f"Loaded {ranked_path.name} ({len(dfs[-1])} rows)")
    # Check if any datasets were loaded
    if not dfs:
        raise FileNotFoundError(f"No datasets found in {MAPPED_DIR}")

    # Combine datasets and drop duplicates
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
    logging.info(f"Combined dataset size: {len(combined)} rows")
    return combined

# Helper function to split TTP tokens from a cell
def split_tokens(cell) -> Set[str]:
    if pd.isna(cell):
        return set()
    return set(m.upper() for m in TTP_PATTERN.findall(str(cell)))

# Expand TTPs to include root techniques
def with_roots(tts: Iterable[str]) -> Set[str]:
    out: Set[str] = set() # Output set of TTPs
    input_roots = {t.split(".", 1)[0] for t in tts if "." in t}
    # Expand each TTP to include its root technique
    for t in tts:
        out.add(t)
        root = t.split(".", 1)[0]
        if "." in t and root not in input_roots:
            out.add(root)

    return out

# Matching logic
def match_ttps(ttps: Tuple[str, ...], MAPPED_DIR: Path) -> pd.DataFrame:
    df = load_combined_dataset(MAPPED_DIR)
    df["_ttp_exact"] = df.apply(
        lambda r: split_tokens(r.get("matched_exact", "")) | split_tokens(r.get("matched_root_only", "")),
        axis=1
    )
    def _expand_with_roots(s: Set[str]) -> Set[str]:
        out = set(s)
        for t in list(s):
            if "." in t:
                out.add(t.split(".", 1)[0])
        return out
    # Keep root-expanded version for later use
    df["_ttp_with_roots"] = df["_ttp_exact"].map(_expand_with_roots)
    input_set = set(ttps)
    mask = df["_ttp_exact"].apply(lambda s: bool(input_set & s))
    matched = df.loc[mask].copy()
    keep_cols = ["group_name", "group_id", "matched_exact", "matched_root_only"]
    others = [c for c in df.columns if c not in keep_cols]
    matched = matched[keep_cols + others]

    logging.info(f"[DEBUG] Strict match rows: {len(matched)} for {input_set}")
    return matched

# Output writing
def write_outputs(matched: pd.DataFrame, ttps: Tuple[str, ...], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_out = out_dir / "matched_groups_rule.csv"

    matched.to_csv(matched_out, index=False)
    return matched_out

# Main execution flow
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        out_dir = PROJECT_ROOT
        user_input = input("Enter up to 5 TTPs (e.g. T1110 T1110.001 ...): ").strip()
        ttps = validate_ttps(user_input.split())

        logging.info("Loading and merging datasets...")
        matched = match_ttps(ttps, MAPPED_DIR)

        m_out, t_out = write_outputs(matched, ttps, out_dir)

        logging.info(f"Matched {len(matched)} rows -> {m_out}")
        logging.info(f"Saved inputted TTPs -> {t_out}")

        if not matched.empty and "group_name" in matched.columns:
            print("\nTop matched groups:")
            for g in matched["group_name"].head(10):
                print("-", g)
        else:
            print("No matches found.")
        return 0

    except Exception as e:
        logging.error(str(e))
        return 1

if __name__ == "__main__":
    sys.exit(main())