from __future__ import annotations
import csv
import logging
from pathlib import Path
import re
import sys
from typing import Iterable, Set, Tuple
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))

from project_paths import (
    PROJECT_ROOT,
    MAPPED_DIR,
    GROUP_TTPS_DETAIL_CSV,
    RANKED_GROUPS_CSV,
)

# ============================================
# Regex definitions
# ============================================
TTP_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")
TTP_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)

PREFERRED_KEYS = [
    "matched_exact", "matched_root_only", "ttps", "ttp",
    "techniques", "technique", "attack"
]

# ============================================
# Validation
# ============================================
def validate_ttps(ttps: Iterable[str]) -> Tuple[str, ...]:
    ttps = tuple(t.strip().upper() for t in ttps if t.strip())
    if not ttps:
        raise ValueError("No TTPs entered.")
    if len(ttps) > 5:
        raise ValueError("Maximum of 5 TTPs allowed.")
    for t in ttps:
        if not TTP_RE.match(t):
            raise ValueError(f"Invalid TTP format: {t}")
    return ttps

# ============================================
# Dataset loading
# ============================================
def load_combined_dataset(MAPPED_DIR: Path) -> pd.DataFrame:
    """
    Load and merge both 'group_ttps_detail.csv' and 'ranked_groups.csv'
    if they exist. Duplicates are dropped.
    """
    dfs = []
    if GROUP_TTPS_DETAIL_CSV.exists():
        dfs.append(pd.read_csv(GROUP_TTPS_DETAIL_CSV))
        logging.info(f"Loaded {GROUP_TTPS_DETAIL_CSV.name} ({len(dfs[-1])} rows)")
    if RANKED_GROUPS_CSV.exists():
        dfs.append(pd.read_csv(RANKED_GROUPS_CSV))
        logging.info(f"Loaded {RANKED_GROUPS_CSV.name} ({len(dfs[-1])} rows)")

    if not dfs:
        raise FileNotFoundError(f"No datasets found in {MAPPED_DIR}")

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
    logging.info(f"Combined dataset size: {len(combined)} rows")
    return combined

# ============================================
# Column identification
# ============================================
def score_column(col: str) -> Tuple[int, int]:
    cl = col.lower()
    exact = any(cl == k for k in PREFERRED_KEYS)
    hits = sum(1 for k in PREFERRED_KEYS if k in cl)
    return (1 if exact else 0, hits)

def find_ttp_column(df: pd.DataFrame) -> str:
    ranked = sorted(df.columns, key=lambda c: score_column(c), reverse=True)
    for c in ranked:
        cl = c.lower()
        if any(k in cl for k in PREFERRED_KEYS):
            return c
    for c in df.columns:
        cl = c.lower()
        if any(x in cl for x in ["ttp", "technique", "attack"]):
            return c
    raise KeyError("Could not find a TTP-related column.")

# ============================================
# Token extraction
# ============================================
def split_tokens(cell) -> Set[str]:
    if pd.isna(cell):
        return set()
    return set(m.upper() for m in TTP_PATTERN.findall(str(cell)))

# ============================================
# Strict matching (no root expansion)
# ============================================
def match_ttps(ttps: Tuple[str, ...], MAPPED_DIR: Path) -> pd.DataFrame:
    """
    STRICT matching logic:
    - Only matches exact input TTPs (no parent/root expansion).
    - If input is T1110.001, it only matches dataset rows containing T1110.001.
    - If input is T1110, it can still match sub-techniques if they explicitly appear in dataset.
    """
    df = load_combined_dataset(MAPPED_DIR)
    ttp_col = find_ttp_column(df)

    # Extract techniques per dataset row
    df["_ttp_set"] = df[ttp_col].map(split_tokens)

    # Build input set (strict)
    input_full = {t.upper() for t in ttps}

    # Match only if any input exactly appears in dataset tokens
    mask = df["_ttp_set"].apply(lambda s: bool(input_full & s))
    matched = df.loc[mask].drop(columns=["_ttp_set"])
    return matched

# ============================================
# CSV output
# ============================================
def write_outputs(matched: pd.DataFrame, ttps: Tuple[str, ...], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_out = out_dir / "matched_groups_rule.csv"
    matched.to_csv(matched_out, index=False)
    return matched_out, out_dir / "inputted_ttps.csv"

# ============================================
# CLI entry point
# ============================================
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        out_dir = PROJECT_ROOT
        user_input = input("Enter up to 5 TTPs (e.g. T1110 T1110.001 ...): ").strip()
        ttps = validate_ttps(user_input.split())

        logging.info("Loading and merging datasets...")
        matched = match_ttps(ttps, MAPPED_DIR)

        m_out, t_out = write_outputs(matched, ttps, out_dir)
        pd.DataFrame({"TTP": ttps}).to_csv(t_out, index=False)

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
