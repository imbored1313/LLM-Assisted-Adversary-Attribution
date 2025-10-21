from __future__ import annotations
import os
import sys
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple
import pandas as pd
from pathlib import Path
# ===========================
# Paths
# ===========================
ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from project_paths import (MITIGATIONS_DIR,MITIGATIONS_CSV,EXCEL_ATTACK_TECHS,
)

# ===========================
# Constants & Configs
# ===========================
REQUIRED_SHEET = "associated mitigations"
DEFAULT_REQUIRED_COLUMNS = ["target id", "target name", "mapping description"]

COLUMN_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "target id": ("target id", "technique id", "targetid", "technique_id"),
    "target name": ("target name", "technique name", "target", "name"),
    "mapping description": (
        "mapping description",
        "relationship description",
        "description",
        "mapping",
    ),
}

_ID_RE = re.compile(r'^\s*[tT]\s*(\d{4})(?:\.(\d{3}))?\s*$')
_WS_RE = re.compile(r'\s+')
_PUNCT_TRIM_RE = re.compile(r'^[\s\-\u2022•·\:;,_\.\|\(\)\[\]\{\}]+|[\s\-\u2022•·\:;,_\.\|\(\)\[\]\{\}]+$')

# ===========================
# Normalizaion Helpers
# ===========================
def _normalize(s: str) -> str:
    return s.strip().lower()

def _norm_ws(s: str) -> str:
    return _WS_RE.sub(' ', s).strip()

def _norm_punct_edges(s: str) -> str:
    return _PUNCT_TRIM_RE.sub('', s).strip()

def _norm_text(s: str) -> str:
    s = str(s)
    s = _norm_ws(s)
    s = _norm_punct_edges(s)
    return s

def _norm_desc_for_dedupe(s: str) -> str:
    s = _norm_text(s)
    s = s.rstrip('.').lower()
    return s

def _norm_tech_id(s: str) -> str:
    if s is None:
        return ''
    m = _ID_RE.match(str(s))
    if not m:
        s = _norm_ws(str(s))
        if s and s[0].lower() == 't':
            s = 'T' + s[1:]
        return s
    major, minor = m.group(1), m.group(2)
    return f"T{major}" + (f".{minor}" if minor else "")

def _id_sort_key(v: str):
    m = _ID_RE.match(v or "")
    if not m:
        return (9999, 999) 
    major = int(m.group(1))
    minor = int(m.group(2) or 999)  
    return (major, minor)
# ===========================
# Sheet & Column Selection
# ===========================
def choose_sheet(excel_path: str, requested: str) -> str | None:
    xls = pd.ExcelFile(excel_path)
    wanted = _normalize(requested)
    lowers = {name: _normalize(name) for name in xls.sheet_names}
    for name, low in lowers.items():
        if low == wanted:
            return name
    for name, low in lowers.items():
        if low.startswith(wanted):
            return name
    for name, low in lowers.items():
        if wanted in low:
            return name
    return None

def pick_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    norm_to_original = {_normalize(col): col for col in df.columns}
    selected = {}
    missing = []

    for canon in required_columns:
        candidates: Iterable[str] = COLUMN_SYNONYMS.get(_normalize(canon), (canon,))
        found_col = None
        for cand in candidates:
            key = _normalize(cand)
            if key in norm_to_original:
                found_col = norm_to_original[key]
                break
        if found_col is None:
            missing.append(canon)
        else:
            selected[canon] = found_col

    if missing:
        return df, missing

    out = df[[selected[c] for c in required_columns]].copy()
    out.columns = list(required_columns)
    return out, []

# ===========================
# DataFrame Cleaning
# ===========================
def clean_text_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else ("" if pd.isna(v) else str(v)))
    return df

# ===========================
# Core Transformations
# ===========================
def tidy_mitigations_dataframe(df: pd.DataFrame,
                               col_id: str = "target id",
                               col_name: str = "target name",
                               col_desc: str = "mapping description",
                               group_by_description_across_ttps: bool = True) -> pd.DataFrame:
    for c in (col_id, col_name, col_desc):
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' not found in mitigations dataframe.")
    # Normalize fields
    df[col_id] = df[col_id].map(_norm_tech_id)
    df[col_name] = df[col_name].map(lambda x: _norm_text(x).strip())
    df[col_desc] = df[col_desc].map(_norm_text)
    df = df[~(df[col_id].eq('') & df[col_name].eq(''))].copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df["_dedupe_desc_norm"] = df[col_desc].map(_norm_desc_for_dedupe)
    df = df.drop_duplicates(subset=[col_id, col_name, "_dedupe_desc_norm"]).reset_index(drop=True)

    if group_by_description_across_ttps:
        out = collapse_across_ttps_by_description(df, col_id, col_name, col_desc)
        return out
    grouped = []
    for (tid, tname), g in df.groupby([col_id, col_name], dropna=False, sort=False):
        seen = set()
        uniq_descs = []
        for desc, normd in zip(g[col_desc].tolist(), g["_dedupe_desc_norm"].tolist()):
            if not normd or normd in seen:
                continue
            seen.add(normd)
            uniq_descs.append(desc)
        merged_desc = "\n".join(f"• {d}" for d in uniq_descs)
        grouped.append({col_id: tid, col_name: tname, col_desc: merged_desc})

    out = pd.DataFrame(grouped, columns=[col_id, col_name, col_desc])

    out = out.sort_values(by=[col_id, col_name],
                          key=lambda s: s.map(_id_sort_key) if s.name == col_id else s.str.lower(),
                          kind="mergesort").reset_index(drop=True)
    return out

def collapse_across_ttps_by_description(df: pd.DataFrame,
                                        col_id: str = "target id",
                                        col_name: str = "target name",
                                        col_desc: str = "mapping description") -> pd.DataFrame:
    if not all(c in df.columns for c in [col_id, col_name, col_desc]):
        raise ValueError("collapse_across_ttps_by_description: missing required columns.")

    if "_dedupe_desc_norm" not in df.columns:
        df = df.copy()
        df["_dedupe_desc_norm"] = df[col_desc].map(_norm_desc_for_dedupe)

    rows = []
    for desc_norm, g in df.groupby("_dedupe_desc_norm", dropna=False):
        if not desc_norm:
            continue
        rep_desc = max(g[col_desc].astype(str), key=lambda s: len(s))
        pairs = []
        seen = set()
        for tid, tname in zip(g[col_id].astype(str), g[col_name].astype(str)):
            tid_n = _norm_tech_id(tid)
            tname_n = _norm_text(tname)
            key = (tid_n.lower(), tname_n.lower())
            if tid_n and key not in seen:
                seen.add(key)
                pairs.append((tid_n, tname_n))
        pairs.sort(key=lambda p: _id_sort_key(p[0]))
        joined_ids = ", ".join(p[0] for p in pairs)
        joined_names = ", ".join(p[1] for p in pairs)
        if joined_ids:
            rows.append({col_id: joined_ids, col_name: joined_names, col_desc: rep_desc})

    out = pd.DataFrame(rows, columns=[col_id, col_name, col_desc])
    def _first_id_sortkey(s: str):
        first = (s or "").split(",")[0].strip()
        return _id_sort_key(first)

    if not out.empty:
        out = out.sort_values(by=[col_id, col_name],
                              key=lambda s: s.map(_first_id_sortkey) if s.name == col_id else s.str.lower(),
                              kind="mergesort").reset_index(drop=True)
    return out

# ===========================
# Run
# ===========================
def main() -> None:
    excel_path = EXCEL_ATTACK_TECHS
    output_dir = MITIGATIONS_DIR
    os.makedirs(output_dir, exist_ok=True)
    out_path = MITIGATIONS_CSV

    required_columns = list(DEFAULT_REQUIRED_COLUMNS)
    sheet_name = choose_sheet(excel_path, REQUIRED_SHEET)
    if sheet_name is None:
        available = pd.ExcelFile(excel_path).sheet_names
        print(f"Error: required sheet '{REQUIRED_SHEET}' not found. Available sheets: {available}")
        sys.exit(2)

    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    out_df, missing = pick_columns(df, required_columns)
    if missing:
        print(f"Error: missing required columns {missing}. Found columns: {list(df.columns)}")
        sys.exit(3)

    out_df = clean_text_columns(out_df, required_columns)

    out_df = tidy_mitigations_dataframe(
        out_df,
        col_id=required_columns[0],         # "target id"
        col_name=required_columns[1],       # "target name"
        col_desc=required_columns[2],       # "mapping description"
    )

    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df):,} cleaned mitigation rows to {out_path}")


if __name__ == "__main__":
    main()
