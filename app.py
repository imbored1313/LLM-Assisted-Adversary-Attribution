from flask import Flask, render_template, request, make_response, jsonify
import pandas as pd
from typing import List, Optional
from datetime import datetime
from collections import defaultdict
import importlib.util
import io
import re
from typing import Iterable
from pathlib import Path
import subprocess, sys
import os, tempfile
import json
# =======================================================
# Project-relative paths
# =======================================================
from project_paths import (
    PROJECT_ROOT, DATA_ROOT, EXPERIMENTS_ROOT, SRC_ROOT, MODELS_ROOT,SCRIPTS_DIR,
    RAW_DIR, PROCESSED_DIR, EXTRACTED_PDFS_DIR,
    MAPPED_DIR, EXCEL_DIR, MITIGATIONS_DIR,
    ATTACK_STIX_DIR,PDFS_DIR,RULES_DIR,EXTRACT_SCRIPT,ATTACK_SCRIPT,MAP_IOCS_SCRIPT,
    BUILD_DATASET_SCRIPT,MITIGATIONS_SCRIPT,
    GROUP_TTPS_DETAIL_CSV,MATCHING_SCRIPT,REPORT_GENERATION_SCRIPT,TECHNIQUE_LABELS_SCRIPT,
    TRAIN_ROBERTA_SCRIPT,PREDICT_SCRIPT,BEST_MODEL_DIR,
    MAPPING_CSV,MITIGATIONS_CSV,EXCEL_ATTACK_TECHS,
    EXTRACTED_IOCS_CSV,TI_GROUPS_TECHS_CSV,DATASET_CSV,LABELS_TXT,GROUP_TTPS_DETAIL_CSV,RANKED_GROUPS_CSV,
    output_dir_for_folds, project_path,ensure_dir_tree,add_src_to_syspath
)
app = Flask(__name__)
sys.path.insert(0, str(SRC_ROOT))  


def _import_from_path(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# load the three script modules by file path
matching           = _import_from_path(MATCHING_SCRIPT, "matching")
technique_labels   = _import_from_path(TECHNIQUE_LABELS_SCRIPT, "technique_labels")
report_generator   = _import_from_path(REPORT_GENERATION_SCRIPT, "report_generator")

# expose the functions you need
validate_ttps         = matching.validate_ttps
match_ttps            = matching.match_ttps

extract_techniques    = technique_labels.extract_techniques

analyze_TTP           = report_generator.analyze_TTP
parse_ai_response     = report_generator.parse_ai_response
load_filtered_mitigations = report_generator.load_filtered_mitigations
summarize_mitigations     = report_generator.summarize_mitigations

# Cache
LAST_RESULTS = {} #Global
LAST_RESULTS_RULE = {}
LAST_RESULTS_ROBERTA = {}

def _ensure_score_and_rank_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has numeric 'score' and 'rank' columns.
    """
    import numpy as np

    if df is None or df.empty:
        return df

    candidates = ["score", "prob", "probability", "confidence", "logit", "logprob"]
    src = next((c for c in candidates if c in df.columns), None)

    if src is None:
        if "rank" in df.columns:
            df["score"] = pd.to_numeric(df["rank"], errors="coerce")
            df["score"] = 1.0 / (1.0 + df["score"].fillna(df["score"].max() or 1))
        else:
            n = len(df)
            df["score"] = np.linspace(1.0, 0.0, n, endpoint=False)
    else:
        df["score"] = pd.to_numeric(df[src], errors="coerce").fillna(0.0)

    if "rank" not in df.columns or df["rank"].isna().all():
        df["rank"] = (-df["score"]).rank(method="first").astype(int)

    return df
#For ROBERTA
def _ensure_score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    if df is None or df.empty:
        return df

    # include group_score here
    candidates = ["score", "group_score", "prob", "probability", "confidence", "logit", "logprob"]
    src = next((c for c in candidates if c in df.columns), None)

    if src is None:
        if "rank" in df.columns:
            df["score"] = pd.to_numeric(df["rank"], errors="coerce")
            df["score"] = 1.0 / (1.0 + df["score"].fillna(df["score"].max() or 1))
        else:
            n = len(df)
            df["score"] = np.linspace(1.0, 0.0, n, endpoint=False)
    else:
        df["score"] = pd.to_numeric(df[src], errors="coerce").fillna(0.0)

    if "rank" not in df.columns or df["rank"].isna().all():
        df["rank"] = (-df["score"]).rank(method="first").astype(int)

    return df

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(map(str, cmd))}")
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

def needs_run(outputs: Iterable[Path], inputs: Iterable[Path] = ()) -> bool:
    outs = list(outputs)
    if not outs or any(not p.exists() for p in outs):
        return True  # missing outputs => run

    # If any input (or the script itself) is newer than any output => run
    out_mtime = min(p.stat().st_mtime for p in outs)
    ins = [p for p in inputs if p is not None and Path(p).exists()]
    if not ins:
        return False
    return max(Path(p).stat().st_mtime for p in ins) > out_mtime

def _atomic_to_csv(df, path: str):
    d = Path(path).parent
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=d, newline="", suffix=".tmp") as tmp:
        tmp_name = tmp.name
        df.to_csv(tmp, index=False)
    os.replace(tmp_name, path)  # atomic on POSIX & Windows

#For mitigations

ID_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
def _norm_ids(items: List[str]) -> List[str]:
    out = []
    for x in items:
        if isinstance(x, str):
            out.extend(m.upper() for m in ID_RE.findall(x))
    def _key(tid: str):
        m = re.match(r"T(\d{4})(?:\.(\d{3}))?$", tid)
        return (int(m.group(1)), int(m.group(2) or 999)) if m else (9999, 999)
    return sorted(set(out), key=_key)

def _parse_id_list(value) -> List[str]:
    # Handles list, JSON string, comma string, or plain string with T-IDs
    if isinstance(value, list):
        strs = [str(v) for v in value]
    elif isinstance(value, str):
        s = value.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                strs = [str(v) for v in parsed]
            else:
                strs = [s]
        except Exception:
            # Comma-separated or raw text with IDs
            strs = [p.strip() for p in s.split(",")] if "," in s else [s]
    else:
        strs = []
    return _norm_ids(strs)

def collect_top_group_ttps(
    df: pd.DataFrame,
    group_ttps_detail_csv: Path,
    use_ml: bool = False,
    tech_lookup_csv: Optional[Path] = None,  # optional: map technique_name -> technique_id
) -> List[str]:
    """
    Unified group → TTPs:
      - If use_ml and df has 'technique_id_list': return those IDs (normalized).
      - Else fall back to rule-based mapping via group_ttps_detail_csv
        using 'matched_exact'/'matched_root_only' for the top-scoring group.
      - If only 'technique_name' exists (ML), optionally map via tech_lookup_csv.
    """
    if df is None or df.empty:
        print("[WARN] Empty df"); return []
    if not group_ttps_detail_csv.exists():
        print("[WARN] Missing mapping CSV"); return []

    # top row by score
    top = df.sort_values("score", ascending=False).iloc[0]
    top_name = str(top.get("group_name", "")).strip().lower()
    top_id   = str(top.get("group_id", "")).strip().lower()

    # -------- ML path --------
    if use_ml:
        # Prefer explicit technique IDs
        if "technique_id_list" in df.columns:
            ids = _parse_id_list(top.get("technique_id_list", ""))
            if ids:
                return ids

        # Optional: names → IDs if provided
        if tech_lookup_csv and tech_lookup_csv.exists() and "technique_name" in df.columns:
            try:
                lut = pd.read_csv(tech_lookup_csv)
                lut.columns = [c.strip().lower() for c in lut.columns]
                if {"technique_id","technique_name"}.issubset(lut.columns):
                    name = str(top.get("technique_name","")).strip().lower()
                    ids = lut.loc[lut["technique_name"].str.lower()==name, "technique_id"].astype(str).tolist()
                    ids = _norm_ids(ids)
                    if ids:
                        return ids
            except Exception as e:
                print(f"[WARN] tech_lookup read failed: {e}")

        # Last resort: extract T-IDs from technique_name text itself
        if "technique_name" in df.columns:
            ids = _norm_ids([str(top.get("technique_name",""))])
            if ids:
                return ids
        # If ML path yields nothing, we fall through to rule-based as a safety net.

    # -------- Rule-based fallback --------
    try:
        gmap = pd.read_csv(group_ttps_detail_csv)
    except Exception as e:
        print(f"[ERROR] Failed reading {group_ttps_detail_csv}: {e}")
        return []
    gmap = gmap.copy()
    gmap.columns = [c.strip().lower() for c in gmap.columns]
    if "group_name" not in gmap.columns or "group_id" not in gmap.columns:
        print("[ERROR] mapping missing 'group_name'/'group_id'"); return []

    mask = pd.Series(False, index=gmap.index)
    if top_id:
        mask = mask | (gmap["group_id"].astype(str).str.lower() == top_id)
    if top_name:
        mask = mask | (gmap["group_name"].astype(str).str.lower() == top_name)
    gsel = gmap.loc[mask]
    if gsel.empty:
        print(f"[INFO] No mapping rows for {top_name or top_id}"); return []

    cols = [c for c in ("matched_exact","matched_root_only") if c in gsel.columns]
    vals = []
    for c in cols:
        vals.extend(gsel[c].fillna("").astype(str).tolist())

    ids = _norm_ids(vals)
    return ids

def _blank_analysis():
    return {
        "summary": "",
        "table": "",
        "attacker": "",
        "mitigation": "",
        "suggestion": "",
    }
# ============================================
# Rule-based flow helper
# ============================================

def _blank_analysis():
    return {
        "summary": "",
        "table": "",
        "attacker": "",
        "mitigation": "",
        "suggestion": "",
    }

def _run_rule_match_flow(ttps: list[str]) -> dict:
    #     """
#     Rule-based threat attribution flow:
#     - Runs rule-based TTP matching
#     - Extracts per-group associated TTPs for display
#     - Filters mitigations to only those related to top group's TTPs
#     - Generates parsed analysis + optional DOCX report
#     """
    LAST_RESULTS_ROBERTA.clear()

    matched_df = match_ttps(ttps, MAPPED_DIR).copy()
    matched_df = _ensure_score_and_rank_rule(matched_df)

    if "rank" in matched_df.columns and matched_df["rank"].notna().any():
        matched_df = matched_df.sort_values(by=["rank", "score"], ascending=[True, False])
    else:
        matched_df = matched_df.sort_values(by="score", ascending=False)

    top3_df = matched_df.head(3)
    matched_df.to_csv("matched_groups_rule.csv", index=False)
    pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps.csv", index=False)

    # ======= EARLY EXIT: no matches / no TTPs =======
    no_matches = matched_df.empty or top3_df.empty
    if no_matches:
        # Ensure the UI sees blank sections and no attachments
        return {
            "ttps": ttps,
            "matched_full_df": matched_df,
            "matched_top3": [],               # no rows for table
            "analysis": _blank_analysis(),    # all fields empty
            "doc_path": None,                 # no DOCX
            "mitigations_csv": None,          # no CSV
            "has_matches": False,             # template guard
        }

    mit_csv_path = MITIGATIONS_CSV
    gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    def _extract_ttps_from_text(s: str):
        if not isinstance(s, str): return []
        return re.findall(r"\bT\d{4}(?:\.\d{3})?\b", s.upper())

    combined_ttps_list = []
    for _, row in matched_df.iterrows():
        ttps_found = set()
        for col in ["matched_exact", "matched_root_only"]:
            if col in row and isinstance(row[col], str):
                ttps_found.update(_extract_ttps_from_text(row[col]))
        combined_ttps_list.append(", ".join(sorted(ttps_found)) if ttps_found else "—")

    matched_df["combined_ttps"] = combined_ttps_list
    top3_df = matched_df.head(3).copy()

    # top group ttps for mitigation filter
    top_rank_row = matched_df.loc[matched_df["rank"] == matched_df["rank"].min()].head(1)
    top_group_ttps = _extract_ttps_from_text(top_rank_row.iloc[0]["combined_ttps"]) if (
        not top_rank_row.empty and "combined_ttps" in top_rank_row.columns
    ) else []

    mit_filtered = load_filtered_mitigations(str(mit_csv_path), top_group_ttps)

    mit_rule_path = PROJECT_ROOT / "mitigations_rule_top.csv"
    if not mit_filtered.empty:
        mit_filtered = mit_filtered.drop_duplicates(
            subset=["target id", "target name", "mapping description"], keep="first"
        )
        parsed["mitigation"] = summarize_mitigations(mit_filtered.to_dict(orient="records"))
        _atomic_to_csv(mit_filtered, mit_rule_path)
        mit_for_docx = str(mit_rule_path)
    else:
        parsed["mitigation"] = ""
        mit_for_docx = None
        try:
            if mit_rule_path.exists():
                mit_rule_path.unlink()
        except Exception as e:
            print(f"[WARN] Could not remove stale {mit_rule_path}: {e}")

    out_path = None
    return {
        "ttps": ttps,
        "matched_full_df": matched_df,
        "matched_top3": top3_df.to_dict(orient="records"),
        "analysis": parsed,
        "doc_path": out_path,
        "mitigations_csv": mit_for_docx,
        "has_matches": True,
    }

# ============================================
# RoBERTa flow helper
# ============================================
def _save_roberta_traces(df_ml):
    _atomic_to_csv(df_ml, "matched_groups_roberta.csv")

def _predict_with_module(payload: dict):
    """
    Try importing predict_roberta.py and using its functions directly.
    Fallback to subprocess if import fails.
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("predict_roberta", str(PREDICT_SCRIPT))
        pr = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(pr)  # type: ignore

        text = pr.build_text(
            report_id=payload.get("id") or "adhoc",
            urls=payload.get("urls") or [],
            domains=payload.get("domains") or [],
            ips=payload.get("ips") or [],
            md5s=payload.get("md5s") or [],
            sha256s=payload.get("sha256s") or [],
            attack_ids=payload.get("attacks") or [],
            free_text=payload.get("text") or None,
        )
        preds = pr.predict(text=text, threshold=float(payload.get("threshold", 0.5)), top_k=int(payload.get("top_k", 10)))
        attack_rows = pr.load_attack_index()

        srcs = pr.resolve_sources_from_inputs(
            payload.get("urls") or [], payload.get("domains") or [], payload.get("ips") or [],
            payload.get("md5s") or [], payload.get("sha256s") or [], payload.get("attacks") or []
        )
        origin_doc = ", ".join(srcs) if srcs else (payload.get("id") or "adhoc")

        attack_ids_in_input = {a.strip().upper() for a in (payload.get("attacks") or []) if a}

        flat = pr.expand_to_attack_rows(
            preds,
            attack_rows,
            attack_ids_in_input=attack_ids_in_input,
            origin_doc=origin_doc,
        )
        group_rows = pr.aggregate_by_group(flat)

        return {"text": text, "groups": group_rows}

    except Exception as e:
        print(f"[WARN] Direct import predict_roberta failed: {e}; falling back to subprocess.")
        args = [
            sys.executable, str(PREDICT_SCRIPT),
            "--id", payload.get("id") or "adhoc",
        ]
        for k, flag in [
            ("urls", "--url"), ("domains", "--domain"), ("ips", "--ip"),
            ("md5s", "--md5"), ("sha256s", "--sha256"), ("attacks", "--attack")
        ]:
            for v in payload.get(k) or []:
                args += [flag, str(v)]
        if payload.get("text"):
            args += ["--text", payload["text"]]
        args += ["--threshold", str(payload.get("threshold", 0.5)),
                 "--top-k", str(payload.get("top_k", 10))]

        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            raise RuntimeError(res.stderr or "predict_roberta failed")

        # NEW: try to parse JSON from CLI and normalize the shape
        try:
            payload = json.loads(res.stdout)
            if isinstance(payload, dict) and "groups" in payload:
                return payload
            # allow older CLIs: maybe the top-level is a list
            if isinstance(payload, list):
                return {"groups": payload}
            # last resort: empty groups
            return {"groups": []}
        except Exception:
            print("[WARN] CLI output was not JSON; returning empty groups.")
            return {"groups": []}
        
def _run_roberta_flow(ttps: list[str]) -> dict:
    """
    RoBERTa-based threat attribution flow:
    - Uses ML model predictions for group ranking
    - Extracts per-group associated TTPs (from matched_exact/root_only/technique_id_list)
    - Filters mitigations based on top group's TTPs
    - Generates parsed analysis and DOCX report
    """
    LAST_RESULTS_RULE.clear()

    # 1️⃣ Run the RoBERTa predictor
    ml = _predict_with_module({
        "id": "from_ttps",
        "attacks": ttps,
        "top_k": 50,
        "threshold": 0.0
    })
    group_rows = ml.get("groups", [])
    if not group_rows:
        raise RuntimeError("RoBERTa returned no groups. Check model artifacts and labels.")
    df_ml = pd.DataFrame(group_rows)

    # Normalize column names
    if "group" in df_ml.columns and "group_name" not in df_ml.columns:
        df_ml.rename(columns={"group": "group_name"}, inplace=True)

    df_ml = _ensure_score_and_rank(df_ml)
    df_ml.sort_values("score", ascending=False, inplace=True)
    _save_roberta_traces(df_ml)
    top3_df = df_ml.head(3).drop(columns=["origin"], errors="ignore")
    # ======= EARLY EXIT: no matches / no TTPs =======
    no_matches = df_ml.empty or top3_df.empty
    if no_matches:
        # Ensure the UI sees blank sections and no attachments
        return {
            "ttps": ttps,
            "matched_full_df": df_ml,
            "matched_top3": [],               # no rows for table
            "analysis": _blank_analysis(),    # all fields empty
            "doc_path": None,                 # no DOCX
            "mitigations_csv": None,          # no CSV
            "has_matches": False,             # template guard
        }

    # 2️⃣ GPT-based analysis
    mit_csv_path = MITIGATIONS_CSV
    gpt_response = analyze_TTP(ttps, df_ml, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    # =========================================================
    # 🧩 Per-group Combined TTPs — Extract for EACH group
    # =========================================================
    def _extract_ttps_from_text(s: str):
        if not isinstance(s, str):
            return []
        return re.findall(r"\bT\d{4}(?:\.\d{3})?\b", s.upper())

    combined_ttps_list = []
    for _, row in df_ml.iterrows():
        ttps_found = set()
        for col in ["matched_exact", "matched_root_only", "technique_id_list"]:
            if col in row and isinstance(row[col], str):
                ttps_found.update(_extract_ttps_from_text(row[col]))
        combined_ttps_list.append(", ".join(sorted(ttps_found)) if ttps_found else "—")

    df_ml["combined_ttps"] = combined_ttps_list
    top3_df = df_ml.head(3).copy()

    # =========================================================
    # 🛡️ Mitigations — use top group's extracted TTPs
    # =========================================================
    # 🛡️ Mitigations — use only the rank-1 group's TTPs
    top_rank_row = df_ml.loc[df_ml["rank"] == df_ml["rank"].min()].head(1)

    if not top_rank_row.empty and "combined_ttps" in top_rank_row.columns:
        top_group_ttps = _extract_ttps_from_text(top_rank_row.iloc[0]["combined_ttps"])
    else:
        top_group_ttps = []

    mit_filtered = load_filtered_mitigations(str(mit_csv_path), top_group_ttps)

    if not top_group_ttps:
        print("[INFO] No top-group TTPs resolved for ROBERTA; skipping mitigations (no group mapping).")

    # 3️⃣ Deduplicate + summarize mitigations
    if not mit_filtered.empty:
        mit_filtered = mit_filtered.drop_duplicates(
            subset=["target id", "target name", "mapping description"],
            keep="first"
        )
        parsed["mitigation"] = summarize_mitigations(mit_filtered.to_dict(orient="records"))
        _atomic_to_csv(mit_filtered, "mitigations_roberta_top.csv")
        mit_for_docx = "mitigations_roberta_top.csv"
    else:
        parsed["mitigation"] = "No mitigations found for these techniques."
        mit_for_docx = None
    out_path = None

    # 5️⃣ Return structured result
    return {
        "ttps": ttps,
        "matched_full_df": df_ml,
        "matched_top3": top3_df.to_dict(orient="records"),
        "analysis": parsed,
        "doc_path": out_path,
    }

# =======================================================
# Pipeline steps (extract pdf → extract stix → build → train)
# =======================================================
def step_extract_pdfs(in_dir: Path = PDFS_DIR, out_dir: Path = EXTRACTED_PDFS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = EXTRACTED_IOCS_CSV

    # Treat the extractor script and input folder as inputs
    inputs = [EXTRACT_SCRIPT] + list(in_dir.glob("*.pdf"))
    if not needs_run([out_csv], inputs=inputs):
        print(f"[SKIP] extract_pdfs.py — up to date: {out_csv}")
        return

    run([sys.executable, str(EXTRACT_SCRIPT), "--in", str(in_dir), "--out", str(out_dir)])

def step_map_iocs_to_attack():
    outputs = [GROUP_TTPS_DETAIL_CSV, RANKED_GROUPS_CSV]
    inputs  = [MAP_IOCS_SCRIPT]
    if not needs_run(outputs, inputs=inputs):
        print(f"[SKIP] map_iocs_to_attack.py — up to date: {EXTRACTED_IOCS_CSV}")
        return
    run([sys.executable, str(MAP_IOCS_SCRIPT)])

def step_enterprise_attack():
    outputs = [TI_GROUPS_TECHS_CSV]
    inputs  = [ATTACK_SCRIPT]  # add STIX source dirs/files if you have them
    if not needs_run(outputs, inputs=inputs):
        print(f"[SKIP] enterprise_attack.py — up to date: {TI_GROUPS_TECHS_CSV}")
        return
    run([sys.executable, str(ATTACK_SCRIPT)])


def step_build_dataset():
    outputs = [DATASET_CSV, LABELS_TXT]
    inputs  = [BUILD_DATASET_SCRIPT, EXTRACTED_IOCS_CSV, TI_GROUPS_TECHS_CSV]
    if not needs_run(outputs, inputs=inputs):
        print(f"[SKIP] build_dataset.py — up to date: {DATASET_CSV}, {LABELS_TXT}")
        return
    run([sys.executable, str(BUILD_DATASET_SCRIPT)])

def step_mitigations(force: bool = False):
    """
    Generate mitigations CSV (Data/mitigations/mitigations.csv).
    Runs mitigations.py if inputs changed or file missing.
    """
    out_csv = MITIGATIONS_CSV
    if not needs_run([out_csv], inputs=[MITIGATIONS_SCRIPT, GROUP_TTPS_DETAIL_CSV, MAPPING_CSV, EXCEL_ATTACK_TECHS]) and not force:
        print(f"[SKIP] mitigations.py — up to date: {out_csv}")
        return

    print(f"[RUN] mitigations.py — generating: {out_csv}")
    res = subprocess.run([sys.executable, str(MITIGATIONS_SCRIPT)], cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"mitigations.py failed with exit code {res.returncode}")
    if not out_csv.exists():
        raise FileNotFoundError(f"Expected mitigations CSV not found at: {out_csv}")
    print(f"[OK] mitigations.csv generated: {out_csv}")

def step_map_iocs_to_attack(force: bool = False):
    """
    Run map_iocs_to_attack.py to produce:
      - ranked_groups.csv
      - group_ttps_detail.csv
    """
    outputs = [RANKED_GROUPS_CSV, GROUP_TTPS_DETAIL_CSV]
    inputs = [MAP_IOCS_SCRIPT, EXTRACTED_IOCS_CSV, TI_GROUPS_TECHS_CSV]
    if not needs_run(outputs, inputs=inputs) and not force:
        print(f"[SKIP] map_iocs_to_attack.py — up to date: {RANKED_GROUPS_CSV}, {GROUP_TTPS_DETAIL_CSV}")
        return

    print(f"[RUN] map_iocs_to_attack.py — generating IOC→ATT&CK mappings…")
    res = subprocess.run([sys.executable, str(MAP_IOCS_SCRIPT)], cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"map_iocs_to_attack.py failed with exit code {res.returncode}")

    for out in outputs:
        if not out.exists():
            raise FileNotFoundError(f"Expected output missing: {out}")

    print(f"[OK] Generated: {RANKED_GROUPS_CSV.name}, {GROUP_TTPS_DETAIL_CSV.name}")

def _is_empty_dir(p: Path) -> bool:
    return (not p.exists()) or (next(p.iterdir(), None) is None)

def step_train_roberta():
    if not _is_empty_dir(BEST_MODEL_DIR):
        print(f"[SKIP] train_roberta.py — best model exists in {BEST_MODEL_DIR}")
        return
    run([sys.executable, str(TRAIN_ROBERTA_SCRIPT)])

def assert_dataset_ready():
    required = {"id", "text", "labels"}  # weight is optional
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"dataset.csv missing: {DATASET_CSV}")
    import csv
    with DATASET_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"dataset.csv missing columns: {missing}")
def ensure_prereqs():
    """Lightweight sanity checks before the pipeline runs."""
    ensure_dir_tree()          # from project_paths
    add_src_to_syspath()       # if your training/build scripts import from src

    # Check PDFs exist for extraction
    if not PDFS_DIR.exists():
        raise FileNotFoundError(f"PDFS_DIR does not exist: {PDFS_DIR}")
    if not any(PDFS_DIR.glob("*.pdf")):
        print(f"[WARN] No PDFs found in {PDFS_DIR}. The pipeline can still run, "
              "but dataset.csv may be empty/unlabeled.")

    # Check that the helper scripts exist
    for p in [EXTRACT_SCRIPT, ATTACK_SCRIPT, MAP_IOCS_SCRIPT, BUILD_DATASET_SCRIPT]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing script: {p}")

def _needs_training(model_dir: Path) -> bool:
    if _is_empty_dir(model_dir):
        return True
    # Check for common artifacts
    required_any = [
        model_dir / "pytorch_model.bin",
        model_dir / "model.safetensors",
    ]
    required_all = [
        model_dir / "config.json",
        model_dir / "tokenizer.json",
        model_dir / "tokenizer_config.json",
    ]
    if not any(p.exists() for p in required_any):
        return True
    if not all(p.exists() for p in required_all):
        return True
    return False

def ensure_best_model():
    """
    Train RoBERTa if MODELS_ROOT/best_roberta_for_predict is missing or empty
    or obviously lacks model artifacts.
    """
    if _needs_training(BEST_MODEL_DIR):
        print(f"[INFO] No usable model in {BEST_MODEL_DIR}; training now…")
        assert_dataset_ready()   # optional guard from my previous message
        step_train_roberta()
    else:
        print(f"[OK] Found a usable model in {BEST_MODEL_DIR}; skipping training.")

def build_everything(force_map=False, force_mitigations=False, train_if_missing=True):
    ensure_prereqs()
    step_enterprise_attack()
    step_extract_pdfs()
    step_map_iocs_to_attack(force=force_map)
    step_build_dataset()
    step_mitigations(force=force_mitigations)

    if train_if_missing:
        ensure_best_model()

# ============================================
# Index & Workflow
#index, workflow, roberta, submit_both, predict with module, predict api, match, export
# ============================================
@app.route('/')
def index():
    ensure_dir_tree()
    try:
        # Always regenerate latest mapping from Excel
        extract_techniques(EXCEL_ATTACK_TECHS, MAPPING_CSV)

        # Load mapping: id, name, label
        df_map = pd.read_csv(MAPPING_CSV)

        # Group main + sub-techniques
        ttp_dict = defaultdict(list)
        for tid, label in zip(df_map["id"], df_map["label"]):
            root = tid.split(".")[0]
            if "." in tid:
                ttp_dict[root].append((tid, label))
            else:
                ttp_dict.setdefault(tid, [])

        # Build grouped list for dropdown
        id_to_label = dict(zip(df_map["id"], df_map["label"]))
        grouped_ttps = []
        for root in sorted(ttp_dict.keys()):
            root_label = id_to_label.get(root, root)
            subs = [lbl for _, lbl in sorted(ttp_dict[root], key=lambda x: x[0])]
            grouped_ttps.append((root_label, subs))
        build_everything()
        return render_template('index.html', grouped_ttps=grouped_ttps)

    except Exception as e:
        return render_template('error.html', error=str(e))
# =======================================================
# WORKFLOW ROUTE
# =======================================================
@app.route('/workflow')
def workflow():
    return render_template('workflow.html')
# =======================================================
# ROBERTA ROUTE 
# =======================================================
@app.route('/roberta', methods=['POST'])
def roberta():
    try:
        ttps_input = [t.split()[0].upper() for t in request.form.getlist('ttps[]')]
        ttps = validate_ttps(ttps_input)

        # ✅ Single source of truth
        res = _run_roberta_flow(ttps)

        # cache for export
        global LAST_RESULTS, LAST_RESULTS_ROBERTA
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        LAST_RESULTS = {
            "ttps": res["ttps"],
            "matched": res["matched_top3"],
            "analysis": res["analysis"],
            "timestamp": timestamp,
        }
        LAST_RESULTS_ROBERTA = LAST_RESULTS.copy()

        # render using roberta keys
        return render_template(
            'results.html',
            ttps=res["ttps"],
            rob_matched=res["matched_top3"],
            rob_analysis=res["analysis"],
            export_mode=False,
            timestamp=timestamp,
        )
    except Exception as e:
        return render_template('error.html', error=str(e))

    
@app.route('/results', methods=['POST'])
def results():
    try:
        ttps_input = [t.split()[0].upper() for t in request.form.getlist('ttps[]')]
        ttps = validate_ttps(ttps_input)

        rule_res = _run_rule_match_flow(ttps)
        rob_res  = _run_roberta_flow(ttps)

        global LAST_RESULTS_RULE, LAST_RESULTS_ROBERTA
        LAST_RESULTS_RULE = {
            "ttps": rule_res["ttps"],
            "matched": rule_res["matched_top3"],
            "analysis": rule_res["analysis"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        LAST_RESULTS_ROBERTA = {
            "ttps": rob_res["ttps"],
            "matched": rob_res["matched_top3"],
            "analysis": rob_res["analysis"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return render_template(
            "results.html",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ttps=ttps,
            rule_matched=rule_res["matched_top3"],
            rule_analysis=rule_res["analysis"],
            rob_matched=rob_res["matched_top3"],
            rob_analysis=rob_res["analysis"],
            export_mode=False,
        )

    except Exception as e:
        return render_template('error.html', error=str(e)), 500

@app.route('/predict', methods=['POST'])
def predict_api():
    """
    JSON body:
    {
      "id": "sample1",
      "urls": [...], "domains": [...], "ips": [...],
      "md5s": [...], "sha256s": [...], "attacks": ["T1566.002", ...],
      "text": "optional free text",
      "threshold": 0.5, "top_k": 10
    }
    """
    try:
        payload = request.get_json(force=True, silent=False) or {}
        result = _predict_with_module(payload)
        return jsonify({"status": "ok", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/match', methods=['POST'])
def match():
    try:
        # Collect only clean Technique IDs
        ttps_input = [t.split()[0].upper() for t in request.form.getlist('ttps[]')]
        ttps = validate_ttps(ttps_input)

        matched_df = match_ttps(ttps, MAPPED_DIR)

        # Convert numeric, sort, top3
        matched_df["rank"]  = pd.to_numeric(matched_df.get("rank", float("nan")), errors="coerce")
        matched_df["score"] = pd.to_numeric(matched_df.get("score", float("nan")), errors="coerce")
        if matched_df["rank"].notna().any():
            matched_df = matched_df.sort_values(by="rank", ascending=True)
        else:
            matched_df = matched_df.sort_values(by="score", ascending=False)
        top3_df = matched_df.head(3)

        # Save traceability outputs
        matched_df.to_csv("matched_groups_rule.csv", index=False)

        # GPT Analysis
        mit_csv_path = MITIGATIONS_CSV
        gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
        parsed = parse_ai_response(gpt_response)

        group_ttps = collect_top_group_ttps(matched_df) or list({t.strip().upper() for t in ttps if t})

        mit_filtered = load_filtered_mitigations(str(mit_csv_path), group_ttps)

        # Remove duplicate mitigation descriptions
        if not mit_filtered.empty:
         mit_filtered = mit_filtered.drop_duplicates(
         subset=["target id", "target name", "mapping description"], keep="first"
        )

        parsed["mitigation"] = mit_filtered.to_dict(orient="records")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        global LAST_RESULTS
        LAST_RESULTS = {
            "ttps": ttps,
            "matched": top3_df.to_dict(orient="records"),
            "analysis": parsed,
            "timestamp": timestamp
        }

        return render_template(
            'results.html',
            ttps=ttps,
            matched=top3_df.to_dict(orient='records'),
            analysis=parsed,
            timestamp=timestamp
        )

    except Exception as e:
        return render_template('error.html', error=str(e))
# =======================================================
# EXPORT ROUTE - For when printing results to document
# =======================================================

@app.route('/export')
def export():
    try:
        which = (request.args.get("which") or "rule").lower()
        if which == "roberta":
            ctx = LAST_RESULTS_ROBERTA
        else:
            ctx = LAST_RESULTS_RULE if LAST_RESULTS_RULE else LAST_RESULTS

        if not ctx:
            return "No results available to export. Please generate a report first."

        # Always provide BOTH sets so the template never complains.
        rendered = render_template(
            "results.html",
            ttps=ctx.get("ttps", []),

            # Rule side
            rule_matched=(ctx.get("matched", []) if which == "rule" else []),
            rule_analysis=(ctx.get("analysis", {}) if which == "rule" else {}),

            # RoBERTa side
            rob_matched=(ctx.get("matched", []) if which == "roberta" else []),
            rob_analysis=(ctx.get("analysis", {}) if which == "roberta" else {}),

            export_mode=True,
            timestamp=ctx.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

        # strip CSS link if needed
        rendered = re.sub(r'<link rel="stylesheet" href="[^"]*attribution\.css">', "", rendered, flags=re.IGNORECASE)

        response = make_response(rendered)
        response.headers["Content-Type"] = "application/msword"
        fname = "threat_report_roberta.doc" if which == "roberta" else "threat_report_rule.doc"
        response.headers["Content-Disposition"] = f"attachment; filename={fname}"
        return response

    except Exception as e:
        return f"Error exporting to .doc: {e}"
    


if __name__ == '__main__':
    app.run(debug=False, threaded=True)