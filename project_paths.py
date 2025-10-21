from __future__ import annotations
from pathlib import Path
import os
import sys

def find_project_root(start: Path | None = None) -> Path:
    """
    Walk up from `start` to find a directory that looks like the repo root.
    We treat a dir with .git/ or pyproject.toml/ or requirements.txt as the root.
    """
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / ".git").exists() or (p / "requirements.txt").exists():
            return p
    return here.parent

PROJECT_ROOT     = Path(os.environ.get("ICT3214_PROJECT_ROOT", find_project_root()))
DATA_ROOT        = Path(os.environ.get("ICT3214_DATA_ROOT", PROJECT_ROOT / "data"))
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT         = PROJECT_ROOT / "src"
SCRIPTS_DIR      = SRC_ROOT / "scripts"     
MODELS_ROOT      = SRC_ROOT / "models"

#Data roots
RAW_DIR            = DATA_ROOT / "raw"
PROCESSED_DIR      = DATA_ROOT / "processed"
EXTRACTED_PDFS_DIR = DATA_ROOT / "extracted_pdfs"

# Raw data subdirs
ATTACK_STIX_DIR = RAW_DIR / "attack_stix"
PDFS_DIR        = RAW_DIR / "pdfs"
EXCEL_DIR       = RAW_DIR / "excel"

# Scripts
EXTRACT_SCRIPT        = SCRIPTS_DIR / "extract_pdfs.py"
ATTACK_SCRIPT         = SCRIPTS_DIR / "enterprise_attack.py"
MAP_IOCS_SCRIPT    =  SCRIPTS_DIR / "map_iocs_to_attack.py"   
BUILD_DATASET_SCRIPT  = SCRIPTS_DIR / "build_dataset.py"
MITIGATIONS_SCRIPT    =  SCRIPTS_DIR / "mitigations.py"   
MATCHING_SCRIPT     =  SCRIPTS_DIR / "matching.py"  
REPORT_GENERATION_SCRIPT    =  SCRIPTS_DIR / "report_generator.py"   
TECHNIQUE_LABELS_SCRIPT    =  SCRIPTS_DIR / "technique_labels.py"   

# Models
TRAIN_ROBERTA_SCRIPT  = MODELS_ROOT / "train_roberta.py"
PREDICT_SCRIPT        = MODELS_ROOT / "predict_roberta.py"
BEST_MODEL_DIR        = MODELS_ROOT / "best_roberta_for_predict"

# Processed subdirs
RULES_DIR       = PROCESSED_DIR / "rules"
MAPPED_DIR      = PROCESSED_DIR / "mapped"  
MITIGATIONS_DIR = PROCESSED_DIR / "mitigations"

# Outputs produced by scripts
EXTRACTED_IOCS_CSV    = PROCESSED_DIR / "extracted_iocs.csv"
TI_GROUPS_TECHS_CSV   = PROCESSED_DIR / "ti_groups_techniques.csv"
DATASET_CSV           = PROCESSED_DIR / "dataset.csv"
LABELS_TXT            = PROCESSED_DIR / "labels.txt"

GROUP_TTPS_DETAIL_CSV = PROCESSED_DIR / "group_ttps_detail.csv"
RANKED_GROUPS_CSV     = PROCESSED_DIR / "ranked_groups.csv"

MAPPING_CSV           = PROCESSED_DIR / "techniques_mapping.csv"
MITIGATIONS_CSV       = MITIGATIONS_DIR / "mitigations.csv"
EXCEL_ATTACK_TECHS    = EXCEL_DIR / "enterprise-attack-v17.1-techniques.xlsx"

def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
def ensure_dir_tree() -> None:
    """Create the directory tree if missing."""
    for d in [
        DATA_ROOT, RAW_DIR, PROCESSED_DIR, EXTRACTED_PDFS_DIR,
        EXPERIMENTS_ROOT, SRC_ROOT, SCRIPTS_DIR, MODELS_ROOT,
        MAPPED_DIR, EXCEL_DIR, MITIGATIONS_DIR, ATTACK_STIX_DIR, PDFS_DIR, RULES_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)

def add_src_to_syspath() -> None:
    """Make src/ and src/scripts importable (so 'import matching' works after move)."""
    for p in (SRC_ROOT, SCRIPTS_DIR):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
