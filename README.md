# LLM-Assisted Adversary Attribution

A modular pipeline for automating cyber threat attribution using MITRE ATT&CK, APTnotes, and GPT-based analysis. 

---

## Table of Contents

- [System Overview](#system-overview)
- [Workflow Overview](#workflow-overview)
- [System Architecture](#system-architecture)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)

---

## System Overview

This toolchain extracts Indicators of Compromise (IOCs) and MITRE ATT&CK Technique IDs (TTPs) from APT reports, maps them to known attacker groups using MITRE STIX data, and generates structured intelligence reports via GPT-based analysis. It supports two analysis modes. Rule-based TTP matching and Roberta model inference, both of which converge into a unified GPT-assisted report generation stage.

---

## System Architecture
```
ICT3214-SEC-ANALYTICS/
├──► app.py - Flask web app
│
├──► data/
│     ├──► raw/
│     │     ├──► attack_stix/ - contains enterprise-attack from MITRE
│     │     │
│     │     ├──► pdfs/ - contains APTnotes PDF files
│     │     │
│     │     └──► excel/ - contains extracted enterprise-attack MITRE ATT&CK & mitigation techniques
│     │
│     └──► mapped/ - contains the pdfs with successful correlation of IOCs and ATT&CK techniques
│
├──► src/
│     ├──► scripts/ - contains scripts to extract files, create datasets or map techniques to threat groups 
│     │
│     └──► models/ - contains latest and best RoBERTa model as well as scripts to train and predict
│
├──► templates/
│       ├──► common/ - contains html that are used by all other pages
│       │
│       ├──► index.html - Serves as the main landing page and user interface for the Flask-based MITRE ATT&CK Threat Attribution system. 
│       │
│       ├──► error.html - Error page rendered when invalid input, missing files, or API-related exceptions occur during app execution.
│       │
│       └──► results.html - Displays the dual-output comparison between the rule-based and RoBERTa-based threat attribution flows.  
│
├──► requirements.txt - list of dependencies that need to be installed via "pip install -r requirements"
│
└──► project_paths.py - script that has static variables used by other scripts that is related to path locations 
```
---

## Workflow Overview

```
src/scripts
│
├──► extract_pdfs.py → Extracts IOCs & TTPs from APTnotes PDF reports 
│
├──► build_dataset.py → Creates labeled dataset for the RoBERTa model
│
├──► enterprise_attack.py → Processes and normalizes the MITRE ATT&CK Enterprise dataset, extracting relationships between techniques, mitigations, and associated tactics
│
├──► map_iocs_to_attack.py → Correlates extracted IOCs and ATT&CK technique IDs with known MITRE ATT&CK threat groups
│
├──► matching.py → Matches input TTPs using both rule-based and RoBERTa inference modes
│
├──► mitigations.py → Retrieves defensive mitigations corresponding to the TTPs associated with matched groups
│
├──► technique_labels.py - Extracts MITRE techniques IDs and technique names for dropdown menu
│
└──► report_generator.py → Generates GenAI-based structured intelligence reports summarizing group matches, mitigations, and analyst insights
```
<img width="450" height="500" alt="image" src="https://github.com/user-attachments/assets/7d663766-5bb2-4217-85bf-bf813b0cc1d1" />
<img width="450" height="500" alt="image" src="https://github.com/user-attachments/assets/3d15d2e7-88b7-4385-b488-935bc2916910" />

<img width="905" height="800" alt="image" src="https://github.com/user-attachments/assets/10f81c44-c773-4fc9-ad50-d609c42c9f51" />

---

## Dependencies

All dependencies are listed in `requirements.txt`.

To install them, follow the setup steps in the Installation section.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/imbored1313/LLM-Assisted-Adversary-Attribution
cd LLM Assisted Adversary Attribution

# 2. Create a virtual environment
# On macOS / Linux:
python3 -m venv .venv
# On Windows:
python -m venv .venv

# 3. Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate 


# 4. Install dependencies
pip install -r requirements.txt

# 5. Create a .env file and add your OpenAI API key
# (Replace YOUR_KEY_HERE with your actual API key, look in user manual for more details)
echo "OPENAI_API_KEY=YOUR_KEY_HERE" > .env
```



---

## Usage

```
## 1. Run the web application

python app.py
navigate to `http://127.0.0.1:5000`
```

---

## Notes

- Limit input to five TTPs max for optimal GPT performance
- Output folders are automatically created if missing

---

## Troubleshooting

| Issue                       | Cause                                                | Solution                                                                          |
|-----------------------------|------------------------------------------------------|-----------------------------------------------------------------------------------|
| OPENAI_API_KEY not found    | `.env` file missing                                  | Add your key to `.env`                                                            |
| No PDFs found               | Incorrect input folder                               | Ensure path to `aptnotes_pdfs/` is correct                                        |
| Empty report output         | Invalid TTP input format                             | Use valid MITRE IDs (e.g., `T1059.003`)                                           |
| Empty Confidence Assessment | Occassional Issues if Submit when Idle               | Return to the homepage and Click "Build/                                          |
| Empty mitigations file      | Missing Data\processed\mitigations\mitigations.csv   | Ensure mitigations.csv exist in the specified directory and restart flask         |
---





