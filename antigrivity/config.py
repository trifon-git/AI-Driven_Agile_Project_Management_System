"""
Antigrivity — Central Configuration
====================================
All paths, team definitions, and constants in one place.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories (supports both old uppercase and new lowercase names)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw") if os.path.isdir(
    os.path.join(PROJECT_ROOT, "data", "raw")) else os.path.join(PROJECT_ROOT, "DATA")
JIRA_DIR = os.path.join(DATA_DIR, "JIRA") if os.path.isdir(
    os.path.join(DATA_DIR, "JIRA")) else os.path.join(DATA_DIR, "jira")
TIMESHEETS_DIR = os.path.join(DATA_DIR, "Timesheets") if os.path.isdir(
    os.path.join(DATA_DIR, "Timesheets")) else os.path.join(DATA_DIR, "timesheets")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed") if os.path.isdir(
    os.path.join(PROJECT_ROOT, "data", "processed")) else os.path.join(PROJECT_ROOT, "OUTPUT")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models") if os.path.isdir(
    os.path.join(PROJECT_ROOT, "models")) else os.path.join(PROJECT_ROOT, "MODELS")

JIRA_WITH_HOURS_CSV = os.path.join(OUTPUT_DIR, "jira_with_timesheet_hours.csv")
UNMATCHED_CSV = os.path.join(OUTPUT_DIR, "unmatched_timesheet_entries.csv")
TRAINING_DATASET_CSV = os.path.join(OUTPUT_DIR, "training_data_v2_no_emb.csv")

EFFORT_MODEL_PATH = os.path.join(MODEL_DIR, "effort_estimation_model.pkl")
EFFORT_MODEL_TUNED_PATH = os.path.join(MODEL_DIR, "effort_estimation_extratrees_tuned.pkl")
EFFORT_MODEL_V2_PATH = os.path.join(MODEL_DIR, "stacked_estimator_v2.pkl")

TEAM = {
    "Marios Anagnostopoulos": {
        "capacity_hours": 35,
        "department": "Developer",
        "role": "Senior Developer",
        "seniority": "senior",
        "skills": ["backend", "frontend", "api", "database", "devops", "architecture", "code review"],
    },
    "Marios Kontis": {
        "capacity_hours": 35,
        "department": "Developer",
        "role": "Junior Developer",
        "seniority": "junior",
        "skills": ["frontend", "backend", "css", "bug fixing", "testing"],
    },
    "Konstantinos Karlis": {
        "capacity_hours": 35,
        "department": "Designer",
        "role": "Designer",
        "seniority": "mid",
        "skills": ["ui design", "ux design", "landing pages", "branding", "figma", "css", "responsive design"],
    },
    "Maria Messari": {
        "capacity_hours": 35,
        "department": "Operations",
        "role": "SEO & Operations",
        "seniority": "mid",
        "skills": ["seo", "content", "analytics", "social media", "operations", "reporting", "copywriting"],
    },
    "Nikos Haliotis": {
        "capacity_hours": 35,
        "department": "Management",
        "role": "CEO / Account Manager",
        "seniority": "senior",
        "skills": ["account management", "business development", "client communication", "strategy", "project management"],
    },
    "Andreas Pelekoudas": {
        "capacity_hours": 35,
        "department": "Developer",
        "role": "Full Stack Developer",
        "seniority": "senior",
        "skills": ["backend", "frontend", "ecommerce", "drupal", "api", "database"],
    },
    "Stavros Messaris": {
        "capacity_hours": 35,
        "department": "Developer",
        "role": "Full Stack Developer",
        "seniority": "senior",
        "skills": ["backend", "bug fixing", "testing", "drupal", "operations"],
    },
}

# Timesheet employee name → Jira assignee full name
EMPLOYEE_MAP = {
    "Anagnostopoulos": "Marios Anagnostopoulos",
    "Kontis": "Marios Kontis",
    "Kostantinos": "Konstantinos Karlis",
    "Maria": "Maria Messari",
    "Nikos": "Nikos Haliotis",
    "Andreas": "Andreas Pelekoudas",
    "Stavros": "Stavros Messaris",
}

SP_TO_HOURS = 0.5        # 1 story point = 0.5 hours (initial estimate)
VALID_YEARS = {2024, 2025}
DONE_KEYWORDS = ("done", "resolved", "closed")

# Sheets to skip in timesheet workbooks
SKIP_SHEETS = {
    "_DATA", "_OLD_AGGREGATION", "_AGGREGATION", "_PROJECT OPTIONS IMPORT",
    "PTO 24", "PTO 25", "_AGGREGATION 2025 VALUES ONLY",
}

# TIMESHEET PROJECT → JIRA KEY MAPPING
TIMESHEET_NAME_TO_JIRA_KEY = {
    # --- Exact or near-exact matches ---
    "Afthonia Studios": "AFTH",
    "Aspa's Verde Palazzo": "ASPAS",
    "Belitsis Accounting": "BEL",
    "Birbilo": "BRB",
    "EPC": "EPC",
    "Eyewear Concept": "EYE",
    "Feedit": "FEED",
    "Forestata Village": "FV",
    "Geusi Vounou": "GV",
    "Haliotis Electric": "HAL",
    "Hobby Art Shop": "HAS",
    "Haliotis Holidays": "HOLIDAYS",
    "Inovtech": "INOV",
    "Kalamatas Wood Expert": "WOOD",
    "Opal Provisions": "OPAL",
    "Physio Rizos": "RIZ",
    "Rivulis RGA": "RGA",
    "Stampaland": "STAM",
    "Stefania Frangista B2B": "SFSB2B",
    "Stova Bambini": "BAMB",
    "TAS Courier": "TAS",
    "Technofil": "TFL",
    "The Production Hub": "HUB",
    "Visiontech": "VT",
    "Vomvikion": "VOMV",
    "web-bunch.com": "WBCOM",
    # --- Fuzzy matches ---
    "HotAF Candles": "HOTAF",
    "Investlaw": "INVL",
    "ManhoodStyle": "MAN",
    "New Style Home": "NSH",
    "PASAP": "PASAP",
    "PASAPP": "PASAP",
    "Profitup": "PROF",
    "Stefania Frangista": "SF",
    "Stefania Frangista B2C": "SF",
    "Stavropoulos Thomas": "STAV",
    "Vivian's Villa Kefalonia": "VIV",
    "Piumosso": "PIUS",
    # --- Name completely different ---
    "Bet & Slot": "BS",
    "Bluewhite Apartments": "BWA",
    "Farmakeio Konstantatos": "ORIMOS",
    "Pagkoi Kouzinas": "PKG",
    "Taxi Kefalonias": "TAXI",
    "Sun Seekers": "SUN",
    "Whitehawk": "WHT",
    "Rivulis Pricing Hub": "RIV",
    # --- Internal / template projects ---
    "Next Base": "NEXTBASE",
    "WB Booking": "BOOKING",
    "WB Commerce": "ES",
    "WB Content": "WBCONTENT",
    "WB Touristic": "TPLTR",
    "WB CRM": "WBCRM",
    "Drupal Contrib": None,
    "Workshops": None,
    # --- Contract projects ---
    "EISA": "EISA",
    "NRG": "NRG",
    "Konecranes": None,
    "NewStyleHome": "NEW",
    "Odyssea": None,
    "Pointblank": None,
    "Timberhub": None,
    "Twipe": None,
    "UN": None,
    # --- Operational ---
    "All": None,
    "Event": None,
    "Social Media": None,
    # --- Timesheets only ---
    "Arsenikos - Salahas": "Stampaland",
    "Ble Holiday Suites": None,
    "Bryq": None,
    "Ethnikos OFPF": None,
    "Farkon": None,
    "FFNA": None,
    "Fortion": None,
    "Georgatos Athanasios": None,
    "Giannopoulos": None,
    "Linardatou Eftixia": "Blue White Apartments",
    "Messaris Gerasimos": None,
    "Mouza Panagiota": None,
    "Ocean Hero": None,
    "Odyssea Platform": None,
    "Papadimitropoulos": None,
    "Physio Keramidas": None,
    "Stelios Manolas": None,
    "Vasiliki Iliopoulou": None,
    "Veranda Suite": None,
    "Haliotis": "Haliotis Electric",
}

import os
os.environ["OLLAMA_HOST"] = "http://130.225.39.230:11434"

OLLAMA_MODEL = "qwen3.6:27b"  # Ollama model name
OLLAMA_NUM_CTX = 8192         # The context window size allocated by Ollama
OLLAMA_NUM_PREDICT = 8192     # Max tokens to generate
OLLAMA_NEGOTIATION_PREDICT = 1024 # Max tokens to generate for negotiation parsing
OLLAMA_SEED = 42              # Seed for reproducible outputs

import subprocess

def ensure_ollama_model_exists(model_name):
    """Ensure the given Ollama model exists on the host, pulling it if necessary."""
    try:
        env = os.environ.copy()
        list_proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, env=env)
        if model_name not in list_proc.stdout:
            print(f"Model '{model_name}' not found on host. Pulling (this may take a while)...")
            subprocess.run(["ollama", "pull", model_name], check=True, env=env)
    except Exception as e:
        print(f"Warning: Failed to check or pull Ollama model '{model_name}': {e}")

ensure_ollama_model_exists(OLLAMA_MODEL)

USE_GNN_ALLOCATION = True


