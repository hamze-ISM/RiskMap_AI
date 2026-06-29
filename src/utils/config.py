# config.py — Chemins et paramètres globaux du projet RiskMap_AI

import os

# ─── Chemin racine ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Données brutes ───────────────────────────────────────────────
DATA_RAW_2024  = os.path.join(BASE_DIR, "data", "raw", "2024", "crime_2024.csv")
DATA_RAW_2025  = os.path.join(BASE_DIR, "data", "raw", "2025", "crime_2025.csv")
DATA_RAW_2026  = os.path.join(BASE_DIR, "data", "raw", "2026", "crime_2026.csv")

# ─── Données externes ─────────────────────────────────────────────
DATA_EXTERNAL  = os.path.join(BASE_DIR, "data", "external", "acled_senegal.csv")

# ─── Données traitées ─────────────────────────────────────────────
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATA_INTERIM   = os.path.join(BASE_DIR, "data", "interim")

# ─── Sorties ──────────────────────────────────────────────────────
MODELS_DIR       = os.path.join(BASE_DIR, "models")
MAPS_DIR         = os.path.join(BASE_DIR, "maps")
REPORTS_FIGURES  = os.path.join(BASE_DIR, "reports", "figures")
REPORTS_TABLES   = os.path.join(BASE_DIR, "reports", "tables")

# ─── Paramètres ML ────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_ESTIMATORS = 100
