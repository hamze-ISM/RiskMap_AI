# setup_project.py
# Ce script crée toute l'arborescence du projet RiskMap_AI

import os

# Liste de tous les dossiers à créer
DOSSIERS = [
    "data/raw/2024",
    "data/raw/2025",
    "data/raw/2026",
    "data/processed",
    "data/interim",
    "data/external",
    "notebooks",
    "src/preprocessing",
    "src/features",
    "src/models",
    "src/mapping",
    "src/alerts",
    "src/dashboard",
    "src/utils",
    "cyber/integrity/hashing",
    "cyber/bias",
    "cyber/attacks",
    "cyber/ethics",
    "mosief/economics/roi",
    "mosief/indicators",
    "reports/figures",
    "reports/tables",
    "models",
    "maps",
    "tests",
    "config",
    "docs/uml",
    "docs/architecture",
    "docs/references",
]

def creer_arborescence():
    print("🚀 Création de l'arborescence du projet RiskMap_AI...\n")
    for dossier in DOSSIERS:
        os.makedirs(dossier, exist_ok=True)
        print(f"  ✅ Créé : {dossier}")
    print("\n✅ Arborescence créée avec succès !")

if __name__ == "__main__":
    creer_arborescence()