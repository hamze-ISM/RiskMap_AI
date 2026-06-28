# setup_files.py
# Ce script crée tous les fichiers vides du projet

import os

# Dictionnaire : chemin du fichier → commentaire d'en-tête
FICHIERS = {
    # Scripts principaux
    "src/preprocessing/merge_data.py": "# merge_data.py — Fusion des datasets Chicago 2024/2025/2026",
    "src/preprocessing/clean_data.py": "# clean_data.py — Nettoyage des données",
    "src/features/feature_engineering.py": "# feature_engineering.py — Construction des features ML",
    "src/models/train_model.py": "# train_model.py — Entraînement Random Forest et XGBoost",
    "src/models/predict.py": "# predict.py — Génération des prédictions de risque",
    "src/mapping/risk_map.py": "# risk_map.py — Carte interactive Folium",
    "src/alerts/alerts.py": "# alerts.py — Système d'alertes automatiques",
    "src/dashboard/dashboard.py": "# dashboard.py — Dashboard Streamlit",
    "src/utils/config.py": "# config.py — Paramètres et chemins globaux du projet",
    # Cybersécurité
    "cyber/integrity/hashing/integrity_check.py": "# integrity_check.py — Vérification SHA256 des données",
    "cyber/bias/bias_analysis.py": "# bias_analysis.py — Analyse des biais du modèle",
    "cyber/attacks/adversarial_test.py": "# adversarial_test.py — Tests d'attaques adversariales",
    "cyber/ethics/ethical_audit.py": "# ethical_audit.py — Audit éthique et anonymisation",
    # MOSIEF
    "mosief/economics/economic_analysis.py": "# economic_analysis.py — Impact économique de la criminalité",
    "mosief/economics/roi/roi_calculator.py": "# roi_calculator.py — Calcul du retour sur investissement",
    "mosief/indicators/kpi_dashboard.py": "# kpi_dashboard.py — KPIs et tableaux de bord décisionnels",
    # Analyse exploratoire
    "notebooks/01_exploration.ipynb": "",
    "notebooks/02_feature_engineering.ipynb": "",
    "notebooks/03_modeling.ipynb": "",
    # Tests
    "tests/test_preprocessing.py": "# test_preprocessing.py — Tests unitaires du nettoyage",
    "tests/test_model.py": "# test_model.py — Tests unitaires du modèle",
    # Config
    "config/settings.yaml": "# settings.yaml — Configuration globale du projet",
    # Fichiers racine
    "README.md": "# RiskMap_AI\n\nIA de prédiction des zones à risque urbain et système d'alerte géolocalisé.",
    "requirements.txt": "# Généré par : pip freeze > requirements.txt",
    ".gitignore": "venv/\n__pycache__/\n*.pyc\n.env\ndata/raw/\n*.pkl\n*.csv\n",
}

def creer_fichiers():
    print("📄 Création des fichiers du projet RiskMap_AI...\n")
    for chemin, contenu in FICHIERS.items():
        # Ne pas écraser un fichier qui existe déjà
        if not os.path.exists(chemin):
            # Créer le dossier parent si nécessaire
            os.makedirs(os.path.dirname(chemin), exist_ok=True) if os.path.dirname(chemin) else None
            with open(chemin, "w", encoding="utf-8") as f:
                f.write(contenu + "\n" if contenu else "")
            print(f"  ✅ Créé : {chemin}")
        else:
            print(f"  ⏭️  Existe déjà (non écrasé) : {chemin}")
    print("\n✅ Tous les fichiers ont été créés !")

if __name__ == "__main__":
    creer_fichiers()