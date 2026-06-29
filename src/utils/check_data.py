# check_data.py
# Vérifie que les fichiers CSV sont bien présents et valides

import os
import pandas as pd

# Chemins des fichiers
FICHIERS = {
    "2024": "data/raw/2024/chicago_crimes_2024.csv",
    "2025": "data/raw/2025/chicago_crimes_2025.csv",
    "2026": "data/raw/2026/chicago_crimes_2026.csv",
}

# Colonnes obligatoires attendues
COLONNES_ATTENDUES = [
    "ID", "Case Number", "Date", "Block", "IUCR",
    "Primary Type", "Description", "Location Description",
    "Arrest", "Domestic", "Beat", "District", "Ward",
    "Community Area", "FBI Code", "X Coordinate", "Y Coordinate",
    "Year", "Updated On", "Latitude", "Longitude", "Location"
]

def verifier_fichier(annee, chemin):
    print(f"\n{'='*50}")
    print(f"📂 Vérification du fichier {annee} : {chemin}")

    # 1. Le fichier existe-t-il ?
    if not os.path.exists(chemin):
        print(f"  ❌ FICHIER MANQUANT : {chemin}")
        print(f"     → Télécharge-le depuis data.cityofchicago.org")
        return False

    # 2. Taille du fichier
    taille_mb = os.path.getsize(chemin) / (1024 * 1024)
    print(f"  📏 Taille : {taille_mb:.1f} MB")

    # 3. Lecture du fichier
    try:
        df = pd.read_csv(chemin, low_memory=False)
    except Exception as e:
        print(f"  ❌ ERREUR DE LECTURE : {e}")
        return False

    # 4. Dimensions
    print(f"  📊 Dimensions : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    # 5. Colonnes présentes
    colonnes_manquantes = [c for c in COLONNES_ATTENDUES if c not in df.columns]
    if colonnes_manquantes:
        print(f"  ⚠️  Colonnes manquantes : {colonnes_manquantes}")
    else:
        print(f"  ✅ Toutes les colonnes attendues sont présentes")

    # 6. Aperçu des colonnes réelles
    print(f"  📋 Colonnes disponibles : {list(df.columns)}")

    # 7. Valeurs nulles dans les colonnes clés
    cles = ["Date", "Primary Type", "Latitude", "Longitude", "District"]
    print(f"\n  🔍 Valeurs nulles dans les colonnes clés :")
    for col in cles:
        if col in df.columns:
            nb_nulls = df[col].isnull().sum()
            pct = (nb_nulls / len(df)) * 100
            statut = "⚠️ " if pct > 5 else "✅"
            print(f"     {statut} {col} : {nb_nulls:,} nulls ({pct:.1f}%)")

    # 8. Distribution des types de crimes
    if "Primary Type" in df.columns:
        print(f"\n  🔍 Top 5 types de crimes :")
        top5 = df["Primary Type"].value_counts().head(5)
        for crime, count in top5.items():
            print(f"     - {crime} : {count:,}")

    # 9. Plage de dates
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        print(f"\n  📅 Plage de dates : {df['Date'].min()} → {df['Date'].max()}")

    print(f"\n  ✅ Fichier {annee} OK")
    return True

def main():
    print("🚀 Vérification des données Chicago Crime Dataset")
    print("="*50)

    resultats = {}
    for annee, chemin in FICHIERS.items():
        resultats[annee] = verifier_fichier(annee, chemin)

    print(f"\n{'='*50}")
    print("📊 RÉSUMÉ DE LA VÉRIFICATION :")
    for annee, ok in resultats.items():
        statut = "✅ OK" if ok else "❌ MANQUANT ou INVALIDE"
        print(f"  {annee} : {statut}")

    tous_ok = all(resultats.values())
    if tous_ok:
        print("\n✅ Tous les fichiers sont valides. Tu peux passer à la Phase 5.")
    else:
        print("\n⚠️  Certains fichiers manquent. Télécharge-les avant de continuer.")

if __name__ == "__main__":
    main()