import pandas as pd
import numpy as np
import os

# --- Chemins ---
CHEMIN_2024 = os.path.join("data", "raw", "2024", "crime_2024.csv")
CHEMIN_2025 = os.path.join("data", "raw", "2025", "crime_2025.csv")
CHEMIN_2026 = os.path.join("data", "raw", "2026", "crime_2026.csv")
DOSSIER_SORTIE = os.path.join("data", "processed")

print("=" * 60)
print("PHASE 5 - Fusion et nettoyage des donnees")
print("=" * 60)

# --- [1/8] Chargement ---
print("\n[1/8] Chargement des datasets...")
df_2024 = pd.read_csv(CHEMIN_2024, low_memory=False)
df_2025 = pd.read_csv(CHEMIN_2025, low_memory=False)
df_2026 = pd.read_csv(CHEMIN_2026, low_memory=False)
print(f"  - 2024 : {len(df_2024):,} lignes, {df_2024.shape[1]} colonnes")
print(f"  - 2025 : {len(df_2025):,} lignes, {df_2025.shape[1]} colonnes")
print(f"  - 2026 : {len(df_2026):,} lignes, {df_2026.shape[1]} colonnes")

# --- [2/8] Normalisation ---
print("\n[2/8] Normalisation des colonnes...")

def normaliser(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df

df_2024 = normaliser(df_2024)
df_2025 = normaliser(df_2025)
df_2026 = normaliser(df_2026)
df_2024["annee_source"] = 2024
df_2025["annee_source"] = 2025
df_2026["annee_source"] = 2026
print("  OK")

# --- [3/8] Fusion ---
print("\n[3/8] Fusion...")
df = pd.concat([df_2024, df_2025, df_2026], ignore_index=True)
print(f"  Total fusionne : {len(df):,} lignes")
os.makedirs(DOSSIER_SORTIE, exist_ok=True)
df.to_csv(os.path.join(DOSSIER_SORTIE, "crime_merged.csv"), index=False)
print("  crime_merged.csv sauvegarde")

# --- [4/8] Doublons ---
print("\n[4/8] Suppression des doublons...")
nb_avant = len(df)
if "id" in df.columns:
    df = df.drop_duplicates(subset=["id"])
else:
    df = df.drop_duplicates()
print(f"  Supprimes : {nb_avant - len(df):,} | Restants : {len(df):,}")

# --- [5/8] Valeurs manquantes ---
print("\n[5/8] Valeurs manquantes...")
pct = (df.isnull().sum() / len(df) * 100).round(2)
manquants = pct[pct > 0].sort_values(ascending=False)
if len(manquants) > 0:
    for col, p in manquants.items():
        print(f"  - {col} : {p}%")
else:
    print("  Aucune valeur manquante")

# --- [6/8] Nettoyage GPS ---
print("\n[6/8] Nettoyage GPS...")
col_lat = next((c for c in ["latitude", "lat", "y"] if c in df.columns), None)
col_lon = next((c for c in ["longitude", "lon", "long", "x"] if c in df.columns), None)

if col_lat and col_lon:
    # Remplacement des virgules par des points (format europeen -> format anglais)
    df[col_lat] = df[col_lat].astype(str).str.replace(",", ".", regex=False)
    df[col_lon] = df[col_lon].astype(str).str.replace(",", ".", regex=False)
    # Conversion en numerique
    df[col_lat] = pd.to_numeric(df[col_lat], errors="coerce")
    df[col_lon] = pd.to_numeric(df[col_lon], errors="coerce")
    # Diagnostic
    print(f"  Latitude  - min: {df[col_lat].min():.4f}, max: {df[col_lat].max():.4f}, NaN: {df[col_lat].isna().sum():,}")
    print(f"  Longitude - min: {df[col_lon].min():.4f}, max: {df[col_lon].max():.4f}, NaN: {df[col_lon].isna().sum():,}")
    # Filtrage dans les limites de Chicago
    nb = len(df)
    masque = (
        df[col_lat].between(41.6, 42.1) &
        df[col_lon].between(-87.95, -87.5)
    )
    if masque.sum() == 0:
        print("  ATTENTION : filtre Chicago vide - on garde toutes les coordonnees non-nulles")
        df = df[df[col_lat].notna() & df[col_lon].notna()].copy()
    else:
        df = df[masque].copy()
    print(f"  Supprimes hors Chicago : {nb - len(df):,} | Restants : {len(df):,}")
else:
    print(f"  ATTENTION : colonnes GPS introuvables. Colonnes disponibles : {list(df.columns)}")

# --- [7/8] Dates ---
print("\n[7/8] Nettoyage des dates...")
col_date = next((c for c in ["date", "date_of_occurrence", "occurred_on", "incident_date"] if c in df.columns), None)
if col_date:
    df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
    nb_sans_date = df[col_date].isnull().sum()
    df = df[df[col_date].notna()].copy()
    print(f"  Sans date supprimees : {nb_sans_date:,}")
    df["annee"]        = df[col_date].dt.year
    df["mois"]         = df[col_date].dt.month
    df["jour"]         = df[col_date].dt.day
    df["heure"]        = df[col_date].dt.hour
    df["jour_semaine"] = df[col_date].dt.dayofweek
    df["weekend"]      = df["jour_semaine"].isin([5, 6]).astype(int)
    print("  Colonnes creees : annee, mois, jour, heure, jour_semaine, weekend")
else:
    print(f"  ATTENTION : colonne date introuvable. Colonnes disponibles : {list(df.columns)}")

# --- [8/8] Suppression colonnes inutiles ---
print("\n[8/8] Suppression colonnes inutiles...")
a_supprimer = [
    "case_number", "updated_on", "fbi_code", "beat",
    "ward", "community_area", "x_coordinate", "y_coordinate", "location"
]
supprimes = [c for c in a_supprimer if c in df.columns]
df = df.drop(columns=supprimes)
print(f"  Supprimees : {supprimes}")

# --- Résumé final ---
print("\n" + "=" * 60)
print("RESUME FINAL")
print("=" * 60)
print(f"  Lignes finales  : {len(df):,}")
print(f"  Nombre colonnes : {df.shape[1]}")
print(f"  Colonnes        : {list(df.columns)}")
print(f"\n  Apercu des 3 premieres lignes :")
print(df.head(3).to_string())

df.to_csv(os.path.join(DOSSIER_SORTIE, "crime_clean.csv"), index=False)
print("\n  crime_clean.csv sauvegarde")
print("\nPhase 5 terminee avec succes.")