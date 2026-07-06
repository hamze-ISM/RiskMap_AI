# =============================================================================
# PHASE 8 — Feature Engineering
# Fichier : src/features/feature_engineering.py
# =============================================================================

import pandas as pd
import numpy as np
import os

# =============================================================================
# CHARGEMENT
# =============================================================================

print("=" * 60)
print("PHASE 8 — Feature Engineering")
print("=" * 60)

print("\nChargement de crime_with_zones.csv...")
df = pd.read_csv(
    os.path.join("data", "processed", "crime_with_zones.csv"),
    low_memory=False
)

# Reconvertir la date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print(f"  {len(df):,} lignes | {df.shape[1]} colonnes")

# =============================================================================
# SECTION 1 — Correction de la classification du risque (quantiles)
# =============================================================================
# On corrige le déséquilibre détecté en Phase 7.
# Au lieu de min-max (qui concentre tout en "Faible"),
# on utilise les quantiles : chaque tiers reçoit exactement 1/3 des zones.

print("\n[1/6] Classification du risque par quantiles...")

# On compte les crimes par zone
crimes_par_zone = df.groupby("zone_id").size().reset_index(name="nb_crimes_zone")
df = df.merge(crimes_par_zone, on="zone_id", how="left")

# Seuils basés sur les quantiles 33% et 66%
q33 = df["nb_crimes_zone"].quantile(0.33)
q66 = df["nb_crimes_zone"].quantile(0.66)

print(f"  Seuil Faible/Moyen : {q33:.0f} crimes par zone")
print(f"  Seuil Moyen/Élevé  : {q66:.0f} crimes par zone")

def classifier_quantile(n):
    if n <= q33:
        return 0   # Faible
    elif n <= q66:
        return 1   # Moyen
    else:
        return 2   # Élevé

df["niveau_risque"] = df["nb_crimes_zone"].apply(classifier_quantile)

# Vérification de la distribution
dist = df["niveau_risque"].value_counts().sort_index()
labels = {0: "Faible", 1: "Moyen", 2: "Élevé"}
print("\n  Distribution après correction :")
for k, v in dist.items():
    print(f"    - {labels[k]} ({k}) : {v:,} lignes ({v/len(df)*100:.1f}%)")

# =============================================================================
# SECTION 2 — Features temporelles
# =============================================================================
# Ces colonnes existent déjà depuis la Phase 5.
# On en crée de nouvelles plus fines.

print("\n[2/6] Features temporelles...")

# Partie de la journée (4 créneaux)
# 0=Nuit (0-5h), 1=Matin (6-11h), 2=Après-midi (12-17h), 3=Soir (18-23h)
def partie_journee(heure):
    if heure < 6:
        return 0
    elif heure < 12:
        return 1
    elif heure < 18:
        return 2
    else:
        return 3

df["partie_journee"] = df["heure"].apply(partie_journee)

# Saison (1=Hiver, 2=Printemps, 3=Été, 4=Automne)
def saison(mois):
    if mois in [12, 1, 2]:
        return 1
    elif mois in [3, 4, 5]:
        return 2
    elif mois in [6, 7, 8]:
        return 3
    else:
        return 4

df["saison"] = df["mois"].apply(saison)

# Heure normalisée (cyclique) — important pour que le modèle comprenne
# que 23h et 0h sont proches, pas opposées
df["heure_sin"] = np.sin(2 * np.pi * df["heure"] / 24)
df["heure_cos"] = np.cos(2 * np.pi * df["heure"] / 24)

# Mois normalisé (cyclique)
df["mois_sin"] = np.sin(2 * np.pi * df["mois"] / 12)
df["mois_cos"] = np.cos(2 * np.pi * df["mois"] / 12)

print("  Créées : partie_journee, saison, heure_sin, heure_cos, mois_sin, mois_cos")

# =============================================================================
# SECTION 3 — Features de densité par zone
# =============================================================================
# On calcule pour chaque zone des statistiques agrégées.
# Ces features donnent au modèle le "contexte historique" de chaque zone.

print("\n[3/6] Features de densité par zone...")

# Densité par zone et par heure
densite_zone_heure = (
    df.groupby(["zone_id", "heure"])
    .size()
    .reset_index(name="incidents_zone_heure")
)
df = df.merge(densite_zone_heure, on=["zone_id", "heure"], how="left")

# Densité par zone et par partie de journée
densite_zone_partie = (
    df.groupby(["zone_id", "partie_journee"])
    .size()
    .reset_index(name="incidents_zone_partie_journee")
)
df = df.merge(densite_zone_partie, on=["zone_id", "partie_journee"], how="left")

# Densité par zone et par jour de la semaine
densite_zone_jour = (
    df.groupby(["zone_id", "jour_semaine"])
    .size()
    .reset_index(name="incidents_zone_jour")
)
df = df.merge(densite_zone_jour, on=["zone_id", "jour_semaine"], how="left")

print("  Créées : incidents_zone_heure, incidents_zone_partie_journee, incidents_zone_jour")

# =============================================================================
# SECTION 4 — Features sur le type de crime
# =============================================================================

print("\n[4/6] Encodage du type de crime...")

if "primary_type" in df.columns:
    # Fréquence de chaque type de crime (frequency encoding)
    # Au lieu d'un one-hot encoding qui créerait 31 colonnes,
    # on remplace chaque type par sa fréquence dans le dataset.
    # Le modèle apprend ainsi que THEFT est plus fréquent que ARSON.
    freq_type = df["primary_type"].value_counts(normalize=True)
    df["freq_type_crime"] = df["primary_type"].map(freq_type)
    print("  Créée : freq_type_crime (frequency encoding)")
else:
    print("  Colonne 'primary_type' absente — ignorée")

# Arrestation (déjà booléen, on s'assure qu'il est en int)
if "arrest" in df.columns:
    df["arrest"] = df["arrest"].astype(int)
    print("  Colonne 'arrest' convertie en int (0/1)")

# Incident domestique
if "domestic" in df.columns:
    df["domestic"] = df["domestic"].astype(int)
    print("  Colonne 'domestic' convertie en int (0/1)")

# =============================================================================
# SECTION 5 — Sélection des features finales
# =============================================================================

print("\n[5/6] Sélection des features pour le modèle ML...")

# Liste des colonnes à conserver pour le Machine Learning.
# On exclut les colonnes texte brut, les identifiants, et les colonnes
# qui "trichent" (nb_crimes_zone contient déjà la réponse).

FEATURES = [
    # Localisation
    "zone_lat",
    "zone_lon",
    # Temporel
    "heure",
    "heure_sin",
    "heure_cos",
    "mois",
    "mois_sin",
    "mois_cos",
    "jour_semaine",
    "weekend",
    "partie_journee",
    "saison",
    "annee_source",
    # Densité historique
    "incidents_zone_heure",
    "incidents_zone_partie_journee",
    "incidents_zone_jour",
    "nb_crimes_zone",
    # Type de crime
    "freq_type_crime",
    "arrest",
    "domestic",
]

# On garde uniquement les features qui existent réellement dans df
FEATURES = [f for f in FEATURES if f in df.columns]
CIBLE = "niveau_risque"

print(f"  {len(FEATURES)} features sélectionnées :")
for f in FEATURES:
    print(f"    - {f}")

# =============================================================================
# SECTION 6 — Sauvegarde
# =============================================================================

print("\n[6/6] Sauvegarde...")

# Dataset complet enrichi
chemin_complet = os.path.join("data", "processed", "crime_features.csv")
df.to_csv(chemin_complet, index=False)
print(f"  Sauvegardé : {chemin_complet}")

# Dataset ML uniquement (features + cible) — plus léger, utilisé en Phase 9
df_ml = df[FEATURES + [CIBLE]].dropna()
chemin_ml = os.path.join("data", "processed", "features.csv")
df_ml.to_csv(chemin_ml, index=False)
print(f"  Sauvegardé : {chemin_ml}")

# =============================================================================
# RÉSUMÉ
# =============================================================================

print("\n" + "=" * 60)
print("RÉSUMÉ PHASE 8")
print("=" * 60)
print(f"  Lignes dans features.csv     : {len(df_ml):,}")
print(f"  Nombre de features           : {len(FEATURES)}")
print(f"  Cible (niveau_risque)        : 0=Faible | 1=Moyen | 2=Élevé")
print(f"  Distribution cible :")
for k, v in df_ml[CIBLE].value_counts().sort_index().items():
    print(f"    - {labels[k]} : {v:,} ({v/len(df_ml)*100:.1f}%)")

print(f"\n  Fichiers produits :")
print(f"    - crime_features.csv (dataset complet enrichi)")
print(f"    - features.csv (prêt pour le ML)")
print("\nPhase 8 terminée avec succès.")
print("Tu peux maintenant passer à la Phase 9 — Machine Learning.")