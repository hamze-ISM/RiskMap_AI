# =============================================================================
# Construction des zones géographiques (Grid System)
# =============================================================================

import pandas as pd
import numpy as np
import os

# =============================================================================
# CONFIGURATION DE LA GRILLE
# =============================================================================

# Limites géographiques de Chicago (identiques au filtre GPS de la Phase 5)
LAT_MIN, LAT_MAX = 41.6, 42.1
LON_MIN, LON_MAX = -87.95, -87.5

# Nombre de cases par axe.
# 50 x 50 = 2500 zones au total, soit environ 1 km² par zone.
# Tu peux augmenter (100x100) pour plus de précision,
# ou diminuer (20x20) si tu manques de données.
NB_CASES = 50

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

print("=" * 60)
print("PHASE 7 — Construction des zones géographiques")
print("=" * 60)

print("\nChargement de crime_clean.csv...")
df = pd.read_csv(
    os.path.join("data", "processed", "crime_clean.csv"),
    low_memory=False
)
print(f"  {len(df):,} lignes chargées")

# =============================================================================
# SECTION 1 — Création de la grille
# =============================================================================

print("\n[1/4] Création de la grille...")

# np.linspace crée NB_CASES+1 valeurs régulièrement espacées entre min et max.
# Ces valeurs sont les "bordures" des cases de la grille.
# Exemple avec NB_CASES=3 :
#   lat_bins = [41.6, 41.767, 41.933, 42.1]
#   → 3 intervalles : [41.6–41.767], [41.767–41.933], [41.933–42.1]

lat_bins = np.linspace(LAT_MIN, LAT_MAX, NB_CASES + 1)
lon_bins = np.linspace(LON_MIN, LON_MAX, NB_CASES + 1)

# pd.cut assigne à chaque crime le numéro de la case correspondante.
# labels=False retourne un entier (0 à NB_CASES-1) au lieu d'un intervalle texte.
# include_lowest=True inclut la borne inférieure dans la première case.

df["zone_lat"] = pd.cut(
    df["latitude"],
    bins=lat_bins,
    labels=False,
    include_lowest=True
)
df["zone_lon"] = pd.cut(
    df["longitude"],
    bins=lon_bins,
    labels=False,
    include_lowest=True
)

# On supprime les rares lignes dont les coordonnées seraient hors de la grille
nb_avant = len(df)
df = df.dropna(subset=["zone_lat", "zone_lon"])
nb_apres = len(df)
print(f"  Lignes hors grille supprimées : {nb_avant - nb_apres}")

# On convertit les numéros de zones en entiers
df["zone_lat"] = df["zone_lat"].astype(int)
df["zone_lon"] = df["zone_lon"].astype(int)

# On crée un identifiant unique de zone : "LAT_LON"
# Exemple : zone_lat=12, zone_lon=34 → zone_id = "12_34"
df["zone_id"] = df["zone_lat"].astype(str) + "_" + df["zone_lon"].astype(str)

nb_zones = df["zone_id"].nunique()
print(f"  Grille : {NB_CASES} x {NB_CASES} = {NB_CASES**2} zones possibles")
print(f"  Zones réellement occupées : {nb_zones} (zones avec au moins 1 crime)")

# =============================================================================
# SECTION 2 — Calcul du score de risque par zone
# =============================================================================

print("\n[2/4] Calcul du score de risque par zone...")

# On compte le nombre de crimes par zone
crimes_par_zone = df.groupby("zone_id").size().reset_index(name="nb_crimes")

# On calcule le score de risque normalisé entre 0 et 1.
# min-max normalisation : (valeur - min) / (max - min)
# Ainsi la zone la moins criminelle → 0.0, la plus criminelle → 1.0

min_crimes = crimes_par_zone["nb_crimes"].min()
max_crimes = crimes_par_zone["nb_crimes"].max()

crimes_par_zone["score_risque_brut"] = (
    (crimes_par_zone["nb_crimes"] - min_crimes) /
    (max_crimes - min_crimes)
)

# On classifie en 3 niveaux : Faible / Moyen / Élevé
# Les seuils 0.33 et 0.66 découpent en trois tiers égaux.
# Tu peux les ajuster si tu veux plus de zones rouges ou vertes.

def classifier_risque(score):
    if score < 0.33:
        return "Faible"
    elif score < 0.66:
        return "Moyen"
    else:
        return "Élevé"

crimes_par_zone["niveau_risque"] = crimes_par_zone["score_risque_brut"].apply(classifier_risque)

# Affichage de la distribution des niveaux de risque
distribution = crimes_par_zone["niveau_risque"].value_counts()
print(f"\n  Distribution des zones :")
for niveau, nb in distribution.items():
    print(f"    - {niveau} : {nb} zones")

# =============================================================================
# SECTION 3 — Ajout des coordonnées du centre de chaque zone
# =============================================================================

print("\n[3/4] Calcul des centres de zones...")

# Pour chaque zone_id, on calcule la latitude/longitude moyenne des crimes.
# Ce sera le point central affiché sur la carte Folium en Phase 11.

centres = df.groupby("zone_id").agg(
    lat_centre=("latitude", "mean"),
    lon_centre=("longitude", "mean")
).reset_index()

# On fusionne les centres avec le tableau des scores de risque
crimes_par_zone = crimes_par_zone.merge(centres, on="zone_id", how="left")

# Affichage des 5 zones les plus dangereuses
print("\n  Top 5 zones les plus dangereuses :")
top5 = crimes_par_zone.nlargest(5, "nb_crimes")[
    ["zone_id", "nb_crimes", "score_risque_brut", "niveau_risque", "lat_centre", "lon_centre"]
]
print(top5.to_string(index=False))

# =============================================================================
# SECTION 4 — Sauvegarde
# =============================================================================

print("\n[4/4] Sauvegarde des fichiers...")

# On sauvegarde le dataset enrichi avec les colonnes de zone
chemin_avec_zones = os.path.join("data", "processed", "crime_with_zones.csv")
df.to_csv(chemin_avec_zones, index=False)
print(f"  Sauvegardé : {chemin_avec_zones}")

# On sauvegarde la table des zones avec les scores de risque
chemin_zones = os.path.join("data", "processed", "zones_risque.csv")
crimes_par_zone.to_csv(chemin_zones, index=False)
print(f"  Sauvegardé : {chemin_zones}")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================

print("\n" + "=" * 60)
print("RÉSUMÉ PHASE 7")
print("=" * 60)
print(f"  Dataset enrichi : {len(df):,} lignes avec zone_lat, zone_lon, zone_id")
print(f"  Table des zones : {len(crimes_par_zone)} zones avec score de risque")
print(f"  Nouvelles colonnes créées : zone_lat, zone_lon, zone_id")
print(f"  Fichiers produits :")
print(f"    - crime_with_zones.csv")
print(f"    - zones_risque.csv")
print("\nPhase 7 terminée avec succès.")
print("Tu peux maintenant passer à la Phase 8 — Feature Engineering.")