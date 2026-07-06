# =============================================================================
# PHASE 10 — Prédictions
# Fichier : src/models/predict.py
# =============================================================================

import pandas as pd
import numpy as np
import os
import joblib

# =============================================================================
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# =============================================================================

print("=" * 60)
print("PHASE 10 — Prédictions")
print("=" * 60)

# Chargement du modèle sauvegardé en Phase 9
print("\nChargement du modèle...")
modele   = joblib.load(os.path.join("models", "best_model.pkl"))
features = joblib.load(os.path.join("models", "features_list.pkl"))

with open(os.path.join("models", "model_info.txt")) as f:
    infos = dict(line.strip().split("=") for line in f)

print(f"  Modèle chargé : {infos['modele']} (F1 = {infos['f1']})")
print(f"  Features attendues : {len(features)}")

# Chargement du dataset enrichi
print("\nChargement de crime_features.csv...")
df = pd.read_csv(
    os.path.join("data", "processed", "crime_features.csv"),
    low_memory=False
)
print(f"  {len(df):,} lignes chargées")

# Chargement des zones avec coordonnées GPS
zones = pd.read_csv(os.path.join("data", "processed", "zones_risque.csv"))
print(f"  {len(zones)} zones géographiques chargées")

# =============================================================================
# SECTION 1 — Prédictions sur les données historiques
# =============================================================================
# On prédit le niveau de risque sur toutes les lignes du dataset.
# Cela permet de vérifier que le modèle se comporte correctement
# et de produire un fichier de prédictions complet.

print("\n[1/4] Prédictions sur les données historiques...")

# On s'assure que les features sont dans le bon ordre
X = df[features]

# Prédiction des classes (0=Faible, 1=Moyen, 2=Élevé)
df["risque_predit"] = modele.predict(X)

# Prédiction des probabilités pour chaque classe
# Cela permet d'afficher un score de confiance sur la carte
proba = modele.predict_proba(X)
df["proba_faible"] = proba[:, 0].round(4)
df["proba_moyen"]  = proba[:, 1].round(4)
df["proba_eleve"]  = proba[:, 2].round(4)

# Conversion en label lisible
mapping = {0: "Faible", 1: "Moyen", 2: "Élevé"}
df["risque_label"] = df["risque_predit"].map(mapping)

# Vérification de la distribution des prédictions
dist = df["risque_label"].value_counts()
print("\n  Distribution des prédictions :")
for label, nb in dist.items():
    print(f"    - {label} : {nb:,} ({nb/len(df)*100:.1f}%)")

# =============================================================================
# SECTION 2 — Agrégation par zone (carte de risque)
# =============================================================================
# Pour la carte Folium (Phase 11), on a besoin d'un risque par zone,
# pas par incident. On agrège en prenant le risque le plus fréquent
# dans chaque zone (mode statistique).

print("\n[2/4] Agrégation par zone géographique...")

risque_par_zone = (
    df.groupby("zone_id")
    .agg(
        risque_predominant=("risque_predit", lambda x: x.mode()[0]),
        proba_eleve_moy=("proba_eleve", "mean"),
        proba_moyen_moy=("proba_moyen", "mean"),
        proba_faible_moy=("proba_faible", "mean"),
        nb_incidents=("risque_predit", "count")
    )
    .reset_index()
)

# Ajout des coordonnées GPS des centres de zones
risque_par_zone = risque_par_zone.merge(
    zones[["zone_id", "lat_centre", "lon_centre"]],
    on="zone_id",
    how="left"
)

# Label lisible
risque_par_zone["risque_label"] = risque_par_zone["risque_predominant"].map(mapping)

# Couleur pour la carte Folium
couleurs = {0: "green", 1: "orange", 2: "red"}
risque_par_zone["couleur"] = risque_par_zone["risque_predominant"].map(couleurs)

print(f"  {len(risque_par_zone)} zones avec prédiction de risque")
print("\n  Distribution des zones prédites :")
dist_zones = risque_par_zone["risque_label"].value_counts()
for label, nb in dist_zones.items():
    print(f"    - {label} : {nb} zones")

# Top 10 zones les plus dangereuses
print("\n  Top 10 zones prédites comme Élevé :")
top10 = (
    risque_par_zone[risque_par_zone["risque_label"] == "Élevé"]
    .nlargest(10, "proba_eleve_moy")
    [["zone_id", "nb_incidents", "proba_eleve_moy", "lat_centre", "lon_centre"]]
)
print(top10.to_string(index=False))

# =============================================================================
# SECTION 3 — Grille horaire (prédictions par heure)
# =============================================================================
# On génère une prédiction pour chaque combinaison zone × heure.
# Utile pour le dashboard (filtrer par heure de la journée).

print("\n[3/4] Génération de la grille zone × heure...")

# On calcule les statistiques moyennes par zone et par heure
grille = (
    df.groupby(["zone_id", "heure"])
    .agg(
        risque_predit=("risque_predit", lambda x: x.mode()[0]),
        proba_eleve=("proba_eleve", "mean"),
        nb_incidents=("risque_predit", "count")
    )
    .reset_index()
)

grille["risque_label"] = grille["risque_predit"].map(mapping)
grille["couleur"]      = grille["risque_predit"].map(couleurs)

# Fusion avec les coordonnées GPS
grille = grille.merge(
    zones[["zone_id", "lat_centre", "lon_centre"]],
    on="zone_id",
    how="left"
)

print(f"  {len(grille):,} combinaisons zone × heure générées")

# =============================================================================
# SECTION 4 — Sauvegarde
# =============================================================================

print("\n[4/4] Sauvegarde des fichiers de prédictions...")

# Prédictions complètes (toutes les lignes)
chemin_pred = os.path.join("data", "processed", "predictions.csv")
df[["zone_id", "heure", "jour_semaine", "mois", "annee_source",
    "risque_predit", "risque_label",
    "proba_faible", "proba_moyen", "proba_eleve"]].to_csv(chemin_pred, index=False)
print(f"  Sauvegardé : {chemin_pred}")

# Carte de risque par zone (utilisée en Phase 11)
chemin_carte = os.path.join("data", "processed", "risk_map_data.csv")
risque_par_zone.to_csv(chemin_carte, index=False)
print(f"  Sauvegardé : {chemin_carte}")

# Grille zone × heure (utilisée en Phase 15 — Dashboard)
chemin_grille = os.path.join("data", "processed", "grille_horaire.csv")
grille.to_csv(chemin_grille, index=False)
print(f"  Sauvegardé : {chemin_grille}")

# =============================================================================
# RÉSUMÉ
# =============================================================================

print("\n" + "=" * 60)
print("RÉSUMÉ PHASE 10")
print("=" * 60)
print(f"  Prédictions générées     : {len(df):,} incidents")
print(f"  Zones avec carte risque  : {len(risque_par_zone)}")
print(f"  Combinaisons zone×heure  : {len(grille):,}")
print(f"\n  Fichiers produits :")
print(f"    - predictions.csv      (prédictions complètes)")
print(f"    - risk_map_data.csv    (carte de risque par zone)")
print(f"    - grille_horaire.csv   (prédictions par heure)")
print("\nPhase 10 terminée avec succès.")
print("Tu peux maintenant passer à la Phase 11 — Cartographie Folium.")