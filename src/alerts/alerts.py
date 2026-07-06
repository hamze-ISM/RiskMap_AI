# =============================================================================
# PHASE 12 — Système d'alertes automatiques
# Fichier : src/alerts/alerts.py
# =============================================================================

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

DOSSIER_ALERTES = os.path.join("reports", "alerts")
os.makedirs(DOSSIER_ALERTES, exist_ok=True)

# Seuils de déclenchement des alertes
SEUIL_ALERTE_ELEVEE  = 0.75   # prob. Élevé > 75% → alerte rouge
SEUIL_ALERTE_MOYENNE = 0.50   # prob. Élevé > 50% → alerte orange

# Heure simulée pour les alertes (tu peux la modifier pour tester)
HEURE_SIMULATION = 22   # 22h — heure de pointe nocturne

# =============================================================================
# CHARGEMENT
# =============================================================================

print("=" * 60)
print("PHASE 12 — Système d'alertes automatiques")
print("=" * 60)

print("\nChargement des données...")
zones   = pd.read_csv(os.path.join("data", "processed", "risk_map_data.csv"))
grille  = pd.read_csv(os.path.join("data", "processed", "grille_horaire.csv"))
print(f"  {len(zones)} zones | {len(grille):,} combinaisons zone×heure")

# =============================================================================
# SECTION 1 — Moteur de classification des alertes
# =============================================================================

print("\n[1/4] Classification des alertes...")

def classifier_alerte(prob_eleve, prob_moyen, nb_incidents):
    """
    Retourne le niveau d'alerte, le message et la recommandation
    selon les probabilités prédites par le modèle.
    """
    if prob_eleve >= SEUIL_ALERTE_ELEVEE:
        return {
            "niveau"         : "ROUGE",
            "emoji"          : "🔴",
            "priorite"       : 1,
            "message"        : "ALERTE CRITIQUE — Zone à risque élevé détectée",
            "recommandation" : "Déploiement immédiat de patrouilles recommandé. "
                               "Éviter la zone si possible.",
            "action"         : "INTERVENTION_IMMEDIATE"
        }
    elif prob_eleve >= SEUIL_ALERTE_MOYENNE:
        return {
            "niveau"         : "ORANGE",
            "emoji"          : "🟠",
            "priorite"       : 2,
            "message"        : "ALERTE MODÉRÉE — Surveillance renforcée requise",
            "recommandation" : "Augmenter la fréquence des patrouilles. "
                               "Vigilance accrue recommandée.",
            "action"         : "SURVEILLANCE_RENFORCEE"
        }
    elif prob_moyen >= SEUIL_ALERTE_MOYENNE:
        return {
            "niveau"         : "JAUNE",
            "emoji"          : "🟡",
            "priorite"       : 3,
            "message"        : "AVIS DE PRÉCAUTION — Zone à surveiller",
            "recommandation" : "Maintenir la présence habituelle. "
                               "Aucune action immédiate requise.",
            "action"         : "SURVEILLANCE_NORMALE"
        }
    else:
        return {
            "niveau"         : "VERT",
            "emoji"          : "🟢",
            "priorite"       : 4,
            "message"        : "Zone calme — Aucune alerte active",
            "recommandation" : "Situation normale.",
            "action"         : "AUCUNE_ACTION"
        }

# Application sur toutes les zones
alertes_zones = []
for _, zone in zones.iterrows():
    infos_alerte = classifier_alerte(
        zone["proba_eleve_moy"],
        zone["proba_moyen_moy"],
        zone["nb_incidents"]
    )
    alertes_zones.append({
        "zone_id"       : zone["zone_id"],
        "lat"           : zone["lat_centre"],
        "lon"           : zone["lon_centre"],
        "nb_incidents"  : int(zone["nb_incidents"]),
        "prob_eleve"    : round(zone["proba_eleve_moy"], 4),
        "prob_moyen"    : round(zone["proba_moyen_moy"], 4),
        **infos_alerte
    })

df_alertes = pd.DataFrame(alertes_zones).sort_values("priorite")

# Distribution
print("\n  Distribution des alertes :")
for niveau, nb in df_alertes["niveau"].value_counts().items():
    emoji = df_alertes[df_alertes["niveau"] == niveau]["emoji"].iloc[0]
    print(f"    {emoji} {niveau} : {nb} zones")

# =============================================================================
# SECTION 2 — Alertes horaires (pour l'heure simulée)
# =============================================================================

print(f"\n[2/4] Alertes pour {HEURE_SIMULATION}h00...")

# Filtrer la grille sur l'heure simulée
grille_heure = grille[grille["heure"] == HEURE_SIMULATION].copy()

# Fusionner avec les probabilités des zones
grille_heure = grille_heure.merge(
    zones[["zone_id", "proba_eleve_moy", "proba_moyen_moy", "nb_incidents"]],
    on="zone_id",
    how="left",
    suffixes=("_grille", "")
)

# Classifier les alertes horaires
alertes_heure = []
for _, row in grille_heure.iterrows():
    infos = classifier_alerte(
        row["proba_eleve_moy"],
        row["proba_moyen_moy"],
        row["nb_incidents"]
    )
    alertes_heure.append({
        "zone_id"    : row["zone_id"],
        "heure"      : int(row["heure"]),
        "lat"        : row["lat_centre"],
        "lon"        : row["lon_centre"],
        "prob_eleve" : round(row["proba_eleve_moy"], 4),
        **infos
    })

df_alertes_heure = pd.DataFrame(alertes_heure).sort_values("priorite")
zones_critiques = df_alertes_heure[df_alertes_heure["niveau"] == "ROUGE"]

print(f"  Zones analysées à {HEURE_SIMULATION}h : {len(df_alertes_heure)}")
print(f"  Alertes ROUGE actives    : {len(zones_critiques)}")

# =============================================================================
# SECTION 3 — Génération des messages d'alerte
# =============================================================================

print("\n[3/4] Génération des messages d'alerte...")

horodatage = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Rapport texte ---
lignes_rapport = [
    "=" * 65,
    "  RAPPORT D'ALERTES — RiskMap_AI",
    f"  Généré le : {horodatage}",
    f"  Heure analysée : {HEURE_SIMULATION}h00",
    "=" * 65,
    "",
]

# Top alertes ROUGE
rouges = df_alertes[df_alertes["niveau"] == "ROUGE"].head(10)
if len(rouges) > 0:
    lignes_rapport.append("🔴 ALERTES CRITIQUES (intervention immédiate)")
    lignes_rapport.append("-" * 45)
    for _, a in rouges.iterrows():
        lignes_rapport.append(
            f"  Zone {a['zone_id']:8s} | Prob. Élevé : {a['prob_eleve']:.1%} | "
            f"Incidents : {a['nb_incidents']:,}"
        )
        lignes_rapport.append(f"  → {a['recommandation']}")
        lignes_rapport.append("")

# Top alertes ORANGE
oranges = df_alertes[df_alertes["niveau"] == "ORANGE"].head(10)
if len(oranges) > 0:
    lignes_rapport.append("🟠 ALERTES MODÉRÉES (surveillance renforcée)")
    lignes_rapport.append("-" * 45)
    for _, a in oranges.iterrows():
        lignes_rapport.append(
            f"  Zone {a['zone_id']:8s} | Prob. Élevé : {a['prob_eleve']:.1%} | "
            f"Incidents : {a['nb_incidents']:,}"
        )
    lignes_rapport.append("")

# Statistiques globales
lignes_rapport += [
    "=" * 65,
    "STATISTIQUES GLOBALES",
    "-" * 45,
    f"  Total zones analysées : {len(df_alertes)}",
    f"  Zones ROUGE  : {len(df_alertes[df_alertes['niveau']=='ROUGE'])}",
    f"  Zones ORANGE : {len(df_alertes[df_alertes['niveau']=='ORANGE'])}",
    f"  Zones JAUNE  : {len(df_alertes[df_alertes['niveau']=='JAUNE'])}",
    f"  Zones VERT   : {len(df_alertes[df_alertes['niveau']=='VERT'])}",
    "=" * 65,
]

rapport_texte = "\n".join(lignes_rapport)
print(rapport_texte)

# --- Export JSON (pour le dashboard Streamlit) ---
alertes_json = df_alertes[df_alertes["niveau"].isin(["ROUGE", "ORANGE"])].to_dict(orient="records")

# =============================================================================
# SECTION 4 — Sauvegarde
# =============================================================================

print("\n[4/4] Sauvegarde...")

# Rapport texte
chemin_rapport = os.path.join(DOSSIER_ALERTES, "rapport_alertes.txt")
with open(chemin_rapport, "w", encoding="utf-8") as f:
    f.write(rapport_texte)
print(f"  Sauvegardé : {chemin_rapport}")

# CSV des alertes
chemin_csv = os.path.join(DOSSIER_ALERTES, "alertes.csv")
df_alertes.to_csv(chemin_csv, index=False)
print(f"  Sauvegardé : {chemin_csv}")

# JSON pour le dashboard
chemin_json = os.path.join(DOSSIER_ALERTES, "alertes_actives.json")
with open(chemin_json, "w", encoding="utf-8") as f:
    json.dump({
        "horodatage"     : horodatage,
        "heure_analysee" : HEURE_SIMULATION,
        "nb_zones_rouge" : len(df_alertes[df_alertes["niveau"] == "ROUGE"]),
        "nb_zones_orange": len(df_alertes[df_alertes["niveau"] == "ORANGE"]),
        "alertes"        : alertes_json
    }, f, ensure_ascii=False, indent=2)
print(f"  Sauvegardé : {chemin_json}")

# =============================================================================
# RÉSUMÉ
# =============================================================================

print("\n" + "=" * 60)
print("RÉSUMÉ PHASE 12")
print("=" * 60)
print(f"  Zones analysées  : {len(df_alertes)}")
print(f"  🔴 ROUGE         : {len(df_alertes[df_alertes['niveau']=='ROUGE'])}")
print(f"  🟠 ORANGE        : {len(df_alertes[df_alertes['niveau']=='ORANGE'])}")
print(f"  🟡 JAUNE         : {len(df_alertes[df_alertes['niveau']=='JAUNE'])}")
print(f"  🟢 VERT          : {len(df_alertes[df_alertes['niveau']=='VERT'])}")
print(f"\n  Fichiers produits :")
print(f"    - rapport_alertes.txt")
print(f"    - alertes.csv")
print(f"    - alertes_actives.json")
print("\nPhase 12 terminée avec succès.")
print("Tu peux maintenant passer à la Phase 13 — Cybersécurité.")