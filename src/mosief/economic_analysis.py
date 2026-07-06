# =============================================================================
# PHASE 14 — Analyse économique MOSIEF
# Fichier : src/mosief/economic_analysis.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import json

# =============================================================================
# CONFIGURATION — Coûts unitaires (sources : études criminologiques US)
# =============================================================================

# Coût moyen par type de crime en USD
# Sources : Institut National de Justice (NIJ), Anderson (1999), McCollister (2010)
COUT_PAR_TYPE = {
    "THEFT"                  : 3_500,
    "BATTERY"                : 15_000,
    "CRIMINAL DAMAGE"        : 4_200,
    "ASSAULT"                : 12_000,
    "BURGLARY"               : 8_500,
    "MOTOR VEHICLE THEFT"    : 10_300,
    "ROBBERY"                : 25_000,
    "DECEPTIVE PRACTICE"     : 6_000,
    "OTHER OFFENSE"          : 5_000,
    "NARCOTICS"              : 7_500,
    "DEFAULT"                : 5_000,   # valeur par défaut
}

# Paramètres du modèle ROI
COUT_SYSTEME_ANNUEL   = 150_000   # coût annuel de RiskMap_AI (serveurs, maintenance)
TAUX_REDUCTION_RISQUE = 0.15      # réduction de 15% de la criminalité avec prévention
POPULATION_CHICAGO    = 2_696_000 # habitants

DOSSIER_FIGURES  = os.path.join("reports", "figures")
DOSSIER_TABLES   = os.path.join("reports", "tables")
DOSSIER_ECONOMIC = os.path.join("reports", "economics")

for d in [DOSSIER_FIGURES, DOSSIER_TABLES, DOSSIER_ECONOMIC]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# CHARGEMENT
# =============================================================================

print("=" * 60)
print("PHASE 14 — Analyse économique MOSIEF")
print("=" * 60)

print("\nChargement des données...")
df = pd.read_csv(
    os.path.join("data", "processed", "crime_features.csv"),
    low_memory=False,
    usecols=["zone_id", "annee_source", "primary_type",
             "niveau_risque", "heure", "mois"]
)
zones = pd.read_csv(os.path.join("data", "processed", "risk_map_data.csv"))
print(f"  {len(df):,} incidents chargés | {len(zones)} zones")

# =============================================================================
# SECTION 1 — Coût de la criminalité par zone
# =============================================================================

print("\n[1/5] Coût de la criminalité par zone...")

# Assignation du coût unitaire à chaque incident
df["cout_unitaire"] = df["primary_type"].map(COUT_PAR_TYPE).fillna(COUT_PAR_TYPE["DEFAULT"])

# Coût total par zone
cout_par_zone = (
    df.groupby("zone_id")
    .agg(
        nb_incidents=("cout_unitaire", "count"),
        cout_total=("cout_unitaire", "sum"),
        cout_moyen=("cout_unitaire", "mean")
    )
    .reset_index()
)

# Fusion avec les niveaux de risque
cout_par_zone = cout_par_zone.merge(
    zones[["zone_id", "risque_label", "lat_centre", "lon_centre"]],
    on="zone_id", how="left"
)

cout_total_global = cout_par_zone["cout_total"].sum()
cout_moyen_par_zone = cout_par_zone["cout_total"].mean()

print(f"  Coût total estimé (2024–2026) : ${cout_total_global:,.0f}")
print(f"  Coût moyen par zone           : ${cout_moyen_par_zone:,.0f}")
print(f"  Coût par habitant             : ${cout_total_global/POPULATION_CHICAGO:,.0f}")

# Top 5 zones les plus coûteuses
print("\n  Top 5 zones les plus coûteuses :")
top5 = cout_par_zone.nlargest(5, "cout_total")[
    ["zone_id", "nb_incidents", "cout_total", "risque_label"]
]
for _, row in top5.iterrows():
    print(f"    Zone {row['zone_id']:8s} | {int(row['nb_incidents']):,} incidents | "
          f"${row['cout_total']:,.0f} | {row['risque_label']}")

# =============================================================================
# SECTION 2 — Coût par type de crime
# =============================================================================

print("\n[2/5] Coût par type de crime...")

cout_par_type = (
    df.groupby("primary_type")
    .agg(
        nb_incidents=("cout_unitaire", "count"),
        cout_total=("cout_unitaire", "sum")
    )
    .sort_values("cout_total", ascending=False)
    .head(10)
    .reset_index()
)

print("\n  Top 10 types de crimes par coût total :")
for _, row in cout_par_type.iterrows():
    print(f"    {row['primary_type']:30s} : ${row['cout_total']:>15,.0f} "
          f"({int(row['nb_incidents']):,} incidents)")

# =============================================================================
# SECTION 3 — Analyse coûts/bénéfices du système préventif
# =============================================================================

print("\n[3/5] Analyse coûts/bénéfices...")

# Coût annuel de la criminalité (ramené sur 1 an, données sur ~2.5 ans)
nb_annees = df["annee_source"].nunique()
cout_annuel = cout_total_global / nb_annees

# Bénéfice annuel estimé avec RiskMap_AI
benefice_annuel = cout_annuel * TAUX_REDUCTION_RISQUE
roi_annuel = (benefice_annuel - COUT_SYSTEME_ANNUEL) / COUT_SYSTEME_ANNUEL * 100
point_mort_mois = (COUT_SYSTEME_ANNUEL / (benefice_annuel / 12))

print(f"\n  Coût annuel de la criminalité   : ${cout_annuel:,.0f}")
print(f"  Réduction estimée (15%)         : ${benefice_annuel:,.0f}")
print(f"  Coût annuel RiskMap_AI          : ${COUT_SYSTEME_ANNUEL:,.0f}")
print(f"  Bénéfice net annuel             : ${benefice_annuel - COUT_SYSTEME_ANNUEL:,.0f}")
print(f"  ROI annuel                      : {roi_annuel:.1f}%")
print(f"  Point mort                      : {point_mort_mois:.1f} mois")

# =============================================================================
# SECTION 4 — KPIs décisionnels
# =============================================================================

print("\n[4/5] KPIs décisionnels...")

# Distribution du coût par niveau de risque
cout_par_niveau = (
    cout_par_zone.groupby("risque_label")
    .agg(
        nb_zones=("zone_id", "count"),
        cout_total=("cout_total", "sum"),
        nb_incidents_total=("nb_incidents", "sum")
    )
    .reset_index()
)
cout_par_niveau["pct_cout"] = (
    cout_par_niveau["cout_total"] / cout_par_niveau["cout_total"].sum() * 100
).round(1)

print("\n  Répartition du coût par niveau de risque :")
for _, row in cout_par_niveau.iterrows():
    print(f"    {row['risque_label']:6s} | {int(row['nb_zones']):3d} zones | "
          f"${row['cout_total']:>15,.0f} | {row['pct_cout']:.1f}% du coût total")

# Coût par créneau horaire (pour prioriser les patrouilles)
cout_heure = df.groupby("heure")["cout_unitaire"].sum()
heure_pic = cout_heure.idxmax()
print(f"\n  Créneau horaire le plus coûteux : {heure_pic}h "
      f"(${cout_heure.max():,.0f})")

# =============================================================================
# SECTION 5 — Visualisations et sauvegarde
# =============================================================================

print("\n[5/5] Visualisations et sauvegarde...")

# --- Graphique 1 : Coût par niveau de risque ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

couleurs = {"Élevé": "#F44336", "Moyen": "#FF9800", "Faible": "#4CAF50"}
coul_list = [couleurs.get(n, "#9E9E9E") for n in cout_par_niveau["risque_label"]]

axes[0].bar(cout_par_niveau["risque_label"], cout_par_niveau["cout_total"] / 1e6,
            color=coul_list, edgecolor="white")
axes[0].set_title("Coût total par niveau de risque (M$)", fontweight="bold")
axes[0].set_ylabel("Millions USD")
for i, (_, row) in enumerate(cout_par_niveau.iterrows()):
    axes[0].text(i, row["cout_total"]/1e6 + 0.5,
                 f"${row['cout_total']/1e6:.0f}M", ha="center", fontsize=10)

axes[1].pie(
    cout_par_niveau["pct_cout"],
    labels=cout_par_niveau["risque_label"],
    colors=coul_list,
    autopct="%1.1f%%",
    startangle=90
)
axes[1].set_title("Répartition du coût (%)", fontweight="bold")

plt.suptitle("Impact économique de la criminalité — Chicago 2024–2026",
             fontsize=13, fontweight="bold")
plt.tight_layout()
chemin_g1 = os.path.join(DOSSIER_FIGURES, "11_cout_par_niveau_risque.png")
plt.savefig(chemin_g1, bbox_inches="tight")
plt.close()
print(f"  Sauvegardée : {chemin_g1}")

# --- Graphique 2 : ROI sur 5 ans ---
annees = np.arange(1, 6)
couts_cumules   = annees * COUT_SYSTEME_ANNUEL
benefices_cumul = annees * benefice_annuel
benefice_net    = benefices_cumul - couts_cumules

plt.figure(figsize=(10, 6))
plt.plot(annees, benefices_cumul / 1e6, "g-o", linewidth=2.5,
         label="Bénéfices cumulés (crimes évités)")
plt.plot(annees, couts_cumules / 1e6,   "r-o", linewidth=2.5,
         label="Coûts cumulés RiskMap_AI")
plt.fill_between(annees,
                 couts_cumules / 1e6,
                 benefices_cumul / 1e6,
                 alpha=0.15, color="green", label="Bénéfice net")
plt.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
plt.title("ROI projeté sur 5 ans — RiskMap_AI", fontweight="bold", fontsize=13)
plt.xlabel("Année")
plt.ylabel("Millions USD")
plt.legend()
plt.xticks(annees, [f"An {a}" for a in annees])
plt.tight_layout()
chemin_g2 = os.path.join(DOSSIER_FIGURES, "12_roi_5_ans.png")
plt.savefig(chemin_g2, bbox_inches="tight")
plt.close()
print(f"  Sauvegardée : {chemin_g2}")

# --- Sauvegarde CSV et JSON ---
cout_par_zone.to_csv(
    os.path.join(DOSSIER_TABLES, "cout_par_zone.csv"), index=False)

kpis = {
    "cout_total_global_usd"    : round(cout_total_global),
    "cout_annuel_usd"          : round(cout_annuel),
    "benefice_annuel_usd"      : round(benefice_annuel),
    "cout_systeme_annuel_usd"  : COUT_SYSTEME_ANNUEL,
    "benefice_net_annuel_usd"  : round(benefice_annuel - COUT_SYSTEME_ANNUEL),
    "roi_annuel_pct"           : round(roi_annuel, 1),
    "point_mort_mois"          : round(point_mort_mois, 1),
    "cout_par_habitant_usd"    : round(cout_total_global / POPULATION_CHICAGO),
    "heure_pic_cout"           : int(heure_pic),
}
with open(os.path.join(DOSSIER_ECONOMIC, "kpis.json"), "w") as f:
    json.dump(kpis, f, indent=2)

print(f"  Sauvegardé : reports/tables/cout_par_zone.csv")
print(f"  Sauvegardé : reports/economics/kpis.json")

# =============================================================================
# RÉSUMÉ
# =============================================================================

print("\n" + "=" * 60)
print("RÉSUMÉ PHASE 14")
print("=" * 60)
print(f"  Coût total criminalité (2024–2026) : ${cout_total_global:,.0f}")
print(f"  Coût annuel estimé                 : ${cout_annuel:,.0f}")
print(f"  ROI annuel RiskMap_AI              : {roi_annuel:.1f}%")
print(f"  Point mort                         : {point_mort_mois:.1f} mois")
print(f"\n  Figures produites :")
print(f"    - 11_cout_par_niveau_risque.png")
print(f"    - 12_roi_5_ans.png")
print("\nPhase 14 terminée avec succès.")
print("Tu peux maintenant passer à la Phase 15 — Dashboard Streamlit.")