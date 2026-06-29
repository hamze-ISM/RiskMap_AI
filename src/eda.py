# =============================================================================
# Analyse Exploratoire des Données (EDA)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# =============================================================================
# CONFIGURATION GÉNÉRALE
# =============================================================================

# Dossier de sauvegarde des figures
DOSSIER_FIGURES = os.path.join("reports", "figures")
os.makedirs(DOSSIER_FIGURES, exist_ok=True)

# Style visuel des graphiques
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 150

def sauvegarder(nom_fichier):
    """Sauvegarde la figure en cours dans reports/figures/"""
    chemin = os.path.join(DOSSIER_FIGURES, nom_fichier)
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardé : {chemin}")

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

print("=" * 60)
print("PHASE 6 — Analyse Exploratoire (EDA)")
print("=" * 60)

print("\nChargement de crime_clean.csv...")
df = pd.read_csv(
    os.path.join("data", "processed", "crime_clean.csv"),
    low_memory=False
)

# Reconvertir la colonne date si elle a été sauvegardée en texte
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

print(f"  {len(df):,} lignes chargées | {df.shape[1]} colonnes")
print(f"  Colonnes : {list(df.columns)}\n")

# =============================================================================
# ANALYSE 1 — Vue d'ensemble
# =============================================================================

print("[1/7] Vue d'ensemble...")

print(f"\n  Nombre total de crimes : {len(df):,}")
print(f"  Période couverte       : {df['annee'].min()} → {df['annee'].max()}")
print(f"  Types de crimes uniques: {df['primary_type'].nunique() if 'primary_type' in df.columns else 'N/A'}")

print("\n  Crimes par année :")
print(df["annee_source"].value_counts().sort_index().to_string())

# =============================================================================
# ANALYSE 2 — Crimes par année (barplot)
# =============================================================================

print("\n[2/7] Graphique : crimes par année...")

crimes_annee = df["annee_source"].value_counts().sort_index()

plt.figure()
ax = crimes_annee.plot(kind="bar", color=["#2196F3", "#FF9800", "#F44336"], edgecolor="white")
plt.title("Nombre de crimes par année", fontsize=14, fontweight="bold")
plt.xlabel("Année")
plt.ylabel("Nombre de crimes")
plt.xticks(rotation=0)

# Afficher les valeurs au-dessus des barres
for i, v in enumerate(crimes_annee):
    ax.text(i, v + 500, f"{v:,}", ha="center", fontsize=10)

sauvegarder("01_crimes_par_annee.png")

# =============================================================================
# ANALYSE 3 — Crimes par mois (saisonnalité)
# =============================================================================

print("[3/7] Graphique : crimes par mois...")

crimes_mois = df.groupby("mois").size()
noms_mois = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
             "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]

plt.figure()
plt.plot(crimes_mois.index, crimes_mois.values,
         marker="o", linewidth=2.5, color="#2196F3", markersize=8)
plt.fill_between(crimes_mois.index, crimes_mois.values, alpha=0.15, color="#2196F3")
plt.title("Nombre de crimes par mois (toutes années confondues)", fontsize=14, fontweight="bold")
plt.xlabel("Mois")
plt.ylabel("Nombre de crimes")
plt.xticks(range(1, 13), noms_mois)
plt.tight_layout()
sauvegarder("02_crimes_par_mois.png")

# =============================================================================
# ANALYSE 4 — Crimes par heure (rythme journalier)
# =============================================================================

print("[4/7] Graphique : crimes par heure...")

crimes_heure = df.groupby("heure").size()

plt.figure()
plt.plot(crimes_heure.index, crimes_heure.values,
         marker="o", linewidth=2.5, color="#E91E63", markersize=6)
plt.fill_between(crimes_heure.index, crimes_heure.values, alpha=0.15, color="#E91E63")
plt.title("Nombre de crimes par heure de la journée", fontsize=14, fontweight="bold")
plt.xlabel("Heure (0 = minuit, 12 = midi)")
plt.ylabel("Nombre de crimes")
plt.xticks(range(0, 24))
plt.tight_layout()
sauvegarder("03_crimes_par_heure.png")

# =============================================================================
# ANALYSE 5 — Crimes par jour de la semaine
# =============================================================================

print("[5/7] Graphique : crimes par jour de la semaine...")

noms_jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
crimes_jour = df.groupby("jour_semaine").size()

plt.figure()
couleurs = ["#78909C"] * 5 + ["#F44336"] * 2  # weekends en rouge
ax = crimes_jour.plot(kind="bar", color=couleurs, edgecolor="white")
plt.title("Nombre de crimes par jour de la semaine", fontsize=14, fontweight="bold")
plt.xlabel("Jour")
plt.ylabel("Nombre de crimes")
plt.xticks(range(7), noms_jours, rotation=30)

for i, v in enumerate(crimes_jour):
    ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=9)

plt.tight_layout()
sauvegarder("04_crimes_par_jour_semaine.png")

# =============================================================================
# ANALYSE 6 — Top 10 types de crimes
# =============================================================================

print("[6/7] Graphique : top 10 types de crimes...")

if "primary_type" in df.columns:
    top10 = df["primary_type"].value_counts().head(10)

    plt.figure(figsize=(12, 7))
    ax = top10.sort_values().plot(kind="barh", color="#7E57C2", edgecolor="white")
    plt.title("Top 10 des types de crimes", fontsize=14, fontweight="bold")
    plt.xlabel("Nombre de crimes")
    plt.ylabel("")

    for i, v in enumerate(top10.sort_values()):
        ax.text(v + 100, i, f"{v:,}", va="center", fontsize=9)

    plt.tight_layout()
    sauvegarder("05_top10_types_crimes.png")
else:
    print("  Colonne 'primary_type' introuvable — graphique ignoré")

# =============================================================================
# ANALYSE 7 — Heatmap heure × jour de la semaine
# =============================================================================

print("[7/7] Heatmap : heure × jour de la semaine...")

heatmap_data = df.groupby(["jour_semaine", "heure"]).size().unstack(fill_value=0)
heatmap_data.index = noms_jours

plt.figure(figsize=(18, 6))
sns.heatmap(
    heatmap_data,
    cmap="YlOrRd",
    linewidths=0.3,
    linecolor="white",
    cbar_kws={"label": "Nombre de crimes"}
)
plt.title("Heatmap : Crimes par heure et jour de la semaine", fontsize=14, fontweight="bold")
plt.xlabel("Heure de la journée")
plt.ylabel("")
plt.tight_layout()
sauvegarder("06_heatmap_heure_jour.png")

# =============================================================================
# RÉSUMÉ FINAL
# =============================================================================

print("\n" + "=" * 60)
print("RÉSUMÉ DE L'EDA")
print("=" * 60)

heure_pic = crimes_heure.idxmax()
mois_pic  = crimes_mois.idxmax()
jour_pic  = crimes_jour.idxmax()

print(f"\n  Heure la plus dangereuse : {heure_pic}h")
print(f"  Mois le plus dangereux   : {noms_mois[mois_pic - 1]}")
print(f"  Jour le plus dangereux   : {noms_jours[jour_pic]}")

if "primary_type" in df.columns:
    print(f"  Crime le plus fréquent   : {df['primary_type'].value_counts().index[0]}")

print(f"\n  6 figures sauvegardées dans : {DOSSIER_FIGURES}/")
print("\nPhase 6 terminée avec succès.")
print("Tu peux maintenant passer à la Phase 7 — Zones géographiques.")