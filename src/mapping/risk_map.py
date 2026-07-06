# =============================================================================
# PHASE 11 — Cartographie interactive Folium
# Fichier : src/mapping/risk_map.py
# =============================================================================

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

DOSSIER_RAPPORTS = os.path.join("reports", "maps")
os.makedirs(DOSSIER_RAPPORTS, exist_ok=True)

# Centre géographique de Chicago
CHICAGO_LAT = 41.8781
CHICAGO_LON = -87.6298

# =============================================================================
# CHARGEMENT
# =============================================================================

print("=" * 60)
print("PHASE 11 — Cartographie Folium")
print("=" * 60)

print("\nChargement des données...")
zones    = pd.read_csv(os.path.join("data", "processed", "risk_map_data.csv"))
grille   = pd.read_csv(os.path.join("data", "processed", "grille_horaire.csv"))
df_crimes = pd.read_csv(
    os.path.join("data", "processed", "crime_features.csv"),
    low_memory=False,
    usecols=["latitude", "longitude", "heure", "primary_type", "risque_predit"]
    if "risque_predit" in pd.read_csv(
        os.path.join("data", "processed", "crime_features.csv"), nrows=1
    ).columns
    else ["latitude", "longitude", "heure", "primary_type"]
)

print(f"  {len(zones)} zones chargées")
print(f"  {len(grille):,} combinaisons zone×heure chargées")

# Supprimer les zones sans coordonnées GPS
zones = zones.dropna(subset=["lat_centre", "lon_centre"])
print(f"  {len(zones)} zones avec coordonnées GPS valides")

# =============================================================================
# SECTION 1 — Carte principale (zones colorées)
# =============================================================================

print("\n[1/4] Carte principale — zones colorées...")

# Couleurs et icônes par niveau de risque
config_risque = {
    "Élevé"  : {"color": "red",    "icon": "exclamation-sign", "fill": "#FF4444"},
    "Moyen"  : {"color": "orange", "icon": "warning-sign",     "fill": "#FF9800"},
    "Faible" : {"color": "green",  "icon": "ok-sign",          "fill": "#4CAF50"},
}

# Création de la carte centrée sur Chicago
carte = folium.Map(
    location=[CHICAGO_LAT, CHICAGO_LON],
    zoom_start=11,
    tiles="CartoDB positron",   # fond de carte clair et lisible
)

# Ajout d'un titre HTML sur la carte
titre_html = """
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
     z-index: 1000; background: white; padding: 10px 20px;
     border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
     font-family: Arial; font-size: 15px; font-weight: bold; color: #333;">
    RiskMap_AI — Carte de risque prédictive — Chicago 2024–2026
</div>
"""
carte.get_root().html.add_child(folium.Element(titre_html))

# Ajout d'une mini-carte de navigation
MiniMap(toggle_display=True).add_to(carte)

# Création de 3 groupes de marqueurs (un par niveau)
# Permet d'activer/désactiver chaque niveau depuis la légende
groupe_eleve  = folium.FeatureGroup(name="🔴 Zones Élevé",  show=True)
groupe_moyen  = folium.FeatureGroup(name="🟠 Zones Moyen",  show=True)
groupe_faible = folium.FeatureGroup(name="🟢 Zones Faible", show=False)

for _, zone in zones.iterrows():
    niveau = zone["risque_label"]
    cfg    = config_risque.get(niveau, config_risque["Faible"])

    # Construction du texte du popup
    popup_html = f"""
    <div style="font-family: Arial; font-size: 13px; min-width: 200px;">
        <b>Zone {zone['zone_id']}</b><br>
        <hr style="margin: 4px 0;">
        Niveau de risque : <b style="color: {cfg['fill']};">{niveau}</b><br>
        Incidents historiques : <b>{int(zone['nb_incidents']):,}</b><br>
        Prob. Élevé : <b>{zone['proba_eleve_moy']:.1%}</b><br>
        Prob. Moyen : <b>{zone['proba_moyen_moy']:.1%}</b><br>
        Prob. Faible : <b>{zone['proba_faible_moy']:.1%}</b><br>
        <hr style="margin: 4px 0;">
        <small>Lat: {zone['lat_centre']:.4f} | Lon: {zone['lon_centre']:.4f}</small>
    </div>
    """

    marqueur = folium.CircleMarker(
        location=[zone["lat_centre"], zone["lon_centre"]],
        radius=10,
        color=cfg["color"],
        fill=True,
        fill_color=cfg["fill"],
        fill_opacity=0.65,
        weight=1.5,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"Zone {zone['zone_id']} — {niveau} ({int(zone['nb_incidents']):,} incidents)"
    )

    if niveau == "Élevé":
        marqueur.add_to(groupe_eleve)
    elif niveau == "Moyen":
        marqueur.add_to(groupe_moyen)
    else:
        marqueur.add_to(groupe_faible)

groupe_eleve.add_to(carte)
groupe_moyen.add_to(carte)
groupe_faible.add_to(carte)

print(f"  {len(zones)} marqueurs ajoutés")

# =============================================================================
# SECTION 2 — Heatmap de densité criminelle
# =============================================================================

print("[2/4] Heatmap de densité...")

# On utilise les coordonnées réelles des crimes pour la heatmap
# On prend un échantillon de 50 000 points pour ne pas alourdir la carte
if "latitude" in df_crimes.columns and "longitude" in df_crimes.columns:
    sample = df_crimes[["latitude", "longitude"]].dropna().sample(
        min(50000, len(df_crimes)), random_state=42
    )
    points_heatmap = sample.values.tolist()
else:
    # Si pas disponible, on utilise les centres de zones pondérés par le nombre d'incidents
    points_heatmap = []
    for _, z in zones.iterrows():
        points_heatmap.append([z["lat_centre"], z["lon_centre"], z["nb_incidents"]])

groupe_heatmap = folium.FeatureGroup(name="🌡️ Heatmap densité", show=False)
HeatMap(
    points_heatmap,
    radius=15,
    blur=10,
    max_zoom=13,
    gradient={"0.4": "blue", "0.6": "lime", "0.8": "orange", "1.0": "red"}
).add_to(groupe_heatmap)
groupe_heatmap.add_to(carte)

print(f"  Heatmap générée avec {len(points_heatmap):,} points")

# =============================================================================
# SECTION 3 — Légende
# =============================================================================

print("[3/4] Légende...")

legende_html = """
<div style="position: fixed; bottom: 30px; right: 20px; z-index: 1000;
     background: white; padding: 15px; border-radius: 10px;
     box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-family: Arial; font-size: 13px;">
    <b style="font-size: 14px;">Niveau de risque</b><br><br>
    <span style="color: #FF4444;">&#9632;</span> Élevé — Intervention prioritaire<br>
    <span style="color: #FF9800;">&#9632;</span> Moyen — Surveillance renforcée<br>
    <span style="color: #4CAF50;">&#9632;</span> Faible — Zone calme<br>
    <hr style="margin: 8px 0;">
    <small>Modèle : XGBoost — F1 = 0.9192<br>
    Données : Chicago 2024–2026<br>
    RiskMap_AI © 2026</small>
</div>
"""
carte.get_root().html.add_child(folium.Element(legende_html))

# Contrôle des couches (checkbox pour activer/désactiver)
folium.LayerControl(collapsed=False).add_to(carte)

# =============================================================================
# SECTION 4 — Sauvegarde
# =============================================================================

print("[4/4] Sauvegarde...")

chemin_carte = os.path.join("reports", "maps", "risk_map.html")
carte.save(chemin_carte)
print(f"  Carte sauvegardée : {chemin_carte}")

# =============================================================================
# RÉSUMÉ
# =============================================================================

nb_eleve  = len(zones[zones["risque_label"] == "Élevé"])
nb_moyen  = len(zones[zones["risque_label"] == "Moyen"])
nb_faible = len(zones[zones["risque_label"] == "Faible"])

print("\n" + "=" * 60)
print("RÉSUMÉ PHASE 11")
print("=" * 60)
print(f"  Zones Élevé   (rouge)  : {nb_eleve}")
print(f"  Zones Moyen   (orange) : {nb_moyen}")
print(f"  Zones Faible  (vert)   : {nb_faible}")
print(f"\n  Carte interactive : {chemin_carte}")
print("  Ouvre ce fichier dans ton navigateur pour visualiser la carte.")
print("\nPhase 11 terminée avec succès.")
print("Tu peux maintenant passer à la Phase 12 — Système d'alertes.")