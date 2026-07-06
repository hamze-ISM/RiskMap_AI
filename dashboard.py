# =============================================================================
# PHASE 15 — Dashboard Streamlit
# Fichier : dashboard.py (à la racine de RiskMap_AI/)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import folium
from streamlit_folium import st_folium

# =============================================================================
# CONFIGURATION DE LA PAGE
# =============================================================================

st.set_page_config(
    page_title="RiskMap_AI — Chicago",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .metric-card {
        background: #1E1E2E;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 4px solid;
    }
    .rouge  { border-color: #F44336; }
    .orange { border-color: #FF9800; }
    .vert   { border-color: #4CAF50; }
    .bleu   { border-color: #2196F3; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CHARGEMENT DES DONNÉES (mis en cache pour performance)
# =============================================================================

@st.cache_data
def charger_donnees():
    zones    = pd.read_csv("data/processed/risk_map_data.csv")
    grille   = pd.read_csv("data/processed/grille_horaire.csv")
    alertes  = pd.read_csv("reports/alerts/alertes.csv")
    kpis_raw = open("reports/economics/kpis.json").read()
    kpis     = json.loads(kpis_raw)
    return zones, grille, alertes, kpis

zones, grille, alertes, kpis = charger_donnees()

# =============================================================================
# SIDEBAR — Filtres
# =============================================================================

st.sidebar.image("https://img.icons8.com/color/96/map-marker.png", width=60)
st.sidebar.title("RiskMap_AI")
st.sidebar.markdown("**Système de prédiction des zones à risque urbain**")
st.sidebar.markdown("---")

# Filtre heure
heure_selectionnee = st.sidebar.slider(
    "🕐 Heure de la journée", 0, 23, 12,
    help="Filtre la carte sur le créneau horaire sélectionné"
)

# Filtre niveau de risque
niveaux_affiches = st.sidebar.multiselect(
    "🎯 Niveaux de risque à afficher",
    options=["Élevé", "Moyen", "Faible"],
    default=["Élevé", "Moyen"],
)

# Filtre nombre minimum d'incidents
min_incidents = st.sidebar.slider(
    "📊 Incidents minimum par zone", 0, 5000, 0, step=100
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Modèle :** XGBoost")
st.sidebar.markdown(f"**F1-Score :** 0.9192")
st.sidebar.markdown(f"**Données :** Chicago 2024–2026")
st.sidebar.markdown(f"**Zones analysées :** {len(zones)}")

# =============================================================================
# TITRE PRINCIPAL
# =============================================================================

st.title("🗺️ RiskMap_AI — Carte de risque prédictive")
st.markdown("**Chicago Crime Prediction System** · XGBoost · F1 = 0.9192")
st.markdown("---")

# =============================================================================
# SECTION 1 — KPIs en haut de page
# =============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

nb_rouge  = len(zones[zones["risque_label"] == "Élevé"])
nb_orange = len(zones[zones["risque_label"] == "Moyen"])
nb_vert   = len(zones[zones["risque_label"] == "Faible"])

with col1:
    st.metric("🔴 Zones Élevé",  nb_rouge,  help="Zones nécessitant intervention immédiate")
with col2:
    st.metric("🟠 Zones Moyen",  nb_orange, help="Surveillance renforcée recommandée")
with col3:
    st.metric("🟢 Zones Faible", nb_vert,   help="Situation normale")
with col4:
    st.metric("💰 Coût annuel",
              f"${kpis['cout_annuel_usd']/1e9:.2f}B",
              help="Coût estimé de la criminalité par an")
with col5:
    st.metric("📈 ROI système",
              f"{kpis['roi_annuel_pct']:,.0f}%",
              help="Retour sur investissement annuel de RiskMap_AI")

st.markdown("---")

# =============================================================================
# SECTION 2 — Carte interactive + Alertes (colonnes)
# =============================================================================

col_carte, col_alertes = st.columns([3, 1])

with col_carte:
    st.subheader(f"🗺️ Carte de risque — {heure_selectionnee}h00")

    # Filtrage des zones selon la sélection
    zones_filtrees = zones[
        (zones["risque_label"].isin(niveaux_affiches)) &
        (zones["nb_incidents"] >= min_incidents)
    ].dropna(subset=["lat_centre", "lon_centre"])

    # Données horaires pour l'heure sélectionnée
    grille_heure = grille[grille["heure"] == heure_selectionnee].copy()
    zones_heure = zones_filtrees.merge(
        grille_heure[["zone_id", "risque_label"]],
        on="zone_id", how="left", suffixes=("", "_heure")
    )

    # Construction de la carte Folium
    carte = folium.Map(
        location=[41.8781, -87.6298],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    config_couleurs = {
        "Élevé" : {"color": "red",    "fill": "#FF4444"},
        "Moyen" : {"color": "orange", "fill": "#FF9800"},
        "Faible": {"color": "green",  "fill": "#4CAF50"},
    }

    for _, zone in zones_filtrees.iterrows():
        cfg = config_couleurs.get(zone["risque_label"], config_couleurs["Faible"])
        popup_txt = (
            f"<b>Zone {zone['zone_id']}</b><br>"
            f"Risque : <b>{zone['risque_label']}</b><br>"
            f"Incidents : {int(zone['nb_incidents']):,}<br>"
            f"Prob. Élevé : {zone['proba_eleve_moy']:.1%}"
        )
        folium.CircleMarker(
            location=[zone["lat_centre"], zone["lon_centre"]],
            radius=9,
            color=cfg["color"],
            fill=True,
            fill_color=cfg["fill"],
            fill_opacity=0.65,
            popup=folium.Popup(popup_txt, max_width=200),
            tooltip=f"{zone['zone_id']} — {zone['risque_label']}"
        ).add_to(carte)

    st_folium(carte, width=750, height=520)
    st.caption(f"Affichage : {len(zones_filtrees)} zones | "
               f"Filtre : {', '.join(niveaux_affiches)} | "
               f"Min. incidents : {min_incidents}")

with col_alertes:
    st.subheader("🚨 Alertes actives")

    alertes_rouges  = alertes[alertes["niveau"] == "ROUGE"].head(8)
    alertes_oranges = alertes[alertes["niveau"] == "ORANGE"].head(5)

    if len(alertes_rouges) > 0:
        st.markdown("**🔴 CRITIQUE**")
        for _, a in alertes_rouges.iterrows():
            st.error(
                f"**Zone {a['zone_id']}**\n\n"
                f"Prob. : {a['prob_eleve']:.1%} | "
                f"{int(a['nb_incidents']):,} incidents"
            )

    if len(alertes_oranges) > 0:
        st.markdown("**🟠 MODÉRÉE**")
        for _, a in alertes_oranges.iterrows():
            st.warning(
                f"**Zone {a['zone_id']}** — "
                f"{a['prob_eleve']:.1%}"
            )

# =============================================================================
# SECTION 3 — Statistiques et graphiques
# =============================================================================

st.markdown("---")
st.subheader("📊 Analyse statistique")

tab1, tab2, tab3 = st.tabs(["Distribution des risques", "Impact économique", "Top zones"])

with tab1:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        dist = zones["risque_label"].value_counts().reset_index()
        dist.columns = ["Niveau", "Nombre de zones"]
        st.bar_chart(dist.set_index("Niveau"))
        st.caption("Distribution des zones par niveau de risque")
    with col_g2:
        grille_dist = (
            grille[grille["heure"] == heure_selectionnee]["risque_label"]
            .value_counts()
            .reset_index()
        )
        grille_dist.columns = ["Niveau", "Zones"]
        st.bar_chart(grille_dist.set_index("Niveau"))
        st.caption(f"Distribution à {heure_selectionnee}h")

with tab2:
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.metric("Coût total 2024–2026",
                  f"${kpis['cout_total_global_usd']/1e9:.2f}B")
    with col_e2:
        st.metric("Bénéfice net annuel",
                  f"${kpis['benefice_net_annuel_usd']/1e6:.0f}M")
    with col_e3:
        st.metric("Coût par habitant",
                  f"${kpis['cout_par_habitant_usd']:,}")

    st.image("reports/figures/12_roi_5_ans.png",
         caption="ROI projeté sur 5 ans", width=700)

with tab3:
    top_zones = zones.nlargest(15, "nb_incidents")[
        ["zone_id", "nb_incidents", "risque_label",
         "proba_eleve_moy", "lat_centre", "lon_centre"]
    ].reset_index(drop=True)
    top_zones.columns = ["Zone", "Incidents", "Risque",
                         "Prob. Élevé", "Latitude", "Longitude"]
    top_zones["Prob. Élevé"] = top_zones["Prob. Élevé"].apply(lambda x: f"{x:.1%}")
    st.dataframe(top_zones, width='stretch')

# =============================================================================
# SECTION 4 — Pied de page
# =============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:grey; font-size:12px;'>"
    "RiskMap_AI © 2026 · XGBoost F1=0.9192 · Chicago Crime Dataset 2024–2026 · "
    "Développé dans le cadre du Projet 17 — ISM Dakar"
    "</div>",
    unsafe_allow_html=True
)