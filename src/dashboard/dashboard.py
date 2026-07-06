# dashboard.py — Dashboard Streamlit RiskMap_AI

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import joblib, json, os

# ── Configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskMap_AI",
    page_icon="🗺️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Chargement des données ─────────────────────────────────────────
@st.cache_data
def charger_donnees():
    chemin = os.path.join(BASE_DIR, "data", "processed", "crimes_features.csv")
    df = pd.read_csv(chemin, parse_dates=["Date"])
    freq_area = df.groupby("Community Area").size()
    seuil_bas  = freq_area.quantile(0.33)
    seuil_haut = freq_area.quantile(0.66)
    def niveau(area):
        f = freq_area.get(area, 0)
        if f >= seuil_haut: return 2
        elif f >= seuil_bas: return 1
        else: return 0
    df["cible"] = df["Community Area"].apply(niveau)
    return df

@st.cache_resource
def charger_modele():
    chemin = os.path.join(BASE_DIR, "models", "random_forest.pkl")
    return joblib.load(chemin)

def charger_alertes():
    chemin = os.path.join(BASE_DIR, "data", "processed", "alertes.json")
    if os.path.exists(chemin):
        with open(chemin) as f:
            return json.load(f)
    return []

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Chicago_flag.svg/200px-Chicago_flag.svg.png", width=150)
st.sidebar.title("RiskMap_AI")
st.sidebar.markdown("IA de prediction des zones a risque urbain")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "Vue d'ensemble",
    "Carte de risque",
    "Alertes",
    "Analyse temporelle",
    "Performance modele",
    "MOSIEF / ROI"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtres**")
annee = st.sidebar.multiselect("Annee", [2024, 2025, 2026], default=[2024, 2025, 2026])

# ── Chargement ────────────────────────────────────────────────────
with st.spinner("Chargement des donnees..."):
    df = charger_donnees()
    modele = charger_modele()
    alertes = charger_alertes()

if annee:
    df = df[df["annee_source"].isin(annee)]

# ══════════════════════════════════════════════════════════════════
# PAGE 1 : Vue d'ensemble
# ══════════════════════════════════════════════════════════════════
if page == "Vue d'ensemble":
    st.title("RiskMap_AI — Vue d'ensemble")
    st.markdown("Systeme d'IA de prediction des zones a risque urbain — Chicago 2024-2026")

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total incidents",    f"{len(df):,}")
    col2.metric("Zones a risque eleve", f"{(df['cible']==2).sum():,}")
    col3.metric("Taux arrestation",   f"{df['Arrest'].mean()*100:.1f}%")
    col4.metric("Crimes domestiques", f"{df['Domestic'].mean()*100:.1f}%")
    col5.metric("Alertes actives",    f"{len(alertes)}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 types de crimes")
        top = df["Primary Type"].value_counts().head(10).reset_index()
        top.columns = ["Crime", "Incidents"]
        fig = px.bar(top, x="Incidents", y="Crime", orientation="h",
                     color="Incidents", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Repartition par niveau de risque")
        risque_counts = df["cible"].value_counts().reset_index()
        risque_counts.columns = ["Niveau", "Count"]
        risque_counts["Label"] = risque_counts["Niveau"].map(
            {0: "Faible", 1: "Moyen", 2: "Eleve"})
        fig2 = px.pie(risque_counts, values="Count", names="Label",
                      color="Label",
                      color_discrete_map={"Faible":"green","Moyen":"orange","Eleve":"red"})
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Evolution mensuelle des incidents")
    monthly = df.groupby(["annee_source","mois"]).size().reset_index(name="incidents")
    monthly["periode"] = monthly["annee_source"].astype(str) + "-" + monthly["mois"].astype(str).str.zfill(2)
    fig3 = px.line(monthly, x="periode", y="incidents", color="annee_source",
                   markers=True, title="Incidents par mois et par annee")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 : Carte de risque
# ══════════════════════════════════════════════════════════════════
elif page == "Carte de risque":
    st.title("Carte interactive de risque — Chicago")

    col1, col2, col3 = st.columns(3)
    col1.metric("Zones Faible",  f"{(df['cible']==0).sum():,}", delta="Vert")
    col2.metric("Zones Moyen",   f"{(df['cible']==1).sum():,}", delta="Orange")
    col3.metric("Zones Eleve",   f"{(df['cible']==2).sum():,}", delta="Rouge")

    st.markdown("---")

    # Carte Folium
    from folium.plugins import HeatMap, MarkerCluster
    carte = folium.Map(location=[41.8781, -87.6298], zoom_start=11,
                       tiles="CartoDB positron")

    df_sample = df[["Latitude","Longitude","cible","Primary Type",
                    "District","heure"]].dropna().sample(n=5000, random_state=42)

    HeatMap(df_sample[["Latitude","Longitude"]].values.tolist(),
            radius=10, blur=15, min_opacity=0.3).add_to(carte)

    couleurs = {0:"green", 1:"orange", 2:"red"}
    labels   = {0:"Faible", 1:"Moyen", 2:"Eleve"}
    cluster  = MarkerCluster().add_to(carte)

    for _, row in df_sample.iterrows():
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=4,
            color=couleurs[row["cible"]],
            fill=True, fill_color=couleurs[row["cible"]], fill_opacity=0.7,
            popup=f"{row['Primary Type']} | {labels[row['cible']]} | {row['heure']}h"
        ).add_to(cluster)

    st_folium(carte, width=1200, height=600)

# ══════════════════════════════════════════════════════════════════
# PAGE 3 : Alertes
# ══════════════════════════════════════════════════════════════════
elif page == "Alertes":
    st.title("Alertes actives")

    if not alertes:
        st.warning("Aucune alerte active.")
    else:
        st.error(f"{len(alertes)} zones a risque ELEVE detectees !")
        for a in alertes:
            with st.expander(f"ALERTE — {a['id']} | District {a['district']}"):
                col1, col2 = st.columns(2)
                col1.metric("Incidents", f"{a['nb_incidents']:,}")
                col2.metric("Crime dominant", a["crime_principal"])
                st.write(f"**Message :** {a['message']}")
                st.write(f"**Timestamp :** {a['timestamp']}")
                st.write(f"**Coordonnees :** {a['latitude']}, {a['longitude']}")

# ══════════════════════════════════════════════════════════════════
# PAGE 4 : Analyse temporelle
# ══════════════════════════════════════════════════════════════════
elif page == "Analyse temporelle":
    st.title("Analyse temporelle")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Incidents par heure")
        h = df.groupby("heure").size().reset_index(name="incidents")
        fig = px.bar(h, x="heure", y="incidents", color="incidents",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Incidents par jour")
        jours = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
        j = df.groupby("jour_semaine").size().reset_index(name="incidents")
        j["jour"] = j["jour_semaine"].map(dict(enumerate(jours)))
        fig2 = px.bar(j, x="jour", y="incidents", color="incidents",
                      color_continuous_scale="Oranges")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Heatmap Heure x Jour")
    pivot = df.groupby(["jour_semaine","heure"]).size().unstack(fill_value=0)
    pivot.index = jours
    fig3 = px.imshow(pivot, color_continuous_scale="YlOrRd",
                     labels={"x":"Heure","y":"Jour","color":"Incidents"})
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE 5 : Performance modele
# ══════════════════════════════════════════════════════════════════
elif page == "Performance modele":
    st.title("Performance du modele")

    col1, col2, col3 = st.columns(3)
    col1.metric("Modele",   "Random Forest")
    col2.metric("Accuracy", "94.65%")
    col3.metric("F1 Macro", "90%")

    st.markdown("---")
    st.subheader("Importance des features")

    with open(os.path.join(BASE_DIR, "models", "features.json")) as f:
        features = json.load(f)

    importances = pd.DataFrame({
        "Feature"   : features,
        "Importance": modele.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig = px.bar(importances, x="Importance", y="Feature",
                 orientation="h", color="Importance",
                 color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Comparaison des modeles")
    comp = pd.DataFrame({
        "Modele"    : ["Random Forest", "XGBoost"],
        "Accuracy"  : [94.65, 100.0],
        "F1 Macro"  : [90.0, 100.0],
        "Note"      : ["Modele retenu", "Overfitting detecte"]
    })
    st.dataframe(comp, use_container_width=True)
    st.info("XGBoost = 100% car overfitting sur Community Area. Random Forest retenu.")

# ══════════════════════════════════════════════════════════════════
# PAGE 6 : MOSIEF / ROI
# ══════════════════════════════════════════════════════════════════
elif page == "MOSIEF / ROI":
    st.title("Analyse economique — MOSIEF / ROI")

    st.subheader("Couts estimés de la criminalité")

    couts = {
        "HOMICIDE": 10_000_000, "CRIMINAL SEXUAL ASSAULT": 240_000,
        "ROBBERY": 67_000,      "ASSAULT": 24_000,
        "BATTERY": 24_000,      "BURGLARY": 18_000,
        "MOTOR VEHICLE THEFT": 10_000, "THEFT": 4_000,
        "NARCOTICS": 5_000,     "CRIMINAL DAMAGE": 6_000,
    }

    total = sum(
        couts.get(c, 3000) * n
        for c, n in df["Primary Type"].value_counts().items()
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Cout total estime",    f"${total/1e9:.2f} Mrd USD")
    col2.metric("Cout moyen/incident",  f"${total/len(df):,.0f} USD")
    col3.metric("Reduction estimee 5%", f"${total*0.05/1e6:.0f} M USD")

    st.markdown("---")
    st.subheader("ROI projete sur 5 ans")

    investissement = 5_000_000
    reduction_pct  = 0.05
    economies      = [total * reduction_pct * (i+1) for i in range(5)]
    couts_cumuls   = [investissement] * 5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1,6)), y=[e/1e6 for e in economies],
                             name="Benefices cumules", line=dict(color="green", width=3)))
    fig.add_trace(go.Scatter(x=list(range(1,6)), y=[c/1e6 for c in couts_cumuls],
                             name="Couts RiskMap_AI", line=dict(color="red", width=3)))
    fig.update_layout(title="ROI projete sur 5 ans",
                      xaxis_title="Annee", yaxis_title="Millions USD")
    st.plotly_chart(fig, use_container_width=True)

    roi = (sum(economies) - investissement*5) / (investissement*5) * 100
    st.success(f"ROI estime sur 5 ans : {roi:.0f}x l'investissement initial") 