# economic_analysis.py — Impact économique de la criminalité

import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COUTS_PAR_CRIME = {
    "HOMICIDE": 10_000_000, "CRIMINAL SEXUAL ASSAULT": 240_000,
    "ROBBERY": 67_000,      "ASSAULT": 24_000,
    "BATTERY": 24_000,      "BURGLARY": 18_000,
    "MOTOR VEHICLE THEFT": 10_000, "THEFT": 4_000,
    "NARCOTICS": 5_000,     "CRIMINAL DAMAGE": 6_000,
    "WEAPONS VIOLATION": 50_000,   "ARSON": 30_000,
}

def analyser_impact():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "crimes_features.csv"))
    print("ANALYSE ECONOMIQUE — Chicago 2024-2026\n" + "="*45)

    # Calcul du coût par type de crime
    resultats = []
    for crime, count in df["Primary Type"].value_counts().items():
        cout_unit = COUTS_PAR_CRIME.get(crime, 3_000)
        cout_total = cout_unit * count
        resultats.append({"Crime": crime, "Incidents": count,
                          "Cout_unitaire": cout_unit, "Cout_total": cout_total})

    df_eco = pd.DataFrame(resultats).sort_values("Cout_total", ascending=False)
    total = df_eco["Cout_total"].sum()

    print(f"\nTop 10 crimes par impact economique :")
    print(df_eco.head(10)[["Crime","Incidents","Cout_total"]].to_string(index=False))
    print(f"\nCout total estime  : ${total/1e9:.2f} milliards USD")
    print(f"Cout moyen/incident: ${total/len(df):,.0f} USD")
    print(f"Reduction 5%       : ${total*0.05/1e6:.0f} M USD/an")

    # Graphique
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    top10 = df_eco.head(10)
    axes[0].barh(top10["Crime"][::-1], top10["Cout_total"][::-1]/1e6, color="tomato")
    axes[0].set_title("Impact economique par type de crime (M$)")
    axes[0].set_xlabel("Millions USD")

    # Par niveau de risque
    freq_area  = df.groupby("Community Area").size()
    seuil_haut = freq_area.quantile(0.66)
    seuil_bas  = freq_area.quantile(0.33)
    df["cible"] = df["Community Area"].apply(
        lambda a: 2 if freq_area.get(a,0) >= seuil_haut
                  else 1 if freq_area.get(a,0) >= seuil_bas else 0)
    df["cout"] = df["Primary Type"].map(COUTS_PAR_CRIME).fillna(3000)
    cout_risque = df.groupby("cible")["cout"].sum() / 1e6
    axes[1].bar(["Faible","Moyen","Eleve"], cout_risque.values,
                color=["green","orange","red"])
    axes[1].set_title("Cout total par niveau de risque (M$)")
    axes[1].set_ylabel("Millions USD")
    for i, v in enumerate(cout_risque.values):
        axes[1].text(i, v + 10, f"${v:.0f}M", ha="center", fontweight="bold")

    plt.tight_layout()
    sortie = os.path.join(BASE_DIR, "reports", "figures", "impact_economique.png")
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    print(f"\nGraphique sauvegarde : {sortie}")
    return df_eco, total

if __name__ == "__main__":
    analyser_impact()