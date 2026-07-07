# bias_analysis.py — Analyse des biais du modèle

import pandas as pd
import matplotlib.pyplot as plt
import os, joblib, json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyser_biais():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "crimes_features.csv"))
    modele = joblib.load(os.path.join(BASE_DIR, "models", "random_forest.pkl"))

    freq_area  = df.groupby("Community Area").size()
    seuil_haut = freq_area.quantile(0.66)
    seuil_bas  = freq_area.quantile(0.33)
    df["cible"] = df["Community Area"].apply(
        lambda a: 2 if freq_area.get(a,0) >= seuil_haut
                  else 1 if freq_area.get(a,0) >= seuil_bas else 0)

    FEATURES = ["heure","jour_semaine","mois","trimestre","est_weekend",
                "est_nuit","District","Community Area","Beat",
                "crime_type_code","Arrest","Domestic"]
    df_clean = df[FEATURES + ["cible"]].dropna()

    print("Analyse des biais par district...\n")
    resultats = []
    for district in sorted(df_clean["District"].unique()):
        df_d = df_clean[df_clean["District"] == district]
        if len(df_d) < 100:
            continue
        X_d = df_d[FEATURES]
        y_d = df_d["cible"]
        y_pred = modele.predict(X_d)
        acc = (y_pred == y_d.values).mean()
        resultats.append({"District": district, "Nb_incidents": len(df_d), "Accuracy": round(acc*100, 2)})

    df_res = pd.DataFrame(resultats).sort_values("Accuracy")
    print(df_res.to_string(index=False))

    # Graphique
    fig, ax = plt.subplots(figsize=(14, 6))
    couleurs = ["red" if a < 85 else "orange" if a < 92 else "green"
                for a in df_res["Accuracy"]]
    ax.bar(df_res["District"].astype(str), df_res["Accuracy"], color=couleurs)
    ax.axhline(y=df_res["Accuracy"].mean(), color="black",
               linestyle="--", label=f"Moyenne : {df_res['Accuracy'].mean():.1f}%")
    ax.set_title("Accuracy du modele par district — Analyse des biais")
    ax.set_xlabel("District")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    plt.tight_layout()
    sortie = os.path.join(BASE_DIR, "reports", "figures", "biais_par_district.png")
    plt.savefig(sortie, dpi=150)
    print(f"\nGraphique sauvegarde : {sortie}")
    return df_res

if __name__ == "__main__":
    analyser_biais()