# adversarial_test.py — Tests d'attaques adversariales

import pandas as pd
import numpy as np
import joblib, json, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_adversarial():
    modele   = joblib.load(os.path.join(BASE_DIR, "models", "random_forest.pkl"))
    with open(os.path.join(BASE_DIR, "models", "features.json")) as f:
        features = json.load(f)

    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "crimes_features.csv"))
    df_test = df[features].dropna().sample(n=1000, random_state=42)

    y_original = modele.predict(df_test)
    print("Tests d attaques adversariales\n")

    # Test 1 : Bruit gaussien
    df_bruit = df_test.copy()
    df_bruit += np.random.normal(0, 0.5, df_bruit.shape)
    y_bruit = modele.predict(df_bruit)
    robustesse_bruit = (y_bruit == y_original).mean()
    print(f"  Test 1 - Bruit gaussien    : robustesse = {robustesse_bruit*100:.1f}%")

    # Test 2 : Valeurs extremes
    df_extreme = df_test.copy()
    df_extreme["heure"] = 23
    df_extreme["est_nuit"] = 1
    y_extreme = modele.predict(df_extreme)
    robustesse_extreme = (y_extreme == y_original).mean()
    print(f"  Test 2 - Valeurs extremes  : robustesse = {robustesse_extreme*100:.1f}%")

    # Test 3 : Permutation aleatoire
    df_perm = df_test.copy()
    df_perm["District"] = np.random.permutation(df_perm["District"].values)
    y_perm = modele.predict(df_perm)
    robustesse_perm = (y_perm == y_original).mean()
    print(f"  Test 3 - Permutation       : robustesse = {robustesse_perm*100:.1f}%")

    robustesse_moy = (robustesse_bruit + robustesse_extreme + robustesse_perm) / 3
    print(f"\n  Robustesse moyenne : {robustesse_moy*100:.1f}%")
    niveau = "ELEVE" if robustesse_moy > 0.85 else "MOYEN" if robustesse_moy > 0.70 else "FAIBLE"
    print(f"  Niveau de robustesse : {niveau}")

if __name__ == "__main__":
    test_adversarial()