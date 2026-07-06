# =============================================================================
# PHASE 9 — Machine Learning : entraînement et évaluation des modèles
# Fichier : src/models/train_model.py
# =============================================================================

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier

# =============================================================================
# CONFIGURATION
# =============================================================================

DOSSIER_MODELES  = os.path.join("models")
DOSSIER_FIGURES  = os.path.join("reports", "figures")
DOSSIER_RAPPORTS = os.path.join("reports", "tables")

os.makedirs(DOSSIER_MODELES,  exist_ok=True)
os.makedirs(DOSSIER_FIGURES,  exist_ok=True)
os.makedirs(DOSSIER_RAPPORTS, exist_ok=True)

LABELS = {0: "Faible", 1: "Moyen", 2: "Élevé"}
NOMS_CLASSES = ["Faible", "Moyen", "Élevé"]

# =============================================================================
# CHARGEMENT
# =============================================================================

print("=" * 60)
print("PHASE 9 — Machine Learning")
print("=" * 60)

print("\nChargement de features.csv...")
df = pd.read_csv(os.path.join("data", "processed", "features.csv"))
print(f"  {len(df):,} lignes | {df.shape[1]} colonnes")

CIBLE    = "niveau_risque"
FEATURES = [c for c in df.columns if c != CIBLE]

X = df[FEATURES]
y = df[CIBLE]

print(f"  Features : {len(FEATURES)}")
print(f"  Cible    : {CIBLE} — {y.nunique()} classes")

# =============================================================================
# SECTION 1 — Train / Test Split
# =============================================================================
# On divise en 80% entraînement et 20% test.
# stratify=y garantit que chaque split a la même proportion de classes.
# random_state=42 rend les résultats reproductibles.

print("\n[1/6] Train / Test Split (80% / 20%)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"  Entraînement : {len(X_train):,} lignes")
print(f"  Test         : {len(X_test):,} lignes")

# =============================================================================
# SECTION 2 — Entraînement Random Forest
# =============================================================================

print("\n[2/6] Entraînement Random Forest...")
print("  (peut prendre 1 à 3 minutes selon ta machine)")

rf = RandomForestClassifier(
    n_estimators=100,    # 100 arbres de décision
    max_depth=15,        # profondeur max de chaque arbre
    min_samples_leaf=10, # au moins 10 exemples par feuille (évite le surapprentissage)
    n_jobs=-1,           # utiliser tous les cœurs du processeur
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

acc_rf  = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average="weighted")
rec_rf  = recall_score(y_test, y_pred_rf, average="weighted")
f1_rf   = f1_score(y_test, y_pred_rf, average="weighted")

print(f"  Accuracy  : {acc_rf:.4f}  ({acc_rf*100:.2f}%)")
print(f"  Precision : {prec_rf:.4f}")
print(f"  Recall    : {rec_rf:.4f}")
print(f"  F1-Score  : {f1_rf:.4f}")

# =============================================================================
# SECTION 3 — Entraînement XGBoost
# =============================================================================

print("\n[3/6] Entraînement XGBoost...")
print("  (peut prendre 2 à 5 minutes selon ta machine)")

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,   # vitesse d'apprentissage
    subsample=0.8,       # 80% des données par arbre (régularisation)
    colsample_bytree=0.8,# 80% des features par arbre
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42,
    verbosity=0
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

acc_xgb  = accuracy_score(y_test, y_pred_xgb)
prec_xgb = precision_score(y_test, y_pred_xgb, average="weighted")
rec_xgb  = recall_score(y_test, y_pred_xgb, average="weighted")
f1_xgb   = f1_score(y_test, y_pred_xgb, average="weighted")

print(f"  Accuracy  : {acc_xgb:.4f}  ({acc_xgb*100:.2f}%)")
print(f"  Precision : {prec_xgb:.4f}")
print(f"  Recall    : {rec_xgb:.4f}")
print(f"  F1-Score  : {f1_xgb:.4f}")

# =============================================================================
# SECTION 4 — Comparaison et sélection du meilleur modèle
# =============================================================================

print("\n[4/6] Comparaison des modèles...")

resultats = pd.DataFrame({
    "Modèle"    : ["Random Forest", "XGBoost"],
    "Accuracy"  : [acc_rf,  acc_xgb],
    "Precision" : [prec_rf, prec_xgb],
    "Recall"    : [rec_rf,  rec_xgb],
    "F1-Score"  : [f1_rf,   f1_xgb],
})
print(resultats.to_string(index=False))

# Le meilleur modèle est celui avec le F1-Score le plus élevé.
# F1 est plus fiable qu'Accuracy quand les classes sont légèrement déséquilibrées.
if f1_rf >= f1_xgb:
    meilleur_modele = rf
    meilleur_nom    = "Random Forest"
    meilleur_f1     = f1_rf
    y_pred_meilleur = y_pred_rf
else:
    meilleur_modele = xgb
    meilleur_nom    = "XGBoost"
    meilleur_f1     = f1_xgb
    y_pred_meilleur = y_pred_xgb

print(f"\n  Meilleur modèle : {meilleur_nom} (F1 = {meilleur_f1:.4f})")

# Sauvegarde du tableau de comparaison
resultats.to_csv(os.path.join(DOSSIER_RAPPORTS, "comparaison_modeles.csv"), index=False)

# =============================================================================
# SECTION 5 — Rapport détaillé + matrices de confusion
# =============================================================================

print("\n[5/6] Rapports détaillés et visualisations...")

# --- Rapport de classification (precision/recall/f1 par classe) ---
print(f"\n  Rapport détaillé — {meilleur_nom} :")
print(classification_report(y_test, y_pred_meilleur, target_names=NOMS_CLASSES))

# --- Matrice de confusion ---
def tracer_confusion(y_vrai, y_pred, titre, nom_fichier):
    cm = confusion_matrix(y_vrai, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=NOMS_CLASSES, yticklabels=NOMS_CLASSES,
        linewidths=0.5
    )
    plt.title(titre, fontsize=13, fontweight="bold")
    plt.ylabel("Vraie classe")
    plt.xlabel("Classe prédite")
    plt.tight_layout()
    chemin = os.path.join(DOSSIER_FIGURES, nom_fichier)
    plt.savefig(chemin, bbox_inches="tight")
    plt.close()
    print(f"  Sauvegardée : {chemin}")

tracer_confusion(y_test, y_pred_rf,  "Matrice de confusion — Random Forest", "07_confusion_rf.png")
tracer_confusion(y_test, y_pred_xgb, "Matrice de confusion — XGBoost",       "08_confusion_xgb.png")

# --- Feature Importance du meilleur modèle ---
if meilleur_nom == "Random Forest":
    importances = meilleur_modele.feature_importances_
else:
    importances = meilleur_modele.feature_importances_

fi = pd.Series(importances, index=FEATURES).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
fi.tail(15).plot(kind="barh", color="#42A5F5", edgecolor="white")
plt.title(f"Top 15 features importantes — {meilleur_nom}", fontsize=13, fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
chemin_fi = os.path.join(DOSSIER_FIGURES, "09_feature_importance.png")
plt.savefig(chemin_fi, bbox_inches="tight")
plt.close()
print(f"  Sauvegardée : {chemin_fi}")

# --- Comparaison visuelle des métriques ---
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(4)
largeur = 0.35
metriques_rf  = [acc_rf,  prec_rf,  rec_rf,  f1_rf]
metriques_xgb = [acc_xgb, prec_xgb, rec_xgb, f1_xgb]

ax.bar(x - largeur/2, metriques_rf,  largeur, label="Random Forest", color="#42A5F5")
ax.bar(x + largeur/2, metriques_xgb, largeur, label="XGBoost",       color="#FF7043")
ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score"])
ax.set_ylim(0, 1.1)
ax.set_title("Comparaison des métriques : RF vs XGBoost", fontsize=13, fontweight="bold")
ax.legend()

for i, v in enumerate(metriques_rf):
    ax.text(i - largeur/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
for i, v in enumerate(metriques_xgb):
    ax.text(i + largeur/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

plt.tight_layout()
chemin_comp = os.path.join(DOSSIER_FIGURES, "10_comparaison_metriques.png")
plt.savefig(chemin_comp, bbox_inches="tight")
plt.close()
print(f"  Sauvegardée : {chemin_comp}")

# =============================================================================
# SECTION 6 — Sauvegarde du meilleur modèle
# =============================================================================

print("\n[6/6] Sauvegarde du meilleur modèle...")

chemin_modele = os.path.join(DOSSIER_MODELES, "best_model.pkl")
joblib.dump(meilleur_modele, chemin_modele)
print(f"  Modèle sauvegardé : {chemin_modele}")

# Sauvegarder aussi la liste des features pour la Phase 10
chemin_features = os.path.join(DOSSIER_MODELES, "features_list.pkl")
joblib.dump(FEATURES, chemin_features)
print(f"  Liste des features : {chemin_features}")

# Sauvegarder le nom du meilleur modèle
chemin_info = os.path.join(DOSSIER_MODELES, "model_info.txt")
with open(chemin_info, "w") as f:
    f.write(f"modele={meilleur_nom}\n")
    f.write(f"f1={meilleur_f1:.4f}\n")
    f.write(f"accuracy={max(acc_rf, acc_xgb):.4f}\n")
print(f"  Infos modèle       : {chemin_info}")

# =============================================================================
# RÉSUMÉ
# =============================================================================

print("\n" + "=" * 60)
print("RÉSUMÉ PHASE 9")
print("=" * 60)
print(f"\n  Random Forest — F1 : {f1_rf:.4f} | Accuracy : {acc_rf*100:.2f}%")
print(f"  XGBoost       — F1 : {f1_xgb:.4f} | Accuracy : {acc_xgb*100:.2f}%")
print(f"\n  Meilleur modèle    : {meilleur_nom}")
print(f"  Sauvegardé dans    : {chemin_modele}")
print(f"\n  Figures produites  :")
print(f"    - 07_confusion_rf.png")
print(f"    - 08_confusion_xgb.png")
print(f"    - 09_feature_importance.png")
print(f"    - 10_comparaison_metriques.png")
print("\nPhase 9 terminée avec succès.")
print("Tu peux maintenant passer à la Phase 10 — Prédictions.")