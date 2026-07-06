# download_data.py — Téléchargement automatique des données Chicago

import urllib.request
import os
import sys

# ─── URLs API officielle Chicago (sans limite) ────────────────────
URLS = {
    "2024": "https://data.cityofchicago.org/api/views/3i3m-jwuy/rows.csv?accessType=DOWNLOAD",
    "2025": "https://data.cityofchicago.org/api/views/t7ek-mgzi/rows.csv?accessType=DOWNLOAD",
    "2026": "https://data.cityofchicago.org/api/views/x2n5-8w5q/rows.csv?accessType=DOWNLOAD",
}

DESTINATIONS = {
    "2024": r"C:\RiskMap_AI\data\raw\2024\crimes_2024.csv",
    "2025": r"C:\RiskMap_AI\data\raw\2025\crimes_2025.csv",
    "2026": r"C:\RiskMap_AI\data\raw\2026\crimes_2026.csv",
}

def afficher_progression(bloc, taille_bloc, taille_totale):
    """Affiche la progression du téléchargement."""
    if taille_totale > 0:
        pct = min(bloc * taille_bloc / taille_totale * 100, 100)
        mo_telecharge = bloc * taille_bloc / (1024 * 1024)
        mo_total = taille_totale / (1024 * 1024)
        print(f"\r  Progression : {pct:.1f}%  ({mo_telecharge:.1f} / {mo_total:.1f} Mo)", end="")
    else:
        mo_telecharge = bloc * taille_bloc / (1024 * 1024)
        print(f"\r  Téléchargé : {mo_telecharge:.1f} Mo", end="")

def telecharger(annee):
    url  = URLS[annee]
    dest = DESTINATIONS[annee]

    # Créer le dossier si nécessaire
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    # Déjà téléchargé ?
    if os.path.exists(dest):
        taille = os.path.getsize(dest) / (1024 * 1024)
        print(f"  ⏭️  crimes_{annee}.csv déjà présent ({taille:.1f} Mo) — ignoré")
        return True

    print(f"\n📥 Téléchargement crimes_{annee}.csv ...")
    print(f"   Source : {url}")
    print(f"   Dest   : {dest}")

    try:
        urllib.request.urlretrieve(url, dest, reporthook=afficher_progression)
        print()  # saut de ligne après la progression
        taille = os.path.getsize(dest) / (1024 * 1024)
        print(f"  ✅ Téléchargé : {taille:.1f} Mo")
        return True
    except Exception as e:
        print(f"\n  ❌ Erreur : {e}")
        return False

def main():
    print("🚀 Téléchargement des données Chicago Crime Dataset")
    print("=" * 55)

    resultats = {}
    for annee in ["2024", "2025", "2026"]:
        resultats[annee] = telecharger(annee)

    print(f"\n{'='*55}")
    print("RÉSUMÉ")
    print(f"{'='*55}")
    for annee, ok in resultats.items():
        statut = "✅ OK" if ok else "❌ ÉCHEC"
        print(f"  crimes_{annee}.csv : {statut}")

    if all(resultats.values()):
        print("\n🎉 Toutes les données sont prêtes !")
    else:
        print("\n⚠️  Certains téléchargements ont échoué.")
        print("   → Essaie le téléchargement manuel (voir instructions ci-dessous)")

if __name__ == "__main__":
    main()