# hash_raw_data.py
# Calcule et sauvegarde les empreintes SHA-256 des données brutes

import hashlib
import json
import os
from datetime import datetime

FICHIERS = {
    "chicago_2024": "data/raw/2024/chicago_crimes_2024.csv",
    "chicago_2025": "data/raw/2025/chicago_crimes_2025.csv",
    "chicago_2026": "data/raw/2026/chicago_crimes_2026.csv",
}

FICHIER_HASHES = "cyber/integrity/hashing/hashes_raw.json"

def calculer_sha256(chemin_fichier):
    """Calcule le hash SHA-256 d'un fichier."""
    sha256 = hashlib.sha256()
    with open(chemin_fichier, "rb") as f:
        for bloc in iter(lambda: f.read(65536), b""):
            sha256.update(bloc)
    return sha256.hexdigest()

def generer_hashes():
    print("🔐 Calcul des empreintes SHA-256 des données brutes...\n")
    hashes = {}

    for nom, chemin in FICHIERS.items():
        if os.path.exists(chemin):
            print(f"  ⏳ Calcul en cours : {chemin}")
            hash_val = calculer_sha256(chemin)
            taille = os.path.getsize(chemin)
            hashes[nom] = {
                "fichier": chemin,
                "sha256": hash_val,
                "taille_bytes": taille,
                "date_hachage": datetime.now().isoformat()
            }
            print(f"  ✅ {nom} : {hash_val[:20]}...")
        else:
            print(f"  ⚠️  Fichier manquant : {chemin}")

    # Sauvegarder les hashes dans un fichier JSON
    with open(FICHIER_HASHES, "w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Hashes sauvegardés dans : {FICHIER_HASHES}")
    return hashes

def verifier_hashes():
    """Vérifie que les fichiers n'ont pas été modifiés depuis le hachage initial."""
    print("\n🔍 Vérification de l'intégrité des fichiers...\n")

    if not os.path.exists(FICHIER_HASHES):
        print("  ❌ Fichier de hashes introuvable. Lance d'abord generer_hashes().")
        return

    with open(FICHIER_HASHES, "r", encoding="utf-8") as f:
        hashes_sauvegardes = json.load(f)

    for nom, info in hashes_sauvegardes.items():
        chemin = info["fichier"]
        if os.path.exists(chemin):
            hash_actuel = calculer_sha256(chemin)
            if hash_actuel == info["sha256"]:
                print(f"  ✅ {nom} : intégrité confirmée")
            else:
                print(f"  ❌ {nom} : FICHIER MODIFIÉ ! Hash différent.")
                print(f"     Attendu  : {info['sha256'][:30]}...")
                print(f"     Actuel   : {hash_actuel[:30]}...")
        else:
            print(f"  ⚠️  {nom} : fichier introuvable")

if __name__ == "__main__":
    generer_hashes()
    verifier_hashes()