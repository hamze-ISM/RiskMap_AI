# integrity_check.py — Vérification SHA256 des données

import hashlib, os, json, datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

FICHIERS = [
    os.path.join(BASE_DIR, "data", "processed", "crimes_clean.csv"),
    os.path.join(BASE_DIR, "data", "processed", "crimes_features.csv"),
    os.path.join(BASE_DIR, "models", "random_forest.pkl"),
]

def calculer_hash(chemin):
    sha256 = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(4096), b""):
            sha256.update(bloc)
    return sha256.hexdigest()

def generer_hashes():
    print("Calcul des hashes SHA256...\n")
    hashes = {}
    for chemin in FICHIERS:
        if os.path.exists(chemin):
            h = calculer_hash(chemin)
            nom = os.path.basename(chemin)
            hashes[nom] = {"hash": h, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "chemin": chemin}
            print(f"  OK {nom}")
            print(f"     {h[:32]}...")
        else:
            print(f"  MANQUANT : {chemin}")
    sortie = os.path.join(BASE_DIR, "cyber", "integrity", "hashes.json")
    with open(sortie, "w") as f:
        json.dump(hashes, f, indent=2)
    print(f"\nHashes sauvegardes : {sortie}")
    return hashes

def verifier_hashes():
    chemin_ref = os.path.join(BASE_DIR, "cyber", "integrity", "hashes.json")
    if not os.path.exists(chemin_ref):
        print("Aucun hash de reference — lance d'abord generer_hashes()")
        return
    with open(chemin_ref) as f:
        hashes_ref = json.load(f)
    print("Verification de l integrite...\n")
    for nom, info in hashes_ref.items():
        if os.path.exists(info["chemin"]):
            h_actuel = calculer_hash(info["chemin"])
            if h_actuel == info["hash"]:
                print(f"  OK {nom} — integre")
            else:
                print(f"  ALERTE {nom} — MODIFIE !")
        else:
            print(f"  MANQUANT {nom}")

if __name__ == "__main__":
    generer_hashes()
    print("\n")
    verifier_hashes()