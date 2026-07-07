# roi_calculator.py — Calcul du retour sur investissement

import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
           os.path.dirname(os.path.abspath(__file__)))))

def calculer_roi(cout_criminalite_annuel=1_650_000_000,
                 investissement=5_000_000,
                 reduction_pct=0.05,
                 nb_annees=5):

    print("CALCUL ROI — RiskMap_AI\n" + "="*40)
    print(f"Cout criminalite/an  : ${cout_criminalite_annuel/1e9:.2f} Mrd USD")
    print(f"Investissement       : ${investissement/1e6:.1f} M USD")
    print(f"Reduction estimee    : {reduction_pct*100:.0f}%")
    print(f"Periode              : {nb_annees} ans\n")

    economies_annuelles = cout_criminalite_annuel * reduction_pct
    print(f"Economies/an         : ${economies_annuelles/1e6:.0f} M USD")

    benefices_cumuls = []
    couts_cumuls     = []
    benefices_nets   = []

    for annee in range(1, nb_annees + 1):
        ben  = economies_annuelles * annee
        cout = investissement * annee
        benefices_cumuls.append(ben)
        couts_cumuls.append(cout)
        benefices_nets.append(ben - cout)
        print(f"  An {annee} : benefice net = ${(ben-cout)/1e6:.0f} M USD")

    roi_total = (sum(benefices_cumuls) - sum(couts_cumuls)) / sum(couts_cumuls) * 100
    print(f"\nROI total sur {nb_annees} ans : {roi_total:.0f}%")
    print(f"Multiplicateur       : x{roi_total/100:.0f} l'investissement")

    # Graphique
    annees = list(range(1, nb_annees + 1))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(annees, [b/1e6 for b in benefices_cumuls],
                 "g-o", linewidth=3, label="Benefices cumules")
    axes[0].plot(annees, [c/1e6 for c in couts_cumuls],
                 "r-o", linewidth=3, label="Couts cumules")
    axes[0].fill_between(annees,
                         [b/1e6 for b in benefices_cumuls],
                         [c/1e6 for c in couts_cumuls],
                         alpha=0.3, color="green", label="Benefice net")
    axes[0].set_title("ROI projete sur 5 ans — RiskMap_AI")
    axes[0].set_xlabel("Annee")
    axes[0].set_ylabel("Millions USD")
    axes[0].legend()
    axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.5)

    axes[1].bar(annees, [b/1e6 for b in benefices_nets],
                color=["green" if b > 0 else "red" for b in benefices_nets])
    axes[1].set_title("Benefice net par annee (M$)")
    axes[1].set_xlabel("Annee")
    axes[1].set_ylabel("Millions USD")
    for i, v in enumerate([b/1e6 for b in benefices_nets]):
        axes[1].text(i+1, v+5, f"${v:.0f}M", ha="center", fontweight="bold")

    plt.tight_layout()
    sortie = os.path.join(BASE_DIR, "reports", "figures", "roi_projection.png")
    plt.savefig(sortie, dpi=150, bbox_inches="tight")
    print(f"Graphique sauvegarde : {sortie}")
    return roi_total

if __name__ == "__main__":
    calculer_roi()