"""Exécute la chaîne v2 complète : les quatre notebooks dans l'ordre, sorties enregistrées.

Usage, depuis le dossier v2 :
    python run_v2.py            # chaîne complète (environ une heure sur 4 cœurs)
    python run_v2.py --rapide   # version courte du notebook 2
"""

import argparse
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

V2 = Path(__file__).resolve().parent
NOTEBOOKS = [
    "1_generation_trajectoires_v2.ipynb",
    "2_classification_ML_v2.ipynb",
    "3_PCA_clustering_v2.ipynb",
    "4_visualisations_geospatiales_v2.ipynb",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rapide", action="store_true", help="moins de tirages dans le notebook 2")
    args = parser.parse_args()
    for nom in NOTEBOOKS:
        nb = nbformat.read(V2 / nom, as_version=4)
        if args.rapide:
            for cellule in nb.cells:
                if cellule.cell_type == "code":
                    cellule.source = cellule.source.replace("RAPIDE = False", "RAPIDE = True")
        t0 = time.time()
        NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(V2)}}).execute()
        nbformat.write(nb, V2 / nom)
        print(f"{nom} exécuté en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
