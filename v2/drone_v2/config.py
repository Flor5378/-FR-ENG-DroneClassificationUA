"""Chemins et constantes globales de la v2."""

from pathlib import Path

RACINE_V2 = Path(__file__).resolve().parents[1]
DATA = RACINE_V2 / "data"
GEO = DATA / "geo"
REFERENTIELS = DATA / "referentiels"
GENEREES = DATA / "generees"
FIGURES = RACINE_V2 / "figures"
CARTES = RACINE_V2 / "cartes"
RAPPORTS = RACINE_V2 / "rapports"

FICHIER_PAYS = GEO / "pays_naturalearth_50m.geojson"

# Graine unique : toute la chaîne (simulation, découpage, modèles) est reproductible.
GRAINE = 2026

# Période simulée (dates fictives, uniquement pour ordonner les données dans le temps).
DATE_DEBUT = "2025-06-01"
NB_JOURS = 120
# Les 30 derniers jours forment le jeu de test temporel, jamais vu pendant la mise au point.
NB_JOURS_TEST = 30

# Radar : période de rafraîchissement des pistes (s).
DT_RADAR_S = 10.0

for _d in (GENEREES, FIGURES, CARTES, RAPPORTS):
    _d.mkdir(parents=True, exist_ok=True)
