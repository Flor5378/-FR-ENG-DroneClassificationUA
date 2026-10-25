"""Catalogue des drones simulés en v2.

Chaque modèle combine :
* des valeurs OSINT publiées (colonnes sans préfixe), avec leurs sources ;
* des hypothèses de simulation (préfixe ``hyp_``) quand aucune valeur publique
  fiable n'existe (altitude d'emploi, SER radar, durée typique de mission...).

Les fourchettes publiques divergent souvent d'une source à l'autre : la v2 tire
les paramètres de chaque vol dans ces fourchettes, ce qui crée des recouvrements
réalistes entre modèles au lieu de classes séparables par construction (défaut
principal de la v1, où vitesse et altitude dépendaient directement de la classe).
"""

from __future__ import annotations

import pandas as pd

ROLES = ["Frappe longue portée", "Leurre", "Munition rôdeuse", "Reconnaissance"]
ORIGINES = ["Russie", "Ukraine"]

SOURCES = {
    "WIKI_SHAHED": "https://en.wikipedia.org/wiki/HESA_Shahed_136",
    "GS_GERAN2": "https://www.globalsecurity.org/military/world/russia/geran-2.html",
    "GUR_GERAN2": "https://war-sanctions.gur.gov.ua/en/uav/336",
    "BI_SHAHED_ALT": "https://www.yahoo.com/news/russia-drone-tactic-dive-bombing-060045080.html",
    "EMP_SHAHED_2025": "https://euromaidanpress.com/2025/06/29/why-cant-ukraine-stop-russias-shahed-drones-anymore/",
    "WIKI_GERBERA": "https://en.wikipedia.org/wiki/Gerbera_(drone)",
    "GUR_GERBERA": "https://war-sanctions.gur.gov.ua/en/uav/329",
    "CAPSS_GERBERA": "https://capssindia.org/?p=16521",
    "ISIS_LEURRES": "https://isis-online.org/isis-reports/a-comprehensive-analytical-review-of-russian-shahed-type-uavs-deployment-against-ukraine-in-2025",
    "WIKI_LANCET": "https://en.wikipedia.org/wiki/ZALA_Lancet",
    "GS_LANCET": "https://www.globalsecurity.org/military/world/russia/lancet-specs.htm",
    "GS_ORLAN10": "https://www.globalsecurity.org/military/world/russia/orlan-10-specs.htm",
    "WIKI_ORLAN10": "https://en.wikipedia.org/wiki/Orlan-10",
    "GUR_S350": "https://war-sanctions.gur.gov.ua/en/uav/217",
    "WIKI_LIUTYI": "https://en.wikipedia.org/wiki/Liutyi",
    "WIKI_BOBER": "https://en.wikipedia.org/wiki/Bober_(drone)",
    "NV_BOBER": "https://english.nv.ua/nation/uj-26-bober-the-drone-hitting-deep-inside-russia-50619436.html",
    "GS_UJ22": "https://www.globalsecurity.org/military/world/ukraine/uj-22-specs.htm",
    "WIKI_WARMATE": "https://en.wikipedia.org/wiki/WB_Electronics_Warmate",
    "AT_LELEKA": "https://www.army-technology.com/projects/leleka-100-unmanned-aerial-vehicle-uav-ukraine/",
    "USS_LELEKA": "https://ukrspecsystems.com/drones/leleka-100-electric-uav",
    "GS_FURIA": "https://www.globalsecurity.org/military/world/ukraine/furia-uav.htm",
    "MSPO_FURIA": "https://www.targikielce.pl/en/mspo/list-of-exhibitors/athlon-avia,3818965",
    "MVS_VALKYRIA": "https://mvs.gov.ua/en/news/u-tilu-voroga-prikordonniki-vikonuiut-zavdannia-zavdiaki-bezpilotnomu-kompleksu-valkiriia-video",
    "UP_VALKYRIA": "https://www.pravda.com.ua/eng/news/2026/02/04/8019364",
}

# Profils de mission utilisés par le simulateur.
PROFIL_FRAPPE = "frappe_points_de_passage"
PROFIL_LEURRE = "leurre_saturation"
PROFIL_RODEUSE = "munition_rodeuse"
PROFIL_RECO = "reconnaissance_patrouille"

# vitesses en km/h, altitudes en m, portées en km, durées en h.
_MODELES = [
    dict(
        modele="Shahed-136 (Geran-2)", origine="Russie", role="Frappe longue portée",
        profil=PROFIL_FRAPPE,
        v_croisiere_min=150, v_croisiere_max=185, v_max=210,
        alt_min=60, alt_max=4000, portee_km=2000, endurance_h=12,
        hyp_alt_basse=(100, 600), hyp_alt_haute=(1500, 3000), hyp_part_haute=0.6,
        hyp_v_pique=250, hyp_ser_m2=0.5, hyp_duree_mission_h=None, hyp_rayon_km=None,
        sources="WIKI_SHAHED;GS_GERAN2;GUR_GERAN2;BI_SHAHED_ALT;EMP_SHAHED_2025",
        note="Croisière 150-185 km/h, bande 60-4000 m ; vol haut 1,5-3 km et piqué terminal observés en 2025.",
    ),
    dict(
        modele="Gerbera", origine="Russie", role="Leurre",
        profil=PROFIL_LEURRE,
        v_croisiere_min=90, v_croisiere_max=130, v_max=160,
        alt_min=100, alt_max=3000, portee_km=600, endurance_h=None,
        hyp_alt_basse=(300, 1200), hyp_alt_haute=(1500, 2800), hyp_part_haute=0.4,
        hyp_v_pique=None, hyp_ser_m2=0.5, hyp_duree_mission_h=None, hyp_rayon_km=None,
        sources="WIKI_GERBERA;GUR_GERBERA;CAPSS_GERBERA;ISIS_LEURRES",
        note="Leurre en mousse et contreplaqué, croisière ~97 km/h, max 160 km/h, portée 300-600 km ; SER augmentée (réflecteurs) par hypothèse pour imiter un Shahed.",
    ),
    dict(
        modele="Lancet-3", origine="Russie", role="Munition rôdeuse",
        profil=PROFIL_RODEUSE,
        v_croisiere_min=80, v_croisiere_max=110, v_max=300,
        alt_min=None, alt_max=None, portee_km=40, endurance_h=1.0,
        hyp_alt_basse=(200, 1200), hyp_alt_haute=None, hyp_part_haute=0.0,
        hyp_v_pique=300, hyp_ser_m2=0.03, hyp_duree_mission_h=(0.3, 0.67), hyp_rayon_km=40,
        sources="WIKI_LANCET;GS_LANCET",
        note="Croisière 80-110 km/h, piqué jusqu'à 300 km/h, portée ~40 km (70 km pour une source), 40 à 60 min de vol selon les sources.",
    ),
    dict(
        modele="Orlan-10", origine="Russie", role="Reconnaissance",
        profil=PROFIL_RECO,
        v_croisiere_min=90, v_croisiere_max=120, v_max=150,
        alt_min=None, alt_max=5000, portee_km=120, endurance_h=16,
        hyp_alt_basse=(800, 2500), hyp_alt_haute=None, hyp_part_haute=0.0,
        hyp_v_pique=None, hyp_ser_m2=0.08, hyp_duree_mission_h=(2.0, 6.0), hyp_rayon_km=110,
        sources="GS_ORLAN10;WIKI_ORLAN10",
        note="Vitesse max 150 km/h, plafond ~5000 m, endurance jusqu'à 16 h, rayon d'action ~110-120 km.",
    ),
    dict(
        modele="Supercam S350", origine="Russie", role="Reconnaissance",
        profil=PROFIL_RECO,
        v_croisiere_min=75, v_croisiere_max=120, v_max=120,
        alt_min=150, alt_max=5000, portee_km=100, endurance_h=4.5,
        hyp_alt_basse=(400, 2500), hyp_alt_haute=None, hyp_part_haute=0.0,
        hyp_v_pique=None, hyp_ser_m2=0.06, hyp_duree_mission_h=(1.5, 4.0), hyp_rayon_km=80,
        sources="GUR_S350",
        note="65-120 km/h, endurance ~4,5 h, liaison radio 70-100 km, bande d'altitude 150-5000 m (valeurs constructeur).",
    ),
    dict(
        modele="An-196 Liutyi", origine="Ukraine", role="Frappe longue portée",
        profil=PROFIL_FRAPPE,
        v_croisiere_min=150, v_croisiere_max=200, v_max=200,
        alt_min=None, alt_max=None, portee_km=2000, endurance_h=None,
        hyp_alt_basse=(100, 800), hyp_alt_haute=(1500, 3000), hyp_part_haute=0.3,
        hyp_v_pique=230, hyp_ser_m2=0.8, hyp_duree_mission_h=None, hyp_rayon_km=None,
        sources="WIKI_LIUTYI",
        note="Portée 1000-2000 km publiée ; vitesse non officielle (hypothèse 150-200 km/h), altitude non publiée.",
    ),
    dict(
        modele="UJ-26 Bober (Babor v1)", origine="Ukraine", role="Frappe longue portée",
        profil=PROFIL_FRAPPE,
        v_croisiere_min=120, v_croisiere_max=200, v_max=200,
        alt_min=None, alt_max=None, portee_km=1000, endurance_h=7,
        hyp_alt_basse=(50, 600), hyp_alt_haute=None, hyp_part_haute=0.0,
        hyp_v_pique=220, hyp_ser_m2=0.3, hyp_duree_mission_h=None, hyp_rayon_km=None,
        sources="WIKI_BOBER;NV_BOBER",
        note="Portée 600-1000 km, vitesse 120-200 km/h non vérifiée, vol ~7 h ; nommé « Babor » dans la v1.",
    ),
    dict(
        modele="UJ-22 Airborne", origine="Ukraine", role="Frappe longue portée",
        profil=PROFIL_FRAPPE,
        v_croisiere_min=120, v_croisiere_max=140, v_max=160,
        alt_min=50, alt_max=6000, portee_km=800, endurance_h=7,
        hyp_alt_basse=(300, 1500), hyp_alt_haute=(2000, 4000), hyp_part_haute=0.4,
        hyp_v_pique=200, hyp_ser_m2=0.4, hyp_duree_mission_h=None, hyp_rayon_km=None,
        sources="GS_UJ22",
        note="Croisière 120-140 km/h, max 160 km/h, plafond 6000 m, 800 km en pilote automatique, ~7 h de vol.",
    ),
    dict(
        modele="Warmate", origine="Ukraine", role="Munition rôdeuse",
        profil=PROFIL_RODEUSE,
        v_croisiere_min=80, v_croisiere_max=120, v_max=150,
        alt_min=None, alt_max=None, portee_km=30, endurance_h=1.0,
        hyp_alt_basse=(150, 1000), hyp_alt_haute=None, hyp_part_haute=0.0,
        hyp_v_pique=150, hyp_ser_m2=0.02, hyp_duree_mission_h=(0.3, 1.0), hyp_rayon_km=30,
        sources="WIKI_WARMATE",
        note="Croisière 80-120 km/h, piqué 150 km/h, portée ~30 km, ~60 min de vol.",
    ),
    dict(
        modele="Leleka-100", origine="Ukraine", role="Reconnaissance",
        profil=PROFIL_RECO,
        v_croisiere_min=65, v_croisiere_max=80, v_max=100,
        alt_min=None, alt_max=1500, portee_km=45, endurance_h=2.5,
        hyp_alt_basse=(300, 1400), hyp_alt_haute=None, hyp_part_haute=0.0,
        hyp_v_pique=None, hyp_ser_m2=0.02, hyp_duree_mission_h=(1.0, 2.3), hyp_rayon_km=40,
        sources="AT_LELEKA;USS_LELEKA",
        note="Croisière 70-75 km/h, plafond 1500 m, 2,5 h d'endurance, liaison 45 km.",
    ),
    dict(
        modele="A1-SM Furia", origine="Ukraine", role="Reconnaissance",
        profil=PROFIL_RECO,
        v_croisiere_min=65, v_croisiere_max=85, v_max=130,
        alt_min=None, alt_max=None, portee_km=50, endurance_h=3.0,
        hyp_alt_basse=(300, 1500), hyp_alt_haute=None, hyp_part_haute=0.0,
        hyp_v_pique=None, hyp_ser_m2=0.02, hyp_duree_mission_h=(1.0, 2.8), hyp_rayon_km=50,
        sources="GS_FURIA;MSPO_FURIA",
        note="Croisière ~65-80 km/h (fiche constructeur), portée 50-70 km, jusqu'à 3-4 h de vol ; altitude non publiée.",
    ),
    dict(
        modele="Valkyria", origine="Ukraine", role="Reconnaissance",
        profil=PROFIL_RECO,
        v_croisiere_min=55, v_croisiere_max=75, v_max=108,
        alt_min=30, alt_max=2000, portee_km=35, endurance_h=2.0,
        hyp_alt_basse=(200, 1500), hyp_alt_haute=None, hyp_part_haute=0.0,
        hyp_v_pique=None, hyp_ser_m2=0.015, hyp_duree_mission_h=(0.7, 1.8), hyp_rayon_km=30,
        sources="MVS_VALKYRIA;UP_VALKYRIA",
        note="Portée 30-35 km, plus de 2 h de vol ; vitesses et altitudes reprises de la v1 (« Valkyrie »).",
    ),
]

CATALOGUE = {m["modele"]: m for m in _MODELES}
MODELES = list(CATALOGUE)


def catalogue_df() -> pd.DataFrame:
    """Catalogue sous forme de table, avec les URL des sources."""
    lignes = []
    for m in _MODELES:
        ligne = {k: v for k, v in m.items() if not isinstance(v, tuple)}
        for k in ("hyp_alt_basse", "hyp_alt_haute", "hyp_duree_mission_h"):
            v = m[k]
            ligne[k] = f"{v[0]}-{v[1]}" if v else ""
        bandes = [b for b in (m["hyp_alt_basse"], m["hyp_alt_haute"]) if b]
        ligne["alt_simulee_min"] = min(b[0] for b in bandes)
        ligne["alt_simulee_max"] = max(b[1] for b in bandes)
        ligne["urls_sources"] = " ".join(SOURCES[s] for s in m["sources"].split(";"))
        lignes.append(ligne)
    return pd.DataFrame(lignes)
