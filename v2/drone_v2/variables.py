"""Extraction des variables par piste radar.

Règle anti-fuite : toutes les variables sont calculées à partir des seuls plots
radar observés (temps, position et altitude bruitées). Aucune grandeur interne du
simulateur (vitesse vraie, modèle, cible, cause de fin, phase de vol...) n'entre
dans la table des variables ; ``verifier_absence_de_fuite`` le contrôle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from . import config
from .geo import Theatre, vers_xy

DT = config.DT_RADAR_S

FAMILLES = {
    "observation": ["duree_min", "n_plots", "taux_detection"],
    "cinematique": [
        "longueur_km", "deplacement_km", "rectitude", "vitesse_moy", "vitesse_med", "vitesse_std",
        "vitesse_p10", "vitesse_p90", "vitesse_p95", "cv_vitesse", "accel_p95", "taux_virage_med",
        "taux_virage_p90", "virage_deg_par_min", "frac_virage", "rayon_giration_km", "compacite",
    ],
    "altitude": [
        "alt_moy", "alt_med", "alt_std", "alt_p05", "alt_p95", "alt_debut", "alt_fin",
        "vz_p05", "vz_p95", "descente_finale",
    ],
    "geographie": [
        "dist_front_debut_km", "dist_front_fin_km", "dist_front_moy_km", "dist_front_min_km",
        "dist_front_max_km", "frac_cote_ua", "vitesse_vers_ua_kmh", "cap_net_sin", "cap_net_cos",
        "dist_ville_fin_km",
    ],
    "temps": ["heure_sin", "heure_cos"],
}
VARIABLES = [v for liste in FAMILLES.values() for v in liste]

# Colonnes interdites dans la matrice X : identifiants, étiquettes et vérité terrain.
INTERDITES = {
    "piste_id", "drone_id", "raid_id", "groupe_id", "jour", "modele", "role", "origine", "fin_vol",
    "fragment", "nb_fragments", "t_debut", "t_fin", "vol_debut_observe", "vol_fin_observee",
    "lon_debut", "lat_debut", "lon_fin", "lat_fin",
    "site_lancement", "cible", "motif", "v_croisiere_kmh", "alt_consigne_m", "phase",
}


def _fenetre(n, souhaitee):
    w = min(souhaitee, n if n % 2 == 1 else n - 1)
    return max(w, 3)


def _villes_xy():
    villes = pd.read_csv(config.REFERENTIELS / "cibles_ukraine_v2.csv")
    return np.column_stack(vers_xy(villes.lon.to_numpy(), villes.lat.to_numpy()))


def variables_piste(t, x, y, z, theatre: Theatre, villes_xy) -> dict:
    """Variables d'une piste (t en s, x et y en m, z en m)."""
    grille = np.arange(t[0], t[-1] + 0.5 * DT, DT)
    n = len(grille)
    xg, yg, zg = (np.interp(grille, t, a) for a in (x, y, z))
    w = _fenetre(n, 9)
    po = min(2, w - 1)
    xs, ys = savgol_filter(xg, w, po), savgol_filter(yg, w, po)
    vx = savgol_filter(xg, w, po, deriv=1, delta=DT)
    vy = savgol_filter(yg, w, po, deriv=1, delta=DT)
    wa = _fenetre(n, 15)
    ax = savgol_filter(xg, wa, min(2, wa - 1), deriv=2, delta=DT)
    ay = savgol_filter(yg, wa, min(2, wa - 1), deriv=2, delta=DT)
    wz = _fenetre(n, 13)
    zs = savgol_filter(zg, wz, 1)
    vz = savgol_filter(zg, wz, 1, deriv=1, delta=DT)

    v_kmh = np.hypot(vx, vy) * 3.6
    cap = np.unwrap(np.arctan2(vy, vx))
    virage = np.abs(np.gradient(cap, DT)) * 180 / np.pi
    pas = np.hypot(np.diff(xs), np.diff(ys))
    longueur = pas.sum() / 1000.0
    dep = np.array([xs[-1] - xs[0], ys[-1] - ys[0]])
    deplacement = np.hypot(*dep) / 1000.0
    duree_min = (t[-1] - t[0]) / 60.0
    rg = np.sqrt(np.mean((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)) / 1000.0
    k = min(6, n)
    k2 = min(12, n - 1)

    # Contexte géographique, sur au plus 200 points régulièrement espacés.
    sel = np.unique(np.linspace(0, n - 1, min(n, 200)).astype(int))
    d_front = theatre.distance_signee_km(xs[sel], ys[sel])
    cote_ua = d_front > 0
    d_ville = np.min(np.hypot(villes_xy[:, 0] - xs[-1], villes_xy[:, 1] - ys[-1])) / 1000.0
    heure = (t[0] % 86400) / 3600.0
    cap_net = np.arctan2(dep[1], dep[0])

    return dict(
        duree_min=duree_min, n_plots=len(t), taux_detection=len(t) / n,
        longueur_km=longueur, deplacement_km=deplacement,
        rectitude=deplacement / max(longueur, 1e-3),
        vitesse_moy=v_kmh.mean(), vitesse_med=np.median(v_kmh), vitesse_std=v_kmh.std(),
        vitesse_p10=np.percentile(v_kmh, 10), vitesse_p90=np.percentile(v_kmh, 90),
        vitesse_p95=np.percentile(v_kmh, 95), cv_vitesse=v_kmh.std() / max(v_kmh.mean(), 1e-3),
        accel_p95=np.percentile(np.hypot(ax, ay), 95),
        taux_virage_med=np.median(virage), taux_virage_p90=np.percentile(virage, 90),
        virage_deg_par_min=np.abs(np.diff(cap)).sum() * 180 / np.pi / max(duree_min, 1e-3),
        frac_virage=float(np.mean(virage > 1.5)), rayon_giration_km=rg,
        compacite=rg / max(longueur, 1e-3),
        alt_moy=zs.mean(), alt_med=np.median(zs), alt_std=zs.std(),
        alt_p05=np.percentile(zs, 5), alt_p95=np.percentile(zs, 95),
        alt_debut=np.median(zs[:k]), alt_fin=np.median(zs[-k:]),
        vz_p05=np.percentile(vz, 5), vz_p95=np.percentile(vz, 95),
        descente_finale=(zs[-1] - zs[-1 - k2]) / (k2 * DT) if k2 > 0 else 0.0,
        dist_front_debut_km=d_front[0], dist_front_fin_km=d_front[-1], dist_front_moy_km=d_front.mean(),
        dist_front_min_km=d_front.min(), dist_front_max_km=d_front.max(),
        frac_cote_ua=float(cote_ua.mean()),
        vitesse_vers_ua_kmh=(d_front[-1] - d_front[0]) / max(duree_min / 60.0, 1e-3),
        cap_net_sin=np.sin(cap_net), cap_net_cos=np.cos(cap_net), dist_ville_fin_km=d_ville,
        heure_sin=np.sin(2 * np.pi * heure / 24), heure_cos=np.cos(2 * np.pi * heure / 24),
    )


def table_variables(pistes: pd.DataFrame, plots: pd.DataFrame, theatre: Theatre,
                    horizon_min: float | None = None) -> pd.DataFrame:
    """Une ligne par piste. ``horizon_min`` limite l'observation aux premières minutes."""
    villes = _villes_xy()
    lignes = []
    plots = plots.sort_values(["piste_id", "t"], kind="stable")
    for piste_id, g in plots.groupby("piste_id", sort=False):
        t = g.t.to_numpy()
        if horizon_min is not None:
            garde = t <= t[0] + horizon_min * 60.0
            if garde.sum() < 6:
                continue
            g, t = g[garde], t[garde]
        x, y = vers_xy(g.lon.to_numpy(), g.lat.to_numpy())
        v = variables_piste(t, x, y, g.alt.to_numpy(float), theatre, villes)
        v["piste_id"] = piste_id
        lignes.append(v)
    X = pd.DataFrame(lignes)
    meta = ["piste_id", "drone_id", "raid_id", "groupe_id", "jour", "modele", "role", "origine",
            "fin_vol", "fragment", "nb_fragments"]
    return pistes[meta].merge(X, on="piste_id", how="inner")


def verifier_absence_de_fuite(colonnes) -> None:
    """Lève une erreur si une colonne interdite figure parmi les variables explicatives."""
    fuites = sorted(set(colonnes) & INTERDITES)
    if fuites:
        raise ValueError(f"Colonnes interdites dans X : {fuites}")
    inconnues = sorted(set(colonnes) - set(VARIABLES))
    if inconnues:
        raise ValueError(f"Colonnes non documentées dans X : {inconnues}")
