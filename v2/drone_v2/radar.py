"""Modèle d'observation : réseau radar de surveillance ukrainien (fictif).

Hypothèses de simulation (aucune position réelle n'est utilisée) :
* sites tirés au hasard (tirage de Poisson-disque) dans la zone contrôlée par
  l'Ukraine, espacés d'au moins 90 km et à plus de 15 km de la ligne de contact ;
* portée limitée par l'horizon radioélectrique, d = 4,12 (sqrt(h_radar) + sqrt(h_cible)) km,
  et par la surface équivalente radar (SER) : R = 250 km x SER^(1/4) (loi du radar) ;
* probabilité de détection par balayage qui chute près de la portée limite et en
  très basse altitude (masquage par le relief et le fouillis de sol) ;
* bruit de mesure croissant avec la distance : 60 m + 2 m/km en position,
  40 m + 3 m/km en altitude (radar 3D) ;
* une piste est coupée après 2 min sans détection et n'est confirmée qu'à partir
  de 18 détections (3 min) : un même vol peut donc produire plusieurs pistes.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from . import catalogue as cat
from . import config
from .geo import Theatre, vers_lonlat

ECART_MAX_S = 120.0
N_MIN_PLOTS = 18


class ReseauRadar:
    def __init__(self, theatre: Theatre, graine: int = config.GRAINE, espacement_km: float = 90.0,
                 recul_km: float = 15.0, portee_ref_km: float = 250.0):
        rng = np.random.default_rng([graine, 99])
        xmin, ymin, xmax, ymax = theatre.ua_controle_xy.bounds
        sites = []
        for _ in range(30000):
            p = rng.uniform([xmin, ymin], [xmax, ymax])
            if not theatre.est_cote_ua(*p) or theatre.distance_signee_km(*p)[0] < recul_km:
                continue
            if sites and np.min(np.hypot(*(np.asarray(sites) - p).T)) < espacement_km * 1000:
                continue
            sites.append(p)
        self.sites = np.asarray(sites)
        self.hauteur_m = rng.uniform(10, 30, len(self.sites))
        self.portee_ref_km = portee_ref_km

    def table_sites(self) -> pd.DataFrame:
        lon, lat = vers_lonlat(self.sites[:, 0], self.sites[:, 1])
        return pd.DataFrame(dict(site=[f"RAD{i:02d}" for i in range(len(self.sites))],
                                 lon=lon, lat=lat, hauteur_m=self.hauteur_m))

    def portee_km(self, z_m, ser_m2):
        """Portée de détection (km) pour une altitude de cible et une SER données."""
        horizon = 4.12 * (np.sqrt(self.hauteur_m)[None, :] + np.sqrt(np.maximum(z_m, 1.0))[:, None])
        return np.minimum(horizon, self.portee_ref_km * ser_m2 ** 0.25)

    def observer(self, x, y, z, ser_m2, rng, facteur_bruit=1.0):
        x, y, z = (np.asarray(a, float) for a in (x, y, z))
        r = np.hypot(x[:, None] - self.sites[None, :, 0], y[:, None] - self.sites[None, :, 1]) / 1000.0
        r_lim = self.portee_km(z, ser_m2)
        pd_site = 0.97 / (1.0 + np.exp(np.clip((r - r_lim) / (0.06 * r_lim), -50, 50)))
        pd_site *= (0.55 + 0.45 * np.clip(z / 300.0, 0.0, 1.0))[:, None]
        p_detection = 1.0 - np.prod(1.0 - pd_site, axis=1)
        detecte = rng.random(len(x)) < p_detection
        meilleur = np.argmax(pd_site, axis=1)
        r_best = r[np.arange(len(x)), meilleur]
        s_xy = (60.0 + 2.0 * r_best) * facteur_bruit
        s_z = (40.0 + 3.0 * r_best) * facteur_bruit
        xm = x + rng.normal(0.0, 1.0, len(x)) * s_xy
        ym = y + rng.normal(0.0, 1.0, len(x)) * s_xy
        zm = np.maximum(z + rng.normal(0.0, 1.0, len(x)) * s_z, 0.0)
        return detecte, xm, ym, zm, r_best


def fragments(t, detecte, ecart_max_s=ECART_MAX_S, n_min=N_MIN_PLOTS):
    """Indices des détections regroupées en pistes confirmées."""
    idx = np.flatnonzero(detecte)
    if idx.size < n_min:
        return []
    coupures = np.flatnonzero(np.diff(t[idx]) > ecart_max_s) + 1
    return [g for g in np.split(idx, coupures) if len(g) >= n_min]


def graine_vol(graine, drone_id, cle=0):
    return np.random.default_rng([graine, 7, cle, zlib.crc32(drone_id.encode())])


def observer_vols(vols, radar: ReseauRadar, graine: int = config.GRAINE, facteur_bruit: float = 1.0, cle: int = 0):
    """Observe chaque vol et renvoie (table des pistes, table des plots radar)."""
    pistes, plots = [], []
    for v in vols:
        rng = graine_vol(graine, v.drone_id, cle)
        ser = cat.CATALOGUE[v.modele]["hyp_ser_m2"]
        detecte, xm, ym, zm, _ = radar.observer(v.x, v.y, v.z, ser, rng, facteur_bruit)
        frags = fragments(v.t, detecte)
        for j, idx in enumerate(frags):
            piste_id = f"{v.drone_id}_P{j}"
            lon, lat = vers_lonlat(xm[idx], ym[idx])
            plots.append(pd.DataFrame(dict(piste_id=piste_id, t=v.t[idx], lon=lon, lat=lat,
                                           alt=zm[idx].astype(np.float32))))
            pistes.append(dict(
                piste_id=piste_id, drone_id=v.drone_id, raid_id=v.raid_id, groupe_id=v.groupe_id,
                jour=v.jour, modele=v.modele, role=v.role, origine=v.origine, fin_vol=v.fin,
                fragment=j, nb_fragments=len(frags), n_plots=len(idx),
                t_debut=float(v.t[idx[0]]), t_fin=float(v.t[idx[-1]]),
                lon_debut=float(lon[0]), lat_debut=float(lat[0]), lon_fin=float(lon[-1]), lat_fin=float(lat[-1]),
                vol_debut_observe=bool(idx[0] == 0), vol_fin_observee=bool(idx[-1] == len(v.t) - 1),
            ))
    colonnes = ["piste_id", "t", "lon", "lat", "alt"]
    plots = pd.concat(plots, ignore_index=True) if plots else pd.DataFrame(columns=colonnes)
    return pd.DataFrame(pistes), plots
