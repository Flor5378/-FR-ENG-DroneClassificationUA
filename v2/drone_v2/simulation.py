"""Simulation des vols (vérité terrain) de la v2.

Chaque vol suit le profil de mission de son modèle (voir ``catalogue``) :
* frappe longue portée : route programmée par points de passage, avec détours,
  feintes et couloirs (frontière biélorusse, mer Noire), vol bas ou haut, piqué
  terminal ;
* leurre : accompagne un groupe de frappe puis s'en écarte et vole jusqu'à
  épuisement du carburant ;
* munition rôdeuse : transit court au-delà du front, orbites de recherche, piqué ;
* reconnaissance : transit, motif de patrouille (hippodrome, orbite, balayage,
  points d'intérêt), retour et atterrissage.

Le vent du jour modifie la vitesse sol, la défense adverse peut abattre ou
brouiller le drone pendant son vol. Les raids russes mêlent drones d'attaque et
leurres (part de leurres de 30 à 55 %, ordre de grandeur des déclarations
publiques de l'armée de l'air ukrainienne en 2025). Les volumes sont réduits
pour le calcul : un raid simulé compte quelques dizaines de drones, pas plusieurs
centaines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from shapely.geometry import Point

from . import catalogue as cat
from . import config
from .geo import Theatre, vers_lonlat, vers_xy

# Phases de vol
PH_TRANSIT, PH_PATROUILLE, PH_RETOUR, PH_PIQUE, PH_CHUTE, PH_PLANE, PH_ATTERRISSAGE = range(1, 8)
NOMS_PHASES = {
    PH_TRANSIT: "transit",
    PH_PATROUILLE: "patrouille",
    PH_RETOUR: "retour",
    PH_PIQUE: "piqué terminal",
    PH_CHUTE: "chute (abattu)",
    PH_PLANE: "descente planée",
    PH_ATTERRISSAGE: "atterrissage",
}

PAS_M = 100.0
KMH = 1 / 3.6


# Géométrie élémentaire

def _segment(a, b, pas=PAS_M):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = max(int(np.ceil(np.hypot(*(b - a)) / pas)), 1)
    f = np.linspace(0.0, 1.0, n + 1)[1:]
    return a + f[:, None] * (b - a)


def _arc(centre, rayon, a0, a1, pas=PAS_M):
    n = max(int(np.ceil(abs(a1 - a0) * rayon / pas)), 2)
    a = np.linspace(a0, a1, n + 1)[1:]
    return np.column_stack([centre[0] + rayon * np.cos(a), centre[1] + rayon * np.sin(a)])


def _polyligne_arrondie(points, rayon, pas=PAS_M):
    """Relie des points de passage par des segments raccordés par des arcs de virage."""
    pts = [np.asarray(p, float) for p in points]
    morceaux = []
    courant = pts[0]
    for i in range(1, len(pts) - 1):
        p, p_suiv = pts[i], pts[i + 1]
        u1 = p - courant
        u2 = p_suiv - p
        l1, l2 = np.hypot(*u1), np.hypot(*u2)
        if l1 < 1.0 or l2 < 1.0:
            continue
        u1, u2 = u1 / l1, u2 / l2
        theta = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
        if theta < 1e-3:
            morceaux.append(_segment(courant, p, pas))
            courant = p
            continue
        theta = min(theta, np.deg2rad(170))
        r = rayon
        d = r * np.tan(theta / 2)
        d_max = 0.45 * min(l1, l2)
        if d > d_max:
            d = d_max
            r = d / np.tan(theta / 2)
        a = p - d * u1
        sens = np.sign(u1[0] * u2[1] - u1[1] * u2[0]) or 1.0
        c = a + r * np.array([-u1[1], u1[0]]) * sens
        a0 = np.arctan2(a[1] - c[1], a[0] - c[0])
        morceaux.append(_segment(courant, a, pas))
        arc = _arc(c, r, a0, a0 + sens * theta, pas)
        morceaux.append(arc)
        courant = arc[-1]
    morceaux.append(_segment(courant, pts[-1], pas))
    return np.vstack(morceaux)


def _hippodrome(centre, cap, longueur, rayon, n_tours, pas=PAS_M):
    u = np.array([np.cos(cap), np.sin(cap)])
    n = np.array([-u[1], u[0]])
    c1, c2 = centre + 0.5 * longueur * u, centre - 0.5 * longueur * u
    a_moins_n = np.arctan2(-n[1], -n[0])
    a_plus_n = np.arctan2(n[1], n[0])
    tour = np.vstack([
        _segment(c2 - rayon * n, c1 - rayon * n, pas),
        _arc(c1, rayon, a_moins_n, a_moins_n + np.pi, pas),
        _segment(c1 + rayon * n, c2 + rayon * n, pas),
        _arc(c2, rayon, a_plus_n, a_plus_n + np.pi, pas),
    ])
    return np.vstack([tour] * max(int(n_tours), 1))


def _orbite(centre, rayon, a0, n_tours, sens, pas=PAS_M):
    return _arc(centre, rayon, a0, a0 + sens * 2 * np.pi * n_tours, pas)


def _balayage(centre, cap, longueur, espacement, n_branches, pas=PAS_M):
    u = np.array([np.cos(cap), np.sin(cap)])
    n = np.array([-u[1], u[0]])
    origine = centre - 0.5 * longueur * u - 0.5 * espacement * (n_branches - 1) * n
    morceaux = []
    for k in range(n_branches):
        signe = 1 if k % 2 == 0 else -1
        debut = origine + k * espacement * n + (0 if signe > 0 else longueur) * u
        fin = debut + signe * longueur * u
        morceaux.append(_segment(debut, fin, pas))
        if k < n_branches - 1:
            c = fin + 0.5 * espacement * n
            a0 = np.arctan2(-n[1], -n[0])
            morceaux.append(_arc(c, 0.5 * espacement, a0, a0 + signe * np.pi, pas))
    return np.vstack(morceaux)


def _axe(xy, debut=True, n=5):
    v = (xy[min(n, len(xy) - 1)] - xy[0]) if debut else (xy[-1] - xy[max(len(xy) - 1 - n, 0)])
    return v / (np.hypot(*v) or 1.0)


def _longueurs(xy):
    d = np.diff(xy, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])


def _couper(xy, phases, s_coupe):
    s = _longueurs(xy)
    k = int(np.searchsorted(s, s_coupe))
    k = max(min(k, len(xy) - 1), 1)
    return xy[: k + 1], phases[: k + 1]


def _cap_final(xy):
    v = xy[-1] - xy[max(len(xy) - 6, 0)]
    return np.arctan2(v[1], v[0])


def _vitesse_sol(xy, va, vent):
    d = np.diff(xy, axis=0)
    ds = np.hypot(d[:, 0], d[:, 1])
    ux, uy = d[:, 0] / ds, d[:, 1] / ds
    w_long = vent[0] * ux + vent[1] * uy
    w_trav = -vent[0] * uy + vent[1] * ux
    vg = w_long + np.sqrt(np.maximum(va**2 - w_trav**2, (0.3 * va) ** 2))
    return ds, np.maximum(vg, 0.3 * va)


def _temps_sans_bruit(xy, va0, vent):
    ds, vg = _vitesse_sol(xy, va0, vent)
    return np.concatenate([[0.0], np.cumsum(ds / vg)])


def _ar1(n, sigma, phi, rng):
    if n <= 0:
        return np.zeros(0)
    e = rng.normal(0.0, sigma * np.sqrt(1 - phi**2), n)
    x0 = rng.normal(0.0, sigma)
    return lfilter([1.0], [1.0, -phi], e, zi=[phi * x0])[0]


@dataclass
class Trajet:
    """Trajet dense (pas ~100 m) et phase de vol de chaque point."""

    morceaux: list = field(default_factory=list)

    def ajouter(self, xy, phase):
        xy = np.asarray(xy, float)
        if len(xy):
            self.morceaux.append((xy, np.full(len(xy), phase, dtype=np.int8)))

    def assembler(self):
        xy = np.vstack([m[0] for m in self.morceaux])
        ph = np.concatenate([m[1] for m in self.morceaux])
        d = np.hypot(*np.diff(xy, axis=0).T)
        garde = np.concatenate([[True], d > 1e-6])
        return xy[garde], ph[garde]

    def position(self):
        return self.morceaux[-1][0][-1]


@dataclass
class Vol:
    drone_id: str
    raid_id: str
    groupe_id: str
    jour: int
    modele: str
    role: str
    origine: str
    profil: str
    site_lancement: str
    cible: str
    t_lancement: float
    v_croisiere_kmh: float
    alt_croisiere_m: float
    fin: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    phase: np.ndarray
    motif: str = ""


def _lire_points(nom_fichier):
    df = pd.read_csv(config.REFERENTIELS / nom_fichier)
    df["x"], df["y"] = vers_xy(df["lon"].to_numpy(), df["lat"].to_numpy())
    return df


class Simulateur:
    """Génère les vols de tous les jours simulés."""

    def __init__(self, theatre: Theatre, graine: int = config.GRAINE, nb_jours: int = config.NB_JOURS):
        self.th = theatre
        self.graine = graine
        self.nb_jours = nb_jours
        self.sites_ru = _lire_points("sites_lancement_russes_v2.csv")
        self.cibles_ua = _lire_points("cibles_ukraine_v2.csv")
        self.cibles_ru = _lire_points("cibles_russie_v2.csv")
        self.zones_ua = _lire_points("zones_lancement_ukrainiennes_v2.csv")
        ligne = self.th.ligne_contact_xy
        self.longueur_contact = ligne.length
        xv, yv = vers_xy(36.95, 50.28)  # Vovtchansk : début du front actif
        self.s_front = ligne.project(Point(float(xv), float(yv)))

    # Points d'appui

    def _abscisse_contact(self, rng):
        """Abscisse sur la ligne de contact : le front actif pèse 3 fois plus que la frontière nord."""
        l_nord = self.s_front
        l_front = self.longueur_contact - self.s_front
        if rng.random() < 3 * l_front / (3 * l_front + l_nord):
            return rng.uniform(self.s_front, self.longueur_contact)
        return rng.uniform(0.0, self.s_front)

    def _point_lateral(self, s, profondeur_km, camp, rng, essais=30):
        """Point à une profondeur donnée d'un camp (positive = chez soi) depuis la ligne de contact."""
        for _ in range(essais):
            p, n_ua = self.th.point_contact(s)
            n = n_ua if camp == "Ukraine" else -n_ua
            q = p + n * profondeur_km * 1000.0
            if profondeur_km >= 0:
                ok = self.th.est_cote_ua(*q) if camp == "Ukraine" else self.th.est_cote_russe(*q)
            else:
                ok = self.th.est_cote_russe(*q) if camp == "Ukraine" else self.th.est_cote_ua(*q)
            if ok:
                return q
            s = float(np.clip(s + rng.normal(0, 15e3), 1.0, self.longueur_contact - 1.0))
        return None

    def _point_zone_ua(self, zone, rng):
        for _ in range(50):
            r = zone.rayon_km * 1000.0 * np.sqrt(rng.random())
            a = rng.uniform(0, 2 * np.pi)
            q = np.array([zone.x + r * np.cos(a), zone.y + r * np.sin(a)])
            if self.th.est_cote_ua(*q) and self.th.distance_signee_km(*q)[0] > 20:
                return q
        return np.array([zone.x, zone.y])

    def _point_lonlat(self, lon, lat):
        return np.array(vers_xy(lon, lat), dtype=float)

    # Profils de mission

    def _route_frappe(self, depart, cible, rng, couloir=None, feinte=False):
        """Points de passage d'une frappe programmée (détours, couloir, feinte éventuelle)."""
        d = np.hypot(*(cible - depart))
        u = (cible - depart) / d
        n = np.array([-u[1], u[0]])
        k = 2 + int(d // 200e3)
        f = np.sort(rng.uniform(0.08, 0.92, k))
        # Détour lissé : deux harmoniques d'amplitude proportionnelle à la distance,
        # puis petites perturbations locales (points de passage programmés).
        a1, a2 = rng.normal(0, 0.15 * d), rng.normal(0, 0.06 * d)
        phi = rng.uniform(0, 2 * np.pi)
        decal = a1 * np.sin(np.pi * f) + a2 * np.sin(2 * np.pi * f + phi) + rng.normal(0, 0.02 * d, k)
        decal = np.clip(decal, -200e3, 200e3)
        points = [depart] + [depart + fi * d * u + oi * n for fi, oi in zip(f, decal)]
        if couloir == "bielorussie":
            entree = self._point_lonlat(rng.uniform(31.8, 33.5), rng.uniform(51.75, 52.15))
            long_frontiere = self._point_lonlat(rng.uniform(29.2, 31.2), rng.uniform(51.25, 51.55))
            points = [depart, entree, long_frontiere]
        elif couloir == "mer":
            points = [depart,
                      self._point_lonlat(rng.uniform(32.0, 33.2), rng.uniform(45.25, 45.75)),
                      self._point_lonlat(rng.uniform(30.8, 31.6), rng.uniform(45.70, 46.15))]
        if feinte:
            a = rng.uniform(0, 2 * np.pi)
            points.append(cible + rng.uniform(20e3, 60e3) * np.array([np.cos(a), np.sin(a)]))
        points.append(cible)
        return points

    def _frappe(self, points, cible_finale, va0, alt, rng):
        """Trajet d'un drone de frappe : route programmée puis piqué sur la cible."""
        trajet = Trajet()
        route = _polyligne_arrondie(points[:-1] + [cible_finale], rayon=rng.uniform(800, 2500))
        # Piqué : les derniers mètres avant la cible, pente de descente 25 à 60 m/s.
        vz = rng.uniform(25, 60)
        d_pique = float(np.clip(alt * va0 / vz, 300.0, 8000.0))
        s = _longueurs(route)
        k = int(np.searchsorted(s, max(s[-1] - d_pique, 0.0)))
        trajet.ajouter(route[: k + 1], PH_TRANSIT)
        trajet.ajouter(route[k + 1:], PH_PIQUE)
        return trajet

    def _leurre(self, points, m, va0, alt, vent, rng):
        trajet = Trajet()
        portee = m["portee_km"] * 1000.0 * rng.uniform(0.55, 1.0)
        if rng.random() < 0.5 and len(points) > 2:
            # Se détache du groupe à un point de passage et part vers une zone voisine.
            j = int(rng.integers(1, len(points) - 1))
            a = rng.uniform(0, 2 * np.pi)
            ecart = points[j] + rng.uniform(40e3, 150e3) * np.array([np.cos(a), np.sin(a)])
            points = points[: j + 1] + [ecart, ecart + rng.normal(0, 60e3, 2)]
        route = _polyligne_arrondie(points, rayon=rng.uniform(800, 2500))
        ph = np.full(len(route), PH_TRANSIT, dtype=np.int8)
        if rng.random() < 0.35:
            # Orbite au-dessus d'une zone pour attirer la défense aérienne.
            r = rng.uniform(1500, 5000)
            orbite = _orbite(route[-1] + np.array([r, 0.0]), r, np.pi, rng.uniform(1, 4), rng.choice([-1, 1]))
            route = np.vstack([route, orbite])
            ph = np.concatenate([ph, np.full(len(orbite), PH_PATROUILLE, dtype=np.int8)])
        # Carburant : la portée publiée s'entend en air calme ; l'autonomie en temps vaut
        # portée / vitesse air. Le vent modifie la distance sol parcourue pendant ce temps.
        d_plane = alt * va0 / rng.uniform(2.5, 5.0)
        autonomie_s = max(portee - d_plane, 10e3) / va0
        t_sol = _temps_sans_bruit(route, va0, vent)
        k = int(np.searchsorted(t_sol, autonomie_s))
        xy, ph = route[: max(k, 2)], ph[: max(k, 2)]
        trajet.morceaux.append((xy, ph))
        cap = _cap_final(xy)
        trajet.ajouter(_segment(xy[-1], xy[-1] + d_plane * np.array([np.cos(cap), np.sin(cap)])), PH_PLANE)
        return trajet

    def _rodeuse(self, depart, cible, m, va0, vent_ms, rng):
        trajet = Trajet()
        d = np.hypot(*(cible - depart))
        u = (cible - depart) / d
        n = np.array([-u[1], u[0]])
        milieu = depart + 0.5 * d * u + rng.normal(0, 0.1 * d) * n
        centre = cible + rng.normal(0, 1500, 2)
        rayon = rng.uniform(400, 1500)
        # Budget de temps calculé au pire cas (vent de face à l'aller) et vitesse
        # moyenne en orbite réduite par le vent.
        duree_max = rng.uniform(*m["hyp_duree_mission_h"]) * 3600
        reste = duree_max - d / max(va0 - vent_ms, 0.4 * va0)
        v_eff = (va0**2 - vent_ms**2) / va0
        t_orbite = float(np.clip(reste * rng.uniform(0.3, 0.9), 60.0, 20 * 60.0))
        n_tours = max(t_orbite * v_eff * 0.9 / (2 * np.pi * rayon), 0.5)
        orbite = _orbite(centre, rayon, 0.0, n_tours, rng.choice([-1, 1]))
        approche = orbite[0] - 1500.0 * _axe(orbite)
        trajet.ajouter(_polyligne_arrondie([depart, milieu, approche, orbite[0]], 600), PH_TRANSIT)
        trajet.ajouter(orbite, PH_PATROUILLE)
        trajet.ajouter(_segment(orbite[-1], cible), PH_PIQUE)
        return trajet

    def _reconnaissance(self, depart, zone, m, va0, vent_ms, rng):
        trajet = Trajet()
        v = va0
        v_eff = (va0**2 - vent_ms**2) / va0  # vitesse sol moyenne sur un circuit fermé
        duree = rng.uniform(*m["hyp_duree_mission_h"]) * 3600
        d = np.hypot(*(zone - depart))
        u = (zone - depart) / max(d, 1.0)
        n = np.array([-u[1], u[0]])
        milieu = depart + 0.5 * d * u + rng.normal(0, 0.08 * d + 1000) * n
        budget = max((duree - 2 * d / v_eff) * v_eff * 0.85, 5e3)
        motif = rng.choice(["hippodrome", "orbite", "balayage", "points"], p=[0.3, 0.25, 0.2, 0.25])
        cap = rng.uniform(0, 2 * np.pi)
        r_action = m["hyp_rayon_km"] * 1000.0  # taille des motifs bornée par le rayon d'action
        if motif == "hippodrome":
            longueur, rayon = rng.uniform(4e3, max(min(15e3, 0.4 * r_action), 4.5e3)), rng.uniform(800, 2000)
            tour = 2 * longueur + 2 * np.pi * rayon
            pat = _hippodrome(zone, cap, longueur, rayon, max(budget // tour, 1))
        elif motif == "orbite":
            rayon = rng.uniform(1000, max(min(4000, 0.1 * r_action), 1200))
            pat = _orbite(zone + np.array([rayon, 0.0]), rayon, np.pi,
                          max(budget / (2 * np.pi * rayon), 1.0), rng.choice([-1, 1]))
        elif motif == "balayage":
            longueur, esp = rng.uniform(3e3, max(min(10e3, 0.3 * r_action), 3.5e3)), rng.uniform(600, 1500)
            nb = int(np.clip(budget // (longueur + np.pi * esp / 2), 3, 30))
            pat = _balayage(zone, cap, longueur, esp, nb)
        else:
            pts = [zone]
            parcours = 0.0
            while parcours < budget and len(pts) < 40:
                a = rng.uniform(0, 2 * np.pi)
                q = zone + rng.uniform(2e3, max(min(15e3, 0.3 * r_action), 2.5e3)) * np.array([np.cos(a), np.sin(a)])
                parcours += np.hypot(*(q - pts[-1]))
                pts.append(q)
            pat = _polyligne_arrondie(pts, rng.uniform(500, 1200))
        approche = pat[0] - 1500.0 * _axe(pat)
        trajet.ajouter(_polyligne_arrondie([depart, milieu, approche, pat[0]], rng.uniform(500, 1200)),
                       PH_TRANSIT)
        trajet.ajouter(pat, PH_PATROUILLE)
        # Retour vers une zone de récupération proche du point de lancement.
        recup = depart + rng.normal(0, 3000, 2)
        alt_ref = 1500.0
        d_atterr = alt_ref * v / 4.0
        p = trajet.position()
        sortie = p + 1500.0 * _axe(pat, debut=False)
        v_ret = recup - sortie
        l_ret = np.hypot(*v_ret)
        if l_ret > d_atterr + 1000:
            debut_desc = recup - v_ret / l_ret * d_atterr
            trajet.ajouter(_polyligne_arrondie([p, sortie, debut_desc], 800), PH_RETOUR)
            trajet.ajouter(_segment(debut_desc, recup), PH_ATTERRISSAGE)
        else:
            trajet.ajouter(_polyligne_arrondie([p, sortie, recup], 800), PH_ATTERRISSAGE)
        return trajet, motif

    # Défense adverse

    def _defense(self, xy, ph, origine, lambdas, rng):
        """Interruption éventuelle du vol (abattu ou brouillé) selon la distance parcourue chez l'adversaire."""
        echant = slice(None, None, 10)
        x, y = xy[echant, 0], xy[echant, 1]
        hostile = self.th.est_cote_ua(x, y) if origine == "Russie" else self.th.est_cote_russe(x, y)
        s = _longueurs(xy)[echant]
        ds = np.diff(s, prepend=0.0)
        d_hostile = np.cumsum(ds * hostile) / 1000.0
        terminal = np.isin(ph[echant], (PH_PIQUE, PH_ATTERRISSAGE, PH_PLANE))
        premier = (None, None)
        for cause, lam in lambdas.items():
            if lam <= 0:
                continue
            seuil = rng.exponential(1.0) / lam
            idx = np.flatnonzero((d_hostile >= seuil) & ~terminal)
            if idx.size and (premier[1] is None or s[idx[0]] < premier[1]):
                premier = (cause, s[idx[0]])
        return premier

    def _interrompre(self, xy, ph, cause, s_coupe, v, alt, rng):
        xy, ph = _couper(xy, ph, s_coupe)
        cap = _cap_final(xy)
        if cause == "abattu":
            l = v * rng.uniform(15, 40)
            fin = xy[-1] + l * np.array([np.cos(cap), np.sin(cap)])
            ajout, phase = _segment(xy[-1], fin), PH_CHUTE
        else:
            u = np.array([np.cos(cap), np.sin(cap)])
            cap += rng.choice([-1, 1]) * rng.uniform(0.5, 1.6)
            l = max(alt, 200.0) * v / rng.uniform(3.0, 8.0)
            fin = xy[-1] + l * np.array([np.cos(cap), np.sin(cap)])
            ajout = _polyligne_arrondie([xy[-1], xy[-1] + 1000.0 * u, fin], 1500.0)[1:]
            phase = PH_PLANE
        return (np.vstack([xy, ajout]),
                np.concatenate([ph, np.full(len(ajout), phase, dtype=np.int8)]))

    # Temps et altitude

    def _temps(self, xy, ph, va0, vent, v_pique, rng):
        p = ph[1:]
        mult = np.ones(len(xy) - 1)
        mult[p == PH_PATROUILLE] = 0.9
        mult[p == PH_CHUTE] = 0.6
        mult[p == PH_PLANE] = 0.85
        mult[p == PH_ATTERRISSAGE] = 0.75
        idx = np.flatnonzero(p == PH_PIQUE)
        if idx.size and v_pique:
            mult[idx] = np.linspace(1.0, max(v_pique / va0, 1.0), idx.size)
        va = va0 * mult * (1.0 + _ar1(len(mult), 0.04, np.exp(-PAS_M / 10e3), rng))
        ds, vg = _vitesse_sol(xy, va, vent)
        return np.concatenate([[0.0], np.cumsum(ds / vg)])

    def _altitude(self, ph_k, alt, m, rng, montee, changements=True):
        n = len(ph_k)
        dt = config.DT_RADAR_S
        cible = np.full(n, alt, dtype=float)
        if changements and m["profil"] in (cat.PROFIL_FRAPPE, cat.PROFIL_LEURRE):
            # Changements de palier en route (consignes programmées).
            t_change = np.cumsum(rng.exponential(5400.0, 8)) / dt
            for k in t_change[t_change < n].astype(int):
                cible[k:] = float(np.clip(cible[k] * rng.uniform(0.7, 1.3), 50.0, 4500.0))
        if m["profil"] == cat.PROFIL_RECO:
            pat = ph_k == PH_PATROUILLE
            cible[pat] = float(np.clip(alt + rng.normal(0, 200), 150, 5000))
        # Montée et changements de palier : réponse du premier ordre bornée par les
        # taux de montée et de descente. Phases terminales (piqué, chute, plané,
        # atterrissage) : descente linéaire jusqu'au sol sur la durée de la phase.
        terminales = np.isin(ph_k, (PH_PIQUE, PH_CHUTE, PH_PLANE, PH_ATTERRISSAGE))
        z = np.empty(n)
        zc = 0.0
        gain = 1.0 - np.exp(-dt / 30.0)
        i = 0
        while i < n:
            if terminales[i]:
                j = i
                while j < n and ph_k[j] == ph_k[i]:
                    j += 1
                z[i:j] = np.linspace(zc, 0.0, j - i + 1)[1:]
                zc = z[j - 1]
                i = j
                continue
            delta = (cible[i] - zc) * gain
            zc = max(zc + min(max(delta, -6.0 * dt), montee * dt), 0.0)
            z[i] = zc
            i += 1
        z += np.where(z > 30, _ar1(n, 12.0, np.exp(-dt / 120.0), rng), 0.0)
        return np.maximum(z, 0.0)

    def _finaliser(self, xy, ph, va0, vent, v_pique, rng, t_lancement, alt, m, montee):
        t_rel = self._temps(xy, ph, va0, vent, v_pique, rng)
        dt = config.DT_RADAR_S
        k0 = int(np.ceil(t_lancement / dt))
        k1 = int(np.floor((t_lancement + t_rel[-1]) / dt))
        t_k = np.arange(k0, k1 + 1) * dt
        tr = t_k - t_lancement
        x = np.interp(tr, t_rel, xy[:, 0])
        y = np.interp(tr, t_rel, xy[:, 1])
        idx = np.clip(np.searchsorted(t_rel, tr), 0, len(ph) - 1)
        ph_k = ph[idx]
        z = self._altitude(ph_k, alt, m, rng, montee)
        return t_k, x, y, z, ph_k

    # Construction des vols

    def _parametres_modele(self, m, rng):
        va0 = rng.uniform(m["v_croisiere_min"], m["v_croisiere_max"]) * KMH
        if m["hyp_alt_haute"] and rng.random() < m["hyp_part_haute"]:
            alt = rng.uniform(*m["hyp_alt_haute"])
        else:
            alt = rng.uniform(*m["hyp_alt_basse"])
        return va0, alt

    def _vol(self, ident, m, trajet_info, rng, t_lancement, vent, lambdas, raid_id, groupe_id, jour,
             site, cible_nom):
        va0, alt = self._parametres_modele(m, rng)
        profil = m["profil"]
        vent_ms = float(np.hypot(*vent))
        if profil in (cat.PROFIL_RODEUSE, cat.PROFIL_RECO) and vent_ms > 0.6 * va0:
            return None  # sortie annulée : vent trop fort pour un drone tactique
        v_pique = (m["hyp_v_pique"] or 0) * KMH
        motif = ""
        montee = {cat.PROFIL_FRAPPE: rng.uniform(4, 9), cat.PROFIL_LEURRE: rng.uniform(3, 6),
                  cat.PROFIL_RODEUSE: rng.uniform(4, 8), cat.PROFIL_RECO: rng.uniform(2.5, 5)}[profil]
        if profil == cat.PROFIL_FRAPPE:
            points, cible_finale, motif = trajet_info
            trajet = self._frappe(points, cible_finale, va0, alt, rng)
        elif profil == cat.PROFIL_LEURRE:
            points, motif = trajet_info
            trajet = self._leurre(points, m, va0, alt, vent, rng)
        elif profil == cat.PROFIL_RODEUSE:
            depart, cible = trajet_info
            trajet = self._rodeuse(depart, cible, m, va0, vent_ms, rng)
            motif = "orbite de recherche"
        else:
            depart, zone = trajet_info
            trajet, motif = self._reconnaissance(depart, zone, m, va0, vent_ms, rng)
        xy, ph = trajet.assembler()
        fin = {cat.PROFIL_FRAPPE: "impact", cat.PROFIL_LEURRE: "carburant",
               cat.PROFIL_RODEUSE: "impact", cat.PROFIL_RECO: "atterrissage"}[profil]
        cause, s_coupe = self._defense(xy, ph, m["origine"], lambdas, rng)
        if cause:
            xy, ph = self._interrompre(xy, ph, cause, s_coupe, va0, alt, rng)
            fin = cause
        if m["origine"] == "Ukraine" and profil == cat.PROFIL_FRAPPE:
            # Au-delà de 350 km en territoire russe, le drone sort de toute couverture
            # radar ukrainienne : inutile de simuler la suite.
            ech = xy[::20]
            dist = self.th.distance_signee_km(ech[:, 0], ech[:, 1])
            loin = np.flatnonzero(dist < -350)
            if loin.size:
                xy, ph = _couper(xy, ph, _longueurs(xy)[min(loin[0] * 20, len(xy) - 1)])
                if fin == "impact":
                    fin = "hors_couverture"
        garde = np.concatenate([[True], np.hypot(*np.diff(xy, axis=0).T) > 1e-6])
        xy, ph = xy[garde], ph[garde]
        t, x, y, z, ph_k = self._finaliser(xy, ph, va0, vent, v_pique, rng, t_lancement, alt, m, montee)
        return Vol(ident, raid_id, groupe_id, jour, m["modele"], m["role"], m["origine"], profil,
                   site, cible_nom, t_lancement, va0 / KMH, alt, fin, t, x, y, z, ph_k, str(motif))

    def _rng(self, *cle):
        return np.random.default_rng([self.graine, *cle])

    def _raid_russe(self, jour, rng, vent, t_jour):
        vols = []
        raid_id = f"J{jour:03d}_RU_RAID"
        t_raid = t_jour + rng.uniform(17.0, 23.5) * 3600
        n_groupes = int(rng.integers(2, 7))
        cibles = self.cibles_ua
        for g in range(n_groupes):
            site = self.sites_ru.iloc[int(rng.integers(len(self.sites_ru)))]
            depart = np.array([site.x, site.y])
            d = np.hypot(cibles.x - site.x, cibles.y - site.y) / 1000.0
            w = cibles.poids * np.exp(-d / 1000.0)
            cible = cibles.iloc[int(rng.choice(len(cibles), p=w / w.sum()))]
            p_cible = np.array([cible.x, cible.y])
            couloir = None
            if site.zone == "nord" and cible.lon < 31.5 and rng.random() < 0.5:
                couloir = "bielorussie"
            elif site.zone == "sud" and cible.lon < 32.3 and cible.lat < 47.5 and rng.random() < 0.7:
                couloir = "mer"
            feinte = rng.random() < 0.2
            points = self._route_frappe(depart, p_cible, rng, couloir, feinte=feinte)
            motif = (couloir or "direct") + (" + feinte" if feinte else "")
            n_drones = int(rng.poisson(6)) + 2
            part_leurres = rng.uniform(0.30, 0.55)
            groupe_id = f"{raid_id}_G{g}"
            t_groupe = t_raid + rng.uniform(0, 2.5) * 3600
            for i in range(n_drones):
                est_leurre = rng.random() < part_leurres
                m = cat.CATALOGUE["Gerbera" if est_leurre else "Shahed-136 (Geran-2)"]
                r = self._rng(jour, 1, g, i)
                pts = [depart] + [p + r.normal(0, 2000, 2) for p in points[1:-1]] + [points[-1]]
                if est_leurre:
                    info = (pts[:-1] + [points[-1] + r.normal(0, 10e3, 2)], motif)
                    lambdas = {"abattu": 0.0012, "brouille": 0.0025}
                else:
                    info = (pts, points[-1] + r.normal(0, 3000, 2), motif)
                    lambdas = {"abattu": 0.0035, "brouille": 0.0012}
                vols.append(self._vol(f"{groupe_id}_D{i:02d}", m, info, r, t_groupe + i * r.uniform(60, 300),
                                      vent, lambdas, raid_id, groupe_id, jour, site.nom, cible.nom))
        return vols

    def _raid_ukrainien(self, jour, rng, vent, t_jour):
        vols = []
        raid_id = f"J{jour:03d}_UA_RAID"
        t_raid = t_jour + rng.uniform(19.0, 26.0) * 3600
        cibles = self.cibles_ru
        for g in range(int(rng.integers(1, 5))):
            zone = self.zones_ua.iloc[int(rng.integers(len(self.zones_ua)))]
            depart_ref = self._point_zone_ua(zone, rng)
            modele = rng.choice(["An-196 Liutyi", "UJ-26 Bober (Babor v1)", "UJ-22 Airborne"],
                                p=[0.5, 0.25, 0.25])
            m = cat.CATALOGUE[modele]
            d = np.hypot(cibles.x - depart_ref[0], cibles.y - depart_ref[1]) / 1000.0
            # Cibles à portée du modèle (marge de 25 % pour les détours de la route).
            w = cibles.poids * np.exp(-d / 600.0) * (d <= 0.75 * m["portee_km"])
            cible = cibles.iloc[int(rng.choice(len(cibles), p=w / w.sum()))]
            p_cible = np.array([cible.x, cible.y])
            couloir = "mer" if cible.lat < 45.6 and cible.lon < 36 else None
            points = self._route_frappe(depart_ref, p_cible, rng, None, feinte=False)
            if couloir == "mer":
                points = [depart_ref,
                          self._point_lonlat(rng.uniform(31.0, 32.6), rng.uniform(45.4, 46.1)),
                          p_cible]
            groupe_id = f"{raid_id}_G{g}"
            t_groupe = t_raid + rng.uniform(0, 2.0) * 3600
            for i in range(int(rng.poisson(4)) + 2):
                r = self._rng(jour, 2, g, i)
                depart = depart_ref + r.normal(0, 4000, 2)
                pts = [depart] + [p + r.normal(0, 2500, 2) for p in points[1:-1]] + [points[-1]]
                info = (pts, points[-1] + r.normal(0, 3000, 2), couloir or "direct")
                lambdas = {"abattu": 0.0025, "brouille": 0.0010}
                vols.append(self._vol(f"{groupe_id}_D{i:02d}", m, info, r, t_groupe + i * r.uniform(60, 400),
                                      vent, lambdas, raid_id, groupe_id, jour, zone.nom, cible.nom))
        return vols

    def _sorties_tactiques(self, jour, rng, vent, t_jour):
        vols = []
        moyennes = {"Lancet-3": 5, "Orlan-10": 5, "Supercam S350": 4,
                    "Warmate": 4, "Leleka-100": 4, "A1-SM Furia": 3.5, "Valkyria": 3}
        for j, (modele, lam) in enumerate(moyennes.items()):
            m = cat.CATALOGUE[modele]
            camp = m["origine"]
            for i in range(int(rng.poisson(lam))):
                r = self._rng(jour, 3, j, i)
                nuit = r.random() < (0.15 if m["profil"] == cat.PROFIL_RODEUSE else 0.25)
                heure = r.uniform(18, 30) if nuit else r.uniform(6, 18.5)
                s = self._abscisse_contact(r)
                rayon = m["hyp_rayon_km"]
                if m["profil"] == cat.PROFIL_RODEUSE:
                    prof_dep = r.uniform(3, 0.4 * rayon)
                    prof_cible = r.uniform(3, 0.85 * rayon - prof_dep)
                    lambdas = {"abattu": 0.004, "brouille": 0.010}
                else:
                    prof_dep = r.uniform(5, min(0.4 * rayon, 40))
                    prof_cible = r.uniform(-5, 0.85 * rayon - prof_dep)
                    lambdas = {"abattu": 0.0006, "brouille": 0.0010}
                depart = self._point_lateral(s, prof_dep, camp, r)
                s2 = float(np.clip(s + r.normal(0, 12e3), 1.0, self.longueur_contact - 1.0))
                cible = self._point_lateral(s2, -prof_cible, camp, r)
                if depart is None or cible is None:
                    continue
                # Le point visé reste dans le rayon d'action publié, avec une marge pour le motif de vol.
                d = np.hypot(*(cible - depart))
                limite = (0.8 if m["profil"] == cat.PROFIL_RODEUSE else 0.7) * rayon * 1000.0
                if d > limite:
                    cible = depart + (cible - depart) * limite / d
                code = "RU" if camp == "Russie" else "UA"
                ident = f"J{jour:03d}_{code}_{modele.split()[0]}_{i:02d}"
                vol = self._vol(ident, m, (depart, cible), r, t_jour + heure * 3600, vent, lambdas,
                                ident, ident, jour, "position tactique", "zone du front")
                if vol is not None:
                    vols.append(vol)
        return vols

    def simuler(self, jours=None, progression=False):
        vols = []
        jours = range(self.nb_jours) if jours is None else jours
        for jour in jours:
            rng = self._rng(jour, 0)
            vitesse = min(rng.weibull(2.0) * 7.0, 20.0)
            direction = rng.vonmises(0.0, 1.0)  # vent dominant d'ouest (vers l'est)
            vent = np.array([vitesse * np.cos(direction), vitesse * np.sin(direction)])
            t_jour = jour * 86400.0
            if rng.random() < 0.85:
                vols += self._raid_russe(jour, rng, vent, t_jour)
            if rng.random() < 0.55:
                vols += self._raid_ukrainien(jour, rng, vent, t_jour)
            vols += self._sorties_tactiques(jour, rng, vent, t_jour)
            if progression and jour % 20 == 0:
                print(f"jour {jour} : {len(vols)} vols")
        return vols


def vols_vers_df(vols) -> pd.DataFrame:
    """Résumé (vérité terrain) d'un vol par ligne."""
    lignes = []
    for v in vols:
        lon0, lat0 = vers_lonlat(v.x[0], v.y[0])
        lon1, lat1 = vers_lonlat(v.x[-1], v.y[-1])
        d = np.hypot(np.diff(v.x), np.diff(v.y)).sum() / 1000.0
        eloignement = np.hypot(v.x - v.x[0], v.y - v.y[0]).max() / 1000.0
        lignes.append(dict(
            drone_id=v.drone_id, raid_id=v.raid_id, groupe_id=v.groupe_id, jour=v.jour,
            modele=v.modele, role=v.role, origine=v.origine, profil=v.profil,
            site_lancement=v.site_lancement, cible=v.cible, motif=v.motif, fin=v.fin,
            heure_lancement=(v.t_lancement % 86400) / 3600.0,
            duree_min=(v.t[-1] - v.t[0]) / 60.0, distance_km=d, eloignement_max_km=eloignement,
            v_croisiere_kmh=v.v_croisiere_kmh, alt_consigne_m=v.alt_croisiere_m,
            alt_max_m=float(v.z.max()), lon_debut=float(lon0), lat_debut=float(lat0),
            lon_fin=float(lon1), lat_fin=float(lat1),
        ))
    return pd.DataFrame(lignes)


def vols_vers_points(vols) -> pd.DataFrame:
    """Vérité terrain point par point (une ligne par instant radar)."""
    morceaux = []
    for v in vols:
        lon, lat = vers_lonlat(v.x, v.y)
        morceaux.append(pd.DataFrame(dict(
            drone_id=v.drone_id, t=v.t, lon=lon, lat=lat,
            alt=v.z.astype(np.float32), phase=v.phase.astype(np.int8),
        )))
    return pd.concat(morceaux, ignore_index=True)
