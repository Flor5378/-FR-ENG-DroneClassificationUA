"""Figures statiques de la v2 (matplotlib).

Charte graphique : palette catégorielle validée (contraste, daltonisme), traits
fins, grilles discrètes, une seule échelle par graphique, légende dès deux séries,
texte toujours dans les encres neutres. Origine : bleu = Ukraine, orange = Russie.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from matplotlib.ticker import FuncFormatter

from . import config

SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
ENCRE_MUETTE = "#898781"
GRILLE = "#e1e0d9"
AXE = "#c3c2b7"
NEUTRE = "#f0efec"
CATEGORIEL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COULEUR_ORIGINE = {"Ukraine": "#2a78d6", "Russie": "#eb6834"}
COULEUR_ROLE = {
    "Frappe longue portée": "#6250d6",
    "Leurre": "#e87ba4",
    "Munition rôdeuse": "#008300",
    "Reconnaissance": "#1baf7a",
}
ACCENT = "#e34948"
SEQ_BLEU = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("bleu_v2", ["#fcfcfb"] + SEQ_BLEU)
CMAP_DIV = LinearSegmentedColormap.from_list("div_v2", ["#2a78d6", NEUTRE, "#e34948"])
MER = tuple(0.30 * np.array(to_rgb("#cde2fb")) + 0.70 * np.array(to_rgb(SURFACE)))
TERRE = NEUTRE
TERRE_UA = "#ffffff"
ASPECT_CARTE = 1 / np.cos(np.deg2rad(48.5))
EMPRISE = (21.8, 42.5, 43.8, 55.4)


def appliquer_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXE, "axes.linewidth": 0.8, "axes.labelcolor": ENCRE_2, "axes.titlecolor": ENCRE,
        "axes.titleweight": "bold", "axes.titlesize": 11.5, "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRILLE, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": AXE, "ytick.color": AXE,
        "xtick.labelcolor": ENCRE_2, "ytick.labelcolor": ENCRE_2, "text.color": ENCRE,
        "font.size": 9.5, "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.titleweight": "bold", "figure.titlesize": 13,
    })


def sauver(fig, nom):
    chemin = config.FIGURES / f"{nom}.png"
    fig.savefig(chemin)
    return chemin


def _legende(ax, elements, **kw):
    kw.setdefault("loc", "upper left")
    return ax.legend(handles=elements, labelcolor=ENCRE_2, **kw)


def ligne_legende(couleur, label, lw=2):
    return Line2D([0], [0], color=couleur, lw=lw, label=label)


def point_legende(couleur, label, marker="o", ms=7, mec=SURFACE):
    return Line2D([0], [0], marker=marker, color="none", markerfacecolor=couleur, markeredgecolor=mec,
                  markersize=ms, label=label)


# Cartes

def fond_carte(ax, theatre, emprise=EMPRISE, villes=True, separation=False, titre=None):
    """Fond de carte : pays, territoire occupé (lavis orange), ligne de contact."""
    import geopandas as gpd

    ax.set_facecolor(MER)
    autres = theatre.pays[~theatre.pays.ADMIN.isin(["Ukraine"])]
    autres.plot(ax=ax, color=TERRE, edgecolor=AXE, linewidth=0.5, zorder=1)
    gpd.GeoSeries([theatre.ua_controle]).plot(ax=ax, color=TERRE_UA, edgecolor=AXE, linewidth=0.6, zorder=1)
    gpd.GeoSeries([theatre.occupe]).plot(ax=ax, color=COULEUR_ORIGINE["Russie"], alpha=0.13, linewidth=0, zorder=2)
    gpd.GeoSeries([theatre.ligne_contact]).plot(ax=ax, color=ENCRE_2, linewidth=1.1, zorder=5)
    if separation:
        gpd.GeoSeries([theatre.ligne_separation]).plot(ax=ax, color=ENCRE_MUETTE, linewidth=0.7, zorder=4)
    if villes:
        for nom, lon, lat in (("Kyiv", 30.52, 50.45), ("Kharkiv", 36.23, 49.99), ("Odesa", 30.73, 46.48),
                              ("Dnipro", 35.05, 48.46), ("Lviv", 24.03, 49.84), ("Moscou", 37.62, 55.75),
                              ("Simferopol", 34.10, 44.95), ("Rostov-sur-le-Don", 39.72, 47.23)):
            if emprise[0] < lon < emprise[1] and emprise[2] < lat < emprise[3]:
                ax.plot(lon, lat, "o", ms=3, color=ENCRE_2, zorder=8)
                ax.annotate(nom, (lon, lat), xytext=(4, 3), textcoords="offset points", fontsize=8,
                            color=ENCRE_2, zorder=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(emprise[0], emprise[1])
    ax.set_ylim(emprise[2], emprise[3])
    ax.set_aspect(ASPECT_CARTE)
    ax.grid(False)
    ax.tick_params(labelsize=7.5, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if titre:
        ax.set_title(titre)
    return ax


def cercle_km(ax, lon, lat, rayon_km, **kw):
    """Cercle de rayon donné en km, dessiné en ellipse lon/lat (la carte corrige l'aspect)."""
    e = Ellipse((lon, lat), width=2 * rayon_km / (111.32 * np.cos(np.deg2rad(lat))), height=2 * rayon_km / 110.57,
                **kw)
    ax.add_patch(e)
    return e


def lignes_carte(ax, segments, couleurs, lw=0.6, alpha=0.45, zorder=6):
    lc = LineCollection(segments, colors=couleurs, linewidths=lw, alpha=alpha, zorder=zorder,
                        capstyle="round", joinstyle="round")
    ax.add_collection(lc)
    return lc


def segments_par_piste(points: pd.DataFrame, cle="piste_id", pas=1):
    segs, cles = [], []
    for k, g in points.groupby(cle, sort=False):
        a = g[["lon", "lat"]].to_numpy()[::pas]
        if len(a) > 1:
            segs.append(a)
            cles.append(k)
    return segs, cles


def fig_theatre(theatre, sites_ru, zones_ua, cibles_ua, cibles_ru, radars):
    fig, ax = plt.subplots(figsize=(12, 8.2))
    fond_carte(ax, theatre, separation=True, villes=False)
    ax.scatter(radars.lon, radars.lat, marker="^", s=26, color=ENCRE_MUETTE, edgecolor=SURFACE, linewidth=0.8,
               zorder=7)
    ax.scatter(cibles_ua.lon, cibles_ua.lat, s=16 + 5 * cibles_ua.poids, color=ENCRE_2, edgecolor=SURFACE,
               linewidth=0.8, zorder=8)
    ax.scatter(cibles_ru.lon, cibles_ru.lat, marker="s", s=22, color=ENCRE_2, edgecolor=SURFACE, linewidth=0.8,
               zorder=8)
    ax.scatter(sites_ru.lon, sites_ru.lat, marker="D", s=70, color=COULEUR_ORIGINE["Russie"], edgecolor=SURFACE,
               linewidth=1.5, zorder=9)
    for _, r in sites_ru.iterrows():
        ax.annotate(r.nom.split(" (")[0], (r.lon, r.lat), xytext=(6, -3), textcoords="offset points",
                    fontsize=7.5, color=ENCRE_2, zorder=9)
    for _, z in zones_ua.iterrows():
        cercle_km(ax, z.lon, z.lat, z.rayon_km, facecolor=COULEUR_ORIGINE["Ukraine"], alpha=0.18,
                  edgecolor=COULEUR_ORIGINE["Ukraine"], linewidth=1.2, zorder=7)
    for nom, lon, lat in (("Kyiv", 30.52, 50.45), ("Kharkiv", 36.23, 49.99), ("Odesa", 30.73, 46.48),
                          ("Lviv", 24.03, 49.84), ("Moscou", 37.62, 55.75), ("Dnipro", 35.05, 48.46)):
        ax.annotate(nom, (lon, lat), xytext=(5, 4), textcoords="offset points", fontsize=8.5, color=ENCRE,
                    weight="bold", zorder=10)
    elements = [
        Patch(facecolor=TERRE_UA, edgecolor=AXE, label="Territoire contrôlé par l'Ukraine"),
        Patch(facecolor=COULEUR_ORIGINE["Russie"], alpha=0.25, label="Territoire occupé (approx. 2025)"),
        ligne_legende(ENCRE_2, "Ligne de contact (front + frontière)", 1.2),
        ligne_legende(ENCRE_MUETTE, "Séparation prolongée (calcul des distances)", 0.8),
        point_legende(COULEUR_ORIGINE["Russie"], "Site de lancement russe (OSINT)", "D", 8),
        Patch(facecolor=COULEUR_ORIGINE["Ukraine"], alpha=0.3, edgecolor=COULEUR_ORIGINE["Ukraine"],
              label="Zone de lancement ukrainienne (fictive)"),
        point_legende(ENCRE_2, "Ville ciblée en Ukraine (taille = poids)", "o", 6),
        point_legende(ENCRE_2, "Cible en Russie ou en Crimée", "s", 6),
        point_legende(ENCRE_MUETTE, "Radar de surveillance (position fictive)", "^", 7),
    ]
    _legende(ax, elements, loc="lower left", fontsize=8.2, bbox_to_anchor=(0.0, 0.0))
    ax.set_title("Théâtre simulé v2 : sites de lancement, cibles et réseau radar")
    return fig


def fig_trajectoires_par_role(theatre, verite: pd.DataFrame, resume: pd.DataFrame, n_par_role=250, graine=config.GRAINE):
    """Petits multiples par rôle, trajectoires vraies colorées par origine."""
    rng = np.random.default_rng(graine)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    for ax, role in zip(axes.ravel(), COULEUR_ROLE):
        ids = resume.loc[resume.role == role, "drone_id"].to_numpy()
        ids = rng.choice(ids, size=min(n_par_role, len(ids)), replace=False)
        sous = verite[verite.drone_id.isin(set(ids))]
        fond_carte(ax, theatre, villes=True)
        origine = resume.set_index("drone_id").origine
        segs, cles = segments_par_piste(sous, "drone_id", pas=3)
        lignes_carte(ax, segs, [COULEUR_ORIGINE[origine[c]] for c in cles], lw=0.7, alpha=0.5)
        n_ru = int((origine[cles] == "Russie").sum())
        ax.set_title(f"{role} ({len(cles)} vols tirés au hasard, dont {n_ru} russes)", fontsize=10.5)
    _legende(axes[0, 0], [ligne_legende(COULEUR_ORIGINE["Russie"], "Vol russe"),
                          ligne_legende(COULEUR_ORIGINE["Ukraine"], "Vol ukrainien")], loc="lower left")
    fig.suptitle("Trajectoires simulées (vérité terrain) par rôle", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_zoom_tactique(theatre, verite, resume, emprise=(34.6, 39.2, 47.0, 50.6), max_vols=220, graine=config.GRAINE):
    rng = np.random.default_rng(graine)
    tactiques = resume[resume.role.isin(["Munition rôdeuse", "Reconnaissance"])]
    ids = rng.choice(tactiques.drone_id.to_numpy(), size=min(max_vols, len(tactiques)), replace=False)
    sous = verite[verite.drone_id.isin(set(ids))]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    origine = resume.set_index("drone_id").origine
    role = resume.set_index("drone_id").role
    for ax, r in zip(axes, ["Reconnaissance", "Munition rôdeuse"]):
        fond_carte(ax, theatre, emprise=emprise, villes=True)
        segs, cles = segments_par_piste(sous[sous.drone_id.map(role) == r], "drone_id")
        lignes_carte(ax, segs, [COULEUR_ORIGINE[origine[c]] for c in cles], lw=0.8, alpha=0.7)
        ax.set_title(f"{r} : motifs de vol près du front ({len(cles)} vols)", fontsize=10.5)
    _legende(axes[0], [ligne_legende(COULEUR_ORIGINE["Russie"], "Vol russe"),
                       ligne_legende(COULEUR_ORIGINE["Ukraine"], "Vol ukrainien")], loc="upper left")
    fig.tight_layout()
    return fig


def fig_profils(vols_exemples: dict):
    """Altitude et vitesse sol en fonction du temps pour un vol type par rôle (deux rangées, une échelle chacune)."""
    roles = list(vols_exemples)
    fig, axes = plt.subplots(2, len(roles), figsize=(15, 6.2), sharex="col")
    for j, role in enumerate(roles):
        v = vols_exemples[role]
        t = (v.t - v.t[0]) / 60.0
        vit = np.hypot(np.diff(v.x), np.diff(v.y)) / np.diff(v.t) * 3.6
        c = COULEUR_ROLE[role]
        axes[0, j].plot(t, v.z, color=c, lw=1.6)
        axes[0, j].set_title(f"{role}\n{v.modele}, fin : {v.fin}", fontsize=9.5)
        axes[1, j].plot(t[1:], vit, color=c, lw=1.0)
        axes[1, j].set_xlabel("Temps depuis le lancement (min)")
        axes[0, j].set_ylim(bottom=0)
        axes[1, j].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Altitude (m)")
    axes[1, 0].set_ylabel("Vitesse sol (km/h)")
    fig.suptitle("Profils de vol simulés : un exemple par rôle", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_radar(theatre, radar, vol, plots_piste: list, alt_basse=150, alt_haute=2000, ser=0.5):
    """Couverture radar (basse et haute altitude) et observation bruitée d'un vol."""
    from .geo import vers_lonlat
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    lon_s, lat_s = vers_lonlat(radar.sites[:, 0], radar.sites[:, 1])
    for ax, alt in zip(axes[:2], (alt_basse, alt_haute)):
        fond_carte(ax, theatre, villes=False)
        portee = radar.portee_km(np.array([float(alt)]), ser)[0]
        for lo, la, r in zip(lon_s, lat_s, portee):
            cercle_km(ax, lo, la, r, facecolor=SEQ_BLEU[6], alpha=0.12, edgecolor="none", zorder=3)
        ax.scatter(lon_s, lat_s, marker="^", s=18, color=ENCRE_2, edgecolor=SURFACE, linewidth=0.6, zorder=7)
        ax.set_title(f"Couverture à {alt} m (SER {ser} m², portée {portee.mean():.0f} km)", fontsize=10)
    ax = axes[2]
    lon_v, lat_v = vers_lonlat(vol.x, vol.y)
    lo_c, la_c = (np.min(lon_v) + np.max(lon_v)) / 2, (np.min(lat_v) + np.max(lat_v)) / 2
    demi_la = max(np.ptp(lat_v), np.ptp(lon_v) / ASPECT_CARTE) * 0.6 + 0.05
    demi_lo = demi_la * ASPECT_CARTE
    fond_carte(ax, theatre, emprise=(lo_c - demi_lo, lo_c + demi_lo, la_c - demi_la, la_c + demi_la), villes=True)
    ax.plot(lon_v, lat_v, color=ENCRE_2, lw=1.0, zorder=6, label="Trajectoire vraie")
    for k, p in enumerate(plots_piste):
        ax.scatter(p.lon, p.lat, s=6, color=CATEGORIEL[k % 3], zorder=7, linewidth=0,
                   label=f"Piste radar {k + 1} ({len(p)} plots)")
    ax.set_title(f"{vol.modele} : vérité et plots radar bruités", fontsize=10)
    ax.legend(loc="upper left", labelcolor=ENCRE_2, markerscale=2.5)
    fig.tight_layout()
    return fig


COULEUR_FIN = {
    "impact": ("Impact", "#e34948"),
    "abattu": ("Abattu", "#6250d6"),
    "brouille": ("Brouillé ou perdu", "#e87ba4"),
    "carburant": ("Panne sèche", "#eda100"),
    "atterrissage": ("Retour et atterrissage", "#008300"),
    "hors_couverture": ("Sorti de la zone simulée", "#1baf7a"),
}


def fig_volumes(pistes: pd.DataFrame):
    ordre = pistes.groupby("modele").size().sort_values()
    origine = pistes.drop_duplicates("modele").set_index("modele").origine
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(ordre))
    ax.barh(y, ordre.to_numpy(), height=0.55, color=[COULEUR_ORIGINE[origine[m]] for m in ordre.index])
    ax.set_yticks(y, ordre.index)
    for yi, val in zip(y, ordre.to_numpy()):
        ax.annotate(f"{val}", (val, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=ENCRE_2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Nombre de pistes radar")
    ax.set_title(f"Pistes radar par modèle ({len(pistes)} pistes)")
    _legende(ax, [Patch(color=COULEUR_ORIGINE["Russie"], label="Russie"),
                  Patch(color=COULEUR_ORIGINE["Ukraine"], label="Ukraine")], loc="lower right")
    fig.tight_layout()
    return fig


def fig_fins_de_vol(resume: pd.DataFrame):
    ordre = resume.groupby("modele").size().sort_values().index
    tab = pd.crosstab(resume.modele, resume.fin, normalize="index").reindex(columns=list(COULEUR_FIN), fill_value=0)
    tab = tab.loc[ordre]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    gauche = np.zeros(len(tab))
    for f, (lib, c) in COULEUR_FIN.items():
        ax.barh(np.arange(len(tab)), tab[f], left=gauche, height=0.55, color=c, label=lib,
                edgecolor=SURFACE, linewidth=2)
        for i, (g, v) in enumerate(zip(gauche, tab[f])):
            if v >= 0.12:
                ax.text(g + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                        color=ENCRE if c == "#eda100" else "#ffffff")
        gauche += tab[f].to_numpy()
    ax.set_yticks(np.arange(len(tab)), tab.index)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Fin de vol simulée (part des vols de chaque modèle)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, labelcolor=ENCRE_2)
    fig.tight_layout()
    return fig


def fig_distributions(variables: pd.DataFrame, colonnes=None):
    colonnes = colonnes or {
        "vitesse_med": "Vitesse médiane (km/h)", "alt_med": "Altitude médiane (m)",
        "duree_min": "Durée de la piste (min)", "rectitude": "Rectitude (déplacement / longueur)",
        "taux_virage_med": "Taux de virage médian (°/s)", "compacite": "Compacité (rayon de giration / longueur)",
    }
    roles = list(COULEUR_ROLE)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    for ax, (col, lib) in zip(axes.ravel(), colonnes.items()):
        donnees = [variables.loc[variables.role == r, col].dropna().to_numpy() for r in roles]
        bp = ax.boxplot(donnees, vert=False, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color=ENCRE, lw=1.6), whiskerprops=dict(color=ENCRE_MUETTE),
                        capprops=dict(color=ENCRE_MUETTE))
        for patch, r in zip(bp["boxes"], roles):
            patch.set_facecolor(COULEUR_ROLE[r])
            patch.set_alpha(0.85)
            patch.set_edgecolor(SURFACE)
        ax.set_yticks(np.arange(1, len(roles) + 1), roles)
        ax.set_title(lib, fontsize=10)
        ax.grid(axis="y", visible=False)
        if col in ("duree_min", "taux_virage_med"):
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for ax in axes[:, 1:].ravel():
        ax.set_yticklabels([])
    fig.suptitle("Distribution des principales variables par rôle (pistes radar, boîtes 25-75 %)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


# Apprentissage supervisé

def fig_comparaison_modeles(res: pd.DataFrame, titre):
    r = res.sort_values("f1_cv")
    y = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(r) + 1.6))
    ax.hlines(y, r.f1_cv - r.f1_cv_ecart_type, r.f1_cv + r.f1_cv_ecart_type, color=SEQ_BLEU[3], lw=2, zorder=2)
    ax.scatter(r.f1_cv, y - 0.0, s=60, color=CATEGORIEL[0], edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Validation croisée groupée (moyenne ± écart-type)")
    if "ic_bas" in r:
        ax.hlines(y - 0.28, r.ic_bas, r.ic_haut, color="#f3b493", lw=2, zorder=2)
    ax.scatter(r.f1_test, y - 0.28, s=60, color=CATEGORIEL[1], edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Jeu de test temporel (IC 95 % bootstrap par jour)")
    for yi, (a, b) in enumerate(zip(r.f1_cv, r.f1_test)):
        ax.annotate(f"{b:.3f}", (max(a, b), yi - 0.14), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=8, color=ENCRE_2)
    ax.set_yticks(y - 0.14, r.modele)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 macro")
    ax.set_xlim(0, 1.06)
    ax.set_title(titre)
    ax.legend(loc="lower right", labelcolor=ENCRE_2)
    fig.tight_layout()
    return fig


def fig_confusion(cm: np.ndarray, classes, titre, ax=None, annoter=True, taille=8):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap=CMAP_SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right", fontsize=taille)
    ax.set_yticks(range(len(classes)), classes, fontsize=taille)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if annoter:
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j] == 0:
                    continue
                v = cmn[i, j]
                ax.text(j, i, f"{v:.0%}" if v >= 0.01 else "<1%", ha="center", va="center", fontsize=taille - 0.5,
                        color="#ffffff" if v > 0.55 else ENCRE_2)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(titre, fontsize=10.5)
    return fig


def fig_courbes_apprentissage(courbes: dict):
    fig, axes = plt.subplots(1, len(courbes), figsize=(5 * len(courbes), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (nom, (tailles, s_tr, s_va)) in zip(axes, courbes.items()):
        for s, c, lab in ((s_tr, CATEGORIEL[1], "Apprentissage"), (s_va, CATEGORIEL[0], "Validation croisée groupée")):
            m, e = s.mean(axis=1), s.std(axis=1)
            ax.fill_between(tailles, m - e, m + e, color=c, alpha=0.12, linewidth=0)
            ax.plot(tailles, m, color=c, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=lab)
        ax.annotate(f"écart final {s_tr.mean(axis=1)[-1] - s_va.mean(axis=1)[-1]:.3f}",
                    (tailles[-1], s_va.mean(axis=1)[-1]), xytext=(-5, -18), textcoords="offset points",
                    ha="right", fontsize=8.5, color=ENCRE_2)
        ax.set_title(nom, fontsize=10.5)
        ax.set_xlabel("Nombre de pistes d'apprentissage")
    axes[0].set_ylabel("F1 macro")
    axes[0].legend(loc="lower right", labelcolor=ENCRE_2)
    fig.suptitle("Courbes d'apprentissage (F1 macro, validation croisée groupée par jour)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_courbe_validation(valeurs, s_tr, s_va, nom_param, titre, log=False):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(valeurs)) if not log else np.asarray(valeurs, float)
    for s, c, lab in ((s_tr, CATEGORIEL[1], "Apprentissage"), (s_va, CATEGORIEL[0], "Validation croisée groupée")):
        m, e = s.mean(axis=1), s.std(axis=1)
        ax.fill_between(x, m - e, m + e, color=c, alpha=0.12, linewidth=0)
        ax.plot(x, m, color=c, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, label=lab)
    if not log:
        ax.set_xticks(x, [str(v) for v in valeurs])
    else:
        ax.set_xscale("log")
    ax.set_xlabel(nom_param)
    ax.set_ylabel("F1 macro")
    ax.set_title(titre)
    ax.legend(loc="lower right", labelcolor=ENCRE_2)
    fig.tight_layout()
    return fig


def fig_permutation(score, scores_perm, p, titre):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    # Classes fixes de largeur 0,01 sans bordure : les scores permutés, très concentrés, restent visibles.
    ax.hist(scores_perm, bins=np.linspace(0, 1, 101), color=ENCRE_MUETTE, edgecolor="none",
            label=f"Étiquettes permutées ({len(scores_perm)} tirages, moyenne {np.mean(scores_perm):.3f})")
    ax.axvline(score, color=CATEGORIEL[0], lw=2, label=f"Étiquettes réelles : {score:.3f}")
    ax.annotate(f"p = {p:.3f}", (score, ax.get_ylim()[1] * 0.9), xytext=(-8, 0), textcoords="offset points",
                ha="right", color=ENCRE_2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 macro en validation croisée groupée")
    ax.set_ylabel("Nombre de tirages")
    ax.set_title(titre)
    ax.legend(loc="center", labelcolor=ENCRE_2)
    fig.tight_layout()
    return fig


def fig_importance(imp: pd.DataFrame, titre, n=15):
    d = imp.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.barh(y, d.importance, xerr=d.ecart_type, height=0.55, color=CATEGORIEL[0],
            error_kw=dict(ecolor=ENCRE_MUETTE, lw=1, capsize=0))
    ax.set_yticks(y, d.variable)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Baisse du F1 macro quand la variable est permutée (jeu de test)")
    ax.set_title(titre)
    fig.tight_layout()
    return fig


def fig_shap(shap_moy: np.ndarray, classes, variables, n=12, titre="Contribution moyenne |SHAP| par classe"):
    total = shap_moy.sum(axis=0)
    idx = np.argsort(total)[::-1][:n]
    data = shap_moy[:, idx].T
    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 4, 0.38 * n + 1.6))
    ax.imshow(data, cmap=CMAP_SEQ, aspect="auto")
    ax.set_xticks(range(len(classes)), classes, rotation=25, ha="right")
    ax.set_yticks(range(n), [variables[i] for i in idx])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    vmax = data.max()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if data[i, j] > 0.55 * vmax else ENCRE_2)
    ax.set_title(titre)
    fig.tight_layout()
    return fig


def fig_calibration(y, proba, titre, n_bins=10, effectif_min=20):
    conf = proba.max(axis=1)
    juste = proba.argmax(axis=1) == y
    bornes = np.linspace(0, 1, n_bins + 1)
    centres, freq, eff = [], [], []
    for a, b in zip(bornes[:-1], bornes[1:]):
        m = (conf > a) & (conf <= b)
        if m.sum() >= effectif_min:
            centres.append(conf[m].mean())
            freq.append(juste[m].mean())
            eff.append(int(m.sum()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw=dict(width_ratios=[1, 1.1]))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=AXE, lw=1, label="Calibration parfaite")
    ax.plot(centres, freq, color=CATEGORIEL[0], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
            label=f"Modèle retenu (classes d'au moins {effectif_min} pistes)")
    for xc, yc, n in zip(centres, freq, eff):
        ax.annotate(f"n = {n}", (xc, yc), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7.5,
                    color=ENCRE_MUETTE)
    ax.set_xlabel("Confiance annoncée (probabilité de la classe prédite)")
    ax.set_ylabel("Taux de bonnes réponses observé")
    ax.set_title("Diagramme de fiabilité (test)")
    ax.legend(loc="lower right", labelcolor=ENCRE_2)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax = axes[1]
    ax.hist(conf, bins=bornes, color=CATEGORIEL[0], edgecolor=SURFACE, linewidth=1.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Confiance annoncée")
    ax.set_ylabel("Nombre de pistes (échelle logarithmique)")
    ax.set_title("Répartition des confiances")
    fig.suptitle(titre, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_barres_simple(etiquettes, valeurs, xlabel, titre, erreurs=None, xlim=(0, 1), fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(etiquettes) + 1.5))
    y = np.arange(len(etiquettes))[::-1]
    ax.barh(y, valeurs, xerr=erreurs, height=0.55, color=CATEGORIEL[0],
            error_kw=dict(ecolor=ENCRE_MUETTE, lw=1, capsize=0))
    for yi, v in zip(y, valeurs):
        ax.annotate(fmt.format(v), (v, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5,
                    color=ENCRE_2)
    ax.set_yticks(y, etiquettes)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(titre)
    fig.tight_layout()
    return fig


def fig_courbes_par_cible(df, x, y, xlabel, titre, libelles, xlog=False, ylim=(0, 1.02), reference=None):
    """Petits multiples : une cible par panneau, une seule série par panneau."""
    cibles = list(dict.fromkeys(df.cible))
    fig, axes = plt.subplots(1, len(cibles), figsize=(4.6 * len(cibles), 3.9), sharey=True)
    for ax, cible in zip(np.atleast_1d(axes), cibles):
        g = df[df.cible == cible].sort_values(x)
        ax.plot(g[x], g[y], color=CATEGORIEL[0], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5)
        for xi, yi in zip(g[x].iloc[[0, -1]], g[y].iloc[[0, -1]]):
            ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, -15 if yi > 0.9 else 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=ENCRE_2)
        if reference is not None and cible in reference:
            ax.axhline(reference[cible], color=AXE, lw=1)
            ax.annotate("hasard", (g[x].iloc[0], reference[cible]), xytext=(2, 4), textcoords="offset points",
                        fontsize=8, color=ENCRE_MUETTE)
        if xlog:
            ax.set_xscale("log")
            ax.set_xticks(sorted(g[x].unique()), [f"{v:g}" for v in sorted(g[x].unique())])
            ax.minorticks_off()
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel)
        ax.set_title(libelles.get(cible, cible), fontsize=10.5)
    np.atleast_1d(axes)[0].set_ylabel("F1 macro (test)")
    fig.suptitle(titre, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_ablation(tables: dict, libelles):
    fig, axes = plt.subplots(1, len(tables), figsize=(6.2 * len(tables), 3.9), sharey=True)
    for ax, (cible, t) in zip(np.atleast_1d(axes), tables.items()):
        y = np.arange(len(t))[::-1]
        ax.barh(y, t.f1_test, height=0.55, color=CATEGORIEL[0])
        ax.errorbar(t.f1_cv, y, xerr=t.f1_cv_ecart_type, fmt="o", ms=5, color=ENCRE_2, mfc=SURFACE, lw=1,
                    label="Validation croisée (moyenne ± écart-type)")
        for yi, v in zip(y, t.f1_test):
            ax.annotate(f"{v:.3f}", (v, yi), xytext=(6, -9), textcoords="offset points", fontsize=8, color=ENCRE_2)
        ax.set_yticks(y, [f"{j} ({n})" for j, n in zip(t.jeu, t.n_variables)])
        ax.set_xlim(0, 1.08)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("F1 macro (barre : test)")
        ax.set_title(libelles.get(cible, cible), fontsize=10.5)
    np.atleast_1d(axes)[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), labelcolor=ENCRE_2, fontsize=8.5)
    fig.suptitle("Ablation : apport de chaque famille de variables (nombre de variables entre parenthèses)",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_coherence_osint(variables: pd.DataFrame, catalogue: pd.DataFrame, colonne, cmin, cmax, xlabel, titre,
                        echelle_log=False):
    """Bande publiée (OSINT) et distribution observée sur les pistes radar, modèle par modèle."""
    cat_ = catalogue.set_index("modele")
    ordre = [m for m in cat_.index if m in set(variables.modele)]
    fig, ax = plt.subplots(figsize=(10, 0.48 * len(ordre) + 1.6))
    for i, mod in enumerate(ordre[::-1]):
        a, b = cat_.loc[mod, cmin], cat_.loc[mod, cmax]
        if pd.notna(a) and pd.notna(b):
            ax.add_patch(plt.Rectangle((a, i - 0.32), b - a, 0.64, facecolor=SEQ_BLEU[0], edgecolor="none", zorder=1))
        v = variables.loc[variables.modele == mod, colonne].dropna()
        q05, q25, q50, q75, q95 = np.percentile(v, [5, 25, 50, 75, 95])
        ax.hlines(i, q05, q95, color=ENCRE_MUETTE, lw=1, zorder=2)
        ax.add_patch(plt.Rectangle((q25, i - 0.14), q75 - q25, 0.28, facecolor=CATEGORIEL[0], edgecolor="none",
                                   zorder=3))
        ax.vlines(q50, i - 0.2, i + 0.2, color=ENCRE, lw=1.6, zorder=4)
    ax.set_yticks(range(len(ordre)), ordre[::-1])
    ax.set_ylim(-0.7, len(ordre) - 0.3)
    if echelle_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ax.set_xlim(left=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(titre)
    _legende(ax, [Patch(facecolor=SEQ_BLEU[0], label="Fourchette publiée ou hypothèse du catalogue"),
                  Patch(facecolor=CATEGORIEL[0], label="Pistes radar simulées (25-75 %, trait : 5-95 %)")],
             loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    return fig


def fig_carte_erreurs(theatre, plots: pd.DataFrame, correct: pd.Series, titre, emprise=EMPRISE, pas=2):
    """Emphase : pistes bien classées en gris, erreurs en rouge."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fond_carte(ax, theatre, emprise=emprise)
    segs, cles = segments_par_piste(plots, pas=pas)
    ok = np.array([bool(correct.get(c, True)) for c in cles])
    lignes_carte(ax, [s for s, o in zip(segs, ok) if o], AXE, lw=0.5, alpha=0.55)
    lignes_carte(ax, [s for s, o in zip(segs, ok) if not o], ACCENT, lw=1.3, alpha=0.95, zorder=7)
    _legende(ax, [ligne_legende(AXE, f"Piste bien classée ({ok.sum()})", 1.5),
                  ligne_legende(ACCENT, f"Piste mal classée ({(~ok).sum()})", 1.8)], loc="lower left")
    ax.set_title(titre)
    fig.tight_layout()
    return fig


# Non supervisé

def fig_variance(pca, n_retenu):
    r = pca.explained_variance_ratio_
    k = np.arange(1, len(r) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(k, r, width=0.6, color=[CATEGORIEL[0] if i < n_retenu else AXE for i in range(len(r))])
    axes[0].set_xlabel("Composante principale")
    axes[0].set_ylabel("Part de variance expliquée")
    axes[0].set_title("Variance expliquée par composante")
    axes[0].grid(axis="x", visible=False)
    axes[1].plot(k, np.cumsum(r), color=CATEGORIEL[0], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5)
    axes[1].axhline(0.9, color=ENCRE_MUETTE, lw=1)
    axes[1].annotate(f"90 % atteints avec {n_retenu} composantes", (n_retenu, 0.9), xytext=(8, -16),
                     textcoords="offset points", color=ENCRE_2, fontsize=9)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Nombre de composantes")
    axes[1].set_ylabel("Variance cumulée")
    axes[1].set_title("Variance cumulée")
    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.tight_layout()
    return fig


def fig_charges(charges: pd.DataFrame, titre):
    ordre = charges.iloc[:, 0].sort_values().index
    d = charges.loc[ordre]
    fig, ax = plt.subplots(figsize=(1.3 * d.shape[1] + 4, 0.3 * d.shape[0] + 1.5))
    v = np.abs(d.to_numpy()).max()
    ax.imshow(d.to_numpy(), cmap=CMAP_DIV, vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(d.shape[1]), d.columns)
    ax.set_yticks(range(d.shape[0]), d.index, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if abs(d.iat[i, j]) >= 0.2:
                ax.text(j, i, f"{d.iat[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if abs(d.iat[i, j]) > 0.7 * v else ENCRE_2)
    ax.set_title(titre)
    fig.tight_layout()
    return fig


def fig_projection_roles(S, roles: pd.Series, pca=None, titre="Projection sur les deux premières composantes"):
    """Petits multiples : chaque rôle en couleur, les autres pistes en gris de contexte."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharex=True, sharey=True)
    for ax, role in zip(axes, COULEUR_ROLE):
        m = (roles == role).to_numpy()
        ax.scatter(S[~m, 0], S[~m, 1], s=3, color=AXE, alpha=0.35, linewidth=0, rasterized=True)
        ax.scatter(S[m, 0], S[m, 1], s=4, color=COULEUR_ROLE[role], alpha=0.6, linewidth=0, rasterized=True)
        ax.set_title(f"{role} ({m.sum()} pistes)", fontsize=10)
        ax.set_xlabel("CP1" + (f" ({pca.explained_variance_ratio_[0]:.0%})" if pca is not None else ""))
    axes[0].set_ylabel("CP2" + (f" ({pca.explained_variance_ratio_[1]:.0%})" if pca is not None else ""))
    fig.suptitle(titre, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_choix_k(ck: pd.DataFrame, k_km, k_gmm):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    for ax, col, lab, k_sel in ((axes[0], "silhouette", "Silhouette (K-Means, plus haut = mieux)", k_km),
                                (axes[1], "davies_bouldin", "Davies-Bouldin (K-Means, plus bas = mieux)", k_km),
                                (axes[2], "bic_gmm", "BIC du mélange gaussien (plus bas = mieux)", k_gmm)):
        ax.plot(ck.k, ck[col], color=CATEGORIEL[0], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5)
        sel = ck.loc[ck.k == k_sel, col].iloc[0]
        ax.scatter([k_sel], [sel], s=90, color=CATEGORIEL[1], edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.annotate(f"k retenu = {k_sel}", (k_sel, sel), xytext=(8, 8), textcoords="offset points", color=ENCRE_2)
        ax.set_xlabel("Nombre de groupes k")
        ax.set_title(lab, fontsize=10)
        ax.set_xticks(ck.k)
    fig.tight_layout()
    return fig


def fig_contingences(parts: dict, etiquettes: pd.Series, titre):
    fig, axes = plt.subplots(1, len(parts), figsize=(4.3 * len(parts), 3.8))
    for ax, (nom, lab) in zip(np.atleast_1d(axes), parts.items()):
        tab = pd.crosstab(pd.Series(lab, name="groupe"), etiquettes.reset_index(drop=True))
        tab = tab.reindex(columns=[c for c in COULEUR_ROLE if c in tab.columns])
        part = tab / tab.sum(axis=1).to_numpy()[:, None]
        ax.imshow(part.to_numpy(), cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(tab.shape[1]), [c.replace(" longue portée", "") for c in tab.columns], rotation=30,
                      ha="right", fontsize=8)
        ax.set_yticks(range(tab.shape[0]), [("Bruit" if g == -1 else f"Groupe {g}") + f" ({n})"
                                            for g, n in zip(tab.index, tab.sum(axis=1))], fontsize=8)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(tab.shape[0]):
            for j in range(tab.shape[1]):
                v = part.iat[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if v > 0.55 else ENCRE_2)
        ax.set_title(nom, fontsize=10)
    fig.suptitle(titre, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def couleurs_groupes(labels):
    uniques = [g for g in sorted(set(labels)) if g >= 0]
    table = {g: CATEGORIEL[i % len(CATEGORIEL)] for i, g in enumerate(uniques)}
    table[-1] = AXE
    return table


def fig_tsne(emb, labels_dict: dict, titre):
    fig, axes = plt.subplots(1, len(labels_dict), figsize=(5.2 * len(labels_dict), 4.8))
    for ax, (nom, lab) in zip(np.atleast_1d(axes), labels_dict.items()):
        if nom == "Rôle réel":
            for role, c in COULEUR_ROLE.items():
                m = lab == role
                ax.scatter(emb[m, 0], emb[m, 1], s=4, color=c, alpha=0.7, linewidth=0, label=role, rasterized=True)
        else:
            table = couleurs_groupes(lab)
            for g, c in table.items():
                m = lab == g
                if m.any():
                    ax.scatter(emb[m, 0], emb[m, 1], s=4, color=c, alpha=0.7, linewidth=0,
                               label="Bruit" if g == -1 else f"Groupe {g}", rasterized=True)
        ax.set_title(nom, fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, markerscale=3, labelcolor=ENCRE_2,
                  fontsize=8)
    fig.suptitle(titre, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def fig_carte_groupes(theatre, plots, groupes: pd.Series, titre, emprise=EMPRISE):
    table = couleurs_groupes(groupes.to_numpy())
    uniques = [g for g in sorted(table) if g >= 0]
    fig, axes = plt.subplots(1, len(uniques), figsize=(5.6 * len(uniques), 5.2))
    for ax, g in zip(np.atleast_1d(axes), uniques):
        fond_carte(ax, theatre, emprise=emprise, villes=False)
        ids = set(groupes.index[groupes == g])
        segs, _ = segments_par_piste(plots[plots.piste_id.isin(ids)])
        lignes_carte(ax, segs, table[g], lw=0.6, alpha=0.55)
        ax.set_title(f"Groupe {g} ({len(ids)} pistes)", fontsize=10.5)
    fig.suptitle(titre, x=0.01, ha="left")
    fig.tight_layout()
    return fig


# Synthèse

def fig_synthese(metriques: dict, theatre, plots_test, correct: pd.Series, cm_role, classes_role,
                 comparaison_role: pd.DataFrame, precoce_role: pd.DataFrame):
    """Planche de synthèse : chiffres clés, carte des erreurs, confusion, algorithmes, classification précoce."""
    c = metriques["cibles"]
    fig = plt.figure(figsize=(16, 13.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.42, 2.2, 1.45], hspace=0.55, wspace=0.32, top=0.92, bottom=0.05)
    tuiles = [("Pistes radar simulées", f"{metriques['jeu']['pistes']:,}".replace(",", " "),
               f"dont {metriques['jeu']['test']} pistes de test (30 derniers jours)")]
    for cible, lib in (("role", "Rôle, 4 classes"), ("origine", "Origine, 2 classes"), ("modele", "Modèle exact, 12 classes")):
        a, b = c[cible]["f1_test_ic95"]
        tuiles.append((f"{lib} : F1 macro test", f"{c[cible]['f1_test']:.3f}",
                       f"IC 95 % {a:.3f} à {b:.3f}, hasard {c[cible]['f1_test_baseline_stratifiee']:.3f}"))
    for j, (lab, val, sous) in enumerate(tuiles):
        ax = fig.add_subplot(gs[0, j])
        ax.axis("off")
        ax.text(0, 0.98, lab, fontsize=10.5, color=ENCRE_2, va="top", transform=ax.transAxes)
        ax.text(0, 0.50, val, fontsize=30, color=ENCRE, va="center", weight="bold", transform=ax.transAxes)
        ax.text(0, 0.06, sous, fontsize=8.8, color=ENCRE_MUETTE, va="center", transform=ax.transAxes)
    ax = fig.add_subplot(gs[1, :2])
    fond_carte(ax, theatre, villes=True)
    segs, cles = segments_par_piste(plots_test, pas=3)
    ok = np.array([bool(correct.get(k, True)) for k in cles])
    lignes_carte(ax, [s for s, o in zip(segs, ok) if o], AXE, lw=0.5, alpha=0.5)
    lignes_carte(ax, [s for s, o in zip(segs, ok) if not o], ACCENT, lw=1.3, alpha=0.95, zorder=7)
    _legende(ax, [ligne_legende(AXE, f"Rôle bien prédit ({ok.sum()} pistes)", 1.5),
                  ligne_legende(ACCENT, f"Rôle mal prédit ({(~ok).sum()} pistes)", 1.8)], loc="lower left")
    ax.set_title("Pistes du jeu de test et erreurs de rôle")
    ax = fig.add_subplot(gs[1, 2:])
    courts = {"Frappe longue portée": "Frappe", "Munition rôdeuse": "Rôdeuse", "Reconnaissance": "Reconnaiss."}
    fig_confusion(cm_role, [courts.get(k, k) for k in classes_role],
                  f"Rôle : matrice de confusion ({c['role']['modele_retenu']})", ax=ax, taille=9)
    ax = fig.add_subplot(gs[2, :2])
    r = comparaison_role[~comparaison_role.modele.str.startswith("Baseline")].sort_values("f1_test")
    y = np.arange(len(r))
    ax.hlines(y, r.ic_bas, r.ic_haut, color=SEQ_BLEU[3], lw=2)
    ax.scatter(r.f1_test, y, s=50, color=CATEGORIEL[0], edgecolor=SURFACE, linewidth=2, zorder=3)
    for yi, v in zip(y, r.f1_test):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(10, 0), textcoords="offset points", va="center", fontsize=8,
                    color=ENCRE_2)
    retenu = c["role"]["modele_retenu"]
    ax.set_yticks(y, [f"{m} (retenu)" if m == retenu else m for m in r.modele], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 macro sur le test (IC 95 % bootstrap par jour) ; modèle retenu sur la validation croisée")
    ax.set_title(f"Rôle : {len(r)} algorithmes comparés (hors références naïves)")
    ax = fig.add_subplot(gs[2, 2:])
    p = precoce_role.sort_values("horizon_min")
    ax.plot(p.horizon_min, p.f1_test, color=CATEGORIEL[0], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5)
    for xi, yi in zip(p.horizon_min.iloc[[0, -1]], p.f1_test.iloc[[0, -1]]):
        ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(0, -15 if yi > 0.9 else 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=ENCRE_2)
    ax.set_xscale("log")
    ax.set_xticks(p.horizon_min, [f"{v:g}" for v in p.horizon_min])
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Minutes de piste observées")
    ax.set_ylabel("F1 macro (test)")
    ax.set_title("Rôle : classification précoce")
    fig.suptitle("Drones Ukraine v2 : synthèse de la classification des pistes radar simulées", x=0.01, ha="left",
                 fontsize=14)
    return fig
