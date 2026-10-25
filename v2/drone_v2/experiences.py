"""Expériences complémentaires de la v2 : ablation, classification précoce,
robustesse au bruit radar, généralisation géographique et comparaison avec les
variables de la v1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shapely
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

from . import config, ml
from .geo import Theatre, vers_xy
from .radar import ReseauRadar
from .simulation import Simulateur
from .variables import FAMILLES, VARIABLES, table_variables


def ablation(modele, train, test, cible, enc, nom=None):
    """Score selon les familles de variables disponibles."""
    physiques = FAMILLES["cinematique"] + FAMILLES["altitude"]
    jeux = {
        "Cinématique seule": FAMILLES["cinematique"],
        "Altitude seule": FAMILLES["altitude"],
        "Cinématique + altitude": physiques,
        "+ observation": physiques + FAMILLES["observation"],
        "+ géographie": physiques + FAMILLES["observation"] + FAMILLES["geographie"],
        "Toutes (+ heure)": VARIABLES,
    }
    ytr, yte = enc.transform(train[cible]), enc.transform(test[cible])
    lignes = []
    for nom_jeu, cols in jeux.items():
        m = clone(modele)
        cv = cross_val_score(m, train[cols], ytr, groups=train.jour, cv=ml.cv_groupee(), scoring="f1_macro",
                             params=ml._params_fit(nom, ytr) or None, n_jobs=4)
        m.fit(train[cols], ytr, **ml._params_fit(nom, ytr))
        lignes.append(dict(jeu=nom_jeu, n_variables=len(cols), f1_cv=cv.mean(), f1_cv_ecart_type=cv.std(),
                           f1_test=f1_score(yte, m.predict(test[cols]), average="macro")))
    return pd.DataFrame(lignes)


def classification_precoce(modele, pistes, plots, theatre, cibles, encs, horizons=(3, 5, 10, 15, 30, 60),
                           noms=None, variables=VARIABLES):
    """F1 macro sur le test quand seules les H premières minutes de chaque piste sont observées."""
    lignes = []
    jour_limite = pistes.jour.max() + 1 - config.NB_JOURS_TEST
    for h in horizons:
        v = table_variables(pistes, plots, theatre, horizon_min=h)
        tr, te = v[v.jour < jour_limite], v[v.jour >= jour_limite]
        for cible in cibles:
            enc = encs[cible]
            ytr, yte = enc.transform(tr[cible]), enc.transform(te[cible])
            nom = noms[cible] if noms else None
            m = clone(modele[cible])
            m.fit(tr[variables], ytr, **ml._params_fit(nom, ytr))
            p = m.predict(te[variables])
            lignes.append(dict(horizon_min=h, cible=cible, f1_test=f1_score(yte, p, average="macro"),
                               exactitude_test=accuracy_score(yte, p), n_test=len(te)))
    return pd.DataFrame(lignes)


def robustesse_bruit(modeles, encs, theatre: Theatre, facteurs=(1.0, 1.5, 2.0, 3.0), graine=config.GRAINE,
                     variables=VARIABLES):
    """Modèles entraînés au bruit nominal, évalués sur une nouvelle observation des vols du test
    avec un bruit radar multiplié."""
    radar = ReseauRadar(theatre, graine)
    jours_test = range(config.NB_JOURS - config.NB_JOURS_TEST, config.NB_JOURS)
    vols_test = Simulateur(theatre, graine).simuler(jours=jours_test)
    from .radar import observer_vols
    lignes = []
    for f in facteurs:
        pistes, plots = observer_vols(vols_test, radar, graine, facteur_bruit=f, cle=1)
        v = table_variables(pistes, plots, theatre)
        for cible, m in modeles.items():
            y = encs[cible].transform(v[cible])
            p = m.predict(v[variables])
            lignes.append(dict(facteur_bruit=f, cible=cible, n_pistes=len(v),
                               f1_test=f1_score(y, p, average="macro"), exactitude_test=accuracy_score(y, p)))
    return pd.DataFrame(lignes)


def secteur(lat_debut: pd.Series) -> pd.Series:
    """Secteur géographique d'une piste d'après la latitude de son premier plot."""
    return pd.Series(np.select([lat_debut < 47.8, lat_debut >= 50.0], ["Sud", "Nord"], default="Est"),
                     index=lat_debut.index)


def generalisation_geographique(modele, train, test, cible, enc, secteurs_train, secteurs_test, exclu="Sud",
                                nom=None, variables=None):
    """Compare un modèle qui a vu le secteur exclu à un modèle qui ne l'a jamais vu."""
    variables = variables or (FAMILLES["cinematique"] + FAMILLES["altitude"] + FAMILLES["observation"])
    ytr, yte = enc.transform(train[cible]), enc.transform(test[cible])
    masque_te = (secteurs_test == exclu).to_numpy()
    lignes = []
    for nom_cas, masque_tr in (("Secteur vu à l'apprentissage", np.ones(len(train), bool)),
                               ("Secteur jamais vu", (secteurs_train != exclu).to_numpy())):
        m = clone(modele)
        m.fit(train.loc[masque_tr, variables], ytr[masque_tr], **ml._params_fit(nom, ytr[masque_tr]))
        p = m.predict(test.loc[masque_te, variables])
        lignes.append(dict(cas=nom_cas, secteur_test=exclu, n_apprentissage=int(masque_tr.sum()),
                           n_test=int(masque_te.sum()), f1_test=f1_score(yte[masque_te], p, average="macro")))
    return pd.DataFrame(lignes)


VARIABLES_V1 = ["Distance_Bordure", "Direction_Ukraine", "Changement_Direction", "Vitesse", "Altitude"]


def variables_v1(plots: pd.DataFrame, theatre: Theatre) -> pd.DataFrame:
    """Variables de la v1 (notebook R) recalculées sur les pistes radar de la v2."""
    bord = theatre.ua_controle_xy.boundary
    lignes = []
    for piste_id, g in plots.sort_values(["piste_id", "t"]).groupby("piste_id", sort=False):
        lon, lat, alt, t = g.lon.to_numpy(), g.lat.to_numpy(), g.alt.to_numpy(float), g.t.to_numpy()
        x, y = vers_xy(lon[:6], lat[:6])
        v = np.hypot(np.diff(x), np.diff(y)) / np.diff(t[:6]) * 3.6
        lignes.append(dict(
            piste_id=piste_id,
            Distance_Bordure=float(shapely.distance(shapely.points(x[0], y[0]), bord)),
            Direction_Ukraine=int(lon[0] > lon[1]),
            Changement_Direction=int(np.var(lon) > 0.1 or np.var(lat) > 0.1),
            Vitesse=float(v[:5].mean()), Altitude=float(alt[:5].mean()),
        ))
    return pd.DataFrame(lignes)
