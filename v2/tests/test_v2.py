"""Tests de la v2 : cohérence physique, reproductibilité et garde-fous anti-fuite.

Lancement depuis le dossier v2 : ``python -m pytest -q tests``
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drone_v2 import catalogue as cat  # noqa: E402
from drone_v2 import config, ml  # noqa: E402
from drone_v2.geo import Theatre, vers_xy  # noqa: E402
from drone_v2.radar import ReseauRadar, observer_vols  # noqa: E402
from drone_v2.simulation import Simulateur, vols_vers_df  # noqa: E402
from drone_v2.variables import INTERDITES, VARIABLES, table_variables, verifier_absence_de_fuite  # noqa: E402


@pytest.fixture(scope="module")
def theatre():
    return Theatre()


@pytest.fixture(scope="module")
def vols(theatre):
    return Simulateur(theatre).simuler(jours=range(4))


def test_catalogue_complet_et_source():
    for m in cat.CATALOGUE.values():
        assert m["role"] in cat.ROLES and m["origine"] in cat.ORIGINES
        assert m["v_croisiere_min"] <= m["v_croisiere_max"] <= m["v_max"]
        for s in m["sources"].split(";"):
            assert s in cat.SOURCES


def test_geographie(theatre):
    kyiv, donetsk = vers_xy(30.52, 50.45), vers_xy(37.80, 48.00)
    assert theatre.est_cote_ua(*kyiv) and not theatre.est_cote_russe(*kyiv)
    assert theatre.est_cote_russe(*donetsk)
    assert theatre.distance_signee_km(*kyiv)[0] > 0 > theatre.distance_signee_km(*donetsk)[0]


def test_vitesses_dans_les_fourchettes_publiees(vols):
    for modele, g in vols_vers_df(vols).groupby("modele"):
        m = cat.CATALOGUE[modele]
        assert g.v_croisiere_kmh.between(m["v_croisiere_min"], m["v_croisiere_max"]).all()


def test_portee_publiee_respectee(vols):
    df = vols_vers_df(vols)
    for modele, g in df[df.fin != "brouille"].groupby("modele"):
        assert g.eloignement_max_km.max() <= cat.CATALOGUE[modele]["portee_km"]


def test_altitudes_physiques(vols):
    for v in vols:
        assert (v.z >= 0).all() and v.z.max() < 6500


def test_simulation_deterministe(theatre):
    a = Simulateur(theatre).simuler(jours=[7])
    b = Simulateur(theatre).simuler(jours=[7])
    assert [v.drone_id for v in a] == [v.drone_id for v in b]
    for va, vb in zip(a, b):
        np.testing.assert_array_equal(va.x, vb.x)
        np.testing.assert_array_equal(va.z, vb.z)


def test_observation_et_variables(theatre, vols):
    pistes, plots = observer_vols(vols, ReseauRadar(theatre))
    assert len(pistes) > 0 and plots.piste_id.isin(pistes.piste_id).all()
    v = table_variables(pistes, plots, theatre)
    X = v[VARIABLES]
    assert np.isfinite(X.to_numpy()).all()
    assert X.rectitude.between(0, 1.0001).all()
    assert X.frac_cote_ua.between(0, 1).all()


def test_garde_anti_fuite():
    verifier_absence_de_fuite(VARIABLES)
    assert not set(VARIABLES) & INTERDITES
    for colonne in ("modele", "role", "origine", "v_croisiere_kmh", "fin_vol", "raid_id"):
        with pytest.raises(ValueError):
            verifier_absence_de_fuite(VARIABLES + [colonne])


def test_decoupage_temporel_sans_recouvrement():
    variables = pd.read_csv(config.GENEREES / "variables_v2.csv.gz")
    train, test = ml.decouper(variables)
    ml.controler_decoupage(train, test)
    assert train.jour.max() < test.jour.min()
    assert test.jour.nunique() == config.NB_JOURS_TEST
