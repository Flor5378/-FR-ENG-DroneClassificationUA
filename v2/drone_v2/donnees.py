"""Génération, sauvegarde et chargement du jeu de données v2."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config
from .geo import Theatre
from .radar import ReseauRadar, observer_vols
from .simulation import Simulateur, vols_vers_df, vols_vers_points
from .variables import table_variables

F_VOLS = config.GENEREES / "vols_v2.csv.gz"
F_PISTES = config.GENEREES / "pistes_v2.csv.gz"
F_VARIABLES = config.GENEREES / "variables_v2.csv.gz"
F_RADARS = config.GENEREES / "radars_v2.csv"
F_ECHANTILLON = config.GENEREES / "plots_echantillon_v2.parquet"
# Fichiers volumineux, régénérables à l'identique (non versionnés).
F_PLOTS = config.GENEREES / "plots_v2.parquet"
F_VERITE = config.GENEREES / "verite_v2.parquet"


def generer(graine: int = config.GRAINE, nb_jours: int = config.NB_JOURS, verbeux: bool = True) -> dict:
    """Simule les vols, les observe au radar et calcule les variables par piste."""
    t0 = time.time()
    theatre = Theatre()
    radar = ReseauRadar(theatre, graine)
    sim = Simulateur(theatre, graine, nb_jours)
    vols = sim.simuler(progression=verbeux)
    if verbeux:
        print(f"{len(vols)} vols simulés en {time.time() - t0:.0f} s")
    pistes, plots = observer_vols(vols, radar, graine)
    if verbeux:
        print(f"{len(pistes)} pistes radar, {len(plots)} plots ({time.time() - t0:.0f} s)")
    resume = vols_vers_df(vols)
    n_pistes = pistes.groupby("drone_id").size()
    resume["nb_pistes"] = resume.drone_id.map(n_pistes).fillna(0).astype(int)
    variables = table_variables(pistes, plots, theatre)
    if verbeux:
        print(f"Variables calculées ({time.time() - t0:.0f} s)")
    return dict(theatre=theatre, radar=radar, simulateur=sim, vols=vols, resume=resume,
                pistes=pistes, plots=plots, variables=variables)


def echantillon_plots(plots: pd.DataFrame, pistes: pd.DataFrame, n_par_modele: int = 60,
                      pas: int = 6, graine: int = config.GRAINE) -> pd.DataFrame:
    """Sous-ensemble léger pour les cartes : n pistes par modèle, un plot sur ``pas``."""
    rng = np.random.default_rng(graine)
    choix = []
    for _, g in pistes.groupby("modele"):
        ids = g.piste_id.to_numpy()
        choix.extend(rng.choice(ids, size=min(n_par_modele, len(ids)), replace=False))
    ech = plots[plots.piste_id.isin(set(choix))]
    ech = ech[ech.groupby("piste_id").cumcount() % pas == 0]
    return ech.reset_index(drop=True)


# Horodatage gzip nul : deux générations identiques donnent des fichiers identiques octet par octet.
GZIP = {"method": "gzip", "mtime": 0}


def sauvegarder(d: dict, complet: bool = True) -> None:
    d["resume"].to_csv(F_VOLS, index=False, float_format="%.5g", compression=GZIP)
    d["pistes"].to_csv(F_PISTES, index=False, float_format="%.10g", compression=GZIP)
    d["variables"].to_csv(F_VARIABLES, index=False, float_format="%.6g", compression=GZIP)
    d["radar"].table_sites().to_csv(F_RADARS, index=False, float_format="%.5f")
    echantillon_plots(d["plots"], d["pistes"]).to_parquet(F_ECHANTILLON, index=False)
    if complet:
        d["plots"].to_parquet(F_PLOTS, index=False)
        vols_vers_points(d["vols"]).to_parquet(F_VERITE, index=False)


def charger_tables() -> dict:
    """Tables versionnées (aucune régénération nécessaire)."""
    return dict(
        resume=pd.read_csv(F_VOLS),
        pistes=pd.read_csv(F_PISTES),
        variables=pd.read_csv(F_VARIABLES),
        radars=pd.read_csv(F_RADARS),
        echantillon=pd.read_parquet(F_ECHANTILLON),
    )


def charger_plots(graine: int = config.GRAINE) -> pd.DataFrame:
    """Plots radar complets ; régénérés à l'identique s'ils sont absents."""
    if not F_PLOTS.exists():
        print("plots_v2.parquet absent : régénération déterministe (environ 2 minutes)...")
        sauvegarder(generer(graine), complet=True)
    return pd.read_parquet(F_PLOTS)


def reobserver(vols, radar: ReseauRadar, theatre: Theatre, facteur_bruit: float,
               graine: int = config.GRAINE, cle: int = 1) -> pd.DataFrame:
    """Nouvelle observation radar des mêmes vols avec un bruit multiplié (test de robustesse)."""
    pistes, plots = observer_vols(vols, radar, graine, facteur_bruit=facteur_bruit, cle=cle)
    return table_variables(pistes, plots, theatre)
