"""Analyse non supervisée v2 : PCA et clustering des comportements de vol.

Les variables géographiques et horaires sont volontairement exclues : le but est
de regrouper les pistes selon leur comportement physique (vitesse, virages,
altitude, durée), pas selon le lieu ou l'heure. Les étiquettes ne servent jamais
à l'ajustement ; elles ne sont utilisées qu'après coup pour mesurer l'accord
entre groupes découverts et classes réelles (ARI, NMI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, completeness_score,
                             davies_bouldin_score, homogeneity_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config
from .variables import FAMILLES

VARIABLES_NS = FAMILLES["observation"] + FAMILLES["cinematique"] + FAMILLES["altitude"]
# Variables positives très asymétriques : passage au logarithme avant standardisation.
ASYMETRIQUES = ["duree_min", "n_plots", "longueur_km", "deplacement_km", "rayon_giration_km",
                "vitesse_std", "accel_p95", "taux_virage_med", "taux_virage_p90", "virage_deg_par_min",
                "alt_std"]


class Preparation:
    """log1p des variables asymétriques puis standardisation, ajustées sur l'apprentissage."""

    def __init__(self, colonnes=VARIABLES_NS):
        self.colonnes = list(colonnes)
        self.scaler = StandardScaler()

    def _brut(self, df):
        X = df[self.colonnes].astype(float).copy()
        for c in ASYMETRIQUES:
            if c in X:
                X[c] = np.log1p(X[c].clip(lower=0))
        return X.fillna(X.median())

    def fit(self, df):
        self.scaler.fit(self._brut(df))
        return self

    def transform(self, df):
        return self.scaler.transform(self._brut(df))


def acp(Z_train, Z_tous, seuil=0.90, graine=config.GRAINE):
    pca = PCA(random_state=graine).fit(Z_train)
    n = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), seuil) + 1)
    return pca, pca.transform(Z_tous), n


def charges(pca, colonnes, n=5):
    return pd.DataFrame(pca.components_[:n].T, index=colonnes, columns=[f"CP{i + 1}" for i in range(n)])


def choix_k(S, ks=range(2, 13), graine=config.GRAINE, n_sil=4000):
    rng = np.random.default_rng(graine)
    idx = rng.choice(len(S), size=min(n_sil, len(S)), replace=False)
    lignes = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=graine).fit(S)
        gmm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=graine).fit(S)
        lignes.append(dict(
            k=k, inertie=km.inertia_, silhouette=silhouette_score(S[idx], km.labels_[idx]),
            davies_bouldin=davies_bouldin_score(S, km.labels_), calinski_harabasz=calinski_harabasz_score(S, km.labels_),
            bic_gmm=gmm.bic(S), aic_gmm=gmm.aic(S),
        ))
    return pd.DataFrame(lignes)


def partitions(S, k_kmeans, k_gmm, graine=config.GRAINE, taille_min_hdbscan=60):
    return {
        "K-Means": KMeans(k_kmeans, n_init=20, random_state=graine).fit_predict(S),
        "Mélange gaussien (GMM)": GaussianMixture(k_gmm, covariance_type="full", n_init=5,
                                                  random_state=graine).fit(S).predict(S),
        "CAH de Ward": AgglomerativeClustering(k_kmeans, linkage="ward").fit_predict(S),
        "HDBSCAN": HDBSCAN(min_cluster_size=taille_min_hdbscan, min_samples=10).fit_predict(S),
    }


def evaluer_partitions(S, parts: dict, etiquettes: pd.DataFrame, graine=config.GRAINE, n_sil=4000):
    rng = np.random.default_rng(graine)
    idx = rng.choice(len(S), size=min(n_sil, len(S)), replace=False)
    lignes = []
    for nom, lab in parts.items():
        valide = lab >= 0  # HDBSCAN marque le bruit par -1
        ligne = dict(methode=nom, n_groupes=len(set(lab[valide])), part_bruit=float(1 - valide.mean()))
        iv = idx[valide[idx]]
        ligne["silhouette"] = silhouette_score(S[iv], lab[iv]) if len(set(lab[iv])) > 1 else np.nan
        ligne["davies_bouldin"] = davies_bouldin_score(S[valide], lab[valide]) if ligne["n_groupes"] > 1 else np.nan
        for cible in ("role", "origine", "modele"):
            y = etiquettes[cible].to_numpy()[valide]
            ligne[f"ARI_{cible}"] = adjusted_rand_score(y, lab[valide])
            ligne[f"NMI_{cible}"] = normalized_mutual_info_score(y, lab[valide])
        y = etiquettes["role"].to_numpy()[valide]
        ligne["homogeneite_role"] = homogeneity_score(y, lab[valide])
        ligne["completude_role"] = completeness_score(y, lab[valide])
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def stabilite(S, k, n=25, fraction=0.8, graine=config.GRAINE):
    """ARI entre la partition de référence et des partitions refaites sur des sous-échantillons."""
    rng = np.random.default_rng(graine)
    ref = KMeans(k, n_init=10, random_state=graine).fit(S)
    aris = []
    for i in range(n):
        idx = rng.choice(len(S), size=int(fraction * len(S)), replace=False)
        km = KMeans(k, n_init=10, random_state=graine + i + 1).fit(S[idx])
        aris.append(adjusted_rand_score(ref.labels_[idx], km.labels_))
    return np.asarray(aris)


def tsne(S, n=3000, graine=config.GRAINE):
    rng = np.random.default_rng(graine)
    idx = np.sort(rng.choice(len(S), size=min(n, len(S)), replace=False))
    emb = TSNE(2, perplexity=40, init="pca", random_state=graine).fit_transform(S[idx])
    return idx, emb


def profils(df: pd.DataFrame, labels, colonnes=("vitesse_med", "alt_med", "duree_min", "rectitude",
                                                   "taux_virage_med", "compacite", "vz_p05")):
    p = df.assign(groupe=labels).groupby("groupe")[list(colonnes)].median()
    p["effectif"] = pd.Series(labels).value_counts().sort_index().to_numpy()
    return p
