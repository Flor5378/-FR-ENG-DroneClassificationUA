"""Apprentissage supervisé v2 : protocole de validation sans fuite.

Protocole :
1. découpage temporel : les 30 derniers jours forment le jeu de test, mis de côté
   avant toute exploration et utilisé une seule fois pour l'évaluation finale ;
2. validation croisée groupée par jour sur l'apprentissage (StratifiedGroupKFold) :
   les pistes d'un même jour (même raid, même vent, mêmes routes) restent ensemble,
   ce qui empêche l'apprentissage de quasi-doublons ;
3. prétraitements (imputation, normalisation) dans des pipelines, ajustés sur les
   seuls plis d'apprentissage ;
4. hyperparamètres choisis par recherche aléatoire en validation croisée, modèle
   retenu sur le score de validation croisée (jamais sur le test) ;
5. baselines naïves, intervalles de confiance par bootstrap groupé, test de
   permutation des étiquettes, courbes d'apprentissage, calibration.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedGroupKFold, learning_curve,
                                     permutation_test_score, validation_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config
from .variables import VARIABLES, verifier_absence_de_fuite

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CIBLES = {
    "role": "Rôle (frappe, leurre, munition rôdeuse, reconnaissance)",
    "origine": "Origine (Russie ou Ukraine)",
    "modele": "Modèle exact (12 classes)",
}


# Découpage

def decouper(variables: pd.DataFrame, nb_jours_test: int = config.NB_JOURS_TEST):
    """Jeu d'apprentissage = jours anciens, jeu de test = derniers jours."""
    jour_limite = variables.jour.max() + 1 - nb_jours_test
    train = variables[variables.jour < jour_limite].reset_index(drop=True)
    test = variables[variables.jour >= jour_limite].reset_index(drop=True)
    return train, test


def controler_decoupage(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Vérifie qu'aucun jour, raid, groupe ni vol n'est partagé entre apprentissage et test."""
    lignes = []
    for col in ("jour", "raid_id", "groupe_id", "drone_id", "piste_id"):
        commun = set(train[col]) & set(test[col])
        lignes.append(dict(identifiant=col, en_commun=len(commun)))
        if commun:
            raise AssertionError(f"Fuite : {len(commun)} valeurs de {col} communes")
    verifier_absence_de_fuite(VARIABLES)
    return pd.DataFrame(lignes)


def cv_groupee(graine=config.GRAINE, n=5):
    return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=graine)


# Modèles candidats

def _pipe(clf, normaliser=False):
    etapes = [("imputation", SimpleImputer(strategy="median"))]
    if normaliser:
        etapes.append(("normalisation", StandardScaler()))
    etapes.append(("clf", clf))
    return Pipeline(etapes)


def modeles_candidats(graine=config.GRAINE) -> dict:
    """Nom -> (pipeline, espace de recherche, nombre de tirages)."""
    g = graine
    return {
        "Baseline (classe majoritaire)": (_pipe(DummyClassifier(strategy="most_frequent")), {}, 1),
        "Baseline (tirage stratifié)": (_pipe(DummyClassifier(strategy="stratified", random_state=g)), {}, 1),
        "Régression logistique": (
            _pipe(LogisticRegression(max_iter=4000, class_weight="balanced"), True),
            {"clf__C": loguniform(1e-3, 1e2)}, 12),
        "k plus proches voisins": (
            _pipe(KNeighborsClassifier(), True),
            {"clf__n_neighbors": randint(3, 60), "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]}, 12),
        "SVM (noyau RBF)": (
            _pipe(SVC(class_weight="balanced", random_state=g), True),
            {"clf__C": loguniform(0.3, 300), "clf__gamma": loguniform(3e-3, 0.3)}, 10),
        "Random Forest": (
            _pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=1, random_state=g)),
            {"clf__max_depth": [None, 8, 12, 16, 24], "clf__min_samples_leaf": [1, 2, 4, 8, 16],
             "clf__max_features": ["sqrt", 0.3, 0.5]}, 12),
        "Extra Trees": (
            _pipe(ExtraTreesClassifier(n_estimators=300, class_weight="balanced", n_jobs=1, random_state=g)),
            {"clf__max_depth": [None, 8, 12, 16, 24], "clf__min_samples_leaf": [1, 2, 4, 8, 16],
             "clf__max_features": ["sqrt", 0.3, 0.5]}, 12),
        "HistGradientBoosting": (
            _pipe(HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=g)),
            {"clf__learning_rate": loguniform(0.02, 0.3), "clf__max_iter": [100, 200, 400],
             "clf__max_leaf_nodes": [15, 31, 63], "clf__min_samples_leaf": [10, 20, 50],
             "clf__l2_regularization": loguniform(1e-3, 10)}, 12),
        "XGBoost": (
            _pipe(XGBClassifier(tree_method="hist", n_jobs=1, random_state=g, eval_metric="mlogloss")),
            {"clf__n_estimators": [150, 300, 600], "clf__learning_rate": loguniform(0.02, 0.3),
             "clf__max_depth": [3, 4, 6], "clf__subsample": [0.7, 0.85, 1.0],
             "clf__colsample_bytree": [0.5, 0.7, 0.9], "clf__min_child_weight": [1, 3, 5, 10],
             "clf__reg_lambda": loguniform(0.1, 10)}, 12),
        "LightGBM": (
            _pipe(LGBMClassifier(class_weight="balanced", random_state=g, n_jobs=1, verbose=-1)),
            {"clf__n_estimators": [150, 300, 600], "clf__learning_rate": loguniform(0.02, 0.3),
             "clf__num_leaves": [15, 31, 63], "clf__min_child_samples": [10, 20, 40],
             "clf__subsample": [0.7, 0.85, 1.0], "clf__subsample_freq": [1],
             "clf__colsample_bytree": [0.5, 0.7, 0.9], "clf__reg_lambda": loguniform(0.01, 10)}, 12),
        "Réseau de neurones (MLP)": (
            _pipe(MLPClassifier(max_iter=800, early_stopping=True, random_state=g), True),
            {"clf__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
             "clf__alpha": loguniform(1e-5, 1e-1), "clf__learning_rate_init": loguniform(1e-4, 1e-2)}, 10),
    }


def _params_fit(nom, y):
    """Pondération des classes pour XGBoost (les autres modèles l'intègrent via class_weight)."""
    if nom == "XGBoost":
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def _avec_probas(modele):
    """Copie du modèle capable de produire des probabilités (SVM : calibration de Platt)."""
    m = clone(modele)
    if isinstance(m.named_steps["clf"], SVC):
        m.set_params(clf__probability=True)
    return m


# Recherche et évaluation

def comparer_modeles(train: pd.DataFrame, test: pd.DataFrame, cible: str, variables=VARIABLES,
                     graine=config.GRAINE, noms=None, n_jobs=4, verbeux=True, n_iter_max=None):
    """Recherche d'hyperparamètres en CV groupée, puis une seule évaluation sur le test."""
    verifier_absence_de_fuite(variables)
    enc = LabelEncoder().fit(pd.concat([train[cible], test[cible]]))
    Xtr, ytr, gtr = train[variables], enc.transform(train[cible]), train.jour.to_numpy()
    Xte, yte = test[variables], enc.transform(test[cible])
    cv = cv_groupee(graine)
    resultats, modeles = [], {}
    for nom, (pipe, espace, n_iter) in modeles_candidats(graine).items():
        if noms is not None and nom not in noms:
            continue
        t0 = time.time()
        fit_params = _params_fit(nom, ytr)
        if n_iter_max is not None:
            n_iter = min(n_iter, n_iter_max)
        if espace:
            rech = RandomizedSearchCV(pipe, espace, n_iter=n_iter, scoring="f1_macro", cv=cv, n_jobs=n_jobs,
                                      random_state=graine, refit=False)
            rech.fit(Xtr, ytr, groups=gtr, **fit_params)
            i = rech.best_index_
            cv_moy, cv_et = rech.cv_results_["mean_test_score"][i], rech.cv_results_["std_test_score"][i]
            meilleur = clone(pipe).set_params(**rech.best_params_)
            params = rech.best_params_
        else:
            from sklearn.model_selection import cross_val_score
            s = cross_val_score(pipe, Xtr, ytr, groups=gtr, cv=cv, scoring="f1_macro")
            cv_moy, cv_et, meilleur, params = s.mean(), s.std(), clone(pipe), {}
        final = _avec_probas(meilleur)
        final.fit(Xtr, ytr, **fit_params)
        p_tr, p_te = final.predict(Xtr), final.predict(Xte)
        proba = final.predict_proba(Xte)
        ligne = dict(
            modele=nom, f1_cv=cv_moy, f1_cv_ecart_type=cv_et,
            f1_apprentissage=f1_score(ytr, p_tr, average="macro"),
            f1_test=f1_score(yte, p_te, average="macro"),
            exactitude_test=accuracy_score(yte, p_te),
            exactitude_equilibree_test=balanced_accuracy_score(yte, p_te),
            log_loss_test=log_loss(yte, proba, labels=np.arange(len(enc.classes_))),
            auc_test=_auc(yte, proba),
            duree_s=time.time() - t0, hyperparametres=_formater(params),
        )
        resultats.append(ligne)
        modeles[nom] = final
        if verbeux:
            print(f"{nom:32s} CV {cv_moy:.3f} ± {cv_et:.3f} | test {ligne['f1_test']:.3f} "
                  f"({ligne['duree_s']:.0f} s)")
    res = pd.DataFrame(resultats)
    res["ecart_apprentissage_cv"] = res.f1_apprentissage - res.f1_cv
    return res, modeles, enc


def _auc(y, proba):
    try:
        if proba.shape[1] == 2:
            return roc_auc_score(y, proba[:, 1])
        return roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=np.arange(proba.shape[1]))
    except ValueError:
        return np.nan


def _formater(params):
    return ", ".join(f"{k.replace('clf__', '')}={v:.3g}" if isinstance(v, float) else f"{k.replace('clf__', '')}={v}"
                     for k, v in params.items())


def choisir(resultats: pd.DataFrame) -> str:
    """Modèle retenu : meilleur score de validation croisée (le test n'intervient pas)."""
    return resultats.sort_values("f1_cv", ascending=False).modele.iloc[0]


def bootstrap_groupe(y, pred, groupes, fonction, n=1000, graine=config.GRAINE):
    """Intervalle de confiance à 95 % par rééchantillonnage des jours du test."""
    rng = np.random.default_rng(graine)
    uniques = np.unique(groupes)
    index = {gr: np.flatnonzero(groupes == gr) for gr in uniques}
    valeurs = []
    for _ in range(n):
        tirage = rng.choice(uniques, size=len(uniques), replace=True)
        idx = np.concatenate([index[gr] for gr in tirage])
        valeurs.append(fonction(y[idx], pred[idx]))
    return np.percentile(valeurs, [2.5, 97.5])


def f1_macro(y, p):
    return f1_score(y, p, average="macro")


def difference_appariee(y, p1, p2, groupes, n=1000, graine=config.GRAINE):
    """Bootstrap apparié de la différence de F1 macro entre deux modèles."""
    rng = np.random.default_rng(graine)
    uniques = np.unique(groupes)
    index = {gr: np.flatnonzero(groupes == gr) for gr in uniques}
    diffs = []
    for _ in range(n):
        idx = np.concatenate([index[gr] for gr in rng.choice(uniques, size=len(uniques))])
        diffs.append(f1_macro(y[idx], p1[idx]) - f1_macro(y[idx], p2[idx]))
    diffs = np.asarray(diffs)
    return dict(diff=float(f1_macro(y, p1) - f1_macro(y, p2)), ic_bas=float(np.percentile(diffs, 2.5)),
                ic_haut=float(np.percentile(diffs, 97.5)), p_inferieur_ou_egal=float(np.mean(diffs <= 0)))


# Diagnostics de sur-apprentissage

def courbes_apprentissage(modele, X, y, groupes, graine=config.GRAINE, nom=None):
    tailles, sc_tr, sc_va = learning_curve(
        modele, X, y, groups=groupes, cv=cv_groupee(graine), scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 7), n_jobs=4, shuffle=True, random_state=graine,
        params=_params_fit(nom, y) or None)
    return tailles, sc_tr, sc_va


def courbe_validation(modele, X, y, groupes, param, valeurs, graine=config.GRAINE, nom=None):
    sc_tr, sc_va = validation_curve(modele, X, y, groups=groupes, param_name=param, param_range=valeurs,
                                    cv=cv_groupee(graine), scoring="f1_macro", n_jobs=4,
                                    params=_params_fit(nom, y) or None)
    return sc_tr, sc_va


def test_permutation(modele, X, y, groupes, n_permutations=50, graine=config.GRAINE):
    """Score réel contre scores obtenus après permutation aléatoire des étiquettes."""
    score, scores_perm, p = permutation_test_score(
        modele, X, y, groups=groupes, cv=cv_groupee(graine), scoring="f1_macro",
        n_permutations=n_permutations, n_jobs=4, random_state=graine)
    return score, scores_perm, p


def importance_par_permutation(modele, X, y, graine=config.GRAINE, n_repeats=10):
    r = permutation_importance(modele, X, y, scoring="f1_macro", n_repeats=n_repeats, random_state=graine, n_jobs=4)
    return (pd.DataFrame(dict(variable=X.columns, importance=r.importances_mean, ecart_type=r.importances_std))
            .sort_values("importance", ascending=False).reset_index(drop=True))


def shap_xgboost(modele_xgb, X):
    """Contributions SHAP exactes (TreeSHAP natif de XGBoost) : |SHAP| moyen par variable et par classe."""
    import xgboost as xgb
    imp = modele_xgb.named_steps["imputation"].transform(X)
    booster = modele_xgb.named_steps["clf"].get_booster()
    contrib = booster.predict(xgb.DMatrix(imp, feature_names=list(X.columns)), pred_contribs=True)
    if contrib.ndim == 2:
        contrib = contrib[:, None, :]
    return np.abs(contrib[:, :, :-1]).mean(axis=0), contrib[:, :, :-1]


def erreur_calibration(y, proba, n_bins=10):
    """Erreur de calibration attendue (ECE) sur la probabilité de la classe prédite."""
    conf = proba.max(axis=1)
    juste = proba.argmax(axis=1) == y
    bornes = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for a, b in zip(bornes[:-1], bornes[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            ece += m.mean() * abs(juste[m].mean() - conf[m].mean())
    return ece


def entrainer(modele, train, cible, enc, variables=VARIABLES, nom=None):
    m = clone(modele)
    y = enc.transform(train[cible])
    m.fit(train[variables], y, **_params_fit(nom, y))
    return m
