"""Cartes interactives de la v2 (Folium / Leaflet) et projection PCA 3D (Plotly).

Les fichiers HTML produits s'ouvrent dans un navigateur ; le fond de carte
(Esri gris clair ou OpenStreetMap) et les bibliothèques JavaScript sont chargés en ligne.
"""

from __future__ import annotations

import json

import folium
import numpy as np
import pandas as pd
from branca.element import Element
from folium import plugins
from shapely.geometry import mapping

from . import config
from .viz import AXE, COULEUR_ORIGINE, COULEUR_ROLE, ENCRE, ENCRE_2, SURFACE, ACCENT, couleurs_groupes



def carte_de_base(theatre, centre=(48.6, 33.0), zoom=6):
    import xyzservices.providers as fournisseurs

    # Fonds de carte sans clé d'API : Esri gris clair par défaut, OpenStreetMap en option.
    m = folium.Map(location=centre, zoom_start=zoom, tiles=None, control_scale=True, prefer_canvas=True)
    folium.TileLayer(fournisseurs.Esri.WorldGrayCanvas, name="Fond gris clair (Esri)").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(m)
    # Frontières intégrées au fichier : la carte reste lisible si le fond de tuiles n'est pas accessible.
    frontieres = theatre.pays[["ADMIN", "geometry"]].copy()
    frontieres["geometry"] = frontieres.geometry.simplify(0.01)
    folium.GeoJson(frontieres.to_json(), name="Frontières (Natural Earth)", control=False,
                   style_function=lambda _: dict(color=ENCRE_2, weight=0.7, opacity=0.6, fill=False)).add_to(m)
    folium.GeoJson(mapping(theatre.occupe), name="Territoire occupé (approx. 2025)",
                   style_function=lambda _: dict(fillColor=COULEUR_ORIGINE["Russie"], fillOpacity=0.13,
                                                 color=COULEUR_ORIGINE["Russie"], weight=0),
                   tooltip="Territoire occupé (tracé approximatif 2025)").add_to(m)
    folium.GeoJson(mapping(theatre.ligne_contact), name="Ligne de contact",
                   style_function=lambda _: dict(color=ENCRE_2, weight=1.6, opacity=0.9),
                   tooltip="Ligne de contact simplifiée (front + frontière russo-ukrainienne)").add_to(m)
    return m


def ajouter_legende(m, titre, elements, cote="left"):
    """Légende fixe en bas de carte (à gauche par défaut) : liste de (couleur, libellé, forme)."""
    lignes = []
    for couleur, libelle, forme in elements:
        if forme == "ligne":
            symbole = f'<span style="display:inline-block;width:18px;height:3px;background:{couleur};' \
                      f'border-radius:2px;margin-right:8px;vertical-align:middle"></span>'
        else:
            symbole = f'<span style="display:inline-block;width:10px;height:10px;background:{couleur};' \
                      f'border-radius:50%;margin:0 12px 0 4px;vertical-align:middle"></span>'
        lignes.append(f'<div style="margin:3px 0">{symbole}{libelle}</div>')
    html = f"""
    <div style="position:fixed;bottom:28px;{cote}:12px;z-index:9999;background:{SURFACE};
                border:1px solid rgba(11,11,11,0.10);border-radius:8px;padding:10px 14px;
                font:12px system-ui,-apple-system,'Segoe UI',sans-serif;color:{ENCRE_2};
                box-shadow:0 1px 4px rgba(0,0,0,0.08);max-width:300px">
      <div style="font-weight:600;color:{ENCRE};margin-bottom:6px">{titre}</div>
      {''.join(lignes)}
    </div>"""
    m.get_root().html.add_child(Element(html))


def ajouter_titre(m, titre, sous_titre=""):
    html = f"""
    <div style="position:fixed;top:12px;left:56px;z-index:9999;background:{SURFACE};
                border:1px solid rgba(11,11,11,0.10);border-radius:8px;padding:8px 14px;
                font:13px system-ui,-apple-system,'Segoe UI',sans-serif;color:{ENCRE};
                box-shadow:0 1px 4px rgba(0,0,0,0.08);max-width:560px">
      <div style="font-weight:600">{titre}</div>
      <div style="color:{ENCRE_2};font-size:12px;margin-top:2px">{sous_titre}</div>
    </div>"""
    m.get_root().html.add_child(Element(html))


def _coords(g, pas=1):
    return [[round(la, 4), round(lo, 4)] for lo, la in zip(g.lon.to_numpy()[::pas], g.lat.to_numpy()[::pas])]


def carte_trajectoires(theatre, echantillon, pistes, sites_ru, zones_ua, radars, fichier):
    m = carte_de_base(theatre)
    info = pistes.set_index("piste_id")
    groupes = {role: folium.FeatureGroup(name=f"Pistes : {role}", show=True) for role in COULEUR_ROLE}
    for piste_id, g in echantillon.groupby("piste_id", sort=False):
        p = info.loc[piste_id]
        duree = (p.t_fin - p.t_debut) / 60
        texte = (f"<b>{p.modele}</b><br>Rôle : {p.role}<br>Origine : {p.origine}<br>"
                 f"Durée observée : {duree:.0f} min<br>Fin du vol : {p.fin_vol}<br>Piste : {piste_id}")
        folium.PolyLine(_coords(g), color=COULEUR_ORIGINE[p.origine], weight=1.6, opacity=0.65,
                        tooltip=f"{p.modele} ({p.origine})", popup=folium.Popup(texte, max_width=260)
                        ).add_to(groupes[p.role])
    for fg in groupes.values():
        fg.add_to(m)
    fg_sites = folium.FeatureGroup(name="Sites de lancement russes (OSINT)", show=True)
    for _, r in sites_ru.iterrows():
        folium.CircleMarker([r.lat, r.lon], radius=7, weight=2, color=SURFACE, fill=True,
                            fill_color=COULEUR_ORIGINE["Russie"], fill_opacity=1, tooltip=r.nom,
                            popup=folium.Popup(f"<b>{r.nom}</b><br><a href='{r.source}' target='_blank'>"
                                               f"source</a>", max_width=250)).add_to(fg_sites)
    fg_sites.add_to(m)
    fg_zones = folium.FeatureGroup(name="Zones de lancement ukrainiennes (fictives)", show=True)
    for _, z in zones_ua.iterrows():
        folium.Circle([z.lat, z.lon], radius=z.rayon_km * 1000, color=COULEUR_ORIGINE["Ukraine"], weight=1.2,
                      fill=True, fill_opacity=0.12, tooltip=z.nom).add_to(fg_zones)
    fg_zones.add_to(m)
    fg_radar = folium.FeatureGroup(name="Radars (positions fictives)", show=False)
    for _, r in radars.iterrows():
        folium.CircleMarker([r.lat, r.lon], radius=3, color=ENCRE_2, weight=1, fill=True, fill_opacity=0.8,
                            tooltip=f"{r.site} (fictif)").add_to(fg_radar)
    fg_radar.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    ajouter_titre(m, "Drones v2 : pistes radar simulées",
                  f"{echantillon.piste_id.nunique()} pistes tirées au hasard, cliquer une piste pour le détail")
    ajouter_legende(m, "Origine du drone", [(COULEUR_ORIGINE["Russie"], "Russie", "ligne"),
                                            (COULEUR_ORIGINE["Ukraine"], "Ukraine", "ligne"),
                                            (COULEUR_ORIGINE["Russie"], "Site de lancement russe", "point")])
    m.save(str(fichier))
    return fichier


def carte_predictions(theatre, plots_test, pred: pd.DataFrame, fichier, n_correct=500, graine=config.GRAINE, pas=6):
    """Pistes du test : bien classées (couleur du rôle prédit) et erreurs (rouge, épaisses)."""
    m = carte_de_base(theatre)
    rng = np.random.default_rng(graine)
    pred = pred.set_index("piste_id")
    correct = pred.index[pred.role_reel == pred.role_predit]
    erreurs = pred.index[pred.role_reel != pred.role_predit]
    choisis = set(rng.choice(correct, size=min(n_correct, len(correct)), replace=False)) | set(erreurs)
    fg_ok = folium.FeatureGroup(name=f"Bien classées ({min(n_correct, len(correct))} sur {len(correct)})")
    fg_err = folium.FeatureGroup(name=f"Mal classées ({len(erreurs)})")
    for piste_id, g in plots_test[plots_test.piste_id.isin(choisis)].groupby("piste_id", sort=False):
        p = pred.loc[piste_id]
        texte = (f"<b>{p.modele}</b> ({p.origine})<br>Rôle réel : {p.role_reel}<br>Rôle prédit : "
                 f"<b>{p.role_predit}</b> (confiance {p.confiance:.0%})<br>Piste : {piste_id}")
        if p.role_reel == p.role_predit:
            folium.PolyLine(_coords(g, pas), color=COULEUR_ROLE[p.role_predit], weight=1.4, opacity=0.6,
                            tooltip=f"{p.role_predit} (correct)", popup=folium.Popup(texte, max_width=280)
                            ).add_to(fg_ok)
        else:
            folium.PolyLine(_coords(g, max(pas // 2, 1)), color=ACCENT, weight=3.5, opacity=0.95,
                            tooltip=f"Erreur : {p.role_reel} prédit {p.role_predit}",
                            popup=folium.Popup(texte, max_width=280)).add_to(fg_err)
    fg_ok.add_to(m)
    fg_err.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    exactitude = (pred.role_reel == pred.role_predit).mean()
    ajouter_titre(m, "Classification des rôles sur le jeu de test (30 derniers jours)",
                  f"Exactitude {exactitude:.1%} sur {len(pred)} pistes jamais vues à l'apprentissage")
    ajouter_legende(m, "Rôle prédit", [(c, r, "ligne") for r, c in COULEUR_ROLE.items()]
                    + [(ACCENT, "Erreur de classification", "ligne")])
    m.save(str(fichier))
    return fichier


def carte_animation_raid(theatre, plots_raid, pistes, fichier, pas_s=60):
    """Animation temporelle d'un raid : ce que voit le radar, minute par minute."""
    info = pistes.set_index("piste_id")
    t0 = plots_raid.t.min()
    base = pd.Timestamp(config.DATE_DEBUT)
    features = []
    for piste_id, g in plots_raid.groupby("piste_id", sort=False):
        g = g.iloc[:: max(int(pas_s // config.DT_RADAR_S), 1)]
        if len(g) < 2:
            continue
        p = info.loc[piste_id]
        temps = [(base + pd.Timedelta(seconds=float(t))).isoformat() for t in g.t]
        features.append(dict(
            type="Feature",
            geometry=dict(type="LineString", coordinates=[[round(lo, 4), round(la, 4)] for lo, la in zip(g.lon, g.lat)]),
            properties=dict(times=temps, popup=f"{p.modele} ({p.role})",
                            style=dict(color=COULEUR_ROLE[p.role], weight=2.5, opacity=0.9),
                            icon="circle",
                            iconstyle=dict(radius=4, fillColor=COULEUR_ROLE[p.role], fillOpacity=1,
                                           color=SURFACE, weight=1.5)),
        ))
    m = carte_de_base(theatre)
    plugins.TimestampedGeoJson(dict(type="FeatureCollection", features=features), period="PT1M",
                               duration="PT20M", add_last_point=True, auto_play=False, loop=False,
                               max_speed=30, loop_button=True, time_slider_drag_update=True,
                               date_options="YYYY-MM-DD HH:mm").add_to(m)
    jour = int(plots_raid.t.min() // 86400)
    ajouter_titre(m, f"Raid simulé du jour {jour} : pistes radar minute par minute",
                  f"{len(features)} pistes ; traînée de 20 min ; utiliser le curseur temporel en bas")
    ajouter_legende(m, "Rôle réel", [(COULEUR_ROLE[r], r, "ligne") for r in ("Frappe longue portée", "Leurre")],
                    cote="right")
    m.save(str(fichier))
    return fichier


def carte_densite(theatre, echantillon, pistes, fichier):
    m = carte_de_base(theatre)
    origine = pistes.set_index("piste_id").origine
    for camp, gradient in (("Russie", {0.2: "#fde7dd", 0.5: "#f3a07c", 0.8: "#eb6834", 1.0: "#b84a1f"}),
                           ("Ukraine", {0.2: "#cde2fb", 0.5: "#6da7ec", 0.8: "#2a78d6", 1.0: "#104281"})):
        pts = echantillon[echantillon.piste_id.map(origine) == camp]
        plugins.HeatMap(pts[["lat", "lon"]].round(3).to_numpy().tolist(), name=f"Densité des détections : {camp}",
                        radius=9, blur=12, min_opacity=0.25, gradient=gradient, show=(camp == "Russie")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    ajouter_titre(m, "Densité des détections radar par origine", "Échantillon de pistes, une couche par origine")
    m.save(str(fichier))
    return fichier


def carte_groupes(theatre, echantillon, groupes: pd.Series, profils: pd.DataFrame, fichier):
    m = carte_de_base(theatre)
    table = couleurs_groupes(groupes.to_numpy())
    couches = {}
    for g in sorted(table):
        if g < 0 or g not in set(groupes):
            continue
        pr = profils.loc[g]
        couches[g] = folium.FeatureGroup(
            name=f"Groupe {g} ({int(pr.effectif)} pistes, {pr.vitesse_med:.0f} km/h, {pr.alt_med:.0f} m)")
    for piste_id, gpl in echantillon[echantillon.piste_id.isin(groupes.index)].groupby("piste_id", sort=False):
        g = groupes[piste_id]
        if g in couches:
            folium.PolyLine(_coords(gpl), color=table[g], weight=1.6, opacity=0.65,
                            tooltip=f"Groupe {g}").add_to(couches[g])
    for fg in couches.values():
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    ajouter_titre(m, "Groupes de comportement découverts sans étiquette (K-Means)",
                  "Chaque couche regroupe des pistes au profil cinématique proche")
    ajouter_legende(m, "Groupe K-Means", [(table[g], f"Groupe {g}", "ligne") for g in couches])
    m.save(str(fichier))
    return fichier


def pca_3d(scores, df: pd.DataFrame, pca, fichier, n=3000, graine=config.GRAINE):
    import plotly.graph_objects as go
    rng = np.random.default_rng(graine)
    idx = np.sort(rng.choice(len(df), size=min(n, len(df)), replace=False))
    symboles = {"Frappe longue portée": "circle", "Leurre": "diamond", "Munition rôdeuse": "square",
                "Reconnaissance": "cross"}
    fig = go.Figure()
    for role, couleur in COULEUR_ROLE.items():
        m = (df.role.to_numpy()[idx] == role)
        sel = idx[m]
        fig.add_trace(go.Scatter3d(
            x=scores[sel, 0], y=scores[sel, 1], z=scores[sel, 2], mode="markers", name=role,
            marker=dict(size=2.6, color=couleur, symbol=symboles[role], opacity=0.75),
            customdata=np.column_stack([df.modele.to_numpy()[sel], df.origine.to_numpy()[sel],
                                        df.piste_id.to_numpy()[sel]]),
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>" + role +
                          "<br>%{customdata[2]}<extra></extra>"))
    v = pca.explained_variance_ratio_
    fig.update_layout(
        title=dict(text="PCA des pistes radar : trois premières composantes", x=0.02),
        scene=dict(xaxis_title=f"CP1 ({v[0]:.0%})", yaxis_title=f"CP2 ({v[1]:.0%})", zaxis_title=f"CP3 ({v[2]:.0%})"),
        legend=dict(itemsizing="constant", title="Rôle (clic pour masquer)"), paper_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=ENCRE_2), height=720,
        margin=dict(l=0, r=0, t=50, b=0))
    fig.write_html(str(fichier), include_plotlyjs="cdn")
    return fichier
