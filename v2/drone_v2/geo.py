"""Géographie du théâtre d'opérations : frontières, ligne de front, projections.

Sources et hypothèses :
* frontières : Natural Earth 1:50m (domaine public). Natural Earth rattache la
  Crimée à la Russie (vue de facto) ; la v2 la rattache à l'Ukraine
  (frontières internationalement reconnues) et la classe en territoire occupé ;
* ligne de front : tracé simplifié (une vingtaine de points) inspiré des cartes
  publiques 2025 (ISW, DeepStateMap). Précision de l'ordre de 10 à 20 km,
  suffisante pour une simulation, inutilisable pour tout usage opérationnel.
"""

from __future__ import annotations

from functools import cached_property

import geopandas as gpd
import numpy as np
import shapely
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, unary_union

from . import config

CRS_WGS84 = CRS.from_epsg(4326)
# Projection azimutale équidistante centrée sur l'Ukraine : coordonnées en mètres,
# déformation des distances inférieure à 0,5 % jusqu'à 1 000 km du centre.
CRS_LOCAL = CRS.from_proj4(
    "+proj=aeqd +lat_0=48.5 +lon_0=34.5 +datum=WGS84 +units=m +no_defs"
)
_VERS_LOCAL = Transformer.from_crs(CRS_WGS84, CRS_LOCAL, always_xy=True)
_VERS_WGS84 = Transformer.from_crs(CRS_LOCAL, CRS_WGS84, always_xy=True)

# Ligne de contact simplifiée (lon, lat), du nord-est (frontière russe près de
# Vovtchansk) au sud-ouest (estuaire du Dniepr, flèche de Kinbourn).
LIGNE_FRONT_2025 = [
    (36.95, 50.45),  # côté russe de la frontière, point de fermeture
    (36.95, 50.28),  # Vovtchansk
    (37.30, 50.05),
    (37.62, 49.80),  # secteur de Koupiansk
    (37.70, 49.55),
    (37.75, 49.30),  # Borova
    (37.90, 49.05),  # Lyman
    (38.05, 48.90),  # Siversk
    (37.92, 48.65),  # Tchassiv Yar
    (37.80, 48.42),  # Toretsk
    (37.45, 48.33),
    (37.18, 48.27),  # Pokrovsk
    (36.95, 48.05),
    (36.80, 47.85),  # Velyka Novosilka
    (36.30, 47.70),  # Houliaïpole
    (35.85, 47.52),  # Orikhiv
    (35.45, 47.52),  # Kamianske
    (35.05, 47.52),  # réservoir de Kakhovka
    (34.50, 47.52),  # entre Nikopol et Enerhodar
    (34.00, 47.25),
    (33.40, 46.80),  # Nova Kakhovka
    (32.65, 46.62),  # Kherson (Dniepr)
    (32.20, 46.50),  # estuaire
    (31.50, 46.50),  # flèche de Kinbourn
    (31.50, 45.80),  # mer Noire
]


def vers_xy(lon, lat):
    """Lon/lat (degrés) vers coordonnées locales (m)."""
    return _VERS_LOCAL.transform(np.asarray(lon, float), np.asarray(lat, float))


def vers_lonlat(x, y):
    """Coordonnées locales (m) vers lon/lat (degrés)."""
    return _VERS_WGS84.transform(np.asarray(x, float), np.asarray(y, float))


def geom_vers_xy(geom):
    return shapely.transform(geom, lambda x, y: _VERS_LOCAL.transform(x, y), interleaved=False)


def _plus_grande_ligne(geom) -> LineString:
    geom = linemerge(geom) if geom.geom_type == "MultiLineString" else geom
    if geom.geom_type == "LineString":
        return geom
    return max(geom.geoms, key=lambda g: g.length)


class Theatre:
    """Géométries du théâtre en lon/lat (affichage) et en mètres (calculs)."""

    def __init__(self, fichier_pays=config.FICHIER_PAYS):
        pays = gpd.read_file(fichier_pays)
        ukr = pays.loc[pays.ADMIN == "Ukraine", "geometry"].iloc[0]
        rus = pays.loc[pays.ADMIN == "Russia", "geometry"].iloc[0]
        crimee = [p for p in rus.geoms if p.contains(Point(34.1, 45.0))][0]

        self.crimee = crimee
        self.ukraine = unary_union([ukr, crimee]).buffer(0)
        self.russie = rus.difference(crimee).buffer(0)
        self.bielorussie = pays.loc[pays.ADMIN == "Belarus", "geometry"].iloc[0]

        fermeture = list(LIGNE_FRONT_2025) + [(31.5, 43.5), (42.0, 43.5), (42.0, 50.45)]
        zone_est = Polygon(fermeture).buffer(0)
        self.occupe = self.ukraine.intersection(zone_est).buffer(0)
        self.ua_controle = self.ukraine.difference(self.occupe).buffer(0)
        self.cote_russe = unary_union([self.russie, self.occupe]).buffer(0)

        # Ligne de contact = limite entre zone contrôlée par l'Ukraine et zone
        # contrôlée par la Russie (front + frontière russo-ukrainienne au nord).
        contact = self.ua_controle.boundary.intersection(self.cote_russe.buffer(0.03))
        self.ligne_contact = _plus_grande_ligne(contact)
        self.ligne_front = LineString(LIGNE_FRONT_2025[1:-1])

        # Ligne de séparation prolongée : frontière Biélorussie-Russie au nord, ligne
        # de contact, puis estuaire du Dniepr et mer Noire au sud. Elle sépare
        # l'espace en deux côtés et rend la distance signée continue (pas de saut
        # au passage d'une côte ou de la frontière biélorusse).
        by_ru = _plus_grande_ligne(self.bielorussie.boundary.intersection(self.russie.buffer(0.03)))
        coords_by = list(by_ru.coords)
        depart_contact = Point(self.ligne_contact.coords[0])
        if Point(coords_by[0]).distance(depart_contact) < Point(coords_by[-1]).distance(depart_contact):
            coords_by = coords_by[::-1]
        coords_contact = list(self.ligne_contact.coords)
        mer = [(31.50, 46.50), (31.00, 45.60), (30.00, 43.50)]
        self.ligne_separation = LineString(coords_by + coords_contact + mer)
        nord, sud = coords_by[0], mer[-1]
        self.zone_cote_russe = Polygon(
            coords_by + coords_contact + mer
            + [(sud[0], 38.0), (65.0, 38.0), (65.0, 62.0), (nord[0], 62.0)]
        ).buffer(0)

        pays = pays.copy()
        pays.loc[pays.ADMIN == "Ukraine", "geometry"] = self.ukraine
        pays.loc[pays.ADMIN == "Russia", "geometry"] = self.russie
        self.pays = pays

    # Versions projetées (m), préparées pour les tests vectorisés.
    @cached_property
    def ua_controle_xy(self):
        g = geom_vers_xy(self.ua_controle)
        shapely.prepare(g)
        return g

    @cached_property
    def cote_russe_xy(self):
        g = geom_vers_xy(self.cote_russe)
        shapely.prepare(g)
        return g

    @cached_property
    def ukraine_xy(self):
        g = geom_vers_xy(self.ukraine)
        shapely.prepare(g)
        return g

    @cached_property
    def ligne_contact_xy(self) -> LineString:
        # Simplification à 500 m : accélère les calculs de distance sans effet notable.
        return geom_vers_xy(self.ligne_contact).simplify(500)

    @cached_property
    def ligne_separation_xy(self) -> LineString:
        return geom_vers_xy(self.ligne_separation).simplify(500)

    @cached_property
    def zone_cote_russe_xy(self):
        g = geom_vers_xy(self.zone_cote_russe)
        shapely.prepare(g)
        return g

    @cached_property
    def bielorussie_xy(self):
        g = geom_vers_xy(self.bielorussie)
        shapely.prepare(g)
        return g

    def est_cote_ua(self, x, y) -> np.ndarray:
        return shapely.contains_xy(self.ua_controle_xy, np.asarray(x), np.asarray(y))

    def est_cote_russe(self, x, y) -> np.ndarray:
        return shapely.contains_xy(self.cote_russe_xy, np.asarray(x), np.asarray(y))

    def distance_signee_km(self, x, y) -> np.ndarray:
        """Distance à la ligne de séparation (km), négative du côté russe, positive de l'autre."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        d = shapely.distance(shapely.points(x, y), self.ligne_separation_xy) / 1000.0
        cote_russe = shapely.contains_xy(self.zone_cote_russe_xy, x, y)
        return np.where(cote_russe, -d, d)

    def point_contact(self, s_m: float):
        """Point de la ligne de contact à l'abscisse curviligne s (m) et normale vers l'Ukraine."""
        ligne = self.ligne_contact_xy
        s = float(np.clip(s_m, 1.0, ligne.length - 1.0))
        p0 = ligne.interpolate(s - 500.0)
        p1 = ligne.interpolate(s + 500.0)
        p = ligne.interpolate(s)
        tx, ty = p1.x - p0.x, p1.y - p0.y
        n = np.hypot(tx, ty) or 1.0
        nx, ny = -ty / n, tx / n
        # Oriente la normale vers le territoire contrôlé par l'Ukraine.
        if not self.est_cote_ua(p.x + 3000 * nx, p.y + 3000 * ny):
            nx, ny = -nx, -ny
        return np.array([p.x, p.y]), np.array([nx, ny])
