from collections import deque

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from tqdm import tqdm


def preprocess_geometry(geom: Polygon | MultiPolygon) -> Polygon:
    """
    Selects the largest polygon and includes all others that touch it.
    This avoids including remote territories like French Guiana (France).
    """
    if isinstance(geom, Polygon):
        return geom

    elif isinstance(geom, MultiPolygon):
        # Find the largest polygon (assumed to be mainland)
        parts = list(geom.geoms)
        mainland = max(parts, key=lambda p: p.area)

        # Include all polygons that touch the mainland
        touching_parts = [p for p in parts if p == mainland or p.touches(mainland)]

        return unary_union(touching_parts)

    return None


def build_border_map(world: gpd.GeoDataFrame) -> dict:
    border_map = {}

    for _, country in tqdm(list(world.iterrows())):
        iso_a3 = country["ISO_A3_EH"]
        geom = preprocess_geometry(country.geometry)

        # Skip if geometry is missing or invalid
        if not isinstance(geom, (Polygon, MultiPolygon)):
            continue

        neighbors = set()

        for idx2, other_country in world.iterrows():
            other_iso = other_country["ISO_A3_EH"]
            if iso_a3 == other_iso:
                continue

            other_geom = preprocess_geometry(other_country.geometry)

            if not isinstance(other_geom, (Polygon, MultiPolygon)):
                continue

            # Check for shared border
            if geom.touches(other_geom):
                neighbors.add(other_iso)

        border_map[iso_a3] = neighbors

    return border_map


def share_common_border(iso3_country1: str, iso3_country2: str, bm: dict) -> bool:
    iso3_country1 = iso3_country1.upper()
    iso3_country2 = iso3_country2.upper()
    return iso3_country2 in bm.get(iso3_country1, set())


def get_connected_country_set(seed_iso3: str, bm: dict, max_size: int = 10) -> set:
    visited: set[str] = set()
    queue = deque([seed_iso3])

    while queue and len(visited) < max_size:
        country = queue.popleft()
        if country in visited:
            continue
        visited.add(country)

        # Enqueue unvisited neighbors
        neighbors = bm.get(country, [])
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append(neighbor)

    return visited


def build_r_countries(
    df: pd.DataFrame, border_map: dict
) -> tuple[np.ndarray, list[str]]:
    iso_list = list(df.index)
    iso_to_idx = {iso: i for i, iso in enumerate(iso_list)}

    N = len(df)
    R = np.zeros((N, N), dtype=int)

    # Build full adjacency matrix
    for i, iso_i in enumerate(iso_list):
        neighbors = border_map.get(iso_i, set())
        for neighbor in neighbors:
            j = iso_to_idx.get(neighbor)
            if j is not None:
                R[i, j] = 1
                R[j, i] = 1  # symmetric

    return R, iso_list
