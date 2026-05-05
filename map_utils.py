"""Shared utilities for cycling analysis maps."""

import folium
import branca.colormap as cm
import geopandas as gpd
import pandas as pd
import numpy as np


COLOR_OPTIONS = [
    'Blues', 'Reds', 'Greens', 'Oranges', 'Purples',
    'YlOrRd', 'YlGnBu', 'BuPu', 'RdPu', 'Greys', 'OrRd',
]

_CICLO_EXTRA_COLS = ['name', 'highway', 'surface', 'maxspeed', 'lanes',
                     'lit', 'oneway', 'segregated', 'smoothness', 'bridge']

_TOOLTIP_ALIASES = {
    'edge_uid': 'Edge UID',
    'osm_reference_id': 'OSM Ref ID',
    'tipo_ciclovia': 'Tipo ciclovia',
    'name': 'Nome da via',
    'highway': 'Tipo de via',
    'surface': 'Superficie',
    'maxspeed': 'Vel. maxima (km/h)',
    'lanes': 'Faixas',
    'lit': 'Iluminacao',
    'oneway': 'Mao unica',
    'segregated': 'Segregada',
    'smoothness': 'Qualidade piso',
    'bridge': 'Ponte',
    'avg_speed_kmh': 'Vel. media ciclistas (km/h)',
    'priority_score': 'Score de prioridade',
    'length_m': 'Extensao (m)',
    'risk_score': 'Risk Score',
    'connectivity_score': 'Connectivity Score',
    'components_connected': 'Componentes conectados',
    'group_pct': '% do grupo',
    'group_count': 'Contagem do grupo',
    'total_people': 'Total de pessoas',
}

_TOOLTIP_INFO_COLS = ['name', 'highway', 'surface', 'maxspeed', 'lanes',
                      'lit', 'oneway', 'segregated', 'smoothness', 'bridge',
                      'avg_speed_kmh']


def add_layers(m):
    """Adds base tile layers to a Folium map.

    Parameters
    ----------
    m : folium.Map
        Map instance.

    Returns
    -------
    folium.Map
        Map with tile layers added.
    """
    folium.TileLayer(tiles="OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
    folium.TileLayer(tiles="CartoDB positron", name="CartoDB positron", control=True).add_to(m)
    folium.TileLayer(tiles="CartoDB dark_matter", name="CartoDB dark_matter", control=True).add_to(m)
    return m


def build_colormap(series, cmap_name, caption):
    """Creates a LinearColormap from a numeric Series.

    Parameters
    ----------
    series : pd.Series
        Numeric values to define vmin/vmax.
    cmap_name : str
        Matplotlib colormap name.
    caption : str
        Colormap caption.

    Returns
    -------
    branca.colormap.LinearColormap
        Configured colormap.
    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    vmin = series.min()
    vmax = series.max()
    if vmin == vmax:
        vmax = vmin + 1
    colors = [
        '#{:02x}{:02x}{:02x}'.format(*[int(c * 255) for c in cmap(i)[:3]])
        for i in np.linspace(0.2, 1.0, 6)
    ]
    return cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax, caption=caption)


def split_outliers(gdf_layer, col, sort_order='desc', iqr_factor=1.5):
    """Splits a GeoDataFrame into normal and outlier rows using IQR.

    Parameters
    ----------
    gdf_layer : gpd.GeoDataFrame
        GeoDataFrame with a numeric column.
    col : str
        Column to detect outliers on.
    sort_order : str
        'desc' = upper outliers, 'asc' = lower outliers.
    iqr_factor : float
        IQR multiplier.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, float]
        (normal rows, outliers, threshold).
    """
    q1 = gdf_layer[col].quantile(0.25)
    q3 = gdf_layer[col].quantile(0.75)
    iqr = q3 - q1
    if sort_order == 'desc':
        threshold = q3 + iqr_factor * iqr
        mask = gdf_layer[col] > threshold
    else:
        threshold = q1 - iqr_factor * iqr
        mask = gdf_layer[col] < threshold
    return gdf_layer[~mask].copy(), gdf_layer[mask].copy(), threshold


def add_batch_geojson(m, gdf_layer, color_col, colormap, layer_name,
                      extra_tooltip_cols=None, show=False):
    """Adds a GeoJson batch layer to the map with an enriched tooltip.

    Parameters
    ----------
    m : folium.Map
        Folium map.
    gdf_layer : gpd.GeoDataFrame
        GeoDataFrame with geometry and data columns.
    color_col : str
        Column used for colouring.
    colormap : branca.colormap.LinearColormap
        Colormap for gradient colouring.
    layer_name : str
        Layer name in the layer control.
    extra_tooltip_cols : list[str] or None
        Additional columns to include in the tooltip beyond the defaults.
    show : bool
        Whether the layer starts visible. Default: False.
    """
    if len(gdf_layer) == 0:
        return

    # Tooltip: base + info da via + extras
    tooltip_candidates = (['edge_uid', 'osm_reference_id', 'tipo_ciclovia', color_col]
                          + _TOOLTIP_INFO_COLS
                          + (extra_tooltip_cols or []))
    # Remover duplicatas mantendo ordem
    seen = set()
    tooltip_cols = []
    for c in tooltip_candidates:
        if c not in seen and c in gdf_layer.columns:
            tooltip_cols.append(c)
            seen.add(c)

    keep = list(set(tooltip_cols + [color_col, 'geometry']))
    layer_data = gdf_layer[keep].copy()
    layer_data[color_col] = layer_data[color_col].fillna(0).astype(float)

    for c in _TOOLTIP_INFO_COLS:
        if c in layer_data.columns:
            layer_data[c] = layer_data[c].fillna('-').astype(str).replace('nan', '-')

    layer_data['_color'] = layer_data[color_col].apply(
        lambda v: '#cccccc' if pd.isna(v) or v == 0 else colormap(v)
    )

    fg = folium.FeatureGroup(name=layer_name, show=show)
    fields = [c for c in tooltip_cols if c in layer_data.columns]
    aliases = [_TOOLTIP_ALIASES.get(c, c) for c in fields]

    geojson = folium.GeoJson(
        layer_data.__geo_interface__,
        style_function=lambda feature: {
            'color': feature['properties'].get('_color', '#cccccc'),
            'weight': 3,
            'opacity': 0.8,
        },
        tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, sticky=True),
    )
    geojson.add_to(fg)
    fg.add_to(m)


def merge_edges_with_infra(edge_count, gdf, ciclovias):
    """Merge edge_count + gdf (geometry) + ciclovias (infra + extras).

    Brings extra columns from ciclovias (name, highway, surface, etc.) and
    computes avg_speed_kmh from edge_count.

    Parameters
    ----------
    edge_count : pd.DataFrame
        DataFrame with edge_uid, osm_reference_id and trip count columns.
    gdf : gpd.GeoDataFrame
        GeoDataFrame with edge_uid and geometry.
    ciclovias : pd.DataFrame
        DataFrame with id, tem_infra, tipo_ciclovia and extra columns.

    Returns
    -------
    gpd.GeoDataFrame
        Fully merged GeoDataFrame.
    """
    merged = gdf[['edge_uid', 'geometry']].merge(edge_count, on='edge_uid', how='inner')

    speed_cols = ['forward_average_speed_meters_per_second',
                  'reverse_average_speed_meters_per_second']
    if all(c in merged.columns for c in speed_cols):
        merged['avg_speed_kmh'] = (merged[speed_cols].max(axis=1) * 3.6).round(1)

    ciclo_cols = ['id', 'tem_infra', 'tipo_ciclovia']
    for col in _CICLO_EXTRA_COLS:
        if col in ciclovias.columns:
            ciclo_cols.append(col)
    ciclo_lookup = ciclovias[ciclo_cols].copy()
    ciclo_lookup['id'] = ciclo_lookup['id'].astype(str)
    merged['osm_reference_id'] = merged['osm_reference_id'].astype(str)
    merged = merged.merge(ciclo_lookup, left_on='osm_reference_id', right_on='id', how='left')
    merged['tem_infra'] = merged['tem_infra'].fillna(False)
    merged['tipo_ciclovia'] = merged['tipo_ciclovia'].fillna('sem_infra')

    return gpd.GeoDataFrame(merged, geometry='geometry')


def create_base_map(map_center=(-23.55, -46.63), zoom_start=12):
    """Creates a base map with tile layers.

    Parameters
    ----------
    map_center : tuple
        Map center coordinates.
    zoom_start : int
        Initial zoom level.

    Returns
    -------
    folium.Map
        Map with tile layers added.
    """
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=None)
    add_layers(m)
    return m