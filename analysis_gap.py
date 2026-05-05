"""Gap analysis: ranking of roads without infrastructure prioritised by demand x length."""

import pandas as pd
import geopandas as gpd
import numpy as np
from map_utils import (merge_edges_with_infra, create_base_map, build_colormap,
                       add_batch_geojson, split_outliers, COLOR_OPTIONS)
import folium


TRIP_COLUMNS = [
    'total_trip_count', 'forward_trip_count', 'reverse_trip_count',
    'forward_commute_trip_count', 'reverse_commute_trip_count',
    'forward_leisure_trip_count', 'reverse_leisure_trip_count',
    'ride_count', 'ebike_ride_count',
]


def gap_analysis_ranking(
    edge_count,
    gdf,
    ciclovias,
    trip_type='total_trip_count',
    top_n=50,
):
    """Generates a ranking of roads without infrastructure prioritised by demand.

    priority_score = trips * length_metres.

    Parameters
    ----------
    edge_count : pd.DataFrame
        DataFrame with edge_uid, osm_reference_id and trip count columns.
    gdf : gpd.GeoDataFrame
        GeoDataFrame with edge_uid and geometry.
    ciclovias : pd.DataFrame
        DataFrame with id, tem_infra, tipo_ciclovia and extra columns.
    trip_type : str
        Trip count column. Default: 'total_trip_count'.
    top_n : int
        Number of roads in the ranking. Default: 50.

    Returns
    -------
    gpd.GeoDataFrame
        Ranking sorted by priority_score descending.
    """
    merged = merge_edges_with_infra(edge_count, gdf, ciclovias)

    merged_proj = merged.set_crs(epsg=4326, allow_override=True).to_crs(epsg=31983)
    merged['length_m'] = merged_proj.geometry.length.round(1)

    sem_infra = merged[merged['tem_infra'] == False].copy()
    sem_infra['priority_score'] = (sem_infra[trip_type] * sem_infra['length_m']).round(0)
    sem_infra = sem_infra.sort_values('priority_score', ascending=False).head(top_n)

    print(f"Total de vias sem infra: {len(merged[merged['tem_infra'] == False])}")
    print(f"Top {top_n} vias por priority_score ({trip_type} * extensao_m)")

    return sem_infra.reset_index(drop=True)


def gap_analysis_map(
    edge_count,
    gdf,
    ciclovias,
    trip_type='total_trip_count',
    top_n=100,
    cmap_name='YlOrRd',
    cmap_outlier='Purples',
    map_center=(-23.55, -46.63),
    zoom_start=12,
    map_name='gap_analysis.html',
    normalize=False,
    sort_order='desc',
    iqr_factor=1.5,
):
    """Generates a map of the highest-priority roads without infrastructure, separating outliers.

    Parameters
    ----------
    edge_count : pd.DataFrame
        DataFrame with edge_uid, osm_reference_id and trip count columns.
    gdf : gpd.GeoDataFrame
        GeoDataFrame with edge_uid and geometry.
    ciclovias : pd.DataFrame
        DataFrame with id, tem_infra, tipo_ciclovia and extra columns.
    trip_type : str
        Trip count column. Default: 'total_trip_count'.
    top_n : int
        Number of roads. Default: 100.
    cmap_name : str
        Colormap for the normal range. Default: 'YlOrRd'.
    cmap_outlier : str
        Colormap for outliers. Default: 'Purples'.
    map_center : tuple
        Map center coordinates.
    zoom_start : int
        Initial zoom level.
    map_name : str
        Output file name. Default: 'gap_analysis.html'.
    normalize : bool
        Min-max normalise to [0,1]. Default: False.
    sort_order : str
        'desc' = more trips (upper outliers), 'asc' = fewer trips (lower outliers).
    iqr_factor : float
        IQR multiplier. Default: 1.5.

    Returns
    -------
    None
        Saves the map to map_name.
    """
    ranking = gap_analysis_ranking(edge_count, gdf, ciclovias, trip_type, top_n)

    if len(ranking) == 0:
        print("Nenhuma via sem infra encontrada.")
        return

    color_col = 'priority_score'
    if normalize:
        vmin, vmax = ranking[color_col].min(), ranking[color_col].max()
        ranking['priority_score_norm'] = (
            (ranking[color_col] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        )
        color_col = 'priority_score_norm'

    normais, outliers, threshold = split_outliers(ranking, 'priority_score', sort_order, iqr_factor)
    outlier_label = '(Upper)' if sort_order == 'desc' else 'inferiores'

    if normalize and len(normais) > 0:
        vmin, vmax = normais['priority_score'].min(), normais['priority_score'].max()
        normais['priority_score_norm'] = (
            (normais['priority_score'] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        )
    if normalize and len(outliers) > 0:
        vmin, vmax = outliers['priority_score'].min(), outliers['priority_score'].max()
        outliers['priority_score_norm'] = (
            (outliers['priority_score'] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        )

    m = create_base_map(map_center, zoom_start)

    extra = ['priority_score', 'length_m', trip_type]

    if len(normais) > 0:
        cmap = build_colormap(normais[color_col], cmap_name, f'Priority Score')
        add_batch_geojson(m, normais, color_col, cmap, 'Vias sem infra', extra)
        cmap.add_to(m)

    if len(outliers) > 0:
        cmap_o = build_colormap(outliers[color_col], cmap_outlier,
                                f'Priority Score - outliers {outlier_label}')
        add_batch_geojson(m, outliers, color_col, cmap_o,
                          f'Outliers {outlier_label}', extra)
        cmap_o.add_to(m)

    white_legend_css = """
    <style>
   .legend text {
        fill: white!important;
        font-weight: bold;
    }
   .legend line {
        stroke: white!important;
    }
    </style>
    """
    m.get_root().header.add_child(folium.Element(white_legend_css))

    folium.LayerControl(collapsed=False).add_to(m)

    print(f"Normais: {len(normais)}, Outliers {outlier_label}: {len(outliers)} (threshold: {threshold:.0f})")
    m.save(map_name)
    return m
