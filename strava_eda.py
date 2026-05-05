"""
Complete EDA - Strava Data
Exploratory Data Analysis of Strava Origin-Destination Data
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium import plugins
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Estatísticas Básicas
#------------------------------------------------------------


def basic_stats(data, origem_destino=2, save_plots=True):
    """
    Computes basic statistics for the dataset.

    Parameters:
    -----------
    data : DataFrame
        Strava dataset
    origem_destino : int (0, 1, or 2)
        0: origins only
        1: destinations only
        2: both (default)
    save_plots : bool
        If True, saves the generated plots
    """
    # Filtrar dados conforme origem/destino
    if origem_destino == 0:
        df = data[data["origins_or_destinations"] == "origins"].copy()
        tipo = "ORIGEM"
    elif origem_destino == 1:
        df = data[data["origins_or_destinations"] == "destinations"].copy()
        tipo = "DESTINO"
    else:
        df = data.copy()
        tipo = "AMBOS (ORIGEM + DESTINO)"
    
    print("="*80)
    print(f"ESTATÍSTICAS BÁSICAS - {tipo}")
    print("="*80)
    print(f"\nTotal de registros: {len(df):,}")
    print(f"Período: {df['date'].min()} até {df['date'].max()}")
    print(f"Número de hexágonos únicos: {df['hex_id'].nunique():,}")
    
    # Estatísticas descritivas numéricas
    print("\n" + "="*80)
    print("ESTATÍSTICAS DESCRITIVAS - VARIÁVEIS NUMÉRICAS")
    print("="*80)
    
    numeric_cols = [
        'count', 'commute_count', 'leisure_count',
        'morning_count', 'midday_count', 'evening_count', 'overnight_count',
        'weekday_count', 'weekend_count',
        'trip_distance_meters_p50', 'trip_duration_seconds_p50'
    ]
    
    # Verificar quais colunas existem
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    stats_df = df[available_cols].describe()
    
    # Adicionar estatísticas adicionais
    stats_adicional = pd.DataFrame({
        'mode': df[available_cols].mode().iloc[0] if len(df) > 0 else np.nan,
        'var': df[available_cols].var(),
        'skewness': df[available_cols].skew(),
        'kurtosis': df[available_cols].kurtosis()
    }).T
    
    stats_completo = pd.concat([stats_df, stats_adicional])
    print(stats_completo.round(2))
    
    # Tipos de atividade
    if 'activity_types' in df.columns:
        print("\n" + "="*80)
        print("DISTRIBUIÇÃO DE TIPOS DE ATIVIDADE")
        print("="*80)
        activity_counts = df['activity_types'].value_counts()
        activity_pct = (activity_counts / activity_counts.sum() * 100).round(2)
        
        activity_summary = pd.DataFrame({
            'Contagem': activity_counts,
            'Percentual (%)': activity_pct
        })
        print(activity_summary)
    
    # Análise temporal
    if 'date' in df.columns:
        print("\n" + "="*80)
        print("DISTRIBUIÇÃO TEMPORAL")
        print("="*80)
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        
        print(f"\nViagens por ano:")
        print(df.groupby('year')['count'].sum().sort_index())
        
        print(f"\nViagens por mês (agregado):")
        month_names = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
                      7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
        monthly = df.groupby('month')['count'].sum().sort_index()
        monthly.index = monthly.index.map(month_names)
        print(monthly)
    
    # VISUALIZAÇÕES
    if save_plots:
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Distribuição de contagens
        plt.subplot(3, 3, 1)
        df['count'].hist(bins=50, edgecolor='black', alpha=0.7)
        plt.title(f'Distribuição de Viagens - {tipo}', fontsize=12, weight='bold')
        plt.xlabel('Número de Viagens')
        plt.ylabel('Frequência')
        plt.axvline(df['count'].mean(), color='red', linestyle='--', label=f'Média: {df["count"].mean():.0f}')
        plt.axvline(df['count'].median(), color='green', linestyle='--', label=f'Mediana: {df["count"].median():.0f}')
        plt.legend()
        
        # 2. Commute vs Leisure
        plt.subplot(3, 3, 2)
        if 'commute_count' in df.columns and 'leisure_count' in df.columns:
            commute_total = df['commute_count'].sum()
            leisure_total = df['leisure_count'].sum()
            plt.pie([commute_total, leisure_total], 
                   labels=['Commute', 'Lazer'],
                   autopct='%1.1f%%',
                   colors=['#3498db', '#e74c3c'])
            plt.title('Commute vs Lazer', fontsize=12, weight='bold')
        
        # 3. Distribuição por período do dia
        plt.subplot(3, 3, 3)
        period_cols = ['morning_count', 'midday_count', 'evening_count', 'overnight_count']
        period_cols = [col for col in period_cols if col in df.columns]
        if period_cols:
            period_data = df[period_cols].sum()
            period_data.index = ['Manhã', 'Meio-dia', 'Tarde/Noite', 'Madrugada']
            period_data.plot(kind='bar', color=['#f39c12', '#e67e22', '#34495e', '#2c3e50'])
            plt.title('Viagens por Período do Dia', fontsize=12, weight='bold')
            plt.ylabel('Total de Viagens')
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        # 4. Weekday vs Weekend
        plt.subplot(3, 3, 4)
        if 'weekday_count' in df.columns and 'weekend_count' in df.columns:
            weekday_total = df['weekday_count'].sum()
            weekend_total = df['weekend_count'].sum()
            plt.bar(['Dias de Semana', 'Fim de Semana'], 
                   [weekday_total, weekend_total],
                   color=['#3498db', '#e74c3c'])
            plt.title('Dias de Semana vs Fim de Semana', fontsize=12, weight='bold')
            plt.ylabel('Total de Viagens')
        
        # 5. Boxplot de distâncias
        plt.subplot(3, 3, 5)
        if 'trip_distance_meters_p50' in df.columns:
            df['trip_distance_km'] = df['trip_distance_meters_p50'] / 1000
            df['trip_distance_km'].plot(kind='box', vert=False)
            plt.title('Distribuição de Distância das Viagens (km)', fontsize=12, weight='bold')
            plt.xlabel('Distância (km)')
        
        # 6. Boxplot de duração
        plt.subplot(3, 3, 6)
        if 'trip_duration_seconds_p50' in df.columns:
            df['trip_duration_minutes'] = df['trip_duration_seconds_p50'] / 60
            df['trip_duration_minutes'].plot(kind='box', vert=False)
            plt.title('Distribuição de Duração das Viagens (min)', fontsize=12, weight='bold')
            plt.xlabel('Duração (minutos)')
        
        # 7. Tipos de atividade
        plt.subplot(3, 3, 7)
        if 'activity_types' in df.columns:
            activity_counts = df['activity_types'].value_counts().head(10)
            activity_counts.plot(kind='barh', color='steelblue')
            plt.title('Top 10 Tipos de Atividade', fontsize=12, weight='bold')
            plt.xlabel('Número de Registros')
        
        # 8. Série temporal - viagens por mês
        plt.subplot(3, 3, 8)
        if 'date' in df.columns:
            df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
            monthly_trips = df.groupby('year_month')['count'].sum()
            monthly_trips.index = monthly_trips.index.astype(str)
            plt.plot(monthly_trips.values, marker='o', linewidth=2, markersize=4)
            plt.title('Evolução Temporal de Viagens', fontsize=12, weight='bold')
            plt.ylabel('Total de Viagens')
            plt.xlabel('Período')
            plt.xticks(range(0, len(monthly_trips), max(1, len(monthly_trips)//10)), 
                      [monthly_trips.index[i] for i in range(0, len(monthly_trips), max(1, len(monthly_trips)//10))],
                      rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
        
        # 9. Heatmap de correlação
        plt.subplot(3, 3, 9)
        corr_cols = [col for col in available_cols if col in df.columns][:8]  # Limitar a 8 colunas
        if len(corr_cols) > 1:
            corr_matrix = df[corr_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlação entre Variáveis', fontsize=12, weight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'analises/eda_basic_stats_{tipo.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráficos salvos: eda_basic_stats_{tipo.lower().replace(' ', '_')}.png")
        plt.close()
    
    return df


# Mapa coroplético interativo
#------------------------------------------------------------

def create_choropleth_map(data, gdf, coluna_count='count',
                         origem_destino=2,
                         tile_type='OpenStreetMap',
                         color_scale='viridis',
                         remove_outliers=False,
                         outlier_color_upper='#FF0000',
                         outlier_color_lower="#0000FF",
                         iqr_k=1.5):
    df = data.copy()

    if origem_destino == 0:
        df = df[df["origins_or_destinations"] == "origins"].copy()
        tipo_label = "Origem"
    elif origem_destino == 1:
        df = df[df["origins_or_destinations"] == "destinations"].copy()
        tipo_label = "Destino"
    else:
        tipo_label = "Origem + Destino"

    df_agg = df.groupby('hex_id')[coluna_count].sum().reset_index()

    if remove_outliers:
        q1 = df_agg[coluna_count].quantile(0.25)
        q3 = df_agg[coluna_count].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_k * iqr
        upper = q3 + iqr_k * iqr
        df_agg["is_outlier_lower"] = df_agg[coluna_count] < lower
        df_agg["is_outlier_upper"] = df_agg[coluna_count] > upper
    else:
        df_agg["is_outlier_lower"] = False
        df_agg["is_outlier_upper"] = False

    gdf_map = gdf.merge(df_agg, on='hex_id', how='left')
    gdf_map[coluna_count] = gdf_map[coluna_count].fillna(0)
    gdf_map["is_outlier_lower"] = gdf_map["is_outlier_lower"].fillna(False).astype(bool)
    gdf_map["is_outlier_upper"] = gdf_map["is_outlier_upper"].fillna(False).astype(bool)

    gdf_normal = gdf_map[~(gdf_map["is_outlier_lower"] | gdf_map["is_outlier_upper"])].copy()
    gdf_outlier_lower = gdf_map[gdf_map["is_outlier_lower"]].copy()
    gdf_outlier_upper = gdf_map[gdf_map["is_outlier_upper"]].copy()

    gdf_map_wgs84 = gdf_map.to_crs(epsg=4326)

    bounds = gdf_map_wgs84.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)

    for name in ["OpenStreetMap", "CartoDB positron", "CartoDB dark_matter"]:
        folium.TileLayer(tiles=name, name=name, control=True).add_to(m)

    folium.Choropleth(
        geo_data=gdf_normal.to_crs(epsg=4326),
        name=f'{coluna_count} (sem outliers)',
        data=gdf_normal,
        columns=['hex_id', coluna_count],
        key_on='feature.properties.hex_id',
        fill_color=color_scale,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'{coluna_count} ({tipo_label})',
        nan_fill_color='white'
    ).add_to(m)

    if remove_outliers and (not gdf_outlier_upper.empty or not gdf_outlier_lower.empty):
        def style_outlier_upper(feature):
            return {'fillColor': outlier_color_upper, 'color': outlier_color_upper,
                    'weight': 0.6, 'fillOpacity': 0.9}
        def style_outlier_lower(feature):
            return {'fillColor': outlier_color_lower, 'color': outlier_color_lower,
                    'weight': 0.6, 'fillOpacity': 0.9}

        folium.GeoJson(
            gdf_outlier_lower.to_crs(epsg=4326),
            name='Outliers Lower (IQR)',
            style_function=style_outlier_lower,
            tooltip=None,
            control=True
        ).add_to(m)

        folium.GeoJson(
            gdf_outlier_upper.to_crs(epsg=4326),
            name='Outliers Upper (IQR)',
            style_function=style_outlier_upper,
            tooltip=None,
            control=True
        ).add_to(m)

    style_function = lambda x: {'fillColor': '#ffffff', 'color': '#000000',
                                'fillOpacity': 0.1, 'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 'color': '#000000',
                                    'fillOpacity': 0.50, 'weight': 0.1}

    tooltip = folium.features.GeoJsonTooltip(
        fields=['hex_id', coluna_count],
        aliases=['Hex ID:', f'{coluna_count}:'],
        localize=True, sticky=False, labels=True,
        style="background-color:#F0EFEF;border:2px solid black;border-radius:3px;box-shadow:3px;",
        max_width=800,
    )

    folium.GeoJson(
        gdf_map_wgs84,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=tooltip
    ).add_to(m)

    folium.LayerControl().add_to(m)
    plugins.MiniMap().add_to(m)
    plugins.Fullscreen().add_to(m)

    remove_outliers_label = "Sim" if remove_outliers else "Nao"
    iqr_label = f"{iqr_k}" if remove_outliers else "-"

    stats_html = f'''
    <div style="position:fixed;top:10px;left:10px;width:300px;height:auto;
                background-color:white;border:2px solid grey;z-index:9999;
                font-size:14px;padding:10px;border-radius:5px;
                box-shadow:2px 2px 6px rgba(0,0,0,0.3);">
    <h4 style="margin-top:0;">Estatisticas</h4>
    <b>Coluna:</b> {coluna_count}<br>
    <b>Tipo:</b> {tipo_label}<br>
    <b>Total de Viagens:</b> {gdf_map[coluna_count].sum():,.0f}<br>
    <b>Media por Hexagono:</b> {gdf_map[coluna_count].mean():.1f}<br>
    <b>Maximo:</b> {gdf_map[coluna_count].max():,.0f}<br>
    <b>Hexagonos com dados:</b> {(gdf_map[coluna_count] > 0).sum()}<br>
    <hr style="margin:6px 0;">
    <b>Remover outliers:</b> {remove_outliers_label}<br>
    <b>IQR (k):</b> {iqr_label}<br>
    <b>Outliers Upper:</b>
    <span style="display:inline-block;width:12px;height:12px;
                 background-color:{outlier_color_upper};border:1px solid #333;
                 vertical-align:middle;"></span>
    <b>Outliers Lower:</b>
    <span style="display:inline-block;width:12px;height:12px;
                 background-color:{outlier_color_lower};border:1px solid #333;
                 vertical-align:middle;"></span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(stats_html))

    return m

# def create_choropleth_map(data, gdf, coluna_count='count', 
#                          data_inicio=None, data_fim=None,
#                          origem_destino=2, 
#                          weekday=None,
#                          tile_type='OpenStreetMap',
#                          color_scale='viridis',
#                          remove_outliers=False,        
#                         outlier_color_upper='#FF0000',     
#                         outlier_color_lower = "#0000FF",
#                         iqr_k=1.5                    
# ):
#     """
#     Cria mapa coroplético interativo com os dados do Strava
    
#     Parameters:
#     -----------
#     data : DataFrame
#         Dataset do Strava
#     gdf : GeoDataFrame
#         GeoDataFrame com geometria dos hexágonos
#     coluna_count : str
#         Coluna de contagem para visualizar
#     data_inicio : str ou datetime
#         Data de início do filtro
#     data_fim : str ou datetime
#         Data de fim do filtro
#     origem_destino : int (0, 1, ou 2)
#         0: apenas origem, 1: apenas destino, 2: ambos
#     tile_type : str
#         Tipo de mapa base ('OpenStreetMap', 'Stamen Terrain', 'Stamen Toner', 'CartoDB positron')
#     color_scale : str
#         Escala de cores ('viridis', 'plasma', 'Blues', 'Reds', 'YlOrRd', 'YlGnBu')
    
#     Returns:
#     --------
#     folium.Map
#         Mapa interativo
#     """
#     # Copiar dados
#     df = data.copy()
    
#     # Filtrar por origem/destino
#     if origem_destino == 0:
#         df = df[df["origins_or_destinations"] == "origins"].copy()
#         tipo_label = "Origem"
#     elif origem_destino == 1:
#         df = df[df["origins_or_destinations"] == "destinations"].copy()
#         tipo_label = "Destino"
#     else:
#         tipo_label = "Origem + Destino"

#     df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    
#     if weekday is not None:
#         # aceita int ou lista/tupla/set
#         if isinstance(weekday, (int, np.integer)):
#             weekday = [int(weekday)]
#         weekday = set(int(x) for x in weekday)  # normaliza
    
#         df = df[df["date_dt"].dt.weekday.isin(weekday)]


#     # Filtrar por data
#     if data_inicio:
#         df = df[df["date_dt"] >= pd.to_datetime(data_inicio)]
#     if data_fim:
#         df = df[df["date_dt"] <= pd.to_datetime(data_fim)]

    
#     # Agregar por hex_id
#     df_agg = df.groupby('hex_id')[coluna_count].sum().reset_index()
#     if remove_outliers:
#         q1 = df_agg[coluna_count].quantile(0.25)
#         q3 = df_agg[coluna_count].quantile(0.75)
#         iqr = q3 - q1
    
#         lower = q1 - iqr_k * iqr
#         upper = q3 + iqr_k * iqr
    
#         df_agg["is_outlier_lower"] = df_agg[coluna_count] < lower
#         df_agg["is_outlier_upper"] = df_agg[coluna_count] > upper
#     else:
#         df_agg["is_outlier_lower"] = False
#         df_agg["is_outlier_upper"] = False


    
#     # Merge com geometria
#     gdf_map = gdf.merge(df_agg, on='hex_id', how='left')
#     gdf_map[coluna_count] = gdf_map[coluna_count].fillna(0)
#     gdf_map["is_outlier_lower"] = gdf_map["is_outlier_lower"].fillna(False).astype(bool)
#     gdf_map["is_outlier_upper"] = gdf_map["is_outlier_upper"].fillna(False).astype(bool)

#     # dois subconjuntos lógicos
#     gdf_normal = gdf_map[
#         ~(gdf_map["is_outlier_lower"] | gdf_map["is_outlier_upper"])
#     ].copy()
    
#     gdf_outlier_lower = gdf_map[gdf_map["is_outlier_lower"]].copy()
#     gdf_outlier_upper = gdf_map[gdf_map["is_outlier_upper"]].copy()


#     # Converter para GeoJSON
#     gdf_map_wgs84 = gdf_map.to_crs(epsg=4326)
    
#     # Calcular centro do mapa
#     bounds = gdf_map_wgs84.total_bounds
#     center_lat = (bounds[1] + bounds[3]) / 2
#     center_lon = (bounds[0] + bounds[2]) / 2
    
#     # Criar mapa
#     m = folium.Map(
#         location=[center_lat, center_lon],
#         zoom_start=11,
#         tiles=None
#     )
#     folium.TileLayer(
#     tiles="OpenStreetMap",
#     name="OpenStreetMap",
#     control=True
#     ).add_to(m)
    
#     folium.TileLayer(
#         tiles="Stamen Terrain",
#         name="Stamen Terrain",
#         attr="Map tiles by Stamen Design, CC BY 3.0 — Data © OpenStreetMap contributors",
#         control=True
#     ).add_to(m)
    
#     folium.TileLayer(
#         tiles="Stamen Toner",
#         name="Stamen Toner",
#         attr="Map tiles by Stamen Design, CC BY 3.0 — Data © OpenStreetMap contributors",
#         control=True
#     ).add_to(m)
    
#     folium.TileLayer(
#         tiles="CartoDB positron",
#         name="CartoDB positron",
#         control=True
#     ).add_to(m)
    
#     folium.TileLayer(
#         tiles="CartoDB dark_matter",
#         name="CartoDB dark_matter",
#         control=True
#     ).add_to(m)

    
#     # Adicionar camada coroplética
#     # folium.Choropleth(
#     #     geo_data=gdf_map_wgs84,
#     #     name=f'choropleth - {coluna_count}',
#     #     data=gdf_map_wgs84,
#     #     columns=['hex_id', coluna_count],
#     #     key_on='feature.properties.hex_id',
#     #     fill_color=color_scale,
#     #     fill_opacity=0.7,
#     #     line_opacity=0.2,
#     #     legend_name=f'{coluna_count} ({tipo_label})',
#     #     nan_fill_color='white'
#     # ).add_to(m)
#     folium.Choropleth(
#         geo_data=gdf_normal.to_crs(epsg=4326),
#         name=f'{coluna_count} (sem outliers)',
#         data=gdf_normal,
#         columns=['hex_id', coluna_count],
#         key_on='feature.properties.hex_id',
#         fill_color=color_scale,
#         fill_opacity=0.7,
#         line_opacity=0.2,
#         legend_name=f'{coluna_count} ({tipo_label})',
#         nan_fill_color='white'
#     ).add_to(m)

#     if remove_outliers and (not gdf_outlier_upper.empty or not gdf_outlier_lower.empty) :
    
#         def style_outlier_upper(feature):
#             return {
#                 'fillColor': outlier_color_upper,
#                 'color': outlier_color_upper,
#                 'weight': 0.6,
#                 'fillOpacity': 0.9
#             }
#         def style_outlier_lower(feature):
#             return {
#                 'fillColor': outlier_color_lower,
#                 'color': outlier_color_lower,
#                 'weight': 0.6,
#                 'fillOpacity': 0.9
#             }
#         tooltip = folium.features.GeoJsonTooltip(
#             fields=["hex_id", coluna_count],
#             aliases=["Hex ID:", f"{coluna_count}:"],
#             localize=True,
#             sticky=False,
#             labels=True,
#             style="""
#                 background-color: #F0EFEF;
#                 border: 2px solid black;
#                 border-radius: 3px;
#                 box-shadow: 3px;
#             """,
#             max_width=800,
#         )
#         print(gdf_outlier_lower.columns)
#         print(gdf_outlier_upper.columns)
#         print("coluna_count:", coluna_count)
#         folium.GeoJson(
#             gdf_outlier_lower.to_crs(epsg=4326),
#             name='Outliers Lower (IQR)',
#             style_function=style_outlier_lower,
#             tooltip=None, # reutiliza o mesmo tooltip
#             control = True
#         ).add_to(m)
        
#         folium.GeoJson(
#             gdf_outlier_upper.to_crs(epsg=4326),
#             name='Outliers Upper (IQR)',
#             style_function=style_outlier_upper,
#             tooltip=None,  # reutiliza o mesmo tooltip
#             control = True
#         ).add_to(m)


    
#     # Adicionar tooltips
#     style_function = lambda x: {'fillColor': '#ffffff', 
#                                 'color':'#000000', 
#                                 'fillOpacity': 0.1, 
#                                 'weight': 0.1}
#     highlight_function = lambda x: {'fillColor': '#000000', 
#                                     'color':'#000000', 
#                                     'fillOpacity': 0.50, 
#                                     'weight': 0.1}
    
#     tooltip = folium.features.GeoJsonTooltip(
#         fields=['hex_id', coluna_count],
#         aliases=['Hex ID:', f'{coluna_count}:'],
#         localize=True,
#         sticky=False,
#         labels=True,
#         style="""
#             background-color: #F0EFEF;
#             border: 2px solid black;
#             border-radius: 3px;
#             box-shadow: 3px;
#         """,
#         max_width=800,
#     )
    
#     folium.GeoJson(
#         gdf_map_wgs84,
#         style_function=style_function, 
#         control=False,
#         highlight_function=highlight_function,
#         tooltip=tooltip
#     ).add_to(m)
    
#     # Adicionar controle de camadas
#     folium.LayerControl().add_to(m)
    
#     # Adicionar minimap
#     minimap = plugins.MiniMap()
#     m.add_child(minimap)
    
#     # Adicionar fullscreen
#     plugins.Fullscreen().add_to(m)

#     # flags simples para o HTML
#     remove_outliers_label = "Sim" if remove_outliers else "Não"
    
#     if remove_outliers:
#         iqr_label = f"{iqr_k}"
#     else:
#         iqr_label = "-"
#     WEEKDAY_NAMES = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    
#     if weekday is None or (isinstance(weekday, (list, tuple, set)) and len(weekday) == 0):
#         weekday_label = "Todos"
#     else:
#         # aceita int ou iterável
#         if isinstance(weekday, (int, np.integer)):
#             weekday = [int(weekday)]
#         weekday_sorted = sorted(set(int(x) for x in weekday))
#         weekday_label = ", ".join(WEEKDAY_NAMES[i] for i in weekday_sorted)

#     # Estatísticas do mapa
#     stats_html = f'''
#     <div style="position: fixed; 
#                 top: 10px; left: 10px; 
#                 width: 300px; height: auto; 
#                 background-color: white; 
#                 border:2px solid grey; 
#                 z-index:9999; 
#                 font-size:14px;
#                 padding: 10px;
#                 border-radius: 5px;
#                 box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
#                 ">
#     <h4 style="margin-top:0;"> Estatísticas</h4>
#     <b>Coluna:</b> {coluna_count}<br>
#     <b>Tipo:</b> {tipo_label}<br>
#     <b>Total de Viagens:</b> {gdf_map[coluna_count].sum():,.0f}<br>
#     <b>Média por Hexágono:</b> {gdf_map[coluna_count].mean():.1f}<br>
#     <b>Máximo:</b> {gdf_map[coluna_count].max():,.0f}<br>
#     <b>Hexágonos com dados:</b> {(gdf_map[coluna_count] > 0).sum()}<br>
#     <hr style="margin:6px 0;">
#     <b>Dia da semana:</b> {weekday_label}<br>

#     <b>Remover outliers:</b> {remove_outliers_label}<br>
#     <b>IQR (k):</b> {iqr_label}<br>
#     <b>Outliers Upper:</b>
#     <span style="display:inline-block;
#                  width:12px;
#                  height:12px;
#                  background-color:{outlier_color_upper};
#                  border:1px solid #333;
#                  vertical-align:middle;"></span>
#     <b>Outliers Lower:</b>
#     <span style="display:inline-block;
#                  width:12px;
#                  height:12px;
#                  background-color:{outlier_color_lower};
#                  border:1px solid #333;
#                  vertical-align:middle;"></span>

#     </div>
    
#     '''
#     m.get_root().html.add_child(folium.Element(stats_html))
    
#     return m


def create_interactive_map_widget(data, gdf):
    """
    Creates an interactive ipywidgets interface for map exploration.

    Parameters:
    -----------
    data : DataFrame
        Strava dataset
    gdf : GeoDataFrame
        GeoDataFrame with geometry
    """
    from ipywidgets import interact, widgets
    from IPython.display import display
    
    # Opções disponíveis
    count_columns = [
        'count', 'commute_count', 'leisure_count',
        'morning_count', 'midday_count', 'evening_count', 'overnight_count',
        'weekday_count', 'weekend_count'
    ]
    
    # Filtrar apenas colunas que existem
    count_columns = [col for col in count_columns if col in data.columns]
    
    # Datas min/max
    date_min = pd.to_datetime(data['date']).min()
    date_max = pd.to_datetime(data['date']).max()
    
    # Criar widgets
    coluna_widget = widgets.Dropdown(
        options=count_columns,
        value='count',
        description='Métrica:',
        style={'description_width': 'initial'}
    )
    
    origem_destino_widget = widgets.Dropdown(
        options=[('Origem', 0), ('Destino', 1), ('Ambos', 2)],
        value=2,
        description='Filtro:',
        style={'description_width': 'initial'}
    )
    
    data_inicio_widget = widgets.DatePicker(
        description='Data Início:',
        value=date_min.date(),
        style={'description_width': 'initial'}
    )
    
    data_fim_widget = widgets.DatePicker(
        description='Data Fim:',
        value=date_max.date(),
        style={'description_width': 'initial'}
    )
    
    # tile_widget = widgets.Dropdown(
    #     options=['OpenStreetMap', 'Stamen Terrain', 'Stamen Toner', 'CartoDB positron', 'CartoDB dark_matter'],
    #     value='OpenStreetMap',
    #     description='Tipo Mapa:',
    #     style={'description_width': 'initial'}
    # )
    weekday_widget = widgets.SelectMultiple(
        options=[
            ("Segunda (0)", 0),
            ("Terça (1)", 1),
            ("Quarta (2)", 2),
            ("Quinta (3)", 3),
            ("Sexta (4)", 4),
            ("Sábado (5)", 5),
            ("Domingo (6)", 6),
        ],
        value=(),  # vazio = sem filtro
        description="Dia semana:",
        style={'description_width': 'initial'}
    )

    color_widget = widgets.Dropdown(
        options=['viridis', 'plasma', 'Blues', 'Reds', 'YlOrRd', 'YlGnBu', 'Greens', 'Purples', 'RdYlGn'],
        value='viridis',
        description='Escala Cor:',
        style={'description_width': 'initial'}
    )

    remove_outliers_widget = widgets.Checkbox(
        value=False,
        description='Tratar outliers (IQR)',
        style={'description_width': 'initial'}
    )

    outlier_upper_color_widget = widgets.ColorPicker(
        concise=False,
        description='Cor Outliers Upper:',
        value='#FF0000',
        style={'description_width': 'initial'}
    )

    outlier_lower_color_widget = widgets.ColorPicker(
        concise=False,
        description='Cor Outliers Lower:',
        value='#0000FF',
        style={'description_width': 'initial'}
    )

    iqr_k_widget = widgets.FloatSlider(
        value=1.5,
        min=0.5,
        max=3.0,
        step=0.1,
        description='IQR (k):',
        continuous_update=False,
        style={'description_width': 'initial'}
    )


    
    # Função de interação
    def update_map(coluna, origem_destino, data_inicio, data_fim,weekday, color, remove_outliers,
    outlier_color_upper, outlier_color_lower, iqr_k):
        mapa = create_choropleth_map(
            data=data,
            gdf=gdf,
            coluna_count=coluna,
            data_inicio=data_inicio,
            data_fim=data_fim,
            weekday=(list(weekday) if len(weekday) > 0 else None),
            origem_destino=origem_destino,
            # tile_type=tile,
            color_scale=color,
            remove_outliers=remove_outliers,
            outlier_color_upper=outlier_color_upper,
            outlier_color_lower = outlier_color_lower,
            iqr_k=iqr_k
        )
        
        # Salvar mapa
        filename = f'analises/mapa_strava_{coluna}_{origem_destino}.html'
        mapa.save(filename)
        print(f"✓ Mapa salvo: {filename}")
        
        return mapa
    
    # Criar interface interativa
    interact(
        update_map,
        coluna=coluna_widget,
        origem_destino=origem_destino_widget,
        data_inicio=data_inicio_widget,
        data_fim=data_fim_widget,
        # tile=tile_widget,
        weekday=weekday_widget,

        color=color_widget,
         remove_outliers=remove_outliers_widget,
        outlier_color_upper=outlier_upper_color_widget,
        outlier_color_lower=outlier_lower_color_widget,
        iqr_k=iqr_k_widget
    )


# Análises Adicionais
#------------------------------------------------------------


def analise_padroes_temporais(data, origem_destino=2):
    """
    Detailed analysis of temporal patterns.
    """
    df = data.copy()
    
    # Filtrar por origem/destino
    if origem_destino == 0:
        df = df[df["origins_or_destinations"] == "origins"].copy()
        tipo = "ORIGEM"
    elif origem_destino == 1:
        df = df[df["origins_or_destinations"] == "destinations"].copy()
        tipo = "DESTINO"
    else:
        tipo = "AMBOS"
    
    print("="*80)
    print(f"ANÁLISE DE PADRÕES TEMPORAIS - {tipo}")
    print("="*80)
    
    # Converter data
    df['datetime'] = pd.to_datetime(df['date'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    
    # Criar visualizações
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Heatmap: Dia da semana vs Hora do dia
    period_cols = ['morning_count', 'midday_count', 'evening_count', 'overnight_count']
    period_cols = [col for col in period_cols if col in df.columns]
    
    if period_cols and 'day_of_week' in df.columns:
        heatmap_data = df.groupby('day_of_week')[period_cols].sum()
        heatmap_data.index = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
        heatmap_data.columns = ['Manhã', 'Meio-dia', 'Tarde', 'Madrugada']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                   ax=axes[0, 0], cbar_kws={'label': 'Viagens'})
        axes[0, 0].set_title('Viagens por Dia da Semana e Período', fontsize=12, weight='bold')
        axes[0, 0].set_xlabel('Período do Dia')
        axes[0, 0].set_ylabel('Dia da Semana')
    
    # 2. Tendência mensal
    monthly_trend = df.groupby(df['datetime'].dt.to_period('M'))['count'].sum()
    axes[0, 1].plot(range(len(monthly_trend)), monthly_trend.values, 
                   marker='o', linewidth=2, markersize=6, color='steelblue')
    axes[0, 1].fill_between(range(len(monthly_trend)), monthly_trend.values, alpha=0.3)
    axes[0, 1].set_title('Tendência Mensal de Viagens', fontsize=12, weight='bold')
    axes[0, 1].set_xlabel('Mês')
    axes[0, 1].set_ylabel('Total de Viagens')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Sazonalidade - Boxplot por mês
    if 'month' in df.columns:
        month_data = df.groupby(['month', 'hex_id'])['count'].sum().reset_index()
        month_order = list(range(1, 13))
        month_labels = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                       'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        
        sns.boxplot(data=month_data, x='month', y='count', 
                   order=month_order, ax=axes[1, 0], palette='Set2')
        axes[1, 0].set_xticklabels(month_labels)
        axes[1, 0].set_title('Sazonalidade - Distribuição por Mês', fontsize=12, weight='bold')
        axes[1, 0].set_xlabel('Mês')
        axes[1, 0].set_ylabel('Viagens por Hexágono')
    
    # 4. Comparação Weekday vs Weekend ao longo do tempo
    if 'weekday_count' in df.columns and 'weekend_count' in df.columns:
        weekly_data = df.groupby(df['datetime'].dt.to_period('W')).agg({
            'weekday_count': 'sum',
            'weekend_count': 'sum'
        })
        
        axes[1, 1].plot(range(len(weekly_data)), weekly_data['weekday_count'], 
                       label='Dias de Semana', marker='o', markersize=3, linewidth=1.5)
        axes[1, 1].plot(range(len(weekly_data)), weekly_data['weekend_count'], 
                       label='Fim de Semana', marker='s', markersize=3, linewidth=1.5)
        axes[1, 1].set_title('Tendência Semanal: Weekday vs Weekend', fontsize=12, weight='bold')
        axes[1, 1].set_xlabel('Semana')
        axes[1, 1].set_ylabel('Total de Viagens')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analises/analise_temporal_{tipo.lower()}.png', 
               dpi=300, bbox_inches='tight')
    print(f"✓ Análise temporal salva: analise_temporal_{tipo.lower()}.png")
    plt.close()


def analise_espacial(data, gdf, origem_destino=2):
    """
    Analysis of spatial patterns.
    """
    df = data.copy()
    
    # Filtrar por origem/destino
    if origem_destino == 0:
        df = df[df["origins_or_destinations"] == "origins"].copy()
        tipo = "ORIGEM"
    elif origem_destino == 1:
        df = df[df["origins_or_destinations"] == "destinations"].copy()
        tipo = "DESTINO"
    else:
        tipo = "AMBOS"
    
    print("="*80)
    print(f"ANÁLISE ESPACIAL - {tipo}")
    print("="*80)
    
    # Agregar por hexágono
    df_agg = df.groupby('hex_id').agg({
        'count': 'sum',
        'commute_count': 'sum',
        'leisure_count': 'sum',
        'trip_distance_meters_p50': 'mean',
        'trip_duration_seconds_p50': 'mean'
    }).reset_index()
    
    # Identificar hotspots
    percentil_95 = df_agg['count'].quantile(0.95)
    percentil_90 = df_agg['count'].quantile(0.90)
    percentil_75 = df_agg['count'].quantile(0.75)
    
    print(f"\n Hotspots (Top 5% - mais de {percentil_95:.0f} viagens):")
    print(f"   Número de hexágonos: {(df_agg['count'] >= percentil_95).sum()}")
    print(f"   Total de viagens: {df_agg[df_agg['count'] >= percentil_95]['count'].sum():,.0f}")
    
    print(f"\n Zonas Ativas (Top 10% - mais de {percentil_90:.0f} viagens):")
    print(f"   Número de hexágonos: {(df_agg['count'] >= percentil_90).sum()}")
    
    print(f"\n Zonas Moderadas (Top 25% - mais de {percentil_75:.0f} viagens):")
    print(f"   Número de hexágonos: {(df_agg['count'] >= percentil_75).sum()}")
    
    # Top 10 hexágonos
    print(f"\n Hexágonos com mais viagens:")
    top_10 = df_agg.nlargest(10, 'count')[['hex_id', 'count', 'commute_count', 'leisure_count']]
    print(top_10.to_string(index=False))
    
    # Análise de distância e duração por intensidade
    df_agg['categoria'] = pd.cut(df_agg['count'], 
                                 bins=[0, percentil_75, percentil_90, percentil_95, float('inf')],
                                 labels=['Baixa', 'Moderada', 'Alta', 'Muito Alta'])
    
    print(f"\n Distância média por categoria:")
    dist_por_categoria = df_agg.groupby('categoria')['trip_distance_meters_p50'].mean() / 1000
    print(dist_por_categoria.round(2))
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Distribuição espacial (top hexágonos)
    top_hex = df_agg.nlargest(20, 'count')
    axes[0, 0].barh(range(len(top_hex)), top_hex['count'].values, color='steelblue')
    axes[0, 0].set_yticks(range(len(top_hex)))
    axes[0, 0].set_yticklabels([f"Hex {i+1}" for i in range(len(top_hex))])
    axes[0, 0].set_xlabel('Total de Viagens')
    axes[0, 0].set_title('Top 20 Hexágonos por Volume', fontsize=12, weight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. Relação Commute vs Leisure
    axes[0, 1].scatter(df_agg['commute_count'], df_agg['leisure_count'], 
                      alpha=0.6, s=50, c=df_agg['count'], cmap='viridis')
    axes[0, 1].set_xlabel('Viagens Commute')
    axes[0, 1].set_ylabel('Viagens Lazer')
    axes[0, 1].set_title('Commute vs Lazer por Hexágono', fontsize=12, weight='bold')
    axes[0, 1].plot([0, df_agg['commute_count'].max()], 
                   [0, df_agg['commute_count'].max()], 
                   'r--', alpha=0.5, label='Linha 1:1')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribuição de distâncias
    if 'trip_distance_meters_p50' in df_agg.columns:
        df_agg['distance_km'] = df_agg['trip_distance_meters_p50'] / 1000
        axes[1, 0].hist(df_agg['distance_km'], bins=50, edgecolor='black', 
                       alpha=0.7, color='coral')
        axes[1, 0].set_xlabel('Distância Média (km)')
        axes[1, 0].set_ylabel('Número de Hexágonos')
        axes[1, 0].set_title('Distribuição de Distância Média', fontsize=12, weight='bold')
        axes[1, 0].axvline(df_agg['distance_km'].median(), 
                          color='red', linestyle='--', 
                          label=f'Mediana: {df_agg["distance_km"].median():.1f} km')
        axes[1, 0].legend()
    
    # 4. Concentração espacial (Curva de Lorenz)
    df_sorted = df_agg.sort_values('count')
    cumsum = df_sorted['count'].cumsum()
    cumsum_pct = cumsum / cumsum.iloc[-1] * 100
    hex_pct = np.linspace(0, 100, len(df_sorted))
    
    axes[1, 1].plot(hex_pct, cumsum_pct, linewidth=2, label='Curva Real')
    axes[1, 1].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Distribuição Uniforme')
    axes[1, 1].fill_between(hex_pct, hex_pct, cumsum_pct, alpha=0.3)
    axes[1, 1].set_xlabel('% de Hexágonos (acumulado)')
    axes[1, 1].set_ylabel('% de Viagens (acumulado)')
    axes[1, 1].set_title('Concentração Espacial (Curva de Lorenz)', fontsize=12, weight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Calcular Gini (medida de desigualdade)
    gini = (hex_pct - cumsum_pct).sum() / hex_pct.sum()
    axes[1, 1].text(0.05, 0.95, f'Índice Gini: {gini:.3f}', 
                   transform=axes[1, 1].transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'analises/analise_espacial_{tipo.lower()}.png', 
               dpi=300, bbox_inches='tight')
    print(f"\n✓ Análise espacial salva: analise_espacial_{tipo.lower()}.png")
    plt.close()


def analise_od_matrix(data, gdf, top_n=20):
    """
    Origin-destination matrix analysis.
    """
    print("="*80)
    print("ANÁLISE DE MATRIZ ORIGEM-DESTINO")
    print("="*80)
    
    # Separar origens e destinos
    origens = (
        data[data["origins_or_destinations"] == "origins"]
        [["hex_id", "count"]]
        .groupby("hex_id", as_index=True)
        .sum()
    )
    # origens = data[data['is_origin'] == True][['hex_id', 'count']].groupby('hex_id').sum()
    # destinos = data[data['is_destination'] == True][['hex_id', 'count']].groupby('hex_id').sum()
    destinos = (
        data[data["origins_or_destinations"] == "destinations"]
        [["hex_id", "count"]]
        .groupby("hex_id", as_index=True)
        .sum()
    )
    
    origens.columns = ['viagens_origem']
    destinos.columns = ['viagens_destino']
    
    # Merge
    od_summary = origens.join(destinos, how='outer').fillna(0)
    od_summary['total'] = od_summary['viagens_origem'] + od_summary['viagens_destino']
    od_summary['saldo'] = od_summary['viagens_origem'] - od_summary['viagens_destino']
    od_summary['taxa_atracao'] = (od_summary['viagens_destino'] / 
                                  (od_summary['viagens_origem'] + od_summary['viagens_destino']))
    
    # Identificar zonas
    print(f"\n Top 10 zonas geradoras (mais origens que destinos):")
    geradores = od_summary.nlargest(10, 'saldo')[['viagens_origem', 'viagens_destino', 'saldo']]
    print(geradores)
    
    print(f"\n Top 10 zonas atratoras (mais destinos que origens):")
    atratores = od_summary.nsmallest(10, 'saldo')[['viagens_origem', 'viagens_destino', 'saldo']]
    print(atratores)
    
    print(f"\nTop 10 zonas balanceadas (origem ≈ destino):")
    od_summary['balance_score'] = abs(od_summary['saldo']) / od_summary['total']
    balanceadas = od_summary.nsmallest(10, 'balance_score')[
        ['viagens_origem', 'viagens_destino', 'saldo', 'balance_score']
    ]
    print(balanceadas)
    
    # Visualizações
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Scatter: Origem vs Destino
    axes[0, 0].scatter(od_summary['viagens_origem'], od_summary['viagens_destino'],
                      alpha=0.6, s=50, c=od_summary['saldo'], cmap='RdBu_r')
    axes[0, 0].plot([0, od_summary['viagens_origem'].max()],
                   [0, od_summary['viagens_origem'].max()],
                   'k--', alpha=0.5, linewidth=2, label='Linha 1:1')
    axes[0, 0].set_xlabel('Viagens como Origem')
    axes[0, 0].set_ylabel('Viagens como Destino')
    axes[0, 0].set_title('Relação Origem-Destino por Hexágono', fontsize=12, weight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribuição do saldo
    axes[0, 1].hist(od_summary['saldo'], bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Equilíbrio')
    axes[0, 1].set_xlabel('Saldo (Origem - Destino)')
    axes[0, 1].set_ylabel('Número de Hexágonos')
    axes[0, 1].set_title('Distribuição do Saldo O-D', fontsize=12, weight='bold')
    axes[0, 1].legend()
    
    # 3. Top geradores vs atratores
    top_geradores = od_summary.nlargest(top_n, 'saldo')
    top_atratores = od_summary.nsmallest(top_n, 'saldo')
    
    x_pos = np.arange(top_n)
    axes[1, 0].barh(x_pos, top_geradores['saldo'].values, color='red', alpha=0.7, label='Geradores')
    axes[1, 0].set_yticks(x_pos)
    axes[1, 0].set_yticklabels([f"G{i+1}" for i in range(top_n)])
    axes[1, 0].set_xlabel('Saldo (Origem - Destino)')
    axes[1, 0].set_title(f'Top {top_n} Zonas Geradoras', fontsize=12, weight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 4. Taxa de atração
    axes[1, 1].hist(od_summary['taxa_atracao'].dropna(), bins=50, 
                   edgecolor='black', alpha=0.7, color='teal')
    axes[1, 1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='50% (Balanceado)')
    axes[1, 1].set_xlabel('Taxa de Atração (Destino / Total)')
    axes[1, 1].set_ylabel('Número de Hexágonos')
    axes[1, 1].set_title('Distribuição da Taxa de Atração', fontsize=12, weight='bold')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('analises/analise_od_matrix.png', 
               dpi=300, bbox_inches='tight')
    print("\n✓ Análise O-D salva: analise_od_matrix.png")
    plt.close()
    
    return od_summary


def analise_atividades(data, origem_destino=2):
    """
    Detailed analysis of activity types.
    """
    df = data.copy()
    
    # Filtrar por origem/destino
    if origem_destino == 0:
        df = df[df["origins_or_destinations"] == "origins"].copy()
        tipo = "ORIGEM"
    elif origem_destino == 1:
        df = df[df["origins_or_destinations"] == "destinations"].copy()
        tipo = "DESTINO"
    else:
        tipo = "AMBOS"
    
    print("="*80)
    print(f"ANÁLISE DE TIPOS DE ATIVIDADE - {tipo}")
    print("="*80)
    
    if 'activity_types' not in df.columns:
        print("Coluna 'activity_types' não encontrada no dataset")
        return
    
    # Contagem de atividades
    activity_counts = df['activity_types'].value_counts()
    activity_pct = (activity_counts / activity_counts.sum() * 100).round(2)
    
    print(f"\nTotal de tipos de atividade: {len(activity_counts)}")
    print(f"\nTop 15 atividades:")
    for i, (activity, count) in enumerate(activity_counts.head(15).items(), 1):
        print(f"{i:2d}. {activity:30s}: {count:8,} ({activity_pct[activity]:5.2f}%)")
    
    # Visualizações
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Top atividades
    ax1 = plt.subplot(2, 3, 1)
    top_15 = activity_counts.head(15)
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_15)))
    top_15.plot(kind='barh', ax=ax1, color=colors)
    ax1.set_xlabel('Número de Registros')
    ax1.set_title('Top 15 Tipos de Atividade', fontsize=12, weight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Percentual acumulado
    ax2 = plt.subplot(2, 3, 2)
    cumsum_pct = activity_pct.cumsum()
    ax2.plot(range(1, len(cumsum_pct) + 1), cumsum_pct.values, 
            marker='o', markersize=3, linewidth=2)
    ax2.axhline(80, color='red', linestyle='--', label='80%')
    ax2.axhline(90, color='orange', linestyle='--', label='90%')
    ax2.set_xlabel('Número de Tipos de Atividade')
    ax2.set_ylabel('Percentual Acumulado (%)')
    ax2.set_title('Curva de Pareto - Atividades', fontsize=12, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Pie chart top 10
    ax3 = plt.subplot(2, 3, 3)
    top_10 = activity_counts.head(10)
    outros = activity_counts[10:].sum()
    pie_data = pd.concat([top_10, pd.Series({'Outros': outros})])
    colors_pie = list(plt.cm.Set3(np.linspace(0, 1, 10))) + ['lightgray']
    ax3.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', 
           colors=colors_pie, startangle=90)
    ax3.set_title('Distribuição Top 10 Atividades', fontsize=12, weight='bold')
    
    # 4. Atividades por período (se disponível)
    if all(col in df.columns for col in ['morning_count', 'midday_count', 'evening_count']):
        ax4 = plt.subplot(2, 3, 4)
        period_by_activity = df.groupby('activity_types')[
            ['morning_count', 'midday_count', 'evening_count', 'overnight_count']
        ].sum()
        top_activities = activity_counts.head(10).index
        period_by_activity.loc[top_activities].plot(kind='bar', ax=ax4, stacked=True)
        ax4.set_xlabel('Tipo de Atividade')
        ax4.set_ylabel('Total de Viagens')
        ax4.set_title('Distribuição por Período - Top 10', fontsize=12, weight='bold')
        ax4.legend(['Manhã', 'Meio-dia', 'Tarde', 'Madrugada'])
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 5. Commute vs Leisure por atividade
    if 'commute_count' in df.columns and 'leisure_count' in df.columns:
        ax5 = plt.subplot(2, 3, 5)
        commute_leisure = df.groupby('activity_types')[['commute_count', 'leisure_count']].sum()
        top_activities = activity_counts.head(10).index
        commute_leisure.loc[top_activities].plot(kind='bar', ax=ax5)
        ax5.set_xlabel('Tipo de Atividade')
        ax5.set_ylabel('Total de Viagens')
        ax5.set_title('Commute vs Lazer - Top 10', fontsize=12, weight='bold')
        ax5.legend(['Commute', 'Lazer'])
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Diversidade de atividades por hexágono
    ax6 = plt.subplot(2, 3, 6)
    diversity = df.groupby('hex_id')['activity_types'].nunique()
    diversity.hist(bins=30, ax=ax6, edgecolor='black', alpha=0.7, color='mediumseagreen')
    ax6.set_xlabel('Número de Tipos de Atividade')
    ax6.set_ylabel('Número de Hexágonos')
    ax6.set_title('Diversidade de Atividades por Hexágono', fontsize=12, weight='bold')
    ax6.axvline(diversity.median(), color='red', linestyle='--', 
               label=f'Mediana: {diversity.median():.0f}')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(f'analises/analise_atividades_{tipo.lower()}.png', 
               dpi=300, bbox_inches='tight')
    print(f"\n✓ Análise de atividades salva: analise_atividades_{tipo.lower()}.png")
    plt.close()


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

def exemplo_uso_completo():
    """
    Example of how to use all EDA functions.
    """
    print("""
    Como usar:
    
    # 1. Carregar seus dados
    import pandas as pd
    import geopandas as gpd
    
    data_strava = pd.read_csv('seu_arquivo.csv')
    gdf = gpd.read_file('geometria_hexagonos.geojson')
    
    # 2. Executar análises básicas
    df_origem = basic_stats(data_strava, origem_destino=0)  # Apenas origem
    df_destino = basic_stats(data_strava, origem_destino=1)  # Apenas destino
    df_todos = basic_stats(data_strava, origem_destino=2)   # Ambos
    
    # 3. Criar mapas coropléticos
    mapa = create_choropleth_map(
        data=data_strava,
        gdf=gdf,
        coluna_count='count',
        data_inicio='2023-01-01',
        data_fim='2023-12-31',
        origem_destino=2,
        tile_type='OpenStreetMap',
        color_scale='viridis',
    )
    mapa.save('mapa_strava.html')
    
    # 4. Interface interativa (Jupyter Notebook)
    create_interactive_map_widget(data_strava, gdf)
    
    # 5. Análises adicionais
    analise_padroes_temporais(data_strava, origem_destino=2)
    analise_espacial(data_strava, gdf, origem_destino=2)
    od_summary = analise_od_matrix(data_strava, gdf)
    analise_atividades(data_strava, origem_destino=2)
    
    OUTPUTS:
    --------
    - Gráficos estatísticos salvos em analises/
    - Mapas interativos em HTML
    - DataFrames processados para análises posteriores
    """)

if __name__ == "__main__":
    exemplo_uso_completo()
