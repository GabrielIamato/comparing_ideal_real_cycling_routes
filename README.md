# BikeScience São Paulo - Cycling Infrastructure Gap Analysis

A geospatial research project analyzing bicycle mobility in São Paulo, Brazil. It integrates Strava Metro trip data with OpenStreetMap road networks and census demographics to identify infrastructure gaps and prioritize cycling investment areas.

---

## Table of Contents

- [BikeScience São Paulo - Cycling Infrastructure Gap Analysis](#bikescience-são-paulo--cycling-infrastructure-gap-analysis)
  - [Table of Contents](#table-of-contents)
  - [Analyses](#analyses)
  - [About the Project](#about-the-project)
  - [Repository Structure](#repository-structure)
  - [Data](#data)
    - [Strava Metro (`data/strava/`)](#strava-metro-datastrava)
    - [OSM Road Network (`data/malha_sp_ops/`)](#osm-road-network-datamalha_sp_ops)
  - [How to Run](#how-to-run)
  - [Acknowledgements](#acknowledgements)

---

## Analyses

| Script | Description |
|--------|-------------|
| `strava_eda.py` | Exploratory Data Analysis - temporal patterns, spatial distributions, and activity type breakdowns across the road network. |
| `analysis_gap.py` | Ranks infrastructure-free roads by a priority score (`trip_count × edge_length_m`), highlighting high-demand unprotected segments. |

Interactive HTML maps and static PNG charts are saved to the `analises/` directory.

---

## About the Project

This work is part of the **BikeScience** research initiative, which applies data science methods to support evidence-based cycling policy in Brazilian cities. The São Paulo case study combines large-scale crowdsourced mobility data with official road geometry to answer two core questions:

- **Where are the structural gaps** in the cycling network - roads that, if upgraded, would most benefit high-demand unprotected segments?

The methodology builds a merged geospatial dataset of OSM road edges enriched with Strava trip counts and infrastructure presence flags. Each analysis module scores and ranks road segments along a different dimension of priority.

---

## Repository Structure

```
0IC_2026/
├── map_utils.py                           # Shared utilities: data loading, merging, colormaps, outlier detection
├── strava_eda.py                          # Exploratory data analysis
├── analysis_gap.py                        # Priority gap ranking
├── analises_strava_EDA.ipynb             # Main organized EDA notebook
├── collect_and_evaluate_choicesets.ipynb  # Route choice set collection and evaluation
└── requirements.txt                       # Python dependencies
```

> Output files (HTML maps, PNG charts, `RELATORIO.txt`) are written to `analises/`. Intermediate geospatial computations are cached in `cache/`.

---

## Data

### Strava Metro (`data/strava/`)

Aggregated, anonymized trip data licensed through the **Strava Metro** program. It includes:
- **Edge trip counts** - number of trips traversing each road segment (forward and reverse directions), with commute/leisure and demographic splits.
- **Origin-destination flows** - aggregated OD pairs for January 2023.

> **These data are private** and are not included in this repository. Access requires a Strava Metro agreement.

### OSM Road Network (`data/malha_sp_ops/`)

Road network geometry and attributes for São Paulo sourced from **OpenStreetMap** via the `osmnx` library. Includes road type (`highway`), speed limits (`maxspeed`), lane counts, and a manually compiled cycling infrastructure inventory (`tem_infra`, `tipo_ciclovia`).

---

## How to Run

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

Or install the core packages manually:

```bash
pip install pandas numpy geopandas folium matplotlib seaborn networkx branca shapely osmnx
```

**2. Run a specific analysis**

```bash
python strava_eda.py
python analysis_connectivity.py
python analysis_demographics.py
python analysis_gap.py
python analysis_hex_coverage.py
python analysis_speed_safety.py
```

**3. Run the notebooks**

Launch Jupyter and run cells sequentially:

```bash
jupyter notebook
```

- `analises_strava_EDA.ipynb` - main organized EDA
- `collect_and_evaluate_choicesets.ipynb` - route choice set evaluation

All outputs are saved to `analises/`.

---

## Acknowledgements

This project was supported by **FAPESP** (grant no. 2025/21593-7) and carried out at the **Instituto de Ciências Matemáticas e de Computação (ICMC)**, **Universidade de São Paulo (USP)**, as part of the **BikeScience** project.
