# 🛰️ AeroRescue-AI: Space-Based Flood Damage & Isolation Intelligence

> **Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from Space**  
> *Rapid, verifiable disaster assessment and road network reachability analysis using cloud-penetrating Sentinel SAR, optical satellite imagery, and grounded LLM agent copilots.*

---

## 📌 Executive Summary

When catastrophic flooding hits high-altitude corridors like the Himalayas, ground communication infrastructure, bridges, and roads are often swept away. **AeroRescue-AI** turns raw, free satellite data into actionable intelligence for disaster rescue teams.

The system automatically ingests pre- and post-event **Sentinel-1 SAR** and **Sentinel-2 Optical** imagery, overlays pre-disaster vector maps (**OpenStreetMap via ohsome API**), computes infrastructure damage, and runs graph pathfinding algorithms to identify cut-off settlements. An integrated **Bilingual Situation-Report Copilot** generates strictly grounded reports in English and Nepali, backed by deterministic tool execution.

---

## ✨ Key Features

- **📡 Multimodal Imagery Fusion**: Fuses Sentinel-1 (C-band SAR) for cloud/monsoon-penetrating change detection with Sentinel-2 optical imagery for high-resolution damage verification.
- **🕸️ Graph Topology Reachability**: Converts pre-event OSM road networks into graph structures (NetworkX) to flag villages with zero active road connectivity to hospitals or towns.
- **🤖 Zero-Hallucination Agentic Copilot**: Powered by an LLM agent with structured tool-calling; every statistic in generated situation reports is strictly pulled from spatial map execution.
- **🏔️ Hydrological Flow Path Tracing**: Uses Copernicus DEM elevation contours to simulate downstream flood/debris propagation paths from any user-defined upstream point.
- **🗺️ Interactive GIS Dashboard**: Streamlit/Pydeck frontend allowing rescue operators to interactively inspect damage layers, override path parameters (Human-in-the-Loop), and download situation reports.
- **📊 Benchmarked Accuracy**: Validated against the official **Copernicus Emergency Management Service (EMSR927)** reference activation for the August 2026 Trishuli flood.

---

## 🏗️ System Architecture

```text
               +----------------------------------+
               |  Copernicus / OpenStreetMap API  |
               +-----------------+----------------+
                                 |
                                 v
                     +-----------+-----------+
                     |  Data Engine & ETL    |
                     | (Co-registration &    |
                     |  Orbit Alignment)     |
                     +-----------+-----------+
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
    +------------+------------+     +------------+------------+
    |  ML Flood & Debris      |     |  OSM Vector Graph Engine|
    |  Segmentation (SAR/S2)  |     |  (Dijkstra / Pathfinding|
    +------------+------------+     +------------+------------+
                 |                               |
                 +---------------+---------------+
                                 |
                                 v
                     +-----------+-----------+
                     | Spatial Overlay Engine|
                     |  (Damaged Roads/Cut-  |
                     |   Off Communities)    |
                     +-----------+-----------+
                                 |
                 +---------------+---------------+
                 |                               |
                 v                               v
    +------------+------------+     +------------+------------+
    | Interactive Map         |     | Agentic Copilot        |
    | Dashboard UI            |     | (EN/NP Grounded Reports|
    +-------------------------+     +-------------------------+
```

## Folder Structure

```text
SentinalGrid/
│
├── .github/
│   └── workflows/
│       ├── ci-cd.yml                # Automated linting, formatting, and unit tests
│       └── docker-build.yml         # Container build and registry push pipelines
│
├── data/                            # Local storage (git-ignored, versioned via DVC)
│   ├── raw/
│   │   ├── sentinel1/               # Pre/post event Sentinel-1 GRD SAR imagery
│   │   ├── sentinel2/               # Pre/post event Sentinel-2 optical imagery
│   │   ├── dem/                     # Copernicus DEM tiles
│   │   └── osm/                     # Pre-event OpenStreetMap snapshots via ohsome API[cite: 2]
│   ├── processed/
│   │   ├── co_registered/           # Aligned rasters on identical orbit tracks[cite: 2]
│   │   └── vector_layers/           # GeoJSON/GeoPackage outputs (buildings, roads)[cite: 2]
│   └── reference/
│       └── emsr927/                 # Copernicus EMS EMSR927 maps for checking/validation[cite: 2]
│
├── config/
│   ├── config.yaml                  # Global spatial parameters, bounding boxes, date ranges[cite: 2]
│   ├── agent_prompts.yaml           # System prompts for English & Nepali Copilot generation[cite: 2]
│   └── logging_config.yaml          # Structured logging and tracing configurations
│
├── models/
│   ├── checkpoints/                 # Fine-tuned segmentation model weights (KuroSiwo/Sen1Floods11)[cite: 2]
│   └── artifact_metadata.json       # Version control metadata for active model artifacts
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_engine/                 # Module 1: Ingestion & Geospatial Preprocessing
│   │   ├── __init__.py
│   │   ├── copernicus_downloader.py # Ingests Sentinel-1/2 rasters via Copernicus API[cite: 2]
│   │   ├── osm_extractor.py         # Pulls pre-event snapshot vector datasets via ohsome API[cite: 2]
│   │   ├── sar_processing.py        # Co-registration, orbit track matching & backscatter delta[cite: 2]
│   │   └── raster_aligner.py        # Spatial reprojection and clipping to ROI
│   │
│   ├── ml_pipeline/                 # Module 2: Flood/Debris Segmentation
│   │   ├── __init__.py
│   │   ├── inference.py             # Model loading and tile-based raster inference
│   │   ├── postprocessing.py        # Vectorization, thresholding, and polygon simplification
│   │   └── model_wrapper.py         # Standardized PyTorch/ONNX model interface
│   │
│   ├── spatial_analysis/            # Module 3: Impact Analysis & Network Routing
│   │   ├── __init__.py
│   │   ├── damage_overlay.py        # Spatial intersections (flooded buildings & damaged roads)[cite: 2]
│   │   ├── network_graph.py         # Converts OSM road networks into graph structures (NetworkX)[cite: 2]
│   │   ├── isolation_routing.py     # Graph reachability analysis to identify cut-off settlements[cite: 2]
│   │   └── hydrology_tracing.py     # DEM flow-path modeling downstream from user points[cite: 2]
│   │
│   ├── agentic_copilot/             # Module 4: Grounded Agent & Tool Execution
│   │   ├── __init__.py
│   │   ├── agent.py                 # Core LangGraph / LlamaIndex agent orchestration
│   │   ├── tools.py                 # Deterministic query tools (fetches metrics directly from Module 3)[cite: 2]
│   │   ├── report_generator.py      # Structured bilingual (EN/NP) situation report generation[cite: 2]
│   │   └── hitl_manager.py          # Human-in-the-Loop review & confidence threshold overrides
│   │
│   └── evaluation/                  # Module 5: Benchmarking & Observability
│       ├── __init__.py
│       ├── emsr927_benchmarker.py   # Computes IoU/F1 scores against official reference maps[cite: 2]
│       └── metrics.py               # Groundedness checks (verifies LLM stats vs spatial data)[cite: 2]
│
├── app/                             # Web Frontend & Dashboard Execution
│   ├── assets/                      # Static assets, styles, and maps
│   ├── components/
│   │   ├── map_viewer.py            # Streamlit/Pydeck interactive GIS map renderer
│   │   ├── copilot_chat.py          # Chat interface with tool execution traces
│   │   └── hitl_controls.py         # Human validation and override controls
│   └── main.py                      # Main entrypoint for the dashboard web application[cite: 2]
│
├── tests/                           # Continuous Integration Test Suite
│   ├── test_data_engine.py          # Verifies SAR processing and coordinate systems
│   ├── test_isolation_routing.py    # Unit tests for graph reachability algorithms[cite: 2]
│   └── test_agent_grounding.py      # Asserts zero hallucinated stats in generated reports[cite: 2]
│
├── scripts/                         # Command Line Utilities
│   ├── run_pipeline.py              # Full CLI command to run pipeline end-to-end for any date/area[cite: 2]
│   └── evaluate_model.py            # Script to run EMSR927 benchmarking suite[cite: 2]
│
├── .dvcignore                       # Data Version Control ignore file
├── .gitignore                       # Standard Git ignore file
├── Dockerfile                       # Multi-stage production container build configuration
├── docker-compose.yml               # Local orchestration for app, GIS tools, and agent logging
├── Makefile                         # Shortcuts for setup, testing, pipeline execution, and deployment
├── pyproject.toml                   # Project metadata and dependencies (Poetry / UV setup)
├── README.md                        # Documentation, architecture guide, and setup instructions[cite: 2]
└── requirements.txt                 # Frozen production dependency locks