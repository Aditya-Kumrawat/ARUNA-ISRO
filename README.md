# ☀️ ARUNA
### Aditya-L1 Real-time Unified Nowcasting & Alert System

**Foresee. Prepare. Protect.**

ARUNA is an open-source, physics-grounded solar flare nowcasting and forecasting platform built around India's Aditya-L1 solar observation mission. It aims to transform multi-instrument solar observations into interpretable flare intelligence, probabilistic forecasts, and operational space-weather alerts.

By combining observations from **SoLEXS and HEL1OS**, physics-informed diagnostics, time-series forecasting, historical event retrieval, and impact assessment, ARUNA explores a unified approach to anticipating solar activity and understanding its potential consequences for Earth-dependent infrastructure.

<p align="center">
  <img src="docs/assets/aruna-banner.png" alt="ARUNA — Solar Intelligence Platform" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Space%20Weather-blue">
  <img src="https://img.shields.io/badge/Mission-Aditya--L1-orange">
  <img src="https://img.shields.io/badge/Status-Active%20Development-yellow">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green">
</p>

---

## 🌍 The Problem

Solar flares and associated space-weather events can affect satellite operations, navigation systems, radio communications, and human spaceflight.

Although solar observation missions continuously collect valuable data, converting those observations into timely, interpretable and operationally useful intelligence remains a challenging engineering problem.

ARUNA explores how this gap can be addressed through an integrated scientific computing pipeline that combines instrument-level observations, physical diagnostics, predictive models, and impact-oriented reporting.

## 🚀 What is ARUNA?

ARUNA is designed as a multi-layered solar intelligence system with four primary operational objectives:

| Capability | Objective |
|---|---|
| WATCH | Continuously monitor solar activity and instrument observations. |
| WARN | Identify developing flare events and estimate their evolution. |
| BRIEF | Translate solar activity into potential infrastructure-level risks. |
| REPORT | Generate interpretable post-event analysis and diagnostics. |

## 🛰️ Scientific Foundation

ARUNA is built around two complementary instruments aboard Aditya-L1:

**1. SoLEXS — Solar Low Energy X-ray Spectrometer**

- Observes soft X-ray emissions.
- Provides insight into thermal plasma evolution.
- Supports temperature and emission-measure diagnostics.

**2. HEL1OS — High Energy L1 Orbiting X-ray Spectrometer**

- Observes hard X-ray emissions.
- Provides information about energetic solar flare processes.
- Complements SoLEXS observations for thermal–nonthermal analysis.

The central research direction is to treat these instruments as complementary observation streams rather than analysing their measurements independently.

## ⚙️ System Architecture

The proposed ARUNA architecture is organised into nine processing layers.

| Layer | Component | Responsibility |
|---|---|---|
| L0 | Source Layer | Scientific data source management |
| L1 | Data Ingestion | FITS ingestion, integrity checks and time alignment |
| L2 | Physics Engine | Plasma diagnostics and physical feature extraction |
| L3 | Nowcasting Engine | Flare detection and event classification |
| L4 | Forecasting Engine | Probabilistic time-series forecasting |
| L5 | Historical Retrieval | Similarity-based historical event retrieval |
| L6 | Threat Intelligence | Potential infrastructure impact assessment |
| L7 | LLM Briefing | Structured, interpretable alert generation |
| L8 | Dashboard Services | Visualisation and operational interfaces |

The architecture also defines ARUNA-XL as an extension for broader solar, heliophysical and Earth-impact intelligence.

## 🧠 Key Research Directions

**Physics-informed diagnostics**
- Plasma temperature estimation.
- Emission measure evolution.
- Dynamic Neupert Integral (DNI).
- Thermal–nonthermal phase-space analysis.
- Spectral hardness evolution.

**Solar flare intelligence**
- Flare onset and evolution analysis.
- GOES-class event classification.
- Precursor identification.
- Event reconstruction and historical comparison.

**Predictive intelligence**
- Time-series forecasting.
- Multi-stream model integration.
- Probabilistic predictions.
- Uncertainty estimation and calibration.

**Space-weather impact assessment**
- NavIC-related risk assessment.
- HF communication disruption analysis.
- Satellite operational risk indicators.
- Human spaceflight radiation-risk considerations.

*These are architectural and research objectives; implementation and validation status varies by module.*

## 🔬 Technology Stack

The repository is organised around scientific computing, machine learning, backend services and visualisation.

The intended technology ecosystem includes:

- **Scientific computing:** Python, NumPy, SciPy, Astropy
- **Astronomical data:** FITS, SunPy and relevant space-weather data sources
- **Machine learning:** PyTorch and time-series forecasting approaches
- **Forecast calibration:** MAPIE
- **Similarity retrieval:** FAISS and Dynamic Time Warping
- **Backend and data services:** API services and event-driven processing
- **Visualisation:** Streamlit and Plotly
- **Deployment:** Docker and containerised services

Refer to the individual modules and configuration files for the actual dependencies and implementation status.

## 📁 Repository Structure

```text
ARUNA-ISRO/
│
├── .agents/                # Agent and development instructions
├── api/                    # API services
├── config/                 # Configuration files
├── data/                   # Data resources and datasets
├── docs/                   # Documentation
├── models/                 # Model-related components
├── notebooks/              # Research and experimentation
├── python/                 # Python implementation
├── web/                    # Web interface
│
├── ARUNA_Pipeline_PRD.md   # Engineering pipeline specification
├── TRAIN_FORECAST.md       # Forecasting documentation
├── eval_model.py           # Model evaluation entry point
├── fit_mapie.py            # Forecast calibration utility
├── docker-compose.yml      # Container configuration
├── docker-compose.dev.yml  # Development configuration
└── .env.example            # Environment template
```

## 🛠️ Getting Started

### Prerequisites

- Python 3.10+
- Git
- Docker and Docker Compose (for supported container workflows)

### Clone the repository

```bash
git clone https://github.com/Aditya-Kumrawat/ARUNA-ISRO.git

cd ARUNA-ISRO
```

### Configure the environment

```bash
cp .env.example .env
```

Configure the required environment variables according to your local setup.

### Install dependencies

Install the dependencies specified by the relevant module's dependency manifest. Setup instructions will be expanded as the individual components are standardised.

### Run

Refer to the module-specific documentation and deployment configuration for the appropriate entry points.

> Note: ARUNA is under active development. A complete, verified end-to-end installation workflow is not yet documented.

## 📊 Research & Engineering Documentation

The repository includes a detailed engineering specification covering:

- End-to-end processing architecture.
- Data contracts and validation.
- Instrument data ingestion.
- Physics and forecasting engines.
- Historical event retrieval.
- Threat intelligence.
- API and deployment architecture.
- Monitoring, testing and validation.

Start here:

- [Engineering Pipeline PRD](ARUNA_Pipeline_PRD.md)
- [Forecasting Documentation](TRAIN_FORECAST.md)

## 🧪 Validation and Scientific Responsibility

Space-weather predictions can influence safety-critical decisions.

ARUNA is a research and engineering project, not an officially operational ISRO warning service.

Predictions, risk assessments and generated briefings should be independently validated against authoritative scientific observations before being used for operational decisions.

The project does not claim official ISRO affiliation, endorsement or operational deployment.

## 🤝 Contributing

Contributions from researchers, developers, astronomers and space-technology enthusiasts are welcome.

Potential contribution areas include:

- Solar data processing and calibration.
- Physics-informed modelling.
- Time-series forecasting.
- Scientific validation and benchmarking.
- API development and system integration.
- Visualisation and documentation.

Suggested contribution workflow:

1. Fork the repository.
2. Create a feature branch.
3. Implement and test your changes.
4. Submit a pull request with a clear technical explanation.

## 🗺️ Roadmap

- [ ] Standardise end-to-end local setup.
- [ ] Strengthen SoLEXS and HEL1OS data ingestion.
- [ ] Validate instrument-level preprocessing.
- [ ] Benchmark physics-informed flare detection.
- [ ] Evaluate probabilistic forecasting.
- [ ] Expand historical event retrieval.
- [ ] Validate infrastructure impact assessment.
- [ ] Improve dashboard and alert integration.
- [ ] Publish reproducible evaluation benchmarks.

## 👨‍💻 Maintainer

**Aditya Kumrawat**

Creator and primary maintainer of ARUNA.

GitHub: [@Aditya-Kumrawat](https://github.com/Aditya-Kumrawat)

Repository: [ARUNA-ISRO](https://github.com/Aditya-Kumrawat/ARUNA-ISRO)

## 📜 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>ARUNA — Foresee. Prepare. Protect.</b><br>
  Exploring the future of physics-grounded solar intelligence.
</p>
