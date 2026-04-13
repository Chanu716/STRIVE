# STRIVE — Product Requirements Document

**Document version:** 1.0.0
**Status:** Draft
**Author:** Karri Chanikya Sri Hari Narayana Dattu
**Last updated:** 2026-04-13

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Goals & Non-Goals](#3-goals--non-goals)
4. [Stakeholders](#4-stakeholders)
5. [User Personas & Use Cases](#5-user-personas--use-cases)
6. [Functional Requirements](#6-functional-requirements)
7. [System Architecture](#7-system-architecture)
8. [Data Requirements](#8-data-requirements)
9. [Machine Learning Requirements](#9-machine-learning-requirements)
10. [API & Integration Requirements](#10-api--integration-requirements)
11. [Non-Functional Requirements](#11-non-functional-requirements)
12. [Security & Privacy Requirements](#12-security--privacy-requirements)
13. [Observability & Operational Requirements](#13-observability--operational-requirements)
14. [Testing & Quality Requirements](#14-testing--quality-requirements)
15. [Deployment & Infrastructure Requirements](#15-deployment--infrastructure-requirements)
16. [Risks & Mitigations](#16-risks--mitigations)
17. [Success Metrics & Acceptance Criteria](#17-success-metrics--acceptance-criteria)
18. [Glossary](#18-glossary)
19. [Revision History](#19-revision-history)

---

## 1. Executive Summary

STRIVE (Spatio-Temporal Risk Intelligence and Vehicular Safety Engine) is a production-grade, real-time safety platform for road networks. It continuously fuses historical crash records, live weather telemetry, and real-time traffic data to produce per-road-segment risk scores and safety-optimised route recommendations — all with sub-300 ms end-to-end latency and full explainability.

STRIVE is designed as a headless backend platform exposed via REST and WebSocket APIs. It is intended to be embedded in:

- Consumer and commercial navigation applications
- Fleet safety and telematics platforms
- Urban mobility and traffic management systems
- Emergency dispatch and first-responder tools
- Insurance telematics pipelines

The platform differentiates itself through three pillars:

1. **Accuracy**: A hybrid Graph Neural Network + Temporal Fusion Transformer architecture that captures road topology, temporal dynamics, and environmental context simultaneously.
2. **Explainability**: Every risk score is accompanied by SHAP factor attributions and natural-language summaries, enabling driver trust and regulatory compliance.
3. **Operability**: A fully containerised, cloud-native architecture built for continuous operation at scale, with automated model retraining and comprehensive observability.

---

## 2. Problem Statement & Motivation

### 2.1 The Safety Gap in Navigation

Modern navigation systems (Google Maps, Apple Maps, Waze) optimise for travel time and distance. Road safety is at best a secondary concern, addressed reactively through live incident overlays. There is no system today that:

- **Proactively scores** the safety of every road segment in real time
- **Explains** why a segment is risky in human-readable terms
- **Routes** around risk while offering a transparent risk/time trade-off

### 2.2 Scale of the Problem

- Road traffic crashes cause **1.35 million deaths** per year globally (WHO, 2023)
- **50 million** more are injured or disabled annually
- Economic cost exceeds **3% of GDP** in most high-income countries
- The majority of crashes involve a combination of speed, weather, and road conditions — all of which are **predictable and quantifiable**

### 2.3 Data Availability

The convergence of three data trends makes STRIVE viable today:

1. **Open crash databases** (FARS, STATS19, city open data) provide decades of labelled incident records
2. **Real-time weather and traffic APIs** provide sub-minute environmental context
3. **Open road network data** (OSM, HERE) provides the structural graph STRIVE needs

### 2.4 Opportunity

A system that quantifies road-segment risk in real time, explains it to users, and routes around it can:

- Reduce crash rates on adopting platforms by an estimated 15–25 % (based on analogous telematics studies)
- Enable data-driven infrastructure investment by municipalities
- Support actuarially sound dynamic insurance pricing
- Improve fleet safety outcomes and reduce employer liability

---

## 3. Goals & Non-Goals

### 3.1 Goals

| ID | Goal |
|---|---|
| G-01 | Provide per-road-segment risk scores updated every ≤ 30 seconds |
| G-02 | Return a safety-optimised route between any two coordinates within ≤ 300 ms |
| G-03 | Explain every risk score with factor-level attributions (SHAP) |
| G-04 | Achieve AUROC ≥ 0.88 on the held-out incident detection task |
| G-05 | Support ≥ 10,000 concurrent API clients at P99 ≤ 300 ms |
| G-06 | Maintain ≥ 99.9 % API availability (three nines) |
| G-07 | Continuously retrain the model as new incident data arrives |
| G-08 | Expose results via a documented REST and WebSocket API |
| G-09 | Support deployment in any cloud provider via Kubernetes + Helm |
| G-10 | Be fully observable with Prometheus metrics, structured logs, and traces |

### 3.2 Non-Goals

| ID | Non-Goal | Rationale |
|---|---|---|
| NG-01 | Consumer-facing map UI | STRIVE is a backend API platform |
| NG-02 | In-vehicle hardware / ADAS | Requires different certification scope |
| NG-03 | Pedestrian or cyclist modes (v1) | Planned for v2; different feature sets required |
| NG-04 | Real-time video analysis | Requires camera infrastructure; separate system |
| NG-05 | Lane-level precision | H3 cell + road-segment granularity is sufficient for v1 |
| NG-06 | Driver behaviour scoring (v1) | Planned for v2 with mobile SDK telemetry |

---

## 4. Stakeholders

| Stakeholder | Role | Interest |
|---|---|---|
| **Product Owner** | Karri Chanikya Sri Hari Narayana Dattu | Define requirements, accept deliverables |
| **Navigation App Partners** | API consumers | Embed STRIVE risk scores in routing UX |
| **Fleet Operators** | API consumers | Monitor driver safety, reduce incidents |
| **Municipal Traffic Authorities** | Data consumers | Infrastructure planning, signal optimisation |
| **Insurance Partners** | Data consumers | Risk-based pricing, claims intelligence |
| **Emergency Dispatch** | API consumers | Route first responders safely |
| **ML Engineering Team** | Builders | Model development, training infrastructure |
| **Platform Engineering Team** | Builders | Streaming pipeline, APIs, infrastructure |
| **Data Engineering Team** | Builders | Data ingestion, storage, quality |
| **Security Team** | Reviewers | Data privacy, access control, pen-testing |
| **Legal / Compliance** | Reviewers | GDPR, CCPA, liability |

---

## 5. User Personas & Use Cases

### 5.1 Personas

#### P1 — Commuter Driver
- Drives in an unfamiliar city during bad weather
- Wants to know *why* a route was changed, not just that it was
- Needs clear, non-technical risk explanations

#### P2 — Fleet Safety Manager
- Manages 200+ vehicles; responsible for driver safety KPIs
- Wants API access to risk scores for post-trip analysis
- Needs aggregate risk heatmaps for driver coaching

#### P3 — Urban Mobility Planner
- Analyses crash hotspots to prioritise road improvements
- Needs historical risk data by road segment and time period
- Wants GeoJSON / MVT tile output for GIS integration

#### P4 — Insurance Actuary
- Prices personal and commercial auto policies
- Needs per-segment risk scores correlated with claims data
- Requires well-calibrated probability estimates, not just rankings

#### P5 — Emergency Dispatcher
- Routes ambulances and fire engines to incident scenes
- Cannot afford to send responders into secondary incidents
- Needs lowest-risk route with ≤ 100 ms latency

### 5.2 Use Cases

| ID | Use Case | Persona | Priority |
|---|---|---|---|
| UC-01 | Request a safety-optimised driving route | P1 | P0 |
| UC-02 | Retrieve real-time risk score for a segment | P2, P5 | P0 |
| UC-03 | Subscribe to live risk updates for a bounding box | P2, P3 | P0 |
| UC-04 | Get risk explanation for a segment | P1, P2 | P0 |
| UC-05 | Download risk heatmap as GeoJSON / MVT | P3 | P1 |
| UC-06 | Query historical risk time series for a segment | P3, P4 | P1 |
| UC-07 | Batch score a set of route options | P2 | P1 |
| UC-08 | Register for risk alerts (geofenced threshold) | P2 | P2 |
| UC-09 | Export segment risk data for actuarial analysis | P4 | P2 |
| UC-10 | Admin: trigger manual model retraining | Internal | P2 |

---

## 6. Functional Requirements

### 6.1 Risk Scoring Engine

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | The system SHALL produce a risk score in the range [0, 100] for every active road segment | P0 |
| FR-02 | Risk scores SHALL be updated at a maximum interval of 30 seconds | P0 |
| FR-03 | Each risk score SHALL include a 90 % confidence interval | P0 |
| FR-04 | Risk scores SHALL incorporate live weather data updated within the last 5 minutes | P0 |
| FR-05 | Risk scores SHALL incorporate traffic speed/density data updated within the last 30 seconds | P0 |
| FR-06 | Risk scores SHALL incorporate historical incident frequency for the same (segment, hour, day-of-week) | P0 |
| FR-07 | The system SHALL classify risk into four levels: LOW (0–25), MODERATE (26–50), HIGH (51–75), CRITICAL (76–100) | P0 |
| FR-08 | The system SHALL support scoring for road networks in any city where OSM data and at least one traffic feed are available | P1 |

### 6.2 Explainability

| ID | Requirement | Priority |
|---|---|---|
| FR-09 | Every risk score response SHALL include the top-5 contributing features with their SHAP values | P0 |
| FR-10 | Every risk score response SHALL include a human-readable natural-language summary of the top factors | P0 |
| FR-11 | The SHAP computation SHALL complete within 60 ms at P99 | P0 |
| FR-12 | The system SHALL expose a `/explain` endpoint that returns the full SHAP explanation for a given segment | P1 |
| FR-13 | Explanation summaries SHALL be available in English; additional languages are a P2 requirement | P2 |

### 6.3 Safety-Aware Routing

| ID | Requirement | Priority |
|---|---|---|
| FR-14 | The system SHALL accept an origin, destination, and safety weight parameter alpha ∈ [0, 1] | P0 |
| FR-15 | The routing algorithm SHALL return a route that minimises a combined cost function: `cost = alpha × risk + (1 − alpha) × travel_time` | P0 |
| FR-16 | The route response SHALL include total risk score, distance, duration, and per-segment risk breakdown | P0 |
| FR-17 | The route response SHALL compare the returned route to the fastest alternative (extra distance, time, risk reduction) | P0 |
| FR-18 | The routing engine SHALL support up to 10 intermediate waypoints | P1 |
| FR-19 | The routing engine SHALL return a route within 300 ms at P99 | P0 |
| FR-20 | The system SHALL support driving, motorcycle, and heavy-goods-vehicle modes (different risk profiles) | P2 |

### 6.4 Data Ingestion

| ID | Requirement | Priority |
|---|---|---|
| FR-21 | The system SHALL ingest live weather data from at least one source at ≤ 5-minute intervals | P0 |
| FR-22 | The system SHALL ingest traffic speed/density from at least one source at ≤ 30-second intervals | P0 |
| FR-23 | The system SHALL ingest new incident reports within 60 seconds of their occurrence | P1 |
| FR-24 | All raw ingested data SHALL be written to the data lake (MinIO) before processing | P1 |
| FR-25 | Malformed or schema-violating records SHALL be routed to a dead-letter queue | P0 |
| FR-26 | The ingestion pipeline SHALL deduplicate events using a Bloom filter | P0 |

### 6.5 Streaming & Real-Time

| ID | Requirement | Priority |
|---|---|---|
| FR-27 | The system SHALL provide a WebSocket endpoint for real-time risk-score updates | P0 |
| FR-28 | WebSocket clients SHALL receive risk delta updates for their subscribed bounding box every 30 seconds | P0 |
| FR-29 | The system SHALL support ≥ 10,000 concurrent WebSocket connections | P0 |
| FR-30 | The system SHALL use Kafka as the internal event backbone with at-least-once delivery semantics | P0 |

### 6.6 Model Lifecycle

| ID | Requirement | Priority |
|---|---|---|
| FR-31 | The system SHALL retrain the risk model at least once every 24 hours on a rolling data window | P0 |
| FR-32 | A new model SHALL only be promoted to production if its AUROC is ≥ the current production model's AUROC | P0 |
| FR-33 | All training runs SHALL be tracked in MLflow with full hyperparameter and metric logging | P0 |
| FR-34 | Model artefacts SHALL be versioned and stored in MinIO | P0 |
| FR-35 | The system SHALL support shadow deployment (A/B testing) of candidate models | P1 |

---

## 7. System Architecture

### 7.1 Architectural Principles

1. **Event-driven**: All data flows through Kafka; services are decoupled and independently scalable
2. **Cache-first**: Redis caching at every I/O boundary to meet latency targets
3. **Immutable data lake**: All raw inputs written to MinIO before any transformation
4. **Stateless services**: All API services are stateless; state lives in Redis and PostgreSQL
5. **Defence in depth**: Auth at gateway, service mesh (mTLS), and database-level encryption
6. **Observability by default**: Every service emits Prometheus metrics, structured JSON logs, and OpenTelemetry traces

### 7.2 Component Inventory

| Component | Technology | Purpose |
|---|---|---|
| API Gateway | Kong 3.x | Ingress, auth, rate-limiting, TLS termination |
| Route Service | Python / FastAPI | Safety-aware A\* routing |
| Risk Score Service | Python / FastAPI | Risk score reads from Redis |
| Explanation Service | Python / FastAPI | SHAP computation and NL generation |
| Ingestion Workers | Python / Faust | Raw event parsing and validation |
| Feature Engineering | Python / Faust + PySpark | Streaming aggregations and spatial joins |
| Inference Engine | Python / ONNX Runtime | GNN + TFT model inference |
| Retraining Pipeline | Apache Airflow + MLflow | Nightly training, evaluation, promotion |
| TimescaleDB | PostgreSQL 16 + TimescaleDB | Time-series incident, weather, traffic storage |
| Redis Cluster | Redis 7.2 | Feature vector and risk score cache |
| PostGIS | PostgreSQL 16 + PostGIS | Road network geometries and spatial queries |
| Neo4j | Neo4j 5.x | Road network graph traversal for routing |
| MinIO | MinIO (S3-compatible) | Data lake and model artefact storage |
| Kafka | Apache Kafka 3.6 | Internal event streaming backbone |
| Kafka Schema Registry | Confluent Schema Registry | Avro schema versioning |
| Prometheus | Prometheus 2.x | Metrics collection |
| Grafana | Grafana 10.x | Dashboards and alerting |
| OpenTelemetry | OTEL Collector | Distributed tracing |

### 7.3 Data Flow Summary

```
External APIs → Ingestion Workers → Kafka (raw-events)
→ Validation → Kafka (clean-events)
→ Spatial Enrichment → Kafka (enriched-events)
→ Feature Engineering → Redis + TimescaleDB (feature store)
→ Inference Engine → Redis (risk scores)
→ Risk Score Service (API reads from Redis)
→ Route Service (uses risk scores for A*)
→ Explanation Service (reads scores + features, computes SHAP)
→ Client (REST response or WebSocket push)
```

---

## 8. Data Requirements

### 8.1 Data Sources

#### 8.1.1 Historical Incident Data

| Source | Format | Update Frequency | Coverage | Licence |
|---|---|---|---|---|
| NHTSA FARS | CSV | Annual | US (fatal crashes) | Public domain |
| UK STATS19 | CSV | Annual | Great Britain | OGL v3 |
| City open data portals | CSV / GeoJSON | Varies | City-specific | Varies |
| Waze CCP | JSON feed | Near real-time | Global (urban) | Partner agreement |

**Minimum historical data requirement:** 3 years of incident records for any city before STRIVE can produce reliable predictions for that city.

#### 8.1.2 Weather Data

| Source | Format | Update Frequency | Resolution |
|---|---|---|---|
| OpenWeatherMap Current | REST JSON | 5 min (cached) | City-level |
| NOAA ASOS (METARs) | Text / JSON | 5 min | Station-level (~50 km) |
| Tomorrow.io Nowcasting | REST JSON | 1 min | 1 km grid |

**Minimum required fields:** precipitation (mm/h), visibility (km), wind speed (m/s), temperature (°C), weather condition code.

#### 8.1.3 Traffic Data

| Source | Format | Update Frequency | Coverage |
|---|---|---|---|
| HERE Traffic Flow | REST JSON | 15 s | Global roads |
| Waze CCP incidents | JSON feed | Real-time | Urban global |
| GTFS-RT vehicle positions | Protobuf | 30 s | Transit systems |

**Minimum required fields:** observed speed (km/h), free-flow speed (km/h), traffic jam factor.

#### 8.1.4 Road Network Data

| Source | Format | Update Frequency | Use |
|---|---|---|---|
| OpenStreetMap (Overpass) | JSON / PBF | Weekly | Road geometry, class, restrictions |
| HERE Map Content | Tiles | Monthly | Speed limits, lane count |
| TIGER/Line (US) | Shapefile | Annual | Administrative boundaries |

### 8.2 Data Schema Standards

All internal data MUST conform to Avro schemas registered in the Confluent Schema Registry. Schema evolution MUST follow backward-compatibility rules. Field additions are permitted; field removals and type changes require a major version bump and a migration plan.

### 8.3 Data Quality Requirements

| Dimension | Requirement |
|---|---|
| Completeness | ≥ 95 % of expected records received within the SLA window |
| Timeliness | Weather records ≤ 5 min stale; traffic records ≤ 30 s stale |
| Accuracy | Lat/lon snapped to road segment with ≤ 50 m error |
| Uniqueness | Deduplication rate ≥ 99.9 % (Bloom filter FPR ≤ 0.1 %) |
| Validity | Schema conformance ≥ 99.5 %; remaining 0.5 % routed to DLQ |

### 8.4 Data Retention

| Store | Retention | Rationale |
|---|---|---|
| Raw data lake (MinIO) | 5 years | Training data, audit trail |
| TimescaleDB (features) | 7 days (hot), 90 days (warm) | Streaming window + historical queries |
| Redis (hot cache) | 30–120 s (TTL) | Real-time serving |
| Kafka topics | 24 h – 7 d (per topic) | Replay and backfill |
| MLflow artefacts | Indefinite | Model versioning |

### 8.5 Data Privacy

- No Personally Identifiable Information (PII) is collected or stored by STRIVE v1
- Aggregate traffic data is used at road-segment level; no individual vehicle tracking
- Incident data from public sources is already anonymised
- All data at rest is encrypted (AES-256); data in transit uses TLS 1.3

---

## 9. Machine Learning Requirements

### 9.1 Problem Formulation

STRIVE solves two coupled supervised learning tasks:

| Task | Type | Label | Positive Rate |
|---|---|---|---|
| **Incident Detection** | Binary classification | Incident occurred in segment in next 30 min | ~2 % |
| **Severity Estimation** | Ordinal classification | Minor / Serious / Fatal | 70 / 25 / 5 % of positives |

The final risk score is a composite of the incident probability and the expected severity, scaled to [0, 100]:

```
risk_score = 100 × P(incident) × (0.3 × P(minor) + 0.6 × P(serious) + 1.0 × P(fatal))
```

### 9.2 Model Architecture Requirements

| Requirement | Specification |
|---|---|
| Must capture road-network topology | Graph Neural Network component required |
| Must capture temporal patterns (time-of-day, weekday, seasonality) | Temporal encoder with ≥ 48 h lookback required |
| Must support explainability | SHAP-compatible architecture required |
| Must run in ≤ 10 ms at P99 | ONNX export and optimisation required |
| Must support uncertainty quantification | MC-Dropout or conformal prediction required |

### 9.3 Training Data Requirements

| Requirement | Specification |
|---|---|
| Minimum history | 3 years of incident records per city |
| Minimum positive samples | ≥ 10,000 incidents per training run |
| Train / val / test split | 80 / 10 / 10, split chronologically |
| Temporal leakage prevention | Strict time-based split; no future features |
| Class imbalance handling | SMOTE oversampling + focal loss weighting |

### 9.4 Evaluation Requirements

| Metric | Minimum Threshold | Production Target |
|---|---|---|
| AUROC (incident detection) | 0.85 | 0.90 |
| AUPRC | 0.40 | 0.48 |
| F1 @ optimal threshold | 0.55 | 0.63 |
| Expected Calibration Error | ≤ 0.08 | ≤ 0.04 |
| Severity macro-F1 | 0.50 | 0.57 |
| Inference latency P99 | ≤ 15 ms | ≤ 10 ms |

A candidate model MUST meet **all** minimum thresholds before promotion to production.

### 9.5 Continuous Learning Requirements

| Requirement | Specification |
|---|---|
| Retraining frequency | Daily (full) + hourly (micro-batch fine-tune) |
| Promotion gate | New model AUROC ≥ current model AUROC on shared holdout set |
| Rollback trigger | AUROC drop > 0.02 from baseline over 1-hour window |
| Retraining data window | Rolling 30 days |
| A/B testing | Shadow mode: new model scores 5 % of traffic; compare outcomes |

### 9.6 Feature Engineering Requirements

| Requirement | Specification |
|---|---|
| Spatial granularity | H3 resolution 8 (avg cell area ~0.74 km²) |
| Temporal resolution | 30-minute bins |
| Lookback window | 48 hours (96 time steps) |
| Spatial context | Neighbour cell aggregates (k=1 ring) |
| Feature freshness | Live features ≤ 30 s stale at inference time |
| Feature count (v1) | ≥ 24 engineered scalar features + 64-dim GNN embeddings |

---

## 10. API & Integration Requirements

### 10.1 API Design Principles

- RESTful design following OpenAPI 3.1 specification
- All responses in JSON; GeoJSON for geographic data
- Pagination via cursor-based tokens for list endpoints
- Versioning via URL path prefix (`/v1/`, `/v2/`)
- HTTP status codes used correctly (200, 201, 400, 401, 403, 404, 422, 429, 500, 503)
- Rate limiting headers (`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`)

### 10.2 Required Endpoints (v1)

| Method | Path | Description | Auth |
|---|---|---|---|
| `POST` | `/auth/token` | Obtain JWT access token | API key |
| `GET` | `/risk/segment/{segment_id}` | Risk score for one segment | JWT |
| `POST` | `/risk/segments/batch` | Risk scores for up to 500 segments | JWT |
| `GET` | `/risk/heatmap` | GeoJSON / MVT heatmap for a bbox | JWT |
| `GET` | `/risk/history/{segment_id}` | Historical risk time series | JWT |
| `POST` | `/route/safe` | Safety-optimised route | JWT |
| `GET` | `/explain/segment/{segment_id}` | Full SHAP explanation | JWT |
| `WebSocket` | `/risk/stream` | Live risk-score delta stream | JWT |
| `GET` | `/health` | Service health check | None |
| `GET` | `/metrics` | Prometheus metrics | Internal |

### 10.3 Rate Limits

| Tier | Requests/min | WebSocket connections | Price |
|---|---|---|---|
| Free | 60 | 1 | $0 |
| Starter | 600 | 10 | $99/mo |
| Professional | 6,000 | 100 | $499/mo |
| Enterprise | Unlimited | Unlimited | Custom |

### 10.4 Integration Requirements

| Integration | Requirement |
|---|---|
| OpenWeatherMap | Must handle rate limits gracefully; implement exponential back-off |
| HERE Traffic API | Must cache responses for ≥ 15 s to avoid per-call pricing overrun |
| Waze CCP | Must comply with Waze data usage policy; no redistribution of raw data |
| Kafka | At-least-once delivery; idempotent consumer logic required |
| MLflow | All training runs tracked; model registry used for promotion workflow |

### 10.5 SDK Requirements (P1)

The STRIVE team will publish official client libraries in:
- Python (PyPI)
- JavaScript / TypeScript (npm)
- Go (pkg.go.dev)

Each SDK MUST implement: authentication, retry logic, WebSocket management, and typed response models.

---

## 11. Non-Functional Requirements

### 11.1 Performance

| Requirement | Target |
|---|---|
| API P50 latency | ≤ 50 ms |
| API P99 latency | ≤ 300 ms |
| Routing endpoint P99 | ≤ 300 ms |
| SHAP explanation P99 | ≤ 200 ms |
| WebSocket message delay | ≤ 5 s from risk score update |
| Inference throughput | ≥ 50,000 segments / second |
| Kafka consumer lag | ≤ 5 s at peak load |

### 11.2 Availability

| Requirement | Target |
|---|---|
| API availability | ≥ 99.9 % (43.8 min/month downtime budget) |
| Planned maintenance | ≤ 2 h/month, communicated 48 h in advance |
| RTO (Recovery Time Objective) | ≤ 15 minutes |
| RPO (Recovery Point Objective) | ≤ 5 minutes |

### 11.3 Scalability

| Requirement | Target |
|---|---|
| Concurrent API clients | ≥ 10,000 |
| Concurrent WebSocket connections | ≥ 10,000 |
| Road segments scored simultaneously | ≥ 500,000 |
| Cities supported | ≥ 50 in v1 |
| Horizontal scaling | All stateless services auto-scale via HPA on CPU/RPS |

### 11.4 Reliability

| Requirement | Target |
|---|---|
| Data loss on component failure | Zero (Kafka at-least-once + idempotent writes) |
| Stale risk scores on inference failure | Fall back to last known score ≤ 5 min old |
| Stale weather on API outage | Fall back to NOAA climatological baseline |
| Model rollback | Automated; previous model version in Redis within 60 s |

### 11.5 Maintainability

- All services MUST have comprehensive unit tests (≥ 80 % line coverage)
- All services MUST have integration tests covering the full request flow
- All public APIs MUST be documented in OpenAPI 3.1 format
- All configuration MUST be externalised via environment variables or Kubernetes ConfigMaps
- All infrastructure MUST be defined as code (Terraform + Helm)

---

## 12. Security & Privacy Requirements

### 12.1 Authentication & Authorisation

| Requirement | Specification |
|---|---|
| Authentication | OAuth 2.0 client credentials flow; JWT (RS256, 15-min expiry) |
| Authorisation | Role-based: `consumer` (read), `partner` (read + bulk export), `admin` |
| API key storage | Hashed with Argon2id; never stored in plaintext |
| Token refresh | Sliding refresh tokens with 7-day max lifetime |
| IP allowlisting | Available for Enterprise tier |

### 12.2 Transport Security

| Requirement | Specification |
|---|---|
| External TLS | TLS 1.3 minimum; TLS 1.2 accepted for compatibility |
| Internal mTLS | Istio service mesh enforces mTLS between all internal services |
| Certificate rotation | Automated via cert-manager (Let's Encrypt for public; internal CA for mesh) |
| HSTS | Strict-Transport-Security with 1-year max-age |

### 12.3 Data Security

| Requirement | Specification |
|---|---|
| Encryption at rest | AES-256 for all database volumes and MinIO objects |
| Encryption in transit | TLS 1.3 for all external; mTLS for internal |
| PII handling | No PII collected or stored; aggregate data only |
| Data residency | EU data stays in EU regions; US data stays in US regions (Enterprise) |
| Audit logging | All API access logged with IP, user ID, endpoint, timestamp |
| Log retention | 90 days hot, 1 year cold (MinIO) |

### 12.4 Vulnerability Management

| Requirement | Specification |
|---|---|
| Dependency scanning | Dependabot + Snyk on every PR |
| Container scanning | Trivy in CI; block Critical/High CVEs |
| SAST | Bandit (Python) + Semgrep on every PR |
| Penetration testing | Annually by external vendor |
| Secrets scanning | GitLeaks in pre-commit hooks; GitHub secret scanning enabled |

### 12.5 Compliance

| Regulation | Applicability | Control |
|---|---|---|
| GDPR | EU deployments | No PII processed; DPA with all sub-processors |
| CCPA | US deployments | Privacy notice; opt-out mechanism |
| SOC 2 Type II | All customers | Annual audit; controls mapped to STRIVE architecture |

---

## 13. Observability & Operational Requirements

### 13.1 Metrics

Every service MUST expose the following Prometheus metrics:

| Metric Name | Type | Description |
|---|---|---|
| `strive_request_duration_seconds` | Histogram | API request duration by endpoint, status |
| `strive_risk_score_value` | Gauge | Current risk score by segment |
| `strive_kafka_consumer_lag` | Gauge | Kafka consumer lag by topic, partition |
| `strive_feature_freshness_seconds` | Gauge | Age of latest feature vector in cache |
| `strive_model_auroc` | Gauge | Rolling AUROC of production model |
| `strive_inference_duration_ms` | Histogram | ONNX inference duration |
| `strive_data_quality_violations_total` | Counter | Schema/quality violations by source |
| `strive_cache_hit_ratio` | Gauge | Redis cache hit ratio by key pattern |

### 13.2 Alerting

| Alert | Condition | Severity | Action |
|---|---|---|---|
| High API error rate | Error rate > 1 % over 5 min | Critical | Page on-call engineer |
| High API latency | P99 > 500 ms over 5 min | Warning | Notify team channel |
| Kafka consumer lag | Lag > 10,000 messages | Warning | Notify team |
| Model AUROC drop | AUROC < (baseline − 0.02) | Critical | Trigger rollback |
| Feature staleness | Feature age > 5 min | Warning | Check data source |
| Data source outage | No messages received > 10 min | Critical | Page on-call |
| High memory usage | Container memory > 80 % limit | Warning | Check for leaks |

### 13.3 Logging

- All logs MUST be structured JSON
- Required fields: `timestamp` (ISO 8601), `service`, `level`, `trace_id`, `span_id`, `message`
- Logs MUST NOT contain PII, secrets, or raw API keys
- Log levels: ERROR (pages), WARN (alerts), INFO (operational), DEBUG (disabled in production)

### 13.4 Distributed Tracing

- All services MUST instrument with OpenTelemetry SDK
- Traces MUST propagate across Kafka messages (W3C TraceContext headers in record headers)
- P99 of trace ingestion overhead MUST be ≤ 1 ms per span
- Traces retained for 7 days in Jaeger / Tempo

### 13.5 Runbooks

The following runbooks MUST be maintained in `docs/runbooks/`:

- `01-api-high-error-rate.md`
- `02-kafka-consumer-lag.md`
- `03-model-auroc-drop.md`
- `04-data-source-outage.md`
- `05-redis-cluster-failure.md`
- `06-database-failover.md`

---

## 14. Testing & Quality Requirements

### 14.1 Test Coverage

| Test Type | Minimum Coverage | Tooling |
|---|---|---|
| Unit tests | 80 % line coverage | pytest, coverage.py |
| Integration tests | All API endpoints | pytest + testcontainers |
| End-to-end tests | Core user journeys (UC-01 to UC-05) | Playwright / pytest |
| Load tests | 10,000 concurrent users | k6 |
| Model evaluation | Full held-out test set | MLflow evaluation |
| Data quality tests | All data pipeline stages | Great Expectations |

### 14.2 CI/CD Quality Gates

Every pull request MUST pass:

1. Unit tests (0 failures)
2. Integration tests (0 failures)
3. Linting (Ruff, mypy with `--strict`)
4. Security scanning (Bandit, Trivy — 0 Critical/High)
5. Test coverage ≥ 80 %
6. OpenAPI spec validation

A merge to `main` additionally triggers:
1. End-to-end tests
2. Load tests (P99 ≤ 300 ms at 1,000 RPS)
3. Docker image build and push
4. Helm chart linting

### 14.3 Performance Testing

Load tests MUST simulate realistic traffic patterns:

- **Baseline**: 1,000 RPS, 10,000 WebSocket connections
- **Peak**: 3,000 RPS (2× Black Friday / major weather event spike)
- **Soak**: 1,000 RPS for 2 hours (memory leak detection)
- **Spike**: Ramp from 100 to 3,000 RPS in 60 seconds

Success criteria: P99 ≤ 300 ms, error rate < 0.1 %, no memory growth > 10 % during soak.

---

## 15. Deployment & Infrastructure Requirements

### 15.1 Cloud Platform

STRIVE is cloud-agnostic and MUST deploy on:
- AWS (EKS + RDS + ElastiCache + MSK)
- GCP (GKE + Cloud SQL + Memorystore + Pub/Sub bridge)
- Azure (AKS + Azure Database + Azure Cache + Event Hubs bridge)

### 15.2 Infrastructure as Code

| Component | Tool |
|---|---|
| Cloud resources | Terraform (HCL) |
| Kubernetes manifests | Helm 3 charts |
| Secrets management | HashiCorp Vault or AWS Secrets Manager |
| CI/CD | GitHub Actions |
| GitOps (production) | ArgoCD |

### 15.3 Kubernetes Requirements

| Requirement | Specification |
|---|---|
| Minimum cluster version | Kubernetes 1.28 |
| Node pools | Compute (API), GPU (inference), Memory (Redis, DB) |
| Pod disruption budgets | ≥ 1 replica available during updates |
| Horizontal Pod Autoscaler | Enabled for all stateless services |
| Cluster autoscaler | Enabled for all node pools |
| Network policies | Default-deny; explicit allow rules per service |
| Pod security standards | Restricted (no root, no host namespaces) |

### 15.4 Disaster Recovery

| Scenario | RTO | RPO | Mechanism |
|---|---|---|---|
| Single pod failure | < 30 s | 0 | K8s liveness probe + restart |
| Node failure | < 2 min | 0 | Pod rescheduling |
| AZ failure | < 5 min | < 30 s | Multi-AZ deployment, Redis Cluster |
| Region failure | < 15 min | < 5 min | Active-passive DR region, DB replica promotion |
| Data corruption | < 30 min | < 5 min | Kafka replay + TimescaleDB PITR |

---

## 16. Risks & Mitigations

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R-01 | Third-party API outage (weather / traffic) | Medium | High | Multiple source fallbacks; stale cache; climatological baseline |
| R-02 | Model accuracy degrades due to concept drift | Medium | High | Automated drift detection; daily retraining; AUROC gate |
| R-03 | OSM road data stale or incorrect | Low | Medium | Weekly refresh; quality checks; user feedback loop |
| R-04 | Kafka consumer lag accumulates during traffic spike | Medium | High | Auto-scaling consumers; priority topic partitioning |
| R-05 | Redis cluster memory exhaustion | Low | Critical | Memory limits + eviction policy; horizontal scaling |
| R-06 | Legal liability if routing leads to an accident | Low | Critical | Prominent disclaimer; not a safety-critical system; advisory-only language |
| R-07 | GDPR violation from unexpected PII ingestion | Low | High | PII scanner in ingestion pipeline; regular audits |
| R-08 | Model bias against certain road types or demographics | Medium | High | Bias audit per road class, time of day, and demographic area; fairness metrics |
| R-09 | GPU availability / cost for inference | Medium | Medium | ONNX CPU fallback; quantised INT8 model for CPU |
| R-10 | Open-source dependency vulnerability | High | Medium | Dependabot; Snyk; weekly review |

---

## 17. Success Metrics & Acceptance Criteria

### 17.1 Launch Criteria (v1 GA)

All of the following MUST be met before v1 general availability:

| Criterion | Target |
|---|---|
| API P99 latency | ≤ 300 ms at 1,000 RPS |
| API error rate | < 0.1 % over 24 h soak test |
| Model AUROC | ≥ 0.88 on held-out test set |
| Model ECE | ≤ 0.05 |
| Risk score freshness | 100 % of active segments updated within 30 s |
| Documentation | OpenAPI spec complete; quickstart guide published |
| Security | 0 Critical/High CVEs in container images |
| Observability | All dashboards and alerts operational |

### 17.2 Business Success Metrics (6 months post-launch)

| Metric | Target |
|---|---|
| API calls per day | ≥ 1 million |
| Active partner integrations | ≥ 5 |
| Cities with active data | ≥ 10 |
| Model AUROC (production) | ≥ 0.90 |
| User-reported route satisfaction | ≥ 4.2 / 5.0 (partner survey) |
| Crash reduction (pilot fleet partners) | ≥ 10 % YoY |

### 17.3 Acceptance Testing Scenarios

| Scenario | Input | Expected Output |
|---|---|---|
| High-risk wet night | Segment with rain=12 mm, speed_ratio=0.4, historical_p95=0.8, night=True | risk_score ≥ 70 |
| Low-risk dry day | Segment with rain=0, speed_ratio=0.95, historical_p95=0.1, night=False | risk_score ≤ 25 |
| Safety route vs fast route | alpha=0.8, route with dangerous shortcut available | Dangerous shortcut NOT selected |
| Explanation factors | High rain + congestion segment | Top SHAP factors include precipitation and speed_ratio |
| WebSocket delta | Risk score changes by >5 points | Client receives delta within 35 s |
| API under load | 1,000 RPS for 60 s | P99 ≤ 300 ms, error rate < 0.1 % |

---

## 18. Glossary

| Term | Definition |
|---|---|
| **AUROC** | Area Under the Receiver Operating Characteristic curve; measures classifier discrimination ability |
| **AUPRC** | Area Under the Precision-Recall Curve; better for imbalanced datasets |
| **Concept drift** | Statistical shift in the relationship between features and labels over time |
| **ECE** | Expected Calibration Error; measures how well predicted probabilities match true frequencies |
| **Focal loss** | Loss function that down-weights easy examples, designed for class imbalance |
| **GNN** | Graph Neural Network; processes graph-structured data |
| **GraphSAGE** | Inductive GNN variant that aggregates neighbour features using a learned aggregator |
| **H3** | Uber's hierarchical hexagonal geospatial indexing system |
| **HPA** | Horizontal Pod Autoscaler (Kubernetes) |
| **METAR** | Meteorological Aerodrome Report; standardised weather observation format |
| **MC-Dropout** | Monte Carlo Dropout; uses dropout at inference time for uncertainty estimation |
| **MVT** | Mapbox Vector Tiles; compressed binary format for geographic data |
| **ONNX** | Open Neural Network Exchange; portable model format for cross-framework inference |
| **OSM** | OpenStreetMap |
| **PII** | Personally Identifiable Information |
| **PSI** | Population Stability Index; measures distribution shift between two datasets |
| **RPO** | Recovery Point Objective; maximum acceptable data loss in time |
| **RTO** | Recovery Time Objective; maximum acceptable downtime |
| **SHAP** | SHapley Additive exPlanations; game-theoretic feature attribution method |
| **SMOTE** | Synthetic Minority Over-sampling Technique |
| **TFT** | Temporal Fusion Transformer; attention-based model for multi-horizon time-series forecasting |
| **VSN** | Variable Selection Network; component of TFT that selects relevant features |
| **W3C TraceContext** | W3C standard for propagating trace context across distributed systems |

---

## 19. Revision History

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0.0 | 2026-04-13 | Karri Chanikya Sri Hari Narayana Dattu | Initial draft |
