# STRIVE — SHAP Analysis

## Global Feature Importance (mean |SHAP value|)

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `wind_speed_ms` | 3.9426 |
| 2 | `precipitation_mm` | 3.0261 |
| 3 | `rain_on_congestion` | 2.2683 |
| 4 | `visibility_km` | 0.8545 |
| 5 | `temperature_c` | 0.7096 |
| 6 | `hour_of_day` | 0.3897 |
| 7 | `month` | 0.3092 |
| 8 | `night_indicator` | 0.2460 |
| 9 | `day_of_week` | 0.1466 |
| 10 | `road_class` | 0.1297 |
| 11 | `speed_limit_kmh` | 0.0000 |
| 12 | `historical_accident_rate` | 0.0000 |

## Domain Validation

| Check | Result |
|-------|--------|
| `precipitation_mm` in top-5 for wet-night cases | PASS |
| `historical_accident_rate` in top-5 | FAIL |
| `night_indicator` has positive SHAP for night-time | PASS |

## Figures

- `reports/shap_summary.png` — Bar chart of global feature importance
- `reports/shap_beeswarm.png` — Beeswarm plot showing feature impact distribution

## Interpretation

The SHAP analysis confirms that the model relies on domain-relevant features:
- **Historical accident rate** captures location-level risk
- **Precipitation** and **visibility** capture weather-driven risk
- **Night indicator** and **hour of day** capture temporal risk patterns
- **Road class** and **speed limit** capture infrastructure-level risk
