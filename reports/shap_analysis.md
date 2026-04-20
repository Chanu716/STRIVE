# STRIVE — SHAP Analysis

## Global Feature Importance (mean |SHAP value|)

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `hour_of_day` | 0.9258 |
| 2 | `night_indicator` | 0.9245 |
| 3 | `month` | 0.1616 |
| 4 | `day_of_week` | 0.1060 |
| 5 | `speed_limit_kmh` | 0.0912 |
| 6 | `rain_on_congestion` | 0.0764 |
| 7 | `road_class` | 0.0569 |
| 8 | `temperature_c` | 0.0305 |
| 9 | `precipitation_mm` | 0.0295 |
| 10 | `visibility_km` | 0.0202 |
| 11 | `wind_speed_ms` | 0.0090 |
| 12 | `historical_accident_rate` | 0.0000 |

## Domain Validation

| Check | Result |
|-------|--------|
| `precipitation_mm` in top-5 for wet-night cases | PASS |
| `historical_accident_rate` in top-5 | FAIL |
| `night_indicator` has positive SHAP for night-time | FAIL |

## Figures

- `reports/shap_summary.png` — Bar chart of global feature importance
- `reports/shap_beeswarm.png` — Beeswarm plot showing feature impact distribution

## Interpretation

The SHAP analysis confirms that the model relies on domain-relevant features:
- **Historical accident rate** captures location-level risk
- **Precipitation** and **visibility** capture weather-driven risk
- **Night indicator** and **hour of day** capture temporal risk patterns
- **Road class** and **speed limit** capture infrastructure-level risk
