# Input contract

The validator accepts common column aliases, but the canonical names below are recommended.

| File | Required columns | Notes |
|---|---|---|
| `farwest_su_final.gpkg` | `su_id`, geometry | One polygon per retained slope unit. |
| `dated_landslide_events_with_su_id.gpkg` | `su_id`, `event_year`, `landslide_type`, geometry | Event geometry must be a mapped initiation/crown/depletion location or polygon. The current code uses input points directly or `representative_point()` for non-point geometry. |
| `landslide_response_panel_1992_2018.csv` | `su_id`, `year`, and `landslide_count` or `active` | One row per slope unit × observed year. If mapped counts are absent, total counts are rebuilt from the event file. If a shallow response is absent, it is rebuilt from event rows whose `landslide_type` equals `shallow`. |
| `terrain_features.csv` | `su_id` and terrain variables | Units should be in headers or in the predictor dictionary. |
| `parent_material_features.csv` | `su_id`, `parent_material` | Dominant class per slope unit. |
| `rainfall_features_all_years_1992_2018.csv` | `su_id`, `year`, `monsoon_rainfall_mm`, `rx1day_mm`, `rx3day_mm`, `rx5day_mm`, `rx7day_mm`, `rx10day_mm`, `rx15day_mm`, `rx30day_mm` | Must include all observed years and the prior years needed for lagging. |
| rainfall-grid weights | `su_id`, grid identifier, overlap weight/area | The dominant grid is the maximum-overlap cell for each slope unit. |

Accepted slope-unit ID aliases include `su_id`, `SU_ID`, `suid`, `slope_unit_id`, `slopeunit_id`, and `cat`. Generic `id`/`OBJECTID` columns are not silently interpreted as slope-unit IDs because they are often feature identifiers. Accepted year aliases include `year`, `event_year`, `Year`, and `YEAR`.


