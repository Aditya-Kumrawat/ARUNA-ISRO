# ARUNA — xfail Test Registry

All `@pytest.mark.xfail` tests across the pipeline. Each entry records why it is failing and exactly what needs to exist before it can be promoted to a real passing test.

Run to see current status:
```
pytest python/ -v --tb=no 2>&1 | grep -E "XFAIL|XPASS"
```

---

## Status Key

| Symbol | Meaning |
|--------|---------|
| `⏳ pending` | Module not yet built |
| `🔒 data-gate` | Module built but needs real PRADAN/external data |
| `✅ resolved` | Promoted to passing test (update this file when done) |

---

## L2 Physics — 5 tests (data-gate)

**File:** `python/tests/test_l2_physics.py`
**Reason:** `"Phase 3 not yet implemented"` (stale label — L2 IS built; these need real PRADAN FITS data)
**What unlocks them:** Real M+ class FITS files from PRADAN in `python/tests/fixtures/`

| Test | Status | Unlock condition |
|------|--------|-----------------|
| `test_isothermal_fit_chi_squared_under_2_on_m_plus_spectra` | 🔒 data-gate | M-class FITS fixtures in tests/fixtures/ |
| `test_dni_hope_threshold_fires_within_2_min_of_impulsive_onset` | 🔒 data-gate | Real X6.3 HEL1OS FITS in tests/fixtures/x63/ |
| `test_tnpsv_in_precursor_quadrant_90s_before_x63_peak` | 🔒 data-gate | Real X6.3 SoLEXS FITS in tests/fixtures/x63/ |
| `test_shei_transitions_from_thermal_to_nonthermal_during_x63` | 🔒 data-gate | Real X6.3 HEL1OS FITS in tests/fixtures/x63/ |
| `test_feka_fluorescence_detected_in_80_pct_of_x_class_events` | 🔒 data-gate | Multiple X-class FITS fixtures |

---

## L3 Nowcasting — 2 tests (data-gate)

**File:** `python/tests/test_l3_nowcasting.py`
**Class:** `TestNowcastOrchestrator`
**What unlocks them:** Real PRADAN M+X event catalog and 100h quiet-Sun telemetry

| Test | Status | Unlock condition |
|------|--------|-----------------|
| `test_tpr_over_0_80_on_real_m_x_validation_set` | 🔒 data-gate | PRADAN catalog 2024-07-01→2024-12-31 |
| `test_far_under_0_05_on_100h_quiet_sun` | 🔒 data-gate | PRADAN 100h quiet-Sun telemetry |

---

## L4 Forecasting — 9 tests (pending)

**File:** `python/tests/test_l4_forecasting.py`
**Reason:** `"Phase 5 not yet implemented"`
**What unlocks them:** Build `python/aruna/l4_forecasting/` (Phase 5)
**Key PRD ref:** §8

| Test | Status | What to implement |
|------|--------|-------------------|
| `test_tss_over_0_65_on_m_x_validation_2024_h2` | ⏳ pending | ForecastOrchestrator + DualStreamTCN |
| `test_roc_auc_over_0_92_on_binary_mx_held_out` | ⏳ pending | Ensemble + calibrator |
| `test_ece_under_0_05_after_mapie_calibration` | ⏳ pending | MAPIE conformal calibration |
| `test_lead_time_mae_under_90s_on_hope_events` | ⏳ pending | HOPE-triggered forecast path |
| `test_ensemble_beats_best_single_model_by_tss_0_03` | ⏳ pending | TCN + Moirai ensemble |
| `test_no_temporal_leakage` | ⏳ pending | Temporal train/val/test split validation |
| `test_forecast_output_within_10_seconds` | ⏳ pending | Latency test for ForecastOrchestrator |
| `test_under_prediction_under_5_pct_for_m5_plus` | ⏳ pending | Asymmetric loss (α=0.2) validation |
| `test_all_forecasts_include_confidence_interval` | ⏳ pending | MAPIE uncertainty bands on every output |
| `test_moirai_context_clip_fallback_bumps_tcn_weight` | ⏳ pending | Moirai >512-step fallback to TCN |

---

## L5 Analogs — 6 tests (pending)

**File:** `python/tests/test_l5_analogs.py`
**Reason:** Phase not yet implemented
**What unlocks them:** Build `python/aruna/l5_analogs/` (Phase 6)
**Key PRD ref:** §9

| Test | Status | What to implement |
|------|--------|-------------------|
| `test_retrieval_precision_over_0_75` | ⏳ pending | FAISS + DTW analog retrieval |
| `test_faiss_search_under_100ms_for_10k_events` | ⏳ pending | FAISS index search latency |
| `test_no_analog_fires_when_dtw_over_3_sigma` | ⏳ pending | DTW similarity threshold gate |
| `test_x63_retrieves_at_least_one_x_class_analog` | ⏳ pending | X6.3 replay analog test |
| `test_faiss_index_auto_rebuilds_after_new_catalog_entry` | ⏳ pending | FAISS auto-rebuild on catalog write |
| `test_cold_start_empty_catalog_no_crash` | ⏳ pending | Empty catalog edge case handler |

---

## L6 Threat — 7 tests (pending)

**File:** `python/tests/test_l6_threat.py`
**Reason:** Phase not yet implemented
**What unlocks them:** Build `python/aruna/l6_threat/` (Phase 7)
**Key PRD ref:** §10

| Test | Status | What to implement |
|------|--------|-------------------|
| `test_threat_flags_within_2_seconds` | ⏳ pending | ThreatOrchestrator latency |
| `test_navic_risk_matches_kintner_model_within_1_level` | ⏳ pending | NavIC Kintner ionospheric model |
| `test_chandrayaan3_is_priority_1_for_x_class` | ⏳ pending | Satellite safing priority queue |
| `test_spacewx_bulletin_validates_against_swmc_template` | ⏳ pending | SpaceWx bulletin generator |
| `test_dos_donts_generated_for_m5_plus_and_x_class` | ⏳ pending | Gaganyaan EVA advisory |
| `test_south_india_highest_eia_amplification` | ⏳ pending | EIA regional amplification model |
| `test_x_class_impact_never_understated` | ⏳ pending | Minimum-floor threat logic for X-class |

---

## L7 LLM Briefing — 7 tests (pending)

**File:** `python/tests/test_l7_llm.py`
**Reason:** Phase not yet implemented
**What unlocks them:** Build `python/aruna/l7_llm/` (Phase 8)
**Key PRD ref:** §11

| Test | Status | What to implement |
|------|--------|-------------------|
| `test_hallucination_guard_passes_on_all_test_alerts` | ⏳ pending | Hallucination guard validator |
| `test_briefing_generated_within_15_seconds` | ⏳ pending | Ollama/Mistral briefing latency |
| `test_fallback_activates_when_guard_fails` | ⏳ pending | Deterministic template fallback |
| `test_every_briefing_has_required_fields` | ⏳ pending | Briefing schema validator |
| `test_no_novel_numbers_in_briefing` | ⏳ pending | Hallucination: no invented numbers |
| `test_llm_runs_fully_offline` | ⏳ pending | Air-gapped Ollama mode |
| `test_deterministic_fallback_valid_without_ollama` | ⏳ pending | Template output when Ollama absent |

---

## Integration — 6 tests (pending)

**File:** `python/tests/test_integration.py`
**Reason:** `"Phase 10 not yet implemented"` — needs full L0-L7 stack + Kafka
**What unlocks them:** All pipeline phases complete + Docker Compose stack running

| Test | Status | What to implement |
|------|--------|-------------------|
| `test_x63_full_pipeline_alert_briefing_within_60s` | ⏳ pending | Full L0-L7 + Kafka + end-to-end timing |
| `test_may2024_superstorm_red_alert_2h_before_kp9` | ⏳ pending | May 2024 counterfactual replay |
| `test_100h_quiet_sun_fewer_than_5_false_alarms` | ⏳ pending | Full pipeline quiet-Sun FAR |
| `test_all_14_kafka_topics_publishing_correct_schemas` | ⏳ pending | Kafka schema validation across all topics |
| `test_health_endpoint_returns_healthy_for_all_stages` | ⏳ pending | Node API /health endpoint + all stages |
| `test_pipeline_auto_recovers_after_kafka_restart` | ⏳ pending | Kafka consumer restart resilience |

---

## Summary

| Layer | Total xfail | Pending (need code) | Data-gate (need PRADAN) |
|-------|------------|---------------------|------------------------|
| L2 Physics | 5 | 0 | 5 |
| L3 Nowcasting | 2 | 0 | 2 |
| L4 Forecasting | 10 | 10 | 0 |
| L5 Analogs | 6 | 6 | 0 |
| L6 Threat | 7 | 7 | 0 |
| L7 LLM | 7 | 7 | 0 |
| Integration | 6 | 6 | 0 |
| **Total** | **43** | **36** | **7** |

As each phase is implemented, update the `Status` column and remove the `xfail` decorator from the test.
