import pytest
import numpy as np
from datetime import datetime, timezone, timedelta


def _make_physics_frame(
    timestamp=None,
    hope_flag=False,
    hope_reason=None,
    hope_lead_time_sec=None,
    T_MK=12.0,
    EM_cm3=1e48,
    DNI=0.1,
    SHEI=-1.5,
    dSHEI_dt=0.0,
    provenance_hash="testhash",
    frame_id=None,
    solexs_flux=500.0,
    hel_flux=100.0,
):
    import uuid
    from aruna.l2_physics.models import PhysicsFrame, PhysicsQuality
    from aruna.models.observation import QualityFlag
    sol = solexs_flux
    hel = hel_flux
    return PhysicsFrame(
        frame_id=frame_id or str(uuid.uuid4()),
        timestamp=timestamp or datetime.now(timezone.utc),
        source_frame_ids=["sol1", "hel1"],
        T_MK=T_MK, T_uncertainty_MK=1.0,
        EM_cm3=EM_cm3, EM_uncertainty=1e46,
        fit_chi_squared=1.2,
        DNI=DNI,
        tnpsv_magnitude=0.1, tnpsv_angle_rad=1.57,
        SHEI=SHEI, dSHEI_dt=dSHEI_dt,
        HFD=float("nan"),
        QPP_period_sec=None, QPP_amplitude=None,
        FeKa_flux=0.0,
        flare_phase="HOPE" if hope_flag else "QUIET",
        hope_flag=hope_flag,
        hope_lead_time_sec=hope_lead_time_sec,
        hope_reason=hope_reason,
        engine_state="ACTIVE",
        data_quality_flag=QualityFlag.GOOD,
        physics_quality_flag=PhysicsQuality.FIT_OK,
        fit_backend="chianti_lookup",
        pipeline_version="2.0.0",
        provenance_hash=provenance_hash,
        solexs_3_6kev=sol * 0.6,
        solexs_6_10kev=sol * 0.4,
        hel1os_10_25kev=hel * 0.5,
        hel1os_25_50kev=hel * 0.3,
        hel1os_50_70kev=hel * 0.2,
    )


class TestNowcastConfig:
    def test_defaults_match_prd(self):
        from aruna.l3_nowcasting.models import NowcastConfig
        cfg = NowcastConfig()
        assert cfg.baseline_window_sec == 10800
        assert cfg.baseline_percentile == 85.0
        assert cfg.baseline_recompute_interval == 60
        assert cfg.trigger_sigma_multiplier == 3.0
        assert cfg.dual_trigger_window_sec == 180
        assert cfg.pelt_penalty == pytest.approx(1.5)
        assert cfg.pelt_min_size == 10
        assert cfg.solexs_to_wm2_factor == pytest.approx(1.0e-8)
        assert cfg.x3_auto_critical is True
        assert cfg.peak_decay_threshold == pytest.approx(0.5)

    def test_overrides_work(self):
        from aruna.l3_nowcasting.models import NowcastConfig
        cfg = NowcastConfig(baseline_window_sec=3600, trigger_sigma_multiplier=4.0)
        assert cfg.baseline_window_sec == 3600
        assert cfg.trigger_sigma_multiplier == pytest.approx(4.0)


class TestRollingP85Baseline:
    def test_none_before_first_recompute(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline(window_sec=100, recompute_interval=60)
        val, thr = b.update(1.0, t_sec=0)
        assert val is None and thr is None

    def test_p85_computed_after_interval(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline(window_sec=200, recompute_interval=60, sigma_multiplier=0.0)
        for i in range(200):
            b.update(float(i), t_sec=i)
        val, thr = b.update(200.0, t_sec=200)
        assert val is not None
        # Last recompute was at t=180 on window [0..180], so p85 of [0..180]
        expected = np.percentile(list(range(181)), 85)
        assert val == pytest.approx(expected, rel=1e-3)
        assert thr == pytest.approx(expected, rel=1e-3)  # sigma=0

    def test_threshold_above_baseline(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline(window_sec=200, recompute_interval=1, sigma_multiplier=3.0)
        rng = np.random.default_rng(0)
        for i in range(200):
            b.update(float(rng.normal(10.0, 1.0)), t_sec=i)
        assert b.threshold > b.baseline

    def test_freeze_holds_baseline_fixed(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline(window_sec=200, recompute_interval=1)
        for i in range(200):
            b.update(1.0, t_sec=i)
        b.freeze()
        baseline_before = b.baseline
        for i in range(200, 400):
            b.update(9999.0, t_sec=i)
        assert b.baseline == baseline_before

    def test_unfreeze_resumes_updates(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline(window_sec=200, recompute_interval=1)
        for i in range(200):
            b.update(1.0, t_sec=i)
        b.freeze()
        b.unfreeze()
        for i in range(200, 400):
            b.update(9999.0, t_sec=i)
        assert b.baseline > 100.0

    def test_is_frozen_flag(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline()
        assert not b.is_frozen()
        b.freeze()
        assert b.is_frozen()
        b.unfreeze()
        assert not b.is_frozen()


class TestChangepointDetector:
    def test_none_during_warmup(self):
        from aruna.l3_nowcasting.changepoint import ChangepointDetector
        d = ChangepointDetector(flux_window=100)
        result = None
        for _ in range(99):
            result = d.update(1.0)
        assert result is None

    def test_detects_step_change(self):
        from aruna.l3_nowcasting.changepoint import ChangepointDetector
        d = ChangepointDetector(flux_window=100, penalty=1.5, min_size=5)
        result = None
        for i in range(100):
            result = d.update(1.0 if i < 50 else 100.0)
        # step at index 50 — PELT should detect it
        assert result is not None
        assert 30 <= result <= 70

    def test_flat_signal_returns_none_or_int(self):
        from aruna.l3_nowcasting.changepoint import ChangepointDetector
        d = ChangepointDetector(flux_window=100, penalty=5.0, min_size=10)
        result = None
        rng = np.random.default_rng(0)
        for _ in range(100):
            result = d.update(float(rng.normal(1.0, 0.001)))
        assert result is None or isinstance(result, int)

    def test_returns_int_not_list(self):
        from aruna.l3_nowcasting.changepoint import ChangepointDetector
        d = ChangepointDetector(flux_window=100, penalty=1.5, min_size=5)
        result = None
        for i in range(100):
            result = d.update(1.0 if i < 50 else 100.0)
        if result is not None:
            assert isinstance(result, int)


class TestGOESClassifier:
    def _gc(self):
        from aruna.l3_nowcasting.goes_class import GOESClassifier
        return GOESClassifier(solexs_to_wm2_factor=1.0e-8)

    def test_x_class(self):
        counts = 1.25e-4 / 1.0e-8   # → 1.25e-4 W/m² → X1.2
        letter, sub = self._gc().classify(counts)
        assert letter == "X"
        assert sub == pytest.approx(1.2, rel=0.05)

    def test_m38_class(self):
        counts = 3.8e-5 / 1.0e-8    # M3.8
        letter, sub = self._gc().classify(counts)
        assert letter == "M"
        assert sub == pytest.approx(3.8, rel=0.01)

    def test_c_class(self):
        counts = 5.0e-6 / 1.0e-8    # C5.0
        letter, sub = self._gc().classify(counts)
        assert letter == "C"

    def test_b_class(self):
        counts = 5.0e-7 / 1.0e-8    # B5.0
        letter, sub = self._gc().classify(counts)
        assert letter == "B"

    def test_sub_a(self):
        letter, sub = self._gc().classify(0.001)
        assert letter == "A"

    def test_zero_flux(self):
        letter, sub = self._gc().classify(0.0)
        assert letter == "A"
        assert sub == 0.0

    def test_x3_is_major(self):
        gc = self._gc()
        assert gc.is_major_xclass("X", 3.0) is True
        assert gc.is_major_xclass("X", 2.9) is False
        assert gc.is_major_xclass("M", 9.0) is False


class TestConsensusEngine:
    def _engine(self, **kwargs):
        from aruna.l3_nowcasting.consensus import ConsensusEngine
        from aruna.l3_nowcasting.models import NowcastConfig
        return ConsensusEngine(NowcastConfig(**kwargs))

    def _ts(self, offset_sec=0):
        return datetime(2024, 2, 22, 4, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=offset_sec)

    def test_initial_state_is_quiet(self):
        assert self._engine().state == "QUIET"

    def test_hope_flag_emits_precursor(self):
        engine = self._engine()
        frame = _make_physics_frame(timestamp=self._ts(), hope_flag=True,
                                    hope_reason="T rising", hope_lead_time_sec=120)
        event = engine.ingest(False, False, True, "T rising", frame, 500.0, 100.0)
        assert event is not None
        assert event.alert_type == "HOPE_PRECURSOR"
        assert event.detection_method == "HOPE"
        assert engine.state == "HOPE"

    def test_hope_fires_only_once(self):
        engine = self._engine()
        frame = _make_physics_frame(timestamp=self._ts(), hope_flag=True)
        engine.ingest(False, False, True, None, frame, 500.0, 100.0)
        event2 = engine.ingest(False, False, True, None,
                                _make_physics_frame(timestamp=self._ts(1), hope_flag=True),
                                500.0, 100.0)
        assert event2 is None  # second hope frame → no duplicate

    def test_solexs_only_no_confirmed_onset(self):
        engine = self._engine()
        for i in range(200):
            event = engine.ingest(True, False, False, None,
                                  _make_physics_frame(timestamp=self._ts(i)),
                                  5000.0, 100.0)
            if event is not None:
                assert event.detection_method != "DUAL_TRIGGER"

    def test_dual_trigger_within_window_fires_onset(self):
        engine = self._engine(dual_trigger_window_sec=180)
        # SoLEXS triggers at t=0
        engine.ingest(True, False, False, None,
                      _make_physics_frame(timestamp=self._ts(0)), 5000.0, 100.0)
        # HEL1OS triggers at t=30 (within 180s)
        event = engine.ingest(False, True, False, None,
                               _make_physics_frame(timestamp=self._ts(30)), 5000.0, 500.0)
        assert event is not None
        assert event.alert_type == "FLARE_ONSET"
        assert event.detection_method == "DUAL_TRIGGER"

    def test_dual_trigger_outside_window_no_onset(self):
        # Dual triggers 200s apart (> 180s window) must not produce a confirmed DUAL_TRIGGER onset.
        # B2: HEL1OS_ONLY advisory may fire on the second call — that is expected and correct.
        engine = self._engine(dual_trigger_window_sec=180)
        engine.ingest(True, False, False, None,
                      _make_physics_frame(timestamp=self._ts(0)), 5000.0, 100.0)
        event = engine.ingest(False, True, False, None,
                               _make_physics_frame(timestamp=self._ts(200)), 5000.0, 500.0)
        # No confirmed DUAL_TRIGGER onset — either None or a degraded advisory is acceptable
        if event is not None:
            assert event.detection_method != "DUAL_TRIGGER", (
                "Window violation: DUAL_TRIGGER onset fired with triggers 200s apart "
                f"(window=180s); got detection_method={event.detection_method}"
            )

    def test_full_lifecycle_quiet_hope_onset_peak_end(self):
        engine = self._engine(dual_trigger_window_sec=180, peak_decay_threshold=0.5)
        events = []
        # HOPE
        e = engine.ingest(False, False, True, "T rising",
                          _make_physics_frame(timestamp=self._ts(0), hope_flag=True),
                          600.0, 110.0)
        if e: events.append(e)
        # Dual trigger at t=1 (both instruments within window)
        engine.ingest(True, False, False, None,
                      _make_physics_frame(timestamp=self._ts(1)), 5000.0, 1000.0)
        e = engine.ingest(False, True, False, None,
                          _make_physics_frame(timestamp=self._ts(2)), 5000.0, 1500.0)
        if e: events.append(e)
        assert engine.state == "ONSET"
        # Rising flux (5 frames)
        for i in range(3, 8):
            e = engine.ingest(False, False, False, None,
                              _make_physics_frame(timestamp=self._ts(i)),
                              5000.0 + i * 1000, 1500.0 + i * 100)
            if e: events.append(e)
        # 3 consecutive declining frames → FLARE_PEAK
        peak_sol = 5000.0 + 7 * 1000
        for j in range(3):
            e = engine.ingest(False, False, False, None,
                              _make_physics_frame(timestamp=self._ts(8 + j)),
                              peak_sol - (j + 1) * 1000, 1000.0)
            if e: events.append(e)
        assert any(ev.alert_type == "FLARE_PEAK" for ev in events), f"events={[ev.alert_type for ev in events]}"
        # Flux < 50% of peak → FLARE_END
        e = engine.ingest(False, False, False, None,
                          _make_physics_frame(timestamp=self._ts(12)),
                          100.0, 10.0)
        if e: events.append(e)
        assert any(ev.alert_type == "FLARE_END" for ev in events)
        assert engine.state == "QUIET"

    def test_overlapping_flare_distinct_event_ids(self):
        engine = self._engine(dual_trigger_window_sec=180, peak_decay_threshold=0.5)
        # Get to PEAK state
        engine.ingest(True, False, False, None,
                      _make_physics_frame(timestamp=self._ts(0)), 5000.0, 1000.0)
        engine.ingest(False, True, False, None,
                      _make_physics_frame(timestamp=self._ts(1)), 5000.0, 1500.0)
        first_onset_id = None
        for i in range(2, 8):
            e = engine.ingest(False, False, False, None,
                              _make_physics_frame(timestamp=self._ts(i)),
                              5000.0 + i * 1000, 1500.0)
            if e and e.alert_type == "FLARE_ONSET":
                first_onset_id = e.event_id
        # force to PEAK via 3 declining
        peak_sol = 5000.0 + 7 * 1000
        for j in range(3):
            engine.ingest(False, False, False, None,
                          _make_physics_frame(timestamp=self._ts(8 + j)),
                          peak_sol - (j+1)*500, 1000.0)
        assert engine.state == "PEAK"
        # New dual trigger during PEAK → overlapping
        engine.ingest(True, False, False, None,
                      _make_physics_frame(timestamp=self._ts(12)), 8000.0, 2000.0)
        e2 = engine.ingest(False, True, False, None,
                           _make_physics_frame(timestamp=self._ts(13)), 8000.0, 2500.0)
        assert e2 is not None
        assert e2.alert_type == "FLARE_ONSET"
        assert e2.flare_type == "OVERLAPPING"
        if first_onset_id:
            assert e2.event_id != first_onset_id

    def test_x3_escalated_to_confidence_1(self):
        engine = self._engine(x3_auto_critical=True, solexs_to_wm2_factor=1.0e-8)
        # 5.0e-4 W/m² → X5.0 (major X-class)
        flux_x5 = 5.0e-4 / 1.0e-8
        engine.ingest(True, False, False, None,
                      _make_physics_frame(timestamp=self._ts(0)), flux_x5, 1000.0)
        event = engine.ingest(False, True, False, None,
                               _make_physics_frame(timestamp=self._ts(1)), flux_x5, 2000.0)
        assert event is not None
        assert event.goes_class == "X"
        assert event.confidence == pytest.approx(1.0)

    def test_sep_event_on_high_kp(self):
        engine = self._engine(sep_kp_threshold=6)
        event = engine.ingest(True, False, False, None,
                               _make_physics_frame(timestamp=self._ts(0)),
                               5000.0, 100.0, kp_index=8.0)
        assert event is not None
        assert event.detection_method == "SEP_EVENT"

    def test_provenance_hash_copied_from_frame(self):
        engine = self._engine()
        frame = _make_physics_frame(timestamp=self._ts(), hope_flag=True,
                                    provenance_hash="myspecialhash")
        event = engine.ingest(False, False, True, None, frame, 500.0, 100.0)
        assert event is not None
        assert event.provenance_hash == "myspecialhash"


class TestCatalogService:
    def _event(self, alert_type="FLARE_ONSET", event_id=None):
        from aruna.models.nowcast import NowcastEvent
        e = NowcastEvent(
            alert_type=alert_type,
            timestamp=datetime(2024, 2, 22, 4, 0, 0, tzinfo=timezone.utc),
            goes_class="X", goes_subclass=6.3,
            confidence=0.9,
            instruments=["SOLEXS", "HEL1OS"],
            physics_frame_id="frame123",
            detection_method="DUAL_TRIGGER" if alert_type != "HOPE_PRECURSOR" else "HOPE",
            flare_type="STANDARD",
            provenance_hash="abc123hash",
        )
        if event_id:
            e.event_id = event_id
        return e

    def test_write_under_100ms(self, tmp_path):
        import time
        from aruna.l3_nowcasting.catalog import CatalogService
        cat = CatalogService(f"sqlite:///{tmp_path}/test.db")
        event = self._event()
        start = time.perf_counter()
        cat.write(event)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100, f"Write took {elapsed_ms:.1f}ms"
        cat.close()

    def test_upsert_idempotent(self, tmp_path):
        from aruna.l3_nowcasting.catalog import CatalogService
        cat = CatalogService(f"sqlite:///{tmp_path}/test.db")
        event = self._event()
        cat.write(event)
        cat.write(event)
        count = cat._conn.execute(
            "SELECT COUNT(*) FROM master_flare_catalog WHERE event_id=?",
            (event.event_id,)
        ).fetchone()[0]
        assert count == 1
        cat.close()

    def test_provenance_hash_stored(self, tmp_path):
        from aruna.l3_nowcasting.catalog import CatalogService
        cat = CatalogService(f"sqlite:///{tmp_path}/test.db")
        event = self._event()
        cat.write(event)
        row = cat.fetch_event(event.event_id)
        assert row is not None
        assert row["provenance_hash"] == "abc123hash"
        cat.close()

    def test_lifecycle_fills_timestamps(self, tmp_path):
        from aruna.l3_nowcasting.catalog import CatalogService
        cat = CatalogService(f"sqlite:///{tmp_path}/test.db")
        eid = "fixed-test-event-id"
        for alert_type in ("HOPE_PRECURSOR", "FLARE_ONSET", "FLARE_PEAK", "FLARE_END"):
            e = self._event(alert_type=alert_type, event_id=eid)
            cat.write(e, peak_T_MK=23.5, peak_flux_solexs=15000.0)
        row = cat.fetch_event(eid)
        assert row["hope_time"] is not None
        assert row["onset_time"] is not None
        assert row["peak_time"] is not None
        assert row["end_time"] is not None
        cat.close()


# ── Task 7: NowcastOrchestrator integration tests ────────────────────────────

class TestNowcastOrchestrator:
    def _run(self, frames, config):
        from aruna.l3_nowcasting.orchestrator import NowcastOrchestrator
        orch = NowcastOrchestrator(config=config)
        events = []
        for f in frames:
            e = orch.process(f)
            if e is not None:
                events.append(e)
        orch.close()
        return events

    def test_x63_detected_within_30s(self, synthetic_physics_frames, test_nowcast_config):
        """Detection of onset must occur within 30s of the synthetic GOES onset time."""
        # GOES onset defined as the first frame of the rise phase (frame 250)
        goes_onset_time = synthetic_physics_frames[250].timestamp
        events = self._run(synthetic_physics_frames, test_nowcast_config)
        onsets = [e for e in events if e.alert_type == "FLARE_ONSET"
                  and e.detection_method == "DUAL_TRIGGER"]
        assert onsets, "No DUAL_TRIGGER FLARE_ONSET emitted"
        detection_delay = (onsets[0].timestamp - goes_onset_time).total_seconds()
        assert detection_delay <= 30, f"Detection delay {detection_delay:.1f}s > 30s"

    def test_hope_fires_90s_before_peak(self, synthetic_physics_frames, test_nowcast_config):
        """HOPE_PRECURSOR must fire >= 90s before FLARE_PEAK."""
        events = self._run(synthetic_physics_frames, test_nowcast_config)
        hope_events = [e for e in events if e.alert_type == "HOPE_PRECURSOR"]
        peak_events = [e for e in events if e.alert_type == "FLARE_PEAK"]
        assert hope_events, "No HOPE_PRECURSOR emitted"
        assert peak_events, "No FLARE_PEAK emitted"
        lead = (peak_events[0].timestamp - hope_events[0].timestamp).total_seconds()
        assert lead >= 90, f"HOPE lead time {lead:.1f}s < 90s"

    def test_tpr_far_tss_synthetic_injection(self, test_nowcast_config):
        """TPR > 0.80, FAR < 0.05, TSS > 0.65 on synthetic M+X flares."""
        from aruna.l3_nowcasting.orchestrator import NowcastOrchestrator
        import uuid

        rng = np.random.default_rng(99)
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)

        def quiet_frame(i, t):
            return _make_physics_frame(
                timestamp=t, solexs_flux=500 + rng.normal(0, 15),
                hel_flux=100 + rng.normal(0, 5),
                provenance_hash=f"q{i}", frame_id=str(uuid.uuid4()),
            )

        def flare_burst(start_t, peak_sol=40000.0):
            """20 quiet + 30 rise + 30 decay frames."""
            frames = []
            for j in range(20):
                frames.append(_make_physics_frame(
                    timestamp=start_t + timedelta(seconds=j),
                    solexs_flux=500.0, hel_flux=100.0,
                    provenance_hash=f"pre{j}", frame_id=str(uuid.uuid4()),
                ))
            for j in range(30):
                frac = j / 30
                frames.append(_make_physics_frame(
                    timestamp=start_t + timedelta(seconds=20 + j),
                    solexs_flux=500 + (peak_sol - 500) * frac,
                    hel_flux=100 + 4900 * frac,
                    SHEI=-1.6 + 2.0 * frac, dSHEI_dt=0.05,
                    hope_flag=(j == 0),
                    hope_reason="T rising" if j == 0 else None,
                    hope_lead_time_sec=90 if j == 0 else None,
                    provenance_hash=f"rise{j}", frame_id=str(uuid.uuid4()),
                ))
            for j in range(30):
                decay = peak_sol * float(np.exp(-3 * j / 30))
                frames.append(_make_physics_frame(
                    timestamp=start_t + timedelta(seconds=50 + j),
                    solexs_flux=decay, hel_flux=decay * 0.2,
                    provenance_hash=f"dec{j}", frame_id=str(uuid.uuid4()),
                ))
            return frames

        # 5 injected flares, each separated by quiet frames
        n_flares = 5
        all_frames = []
        flare_windows = []
        t = base
        for fi in range(n_flares):
            # 50 quiet frames
            for qi in range(50):
                all_frames.append(quiet_frame(len(all_frames), t))
                t += timedelta(seconds=1)
            flare_start = t
            burst = flare_burst(flare_start, peak_sol=40000.0)
            all_frames.extend(burst)
            flare_windows.append((flare_start, flare_start + timedelta(seconds=80)))
            t += timedelta(seconds=80)

        orch = NowcastOrchestrator(config=test_nowcast_config)
        detected_onsets = []
        for f in all_frames:
            e = orch.process(f)
            if e is not None and e.alert_type == "FLARE_ONSET" and e.detection_method == "DUAL_TRIGGER":
                detected_onsets.append(e.timestamp)
        orch.close()

        # TPR: fraction of injected flares detected
        hits = sum(
            any(w[0] <= dt <= w[1] for dt in detected_onsets)
            for w in flare_windows
        )
        tpr = hits / n_flares

        # FAR: false alarms across all 5 inter-flare quiet windows (50s before each flare)
        quiet_intervals = [
            (w[0] - timedelta(seconds=50), w[0])
            for w in flare_windows
        ]
        false_alarms = sum(
            any(qi[0] <= dt < qi[1] for qi in quiet_intervals)
            for dt in detected_onsets
        )
        far = false_alarms / n_flares

        tss = tpr - far
        assert tpr >= 0.80, f"TPR={tpr:.2f} < 0.80"
        assert far < 0.05, f"FAR={far:.2f} >= 0.05"
        assert tss >= 0.65, f"TSS={tss:.2f} < 0.65"

    def test_all_events_have_provenance_hash(self, synthetic_physics_frames, test_nowcast_config):
        events = self._run(synthetic_physics_frames, test_nowcast_config)
        assert events, "No events emitted"
        for e in events:
            assert e.provenance_hash, f"Empty provenance_hash on {e.alert_type}"

    def test_x3_plus_escalated_to_critical(self, test_nowcast_config):
        """X3+ event (X5.0) must get confidence=1.0."""
        from aruna.l3_nowcasting.orchestrator import NowcastOrchestrator
        # X5.0: 5.0e-4 W/m² → counts = 5.0e-4 / 1e-8 = 50000
        x5_sol = 50000.0
        base = datetime(2024, 2, 22, tzinfo=timezone.utc)
        orch = NowcastOrchestrator(config=test_nowcast_config)
        # Establish baseline (200 quiet frames)
        for i in range(200):
            orch.process(_make_physics_frame(
                timestamp=base + timedelta(seconds=i),
                solexs_flux=500.0, hel_flux=100.0,
                provenance_hash=f"q{i}",
            ))
        # Sol triggers at t=200
        orch.process(_make_physics_frame(
            timestamp=base + timedelta(seconds=200),
            solexs_flux=x5_sol, hel_flux=100.0,
            SHEI=-0.5, dSHEI_dt=0.1,
            provenance_hash="x5_sol",
        ))
        # Hel triggers at t=201 (within 180s window)
        event = orch.process(_make_physics_frame(
            timestamp=base + timedelta(seconds=201),
            solexs_flux=x5_sol, hel_flux=5000.0,
            SHEI=-0.5, dSHEI_dt=0.1,
            provenance_hash="x5_hel",
        ))
        orch.close()
        # Event may come from consensus — check any FLARE_ONSET with X class
        assert event is not None
        assert event.goes_class == "X"
        assert event.confidence == pytest.approx(1.0)

    def test_solexs_only_never_produces_dual_trigger_onset(self, test_nowcast_config):
        from aruna.l3_nowcasting.orchestrator import NowcastOrchestrator
        base = datetime(2024, 2, 22, tzinfo=timezone.utc)
        orch = NowcastOrchestrator(config=test_nowcast_config)
        all_events = []
        for i in range(400):
            # SoLEXS goes high from frame 200 onward; HEL1OS stays low with no SHEI hardening
            sol = 500.0 if i < 200 else 50000.0
            e = orch.process(_make_physics_frame(
                timestamp=base + timedelta(seconds=i),
                solexs_flux=sol, hel_flux=100.0,
                SHEI=-1.8, dSHEI_dt=-0.01,  # no hardening
                provenance_hash=f"p{i}",
            ))
            if e is not None:
                all_events.append(e)
        orch.close()
        # Verify no DUAL_TRIGGER FLARE_ONSET when HEL1OS offline
        confirmed = [e for e in all_events if e.alert_type == "FLARE_ONSET" and e.detection_method == "DUAL_TRIGGER"]
        assert not confirmed, f"Got {len(confirmed)} unexpected DUAL_TRIGGER onsets"
        # Verify SOLEXS_ONLY advisory IS emitted with confidence <= 0.5
        solexs_only_events = [e for e in all_events if e.detection_method == "SOLEXS_ONLY"]
        assert solexs_only_events, "No SOLEXS_ONLY advisory emitted when HEL1OS quiet"
        assert all(e.confidence <= 0.5 for e in solexs_only_events)

    def test_catalog_write_under_100ms(self, synthetic_physics_frames, test_nowcast_config):
        import time
        from aruna.l3_nowcasting.orchestrator import NowcastOrchestrator
        orch = NowcastOrchestrator(config=test_nowcast_config)
        write_times = []
        for f in synthetic_physics_frames:
            t0 = time.perf_counter()
            e = orch.process(f)
            if e is not None:
                write_times.append((time.perf_counter() - t0) * 1000)
        orch.close()
        if write_times:
            assert max(write_times) < 100, f"Slowest write: {max(write_times):.1f}ms"

    def test_tpr_over_0_80_on_real_m_x_validation_set(self, test_nowcast_config):
        """TPR > 0.80 on synthetic M+X validation set (real PRADAN data absent)."""
        from aruna.l3_nowcasting.orchestrator import NowcastOrchestrator
        import uuid

        rng = np.random.default_rng(42)
        base = datetime(2024, 7, 1, tzinfo=timezone.utc)

        def _quiet(i, t):
            return _make_physics_frame(
                timestamp=t, solexs_flux=500 + rng.normal(0, 15),
                hel_flux=100 + rng.normal(0, 5),
                provenance_hash=f"q{i}", frame_id=str(uuid.uuid4()),
            )

        def _flare_burst(start_t, peak_sol=45000.0):
            frames = []
            for j in range(20):
                frames.append(_make_physics_frame(
                    timestamp=start_t + timedelta(seconds=j),
                    solexs_flux=500.0, hel_flux=100.0,
                    provenance_hash=f"pre{j}", frame_id=str(uuid.uuid4()),
                ))
            for j in range(30):
                frac = j / 30
                frames.append(_make_physics_frame(
                    timestamp=start_t + timedelta(seconds=20 + j),
                    solexs_flux=500 + (peak_sol - 500) * frac,
                    hel_flux=100 + 4900 * frac,
                    SHEI=-1.6 + 2.0 * frac, dSHEI_dt=0.05,
                    hope_flag=(j == 0),
                    hope_reason="T rising" if j == 0 else None,
                    hope_lead_time_sec=90 if j == 0 else None,
                    provenance_hash=f"rise{j}", frame_id=str(uuid.uuid4()),
                ))
            for j in range(30):
                frames.append(_make_physics_frame(
                    timestamp=start_t + timedelta(seconds=50 + j),
                    solexs_flux=peak_sol * float(np.exp(-3 * j / 30)),
                    hel_flux=peak_sol * 0.2 * float(np.exp(-3 * j / 30)),
                    provenance_hash=f"dec{j}", frame_id=str(uuid.uuid4()),
                ))
            return frames

        n_flares = 10
        all_frames = []
        flare_windows = []
        t = base
        for fi in range(n_flares):
            for qi in range(50):
                all_frames.append(_quiet(len(all_frames), t))
                t += timedelta(seconds=1)
            flare_start = t
            burst = _flare_burst(flare_start, peak_sol=45000.0)
            all_frames.extend(burst)
            flare_windows.append((flare_start, flare_start + timedelta(seconds=80)))
            t += timedelta(seconds=80)

        orch = NowcastOrchestrator(config=test_nowcast_config)
        detected = []
        for f in all_frames:
            e = orch.process(f)
            if e is not None and e.alert_type == "FLARE_ONSET" and e.detection_method == "DUAL_TRIGGER":
                detected.append(e.timestamp)
        orch.close()

        hits = sum(any(w[0] <= dt <= w[1] for dt in detected) for w in flare_windows)
        tpr = hits / n_flares
        assert tpr >= 0.80, f"TPR={tpr:.2f} < 0.80 on synthetic M+X validation set"

    def test_far_under_0_05_on_100h_quiet_sun(self, test_nowcast_config):
        """< 5 false FLARE_ONSET alerts expected over 100h quiet sun (tested over 1h)."""
        from aruna.l3_nowcasting.orchestrator import NowcastOrchestrator
        import uuid

        rng = np.random.default_rng(99)
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        orch = NowcastOrchestrator(config=test_nowcast_config)

        false_alarms = 0
        for i in range(3600):  # 1 hour at 1-second cadence
            frame = _make_physics_frame(
                timestamp=base + timedelta(seconds=i),
                solexs_flux=500.0 + rng.normal(0, 15),
                hel_flux=100.0 + rng.normal(0, 5),
                SHEI=-1.8, dSHEI_dt=-0.01,
                provenance_hash=f"qs{i}", frame_id=str(uuid.uuid4()),
            )
            event = orch.process(frame)
            if event is not None and event.alert_type == "FLARE_ONSET":
                false_alarms += 1
        orch.close()

        # 0 false alarms in 1h → 0 in 100h (well below the < 5 limit)
        assert false_alarms == 0, (
            f"{false_alarms} false FLARE_ONSET events in 1h quiet sun "
            f"(projects to ~{false_alarms * 100} per 100h, limit: < 5)"
        )
