"""L5 — Historical Analog Retrieval Tests. PRD §9.3 + brainstorm."""
import pytest
import os
import sqlite3
import time
import numpy as np
from datetime import datetime, timezone, timedelta

from aruna.models.forecast import ForecastFrame
from aruna.models.physics import PhysicsFrame
from aruna.models.observation import QualityFlag
from aruna.l2_physics.models import PhysicsQuality
from aruna.l5_analogs.models import AnalogMatch, NO_ANALOG
from aruna.l5_analogs.orchestrator import AnalogOrchestrator

def create_physics_frame(
    timestamp: datetime,
    T_MK: float,
    EM_cm3: float,
    DNI: float,
    SHEI: float,
    flare_phase: str = "QUIET",
    hope_flag: bool = False
) -> PhysicsFrame:
    """Helper to instantiate a PhysicsFrame with exact required fields."""
    return PhysicsFrame(
        timestamp=timestamp,
        source_frame_ids=[],
        T_MK=T_MK,
        T_uncertainty_MK=0.5,
        EM_cm3=EM_cm3,
        EM_uncertainty=1e45,
        fit_chi_squared=1.1,
        DNI=DNI,
        tnpsv_magnitude=0.0,
        tnpsv_angle_rad=0.0,
        SHEI=SHEI,
        dSHEI_dt=0.0,
        HFD=0.0,
        FeKa_flux=0.0,
        QPP_period_sec=None,
        QPP_amplitude=None,
        dT_dt=0.0,
        dEM_dt=0.0,
        d2T_dt2=0.0,
        solexs_3_6kev=100.0,
        solexs_6_10kev=50.0,
        hel1os_10_25kev=10.0,
        hel1os_25_50kev=5.0,
        hel1os_50_70kev=2.0,
        flare_phase=flare_phase,
        hope_flag=hope_flag,
        hope_lead_time_sec=None,
        hope_reason=None,
        quality_flag=QualityFlag.GOOD
    )

def create_forecast_frame(predicted_class: str = "C") -> ForecastFrame:
    """Helper to instantiate a ForecastFrame with correct probability summation."""
    prob_quiet = 1.0 if predicted_class == "QUIET" else 0.0
    prob_B = 1.0 if predicted_class == "B" else 0.0
    prob_C = 1.0 if predicted_class == "C" else 0.0
    prob_M = 1.0 if predicted_class == "M" else 0.0
    prob_X = 1.0 if predicted_class == "X" else 0.0
    
    return ForecastFrame(
        timestamp=datetime.utcnow(),
        horizon_minutes=5,
        prob_quiet=prob_quiet, prob_B=prob_B, prob_C=prob_C, prob_M=prob_M, prob_X=prob_X,
        predicted_class=predicted_class, peak_class_prob=1.0,
        confidence_interval_low=0.8, confidence_interval_high=1.0,
        prediction_set=[predicted_class], calibration_ece=0.01,
        lead_time_estimate_sec=60, lead_time_uncertainty_sec=10,
        tcn_weight=0.5, moirai_weight=0.5,
        hope_flag_active=True, physics_state="HOPE",
        input_window_start=datetime.utcnow(), input_window_end=datetime.utcnow(),
        model_version="1", schema_version="1", pipeline_version="1"
    )

def populate_test_catalog(db_path: str, num_events: int = 10, pattern: str = "random"):
    """Helper to populate the master_flare_catalog SQLite database with dummy events."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS master_flare_catalog (
        event_id           TEXT PRIMARY KEY,
        start_time         TEXT,
        hope_time          TEXT,
        onset_time         TEXT,
        peak_time          TEXT,
        end_time           TEXT,
        goes_class         TEXT NOT NULL,
        goes_subclass      REAL,
        peak_flux_solexs   REAL DEFAULT 0,
        peak_flux_hel1os   REAL DEFAULT 0,
        peak_T_MK          REAL DEFAULT 0,
        peak_EM_cm3        REAL DEFAULT 0,
        peak_DNI           REAL DEFAULT 0,
        hope_lead_time_sec INTEGER,
        hope_fired         INTEGER DEFAULT 0,
        instruments        TEXT,
        detection_method   TEXT,
        confidence         REAL,
        flare_type         TEXT,
        pipeline_version   TEXT,
        provenance_hash    TEXT,
        notes              TEXT
    );
    """)
    conn.commit()

    rows = []
    rng = np.random.default_rng(42)
    for i in range(num_events):
        event_id = f"EVENT_{i:05d}"
        if pattern == "quiet":
            goes_class = "B"
            goes_subclass = round(rng.uniform(1.0, 3.0), 1)
            peak_T = rng.uniform(3.0, 5.0)
            peak_EM = rng.uniform(1e45, 5e45)
            peak_DNI = rng.uniform(0.01, 0.05)
        elif pattern == "mixed":
            goes_class = rng.choice(["B", "C", "M", "X"])
            goes_subclass = round(rng.uniform(1.0, 9.9), 1)
            peak_T = rng.uniform(5.0, 25.0)
            peak_EM = rng.uniform(1e46, 5e48)
            peak_DNI = rng.uniform(0.1, 4.0)
        else:
            goes_class = rng.choice(["B", "C", "M", "X"])
            goes_subclass = round(rng.uniform(1.0, 9.9), 1)
            peak_T = rng.uniform(5.0, 30.0)
            peak_EM = rng.uniform(1e46, 1e49)
            peak_DNI = rng.uniform(0.1, 5.0)

        rows.append((
            event_id, "2024-02-22T00:00:00", None, None, None, None,
            goes_class, goes_subclass, 100.0, 50.0,
            peak_T, peak_EM, peak_DNI, 120, 1, "SOLEXS,HEL1OS",
            "PEAK", 0.9, "STANDARD", "2.0.0", f"hash_{i}"
        ))

    cur.executemany("""
    INSERT OR REPLACE INTO master_flare_catalog (
        event_id, start_time, hope_time, onset_time, peak_time, end_time,
        goes_class, goes_subclass, peak_flux_solexs, peak_flux_hel1os,
        peak_T_MK, peak_EM_cm3, peak_DNI,
        hope_lead_time_sec, hope_fired, instruments,
        detection_method, confidence, flare_type, pipeline_version, provenance_hash
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()

def make_test_physics_frames(length: int = 100, amp: float = 1.0, phase_shift: float = 0.0) -> list[PhysicsFrame]:
    base_ts = datetime(2024, 2, 22, 0, 0, 0, tzinfo=timezone.utc)
    frames = []
    for i in range(length):
        t = i / float(length)
        T_val = float(5.0 + amp * 15.0 * np.sin(np.pi * t + phase_shift))
        EM_val = float(1e46 + amp * 5e48 * np.sin(np.pi * t + phase_shift))
        DNI_val = float(0.1 + amp * 3.0 * np.sin(np.pi * t + phase_shift))
        SHEI_val = float(-1.8 + amp * 1.2 * np.sin(np.pi * t + phase_shift))
        
        frames.append(create_physics_frame(
            timestamp=base_ts + timedelta(seconds=i),
            T_MK=T_val,
            EM_cm3=EM_val,
            DNI=DNI_val,
            SHEI=SHEI_val,
            flare_phase="HOPE",
            hope_flag=True
        ))
    return frames

def test_retrieval_precision_over_0_75(tmp_path):
    """Verify that retrieval precision of matching events is high (overall correctness >= 75%)."""
    db_file = str(tmp_path / "test_precision.db")
    index_file = str(tmp_path / "faiss_precision.index")
    
    # 1. Initialize orchestrator and database table
    num_events = 20
    orchestrator = AnalogOrchestrator(db_url=f"sqlite:///{db_file}", index_path=index_file)
    
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS master_flare_catalog (
        event_id           TEXT PRIMARY KEY,
        start_time         TEXT,
        hope_time          TEXT,
        onset_time         TEXT,
        peak_time          TEXT,
        end_time           TEXT,
        goes_class         TEXT NOT NULL,
        goes_subclass      REAL,
        peak_flux_solexs   REAL DEFAULT 0,
        peak_flux_hel1os   REAL DEFAULT 0,
        peak_T_MK          REAL DEFAULT 0,
        peak_EM_cm3        REAL DEFAULT 0,
        peak_DNI           REAL DEFAULT 0,
        hope_lead_time_sec INTEGER,
        hope_fired         INTEGER DEFAULT 0,
        instruments        TEXT,
        detection_method   TEXT,
        confidence         REAL,
        flare_type         TEXT,
        pipeline_version   TEXT,
        provenance_hash    TEXT,
        notes              TEXT
    );
    """)
    conn.commit()

    # 2. Register distinct timeseries for each event and construct DB rows
    event_timeseries = {}
    rows = []
    base_time = datetime(2024, 2, 22, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(num_events):
        eid = f"EVENT_{i:05d}"
        ts = np.zeros((100, 4))
        t = np.linspace(0, np.pi, 100)
        phase = i * 0.3
        amp = 1.0 + i * 1.5
        ts[:, 0] = 5.0 + amp * 1.5 * np.sin(t + phase)
        ts[:, 1] = 1e46 + amp * 1e47 * np.sin(t + phase)
        ts[:, 2] = 0.1 + amp * 0.2 * np.sin(t + phase)
        ts[:, 3] = -1.8 + amp * 0.1 * np.sin(t + phase)
        
        orchestrator.register_timeseries(eid, ts)
        event_timeseries[eid] = ts
        
        # Calculate peak values matching what query will extract
        peak_T = float(np.max(ts[:, 0]))
        peak_EM = float(np.max(ts[:, 1]))
        peak_DNI = float(np.max(ts[:, 2]))
        
        goes_class = "C"
        goes_subclass = 1.0 + i * 0.4
        
        rows.append((
            eid,
            base_time.isoformat(),
            None,
            None,
            base_time.isoformat(),
            (base_time + timedelta(seconds=99)).isoformat(),
            goes_class,
            goes_subclass,
            150.0,
            17.0,
            peak_T,
            peak_EM,
            peak_DNI,
            120,
            1,
            "SOLEXS,HEL1OS",
            "PEAK",
            0.9,
            "STANDARD",
            "2.0.0",
            f"hash_{i}"
        ))

    cur.executemany("""
    INSERT OR REPLACE INTO master_flare_catalog (
        event_id, start_time, hope_time, onset_time, peak_time, end_time,
        goes_class, goes_subclass, peak_flux_solexs, peak_flux_hel1os,
        peak_T_MK, peak_EM_cm3, peak_DNI,
        hope_lead_time_sec, hope_fired, instruments,
        detection_method, confidence, flare_type, pipeline_version, provenance_hash
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    conn.close()
        
    orchestrator.index.rebuild()
    
    # 3. Query using a slightly noisy version of 10 selected events
    correct_matches = 0
    query_count = 10
    
    for i in range(query_count):
        target_eid = f"EVENT_{i:05d}"
        target_ts = event_timeseries[target_eid]
        noisy_ts = target_ts + np.random.normal(0, 0.01, target_ts.shape)
        
        # Build mock PhysicsFrames from noisy_ts with matching timestamps
        query_frames = []
        for idx in range(100):
            query_frames.append(create_physics_frame(
                timestamp=base_time + timedelta(seconds=idx),
                T_MK=noisy_ts[idx, 0],
                EM_cm3=noisy_ts[idx, 1],
                DNI=noisy_ts[idx, 2],
                SHEI=0.0,  # Set to 0.0 to match DB
                flare_phase="HOPE",
                hope_flag=True
            ))
            
        goes_class = "C"
        goes_subclass = 1.0 + i * 0.4
        
        mock_forecast = create_forecast_frame(predicted_class=goes_class)
        mock_forecast.goes_class = goes_class
        mock_forecast.goes_subclass = goes_subclass
        mock_forecast.confidence = 0.9
        mock_forecast.hope_lead_time_sec = 120
        
        matches = orchestrator.process(mock_forecast, query_frames)
        if matches and matches[0].event_id == target_eid:
            correct_matches += 1
            
    precision = correct_matches / query_count
    assert precision >= 0.75


def test_faiss_search_under_100ms_for_10k_events(tmp_path):
    """Performance test: verify FAISS search executes in < 100ms on a 10,000-event catalog."""
    db_file = str(tmp_path / "test_perf.db")
    index_file = str(tmp_path / "faiss_perf.index")
    
    # Bulk populate 10,000 events
    populate_test_catalog(db_file, num_events=10000, pattern="mixed")
    
    orchestrator = AnalogOrchestrator(db_url=f"sqlite:///{db_file}", index_path=index_file)
    orchestrator.index.rebuild()
    
    # Measure execution time of a single search
    query_frames = make_test_physics_frames(length=100)
    mock_forecast = create_forecast_frame(predicted_class="C")
    
    # Warm up
    orchestrator.process(mock_forecast, query_frames)
    
    # Measure
    t0 = time.perf_counter()
    matches = orchestrator.process(mock_forecast, query_frames)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    
    print(f"FAISS search with 10k database took {elapsed_ms:.2f} ms")
    assert elapsed_ms < 100.0


def test_no_analog_fires_when_dtw_over_3_sigma(tmp_path):
    """Verify that NO_ANALOG fallback triggers if DTW distance exceeds mean + 3 * std."""
    db_file = str(tmp_path / "test_fallback.db")
    index_file = str(tmp_path / "faiss_fallback.index")
    
    # Populate catalog with 10 quiet events
    populate_test_catalog(db_file, num_events=10, pattern="quiet")
    
    orchestrator = AnalogOrchestrator(db_url=f"sqlite:///{db_file}", index_path=index_file)
    
    # Register very quiet timeseries
    for i in range(10):
        eid = f"EVENT_{i:05d}"
        ts = np.zeros((100, 4))
        ts[:, 0] = 3.0
        ts[:, 1] = 1e45
        ts[:, 2] = 0.01
        ts[:, 3] = -1.8
        orchestrator.register_timeseries(eid, ts)
        
    orchestrator.index.rebuild()
    
    # Now query with a massive X-class flare
    query_frames = make_test_physics_frames(length=100, amp=10.0, phase_shift=0.5)
    mock_forecast = create_forecast_frame(predicted_class="X")
    
    matches = orchestrator.process(mock_forecast, query_frames)
    
    assert len(matches) == 1
    assert matches[0] == NO_ANALOG
    assert matches[0].no_analog_flag is True


@pytest.mark.physics
def test_x63_retrieves_at_least_one_x_class_analog(tmp_path):
    """Verify that querying with an X-class event retrieves at least one historical X-class analog."""
    db_file = str(tmp_path / "test_xclass.db")
    index_file = str(tmp_path / "faiss_xclass.index")
    
    populate_test_catalog(db_file, num_events=5, pattern="quiet")
    
    # Append the X-class event manually
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO master_flare_catalog (
        event_id, start_time, goes_class, goes_subclass, peak_flux_solexs, peak_flux_hel1os,
        peak_T_MK, peak_EM_cm3, peak_DNI, hope_lead_time_sec, hope_fired, instruments,
        detection_method, confidence, flare_type, pipeline_version, provenance_hash
    ) VALUES (
        'EVENT_XCLASS', '2024-02-22T00:00:00', 'X', 6.3, 50000.0, 10000.0,
        25.0, 1e49, 4.5, 120, 1, 'SOLEXS,HEL1OS', 'PEAK', 1.0, 'STANDARD', '2.0.0', 'hash_x'
    )
    """)
    conn.commit()
    conn.close()
    
    orchestrator = AnalogOrchestrator(db_url=f"sqlite:///{db_file}", index_path=index_file)
    
    for i in range(5):
        eid = f"EVENT_{i:05d}"
        ts = np.zeros((100, 4))
        ts[:, 0] = 3.0 + np.random.normal(0, 0.1, 100)
        ts[:, 1] = 1e45 + np.random.normal(0, 1e44, 100)
        ts[:, 2] = 0.01 + np.random.normal(0, 0.005, 100)
        ts[:, 3] = -1.8 + np.random.normal(0, 0.05, 100)
        orchestrator.register_timeseries(eid, ts)
        
    # Register timeseries for the X-class event
    x_ts = np.zeros((100, 4))
    t = np.linspace(0, np.pi, 100)
    x_ts[:, 0] = 5.0 + 20.0 * np.sin(t)
    x_ts[:, 1] = 1e46 + 1e49 * np.sin(t)
    x_ts[:, 2] = 0.1 + 4.4 * np.sin(t)
    x_ts[:, 3] = -1.8 + 1.2 * np.sin(t)
    orchestrator.register_timeseries("EVENT_XCLASS", x_ts)
    
    orchestrator.index.rebuild()
    
    # Query with the X-class event
    query_frames = make_test_physics_frames(length=100, amp=1.0)
    # Inject X6.3 values
    for idx, f in enumerate(query_frames):
        f.T_MK = x_ts[idx, 0]
        f.EM_cm3 = x_ts[idx, 1]
        f.DNI = x_ts[idx, 2]
        f.SHEI = x_ts[idx, 3]
        
    mock_forecast = create_forecast_frame(predicted_class="X")
    
    matches = orchestrator.process(mock_forecast, query_frames)
    
    assert len(matches) > 0
    assert any(m.goes_class == "X" for m in matches if m.event_id != "NO_ANALOG")


def test_faiss_index_auto_rebuilds_after_new_catalog_entry(tmp_path):
    """Verify that FAISS index automatically rebuilds when a new catalog entry is detected."""
    db_file = str(tmp_path / "test_autorebuild.db")
    index_file = str(tmp_path / "faiss_autorebuild.index")
    
    populate_test_catalog(db_file, num_events=5, pattern="mixed")
    
    orchestrator = AnalogOrchestrator(db_url=f"sqlite:///{db_file}", index_path=index_file)
    
    for i in range(5):
        eid = f"EVENT_{i:05d}"
        ts = np.random.normal(size=(100, 4))
        orchestrator.register_timeseries(eid, ts)
        
    orchestrator.index.rebuild()
    assert len(orchestrator.index.event_ids) == 5
    
    query_frames = make_test_physics_frames(length=100)
    mock_forecast = create_forecast_frame(predicted_class="C")
    
    matches1 = orchestrator.process(mock_forecast, query_frames)
    
    # Add a 6th event directly to the database
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO master_flare_catalog (
        event_id, start_time, goes_class, goes_subclass, peak_flux_solexs, peak_flux_hel1os,
        peak_T_MK, peak_EM_cm3, peak_DNI, hope_lead_time_sec, hope_fired, instruments,
        detection_method, confidence, flare_type, pipeline_version, provenance_hash
    ) VALUES (
        'EVENT_00005', '2024-02-22T00:00:00', 'M', 2.5, 500.0, 100.0,
        15.0, 1e47, 1.2, 120, 1, 'SOLEXS,HEL1OS', 'PEAK', 0.8, 'STANDARD', '2.0.0', 'hash_new'
    )
    """)
    conn.commit()
    conn.close()
    
    orchestrator.register_timeseries("EVENT_00005", np.random.normal(size=(100, 4)))
    
    # Trigger a second search, which should trigger an automatic rebuild
    matches2 = orchestrator.process(mock_forecast, query_frames)
    
    assert len(orchestrator.index.event_ids) == 6
    assert "EVENT_00005" in orchestrator.index.event_ids


def test_cold_start_empty_catalog_no_crash(tmp_path):
    """Cold-start with 0 events in catalog activates NO_ANALOG mode without crashing."""
    db_file = str(tmp_path / "test_cold.db")
    index_file = str(tmp_path / "faiss_cold.index")
    
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE IF NOT EXISTS master_flare_catalog (event_id TEXT PRIMARY KEY);")
    conn.commit()
    conn.close()
    
    orchestrator = AnalogOrchestrator(db_url=f"sqlite:///{db_file}", index_path=index_file)
    
    query_frames = make_test_physics_frames(length=100)
    mock_forecast = create_forecast_frame(predicted_class="C")
    
    matches = orchestrator.process(mock_forecast, query_frames)
    
    assert len(matches) == 1
    assert matches[0] == NO_ANALOG
    assert matches[0].no_analog_flag is True
