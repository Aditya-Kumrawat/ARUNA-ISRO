"""Debug why L2 physics engine produces no frames from real data."""
import sys, logging
sys.path.insert(0, 'python')
logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s %(name)s - %(message)s')

from aruna.pipeline.run import _find_fits, run_l1_pradan
from aruna.l2_physics import PhysicsEngine, PhysicsConfig
from pathlib import Path
import numpy as np

fits_dir = Path('data/solexs_20240222/')
fits_files = _find_fits(fits_dir)
print(f"FITS files: {fits_files}")

obs_frames = run_l1_pradan(fits_files, single_instrument=True, window_sec=300)
print(f"\nObs frames: {len(obs_frames)}")
if obs_frames:
    f = obs_frames[0]
    print(f"  [0] instrument={f.instrument} ts={f.timestamp}")
    print(f"  [0] counts_per_s_kev shape={np.array(f.counts_per_s_kev).shape}")
    print(f"  [0] energy_bins_kev shape={np.array(f.energy_bins_kev).shape}")

sol_frames = [f for f in obs_frames if "SOLEXS" in f.instrument]
hel_frames = [f for f in obs_frames if "HEL1OS" in f.instrument or "HEL" in f.instrument]
print(f"\nSOLEXS: {len(sol_frames)}  HEL1OS: {len(hel_frames)}")

if sol_frames and hel_frames:
    engine = PhysicsEngine(config=PhysicsConfig())
    sol = sol_frames[0]
    hel = hel_frames[0]
    print(f"\nTrying process() on first pair:")
    print(f"  SOL ts={sol.timestamp}  HEL ts={hel.timestamp}")
    try:
        pf = engine.process(sol, hel)
        print(f"  SUCCESS: {pf}")
    except Exception as exc:
        import traceback
        print(f"  FAILED: {exc}")
        traceback.print_exc()
