from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

_DATA_DIR   = Path(__file__).parent / "data"
_TABLE_PATH = _DATA_DIR / "chianti_table.npz"

T_GRID  = np.logspace(np.log10(1.0),  np.log10(100.0), 50)
EM_GRID = np.logspace(44.0, 52.0, 50)


def _bremsstrahlung(energies_kev: np.ndarray, T_MK: float, EM_cm3: float) -> np.ndarray:
    T_K  = T_MK * 1e6
    kT   = 8.617e-8 * T_K        # keV
    A    = 9.52e-24               # MKG normalisation (cm^3 s^-1 keV^-1)
    spec = A * EM_cm3 * (T_K ** -0.5) * np.exp(-energies_kev / (kT + 1e-30)) * 1.2
    return np.clip(spec, 0.0, None)


class CHIANTILookupTable:
    def __init__(self) -> None:
        self._interp: RegularGridInterpolator | None = None

    def generate(self, energy_bins: np.ndarray) -> None:
        centres = (energy_bins[:-1] + energy_bins[1:]) / 2.0
        n_e     = len(centres)
        table   = np.zeros((len(T_GRID), len(EM_GRID), n_e), dtype=np.float32)
        for i, T in enumerate(T_GRID):
            for j, EM in enumerate(EM_GRID):
                table[i, j, :] = _bremsstrahlung(centres, T, EM)
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            _TABLE_PATH, table=table,
            T_grid=T_GRID, EM_grid=EM_GRID, energy_bins=energy_bins,
        )
        logger.info("CHIANTILookupTable generated at %s", _TABLE_PATH)
        self._load(energy_bins)

    def _load(self, energy_bins: np.ndarray) -> None:
        data = np.load(_TABLE_PATH)
        self._interp = RegularGridInterpolator(
            (np.log10(data["T_grid"]), np.log10(data["EM_grid"])),
            data["table"].astype(np.float64),
            method="linear", bounds_error=False, fill_value=0.0,
        )

    def lookup(self, T_MK: float, EM_cm3: float, energy_bins: np.ndarray) -> np.ndarray:
        if self._interp is None:
            if _TABLE_PATH.exists():
                self._load(energy_bins)
            else:
                self.generate(energy_bins)
        T_log  = np.clip(np.log10(max(T_MK,  1e-30)), np.log10(T_GRID[0]),  np.log10(T_GRID[-1]))
        EM_log = np.clip(np.log10(max(EM_cm3, 1e-30)), np.log10(EM_GRID[0]), np.log10(EM_GRID[-1]))
        return self._interp([[T_log, EM_log]])[0]
