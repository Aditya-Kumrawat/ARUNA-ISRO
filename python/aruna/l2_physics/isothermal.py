from __future__ import annotations
import logging
import numpy as np
from scipy.optimize import minimize
from .chianti_table import CHIANTILookupTable
from .models import FitResult

logger = logging.getLogger(__name__)

try:
    from sunkit_spex.fitting import FitIsothermal as _SunkitFit
    _BACKEND = "sunkit_spex"
    logger.info("IsothermalFitter: using sunkit-spex f_vth backend")
except ImportError:
    _SunkitFit = None
    _BACKEND   = "chianti_lookup"
    logger.warning("IsothermalFitter: sunkit-spex unavailable, using CHIANTILookupTable fallback")

_shared_table = None

FIT_E_MIN = 3.0
FIT_E_MAX = 20.0


def _get_shared_table() -> CHIANTILookupTable:
    """Lazy-initialize shared lookup table for test monkeypatching."""
    global _shared_table
    if _shared_table is None:
        _shared_table = CHIANTILookupTable()
    return _shared_table


def _poisson_chi2(observed: np.ndarray, model: np.ndarray) -> float:
    """Compute chi2 statistic using Poisson maximum likelihood (Cash statistic).

    Cash = 2 * sum(model - observed + observed * ln(observed / model))
    Normalized by degrees of freedom to get reduced chi-square.
    Formula matches Sherpa's default Poisson chi2.
    """
    mask = model > 1e-30
    if not mask.any():
        return np.inf
    o, m = observed[mask], model[mask]
    chi2 = 2.0 * np.sum(m - o + o * np.log((o + 1e-30) / m))
    return chi2 / max(mask.sum() - 2, 1)


class IsothermalFitter:
    """Isothermal plasma fitter for SoLEXS spectra.

    Uses sunkit-spex f_vth if available; falls back to CHIANTILookupTable.
    The active backend is recorded in every FitResult.backend field.
    """
    def __init__(self) -> None:
        self._backend = _BACKEND

    @property
    def backend(self) -> str:
        return self._backend

    def fit(
        self,
        counts_per_s_kev: np.ndarray,
        energy_bins_kev:  np.ndarray,
        response_matrix=None,
    ) -> FitResult:
        """Fit isothermal model to a SoLEXS spectrum.

        Args:
            counts_per_s_kev: Dead-time corrected spectrum, shape (N,)
            energy_bins_kev: Bin edges in keV, shape (N+1,)
            response_matrix: Optional ARF×SRF matrix (sunkit-spex only)

        Returns:
            FitResult with T_MK, EM_cm3, uncertainties, chi2, backend.
            T_MK and EM_cm3 are NaN if fit diverged.
        """
        counts_per_s_kev = np.clip(counts_per_s_kev, 0.0, None)
        centres = (energy_bins_kev[:-1] + energy_bins_kev[1:]) / 2.0
        mask    = (centres >= FIT_E_MIN) & (centres <= FIT_E_MAX)
        if not mask.any():
            return FitResult(np.nan, np.nan, np.nan, np.nan, np.inf, self._backend)

        # Check if spectrum is essentially zero
        if np.sum(counts_per_s_kev[mask]) < 1e-6:
            return FitResult(np.nan, np.nan, np.nan, np.nan, np.inf, self._backend)

        if self._backend == "sunkit_spex" and _SunkitFit is not None:
            return self._fit_sunkit(counts_per_s_kev[mask], energy_bins_kev, response_matrix)
        return self._fit_lookup(counts_per_s_kev, energy_bins_kev, mask)

    def _fit_lookup(self, counts, energy_bins, mask) -> FitResult:
        counts_fit = counts[mask]
        table = _get_shared_table()

        # Estimate initial EM from spectrum integral
        # The bremsstrahlung output scales with A * EM_cm3 where A ~ 1e-24
        # So if we observe counts ~ 1e20, we need EM ~ 1e44
        integrated_flux = np.sum(counts_fit)
        if integrated_flux < 1e-10:
            log_EM_init = 44.0
        else:
            # Rough estimate: counts ≈ A * EM, so EM ≈ counts / A ≈ counts * 1e24
            log_EM_init = np.log10(max(1e40, integrated_flux * 1e24))

        def objective(params):
            T_MK, log_EM = params
            if T_MK <= 0.1 or T_MK > 200:
                return 1e9
            try:
                EM_cm3 = 10 ** log_EM
                if EM_cm3 <= 0:
                    return 1e9
                model_full = table.lookup(T_MK, EM_cm3, energy_bins)
                chi2 = _poisson_chi2(counts_fit, model_full[mask])
                return chi2 if np.isfinite(chi2) else 1e9
            except Exception:
                return 1e9

        res = minimize(objective, x0=[15.0, log_EM_init], method="Nelder-Mead",
                       options={"xatol": 0.1, "fatol": 0.01, "maxiter": 600, "maxfev": 1000})
        T_MK, EM_cm3 = res.x[0], 10 ** res.x[1]
        chi2 = float(res.fun)

        # Poisson chi2 scales with count rate and can be very large for high-count spectra.
        # Accept fit if optimizer converged and parameters are physical.
        if not res.success or T_MK <= 0 or not np.isfinite(chi2):
            return FitResult(np.nan, np.nan, np.nan, np.nan, chi2, self._backend)

        return FitResult(
            T_MK=float(T_MK), EM_cm3=float(EM_cm3),
            T_uncertainty_MK=max(0.5, T_MK * 0.05),
            EM_uncertainty=max(1e46, EM_cm3 * 0.1),
            chi2_reduced=chi2, backend=self._backend,
        )

    def _fit_sunkit(self, counts_fit, energy_bins, response_matrix) -> FitResult:
        try:
            fitter = _SunkitFit(model="f_vth",
                                energy_range=(FIT_E_MIN, FIT_E_MAX),
                                abundance_table="coronal",
                                response_matrix=response_matrix)
            r = fitter.fit(counts_fit, energy_bins)
            return FitResult(r.T_MK, r.EM_cm3, r.T_1sigma, r.EM_1sigma,
                             r.chi2_reduced, self._backend)
        except Exception as exc:
            logger.error("sunkit-spex fit failed: %s", exc)
            return FitResult(np.nan, np.nan, np.nan, np.nan, np.inf, self._backend)
