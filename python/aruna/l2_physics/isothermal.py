from __future__ import annotations
import logging
import numpy as np
from scipy.optimize import minimize
from .chianti_table import CHIANTILookupTable
from .models import FitResult

logger = logging.getLogger(__name__)

_SunkitFit = None
_BACKEND   = "chianti_lookup"

try:
    # sunkit-spex fitting API: SpexFitter with f_vth thermal model
    from sunkit_spex.fitting.fitter import SpexFitter as _SunkitFit
    _BACKEND = "sunkit_spex"
    logger.info("IsothermalFitter: using sunkit-spex SpexFitter (f_vth) backend")
except ImportError as _spex_err:
    try:
        import sunkit_spex as _probe
        # Package is installed but the fitter module path is different in this version
        logger.warning(
            "IsothermalFitter: sunkit-spex %s installed but SpexFitter not found (%s) "
            "— using CHIANTILookupTable fallback. "
            "Try: pip install --upgrade sunkit-spex",
            getattr(_probe, '__version__', 'unknown'), _spex_err,
        )
    except ImportError:
        logger.warning("IsothermalFitter: sunkit-spex not installed — using CHIANTILookupTable fallback")

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

    def _fit_lookup_direct(self, counts_fit, energy_bins) -> FitResult:
        """Lookup-table path called as fallback when sunkit-spex fit raises."""
        centres = (energy_bins[:-1] + energy_bins[1:]) / 2.0
        mask    = (centres >= FIT_E_MIN) & (centres <= FIT_E_MAX)
        return self._fit_lookup(counts_fit, energy_bins, mask)

    def _fit_lookup(self, counts, energy_bins, mask) -> FitResult:
        counts_fit = counts[mask]
        table = _get_shared_table()

        data_total = float(np.sum(counts_fit))
        if data_total < 1e-10:
            return FitResult(np.nan, np.nan, np.nan, np.nan, np.inf, self._backend)

        # 1-D optimisation over T only.
        # For each T, the analytically-optimal EM is the one that scales the
        # model's total flux to match the data's total flux.  This is exact
        # because _bremsstrahlung scales linearly with EM, so:
        #   EM_opt = EM_ref * (data_total / model_total_at_EM_ref)
        # Temperature is recovered from the spectral *shape* alone, decoupled
        # from the absolute amplitude.  Works for synthetic spectra (EM ~ 1e48)
        # and real detector counts (effective EM ~ 1e30) without a response matrix.
        _EM_REF = 1e48
        _best   = [np.inf, _EM_REF]   # [best_chi2, best_EM] — avoids re-lookup after opt

        def objective(T_arr):
            T_MK = T_arr[0]
            if T_MK <= 0.5 or T_MK > 200:
                return 1e9
            try:
                model_ref = table.lookup(T_MK, _EM_REF, energy_bins)[mask]
                ref_total = float(np.sum(model_ref))
                if ref_total < 1e-30:
                    return 1e9
                scale         = data_total / ref_total
                model_scaled  = model_ref * scale
                chi2          = _poisson_chi2(counts_fit, model_scaled)
                val           = float(chi2) if np.isfinite(chi2) else 1e9
                if val < _best[0]:
                    _best[0] = val
                    _best[1] = _EM_REF * scale
                return val
            except Exception:
                return 1e9

        res  = minimize(objective, x0=[15.0], method="Nelder-Mead",
                        options={"xatol": 0.05, "fatol": 1e-4, "maxiter": 400, "maxfev": 600})
        T_MK = float(res.x[0])
        chi2 = float(res.fun)

        if not res.success or T_MK <= 0 or not np.isfinite(chi2):
            return FitResult(np.nan, np.nan, np.nan, np.nan, chi2, self._backend)

        EM_cm3 = _best[1]   # analytically-optimal EM captured during optimisation

        return FitResult(
            T_MK=float(T_MK), EM_cm3=float(EM_cm3),
            T_uncertainty_MK=max(0.5, T_MK * 0.05),
            EM_uncertainty=max(1e46, abs(EM_cm3) * 0.1),
            chi2_reduced=chi2, backend=self._backend,
        )

    def _fit_sunkit(self, counts_fit, energy_bins, response_matrix) -> FitResult:
        try:
            import astropy.units as u

            energies = energy_bins * u.keV
            counts_u = counts_fit * u.ct / u.s / u.keV

            fitter = _SunkitFit()
            fitter.add_data(counts_u, energies)
            fitter.add_model("f_vth")
            fitter.fit(energy_range=[FIT_E_MIN, FIT_E_MAX] * u.keV)

            params = fitter.params
            T_MK   = float(params["f_vth_T"].value)      # MK
            EM_cm3 = float(params["f_vth_EM"].value)     # cm^-3
            chi2   = float(fitter.rstat)

            # Uncertainties: use 5 % of value as a conservative estimate
            # when the fitter does not expose 1-sigma directly
            T_unc  = max(0.5, T_MK  * 0.05)
            EM_unc = max(1e46, abs(EM_cm3) * 0.1)

            return FitResult(T_MK, EM_cm3, T_unc, EM_unc, chi2, self._backend)
        except Exception as exc:
            logger.warning("sunkit-spex fit failed (%s) — falling back to lookup table", exc)
            return self._fit_lookup_direct(counts_fit, energy_bins)
