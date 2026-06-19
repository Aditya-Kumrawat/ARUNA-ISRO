"""
ARUNA Validation Engine — PRD §19, §20.

Usage:
    from aruna.validation import ValidationEngine
    engine = ValidationEngine()
    result = engine.evaluate(nowcast_events, period_start, period_end)
    print(f"TSS = {result.tss:.3f}  Kappa = {result.kappa:.3f}")
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from aruna.models.nowcast import NowcastEvent
from aruna.models.forecast import ForecastFrame
from aruna.validation.models import MatchedEvent, ValidationResult
from aruna.validation.ground_truth import fetch_goes_catalog, match_all
from aruna.validation.metrics import compute_all

logger = logging.getLogger(__name__)


class ValidationEngine:
    """
    Computes full WMO-grade validation metrics for a batch of ARUNA detections.

    Parameters
    ----------
    catalog_source : "sunpy_hek" (default) | "mock" (inject via _mock_catalog)
    bootstrap_n    : number of bootstrap resamples (default 2000, per PRD §20.4)
    """

    def __init__(
        self,
        catalog_source: str = "sunpy_hek",
        bootstrap_n: int = 2000,
    ) -> None:
        self.catalog_source = catalog_source
        self.bootstrap_n = bootstrap_n

    def evaluate(
        self,
        nowcast_events: List[NowcastEvent],
        period_start: datetime,
        period_end: datetime,
        _mock_catalog: Optional[List[dict]] = None,
    ) -> ValidationResult:
        """
        Run full validation against the GOES catalog.

        Parameters
        ----------
        nowcast_events : list of NowcastEvents from L3
        period_start   : start of evaluation window
        period_end     : end of evaluation window
        _mock_catalog  : inject synthetic catalog for tests (skips HEK query)

        Returns
        -------
        ValidationResult with confusion matrix, TSS, Kappa, ECE, bootstrap CIs, etc.
        """
        catalog = fetch_goes_catalog(period_start, period_end, _mock_catalog=_mock_catalog)
        matched = match_all(nowcast_events, catalog)
        result  = compute_all(
            matched,
            period_start=period_start,
            period_end=period_end,
            bootstrap_n=self.bootstrap_n,
        )
        logger.info(
            "Validation complete: n=%d matched=%d TSS=%.3f Kappa=%.3f ECE=%.4f",
            result.n_events, result.n_matched, result.tss, result.kappa, result.ece,
        )
        return result

    def evaluate_from_matched(
        self,
        matched_events: List[MatchedEvent],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> ValidationResult:
        """Run metrics on pre-matched events (skip catalog fetch)."""
        return compute_all(
            matched_events,
            period_start=period_start,
            period_end=period_end,
            bootstrap_n=self.bootstrap_n,
        )
