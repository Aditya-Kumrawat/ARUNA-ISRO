import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from dtaidistance import dtw_ndim
    _DTAIDISTANCE_AVAILABLE = True
except ImportError:
    _DTAIDISTANCE_AVAILABLE = False
    logger.warning("dtaidistance package not available. Falling back to pure NumPy DTW.")

def z_score_sequence(seq: np.ndarray) -> np.ndarray:
    """z-score standardize each timeseries feature sequence."""
    if len(seq) == 0:
        return seq
    mean = np.mean(seq, axis=0)
    std = np.std(seq, axis=0)
    # Prevent division by zero
    std = np.where(std == 0.0, 1.0, std)
    return (seq - mean) / std

def compute_numpy_dtw(s1: np.ndarray, s2: np.ndarray) -> float:
    """Pure NumPy fallback for multi-dimensional DTW."""
    n = len(s1)
    m = len(s2)
    dtw_matrix = np.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0.0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )
    return float(dtw_matrix[n, m])

def compute_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Computes the DTW distance between two multidimensional time series.
    Both sequences are z-score standardized prior to matching.
    """
    if len(seq1) == 0 or len(seq2) == 0:
        return float('inf')

    s1 = z_score_sequence(seq1)
    s2 = z_score_sequence(seq2)

    if _DTAIDISTANCE_AVAILABLE:
        try:
            # dtaidistance expects float64 contiguous arrays
            s1_c = np.ascontiguousarray(s1, dtype=np.float64)
            s2_c = np.ascontiguousarray(s2, dtype=np.float64)
            dist = dtw_ndim.distance(s1_c, s2_c)
            # If the library returns NaN or invalid distance, use fallback
            if np.isnan(dist):
                return compute_numpy_dtw(s1, s2)
            return float(dist)
        except Exception as e:
            logger.debug("dtaidistance failed: %s. Using NumPy fallback.", e)
            return compute_numpy_dtw(s1, s2)
    else:
        return compute_numpy_dtw(s1, s2)
