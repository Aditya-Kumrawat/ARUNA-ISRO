from dataclasses import dataclass
import numpy as np

@dataclass
class AnalogMatch:
    event_id: str
    goes_class: str
    dtw_distance: float
    similarity: float         # 1 / (1 + dtw_distance)
    feature_vector: np.ndarray
    no_analog_flag: bool = False

NO_ANALOG = AnalogMatch(
    event_id="NO_ANALOG",
    goes_class="UNKNOWN",
    dtw_distance=float("inf"),
    similarity=0.0,
    feature_vector=np.zeros(17),
    no_analog_flag=True
)

