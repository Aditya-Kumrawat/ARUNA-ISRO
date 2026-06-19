import numpy as np
from typing import List, Union, Optional
from datetime import datetime
from aruna.models.physics import PhysicsFrame
from aruna.models.nowcast import NowcastEvent

class PhysicsFeatureEmbedder:
    """
    Extracts 17 physics features from NowcastEvent metadata / DB row or PhysicsFrame list.
    Normalizes via a fitted StandardScaler (if provided) and L2 normalization.
    """
    FEATURES = [
        # Thermal (4)
        "peak_T_MK", "peak_EM_cm3", "dT_dt_peak", "dEM_dt_peak",
        # Nonthermal (3)
        "peak_SHEI", "peak_dSHEI_dt", "peak_HEL_flux",
        # Dynamics (7)
        "peak_DNI", "peak_TNPSV_mag", "peak_TNPSV_angle",
        "hope_lead_time_s", "flare_duration_s", "rise_time_s", "decay_time_s",
        # Classification (3)
        "goes_class_numeric", "goes_subclass", "confidence"
    ]

    def __init__(self, scaler=None) -> None:
        self.scaler = scaler

    def encode(self, item: Union[List[PhysicsFrame], dict, NowcastEvent], event: Optional[NowcastEvent] = None) -> np.ndarray:
        """
        Encodes the flare event into a 17-dimensional vector, scales it, and L2 normalizes it.
        """
        raw_vec = self.extract_raw(item, event)
        if self.scaler is not None:
            scaled = self.scaler.transform(raw_vec.reshape(1, -1))[0]
        else:
            scaled = raw_vec

        norm = np.linalg.norm(scaled)
        if norm > 0:
            return scaled / norm
        return scaled

    def extract_raw(self, item: Union[List[PhysicsFrame], dict, NowcastEvent, np.ndarray], event: Optional[NowcastEvent] = None) -> np.ndarray:
        """
        Extracts the 17 raw features as a numpy array of shape (17,).
        """
        if isinstance(item, np.ndarray):
            return item

        features = {k: 0.0 for k in self.FEATURES}

        # GOES class helper
        def _class_num(c: str) -> float:
            mapping = {"B": 1.0, "C": 2.0, "M": 3.0, "X": 4.0}
            return mapping.get(c, 0.0)

        def _signed_log(val: float) -> float:
            return float(np.sign(val) * np.log10(np.abs(val) + 1.0))

        # 1. Dictionary / DB row
        if isinstance(item, dict):
            features["peak_T_MK"] = float(item.get("peak_T_MK") or 0.0)
            features["peak_EM_cm3"] = _signed_log(float(item.get("peak_EM_cm3") or 0.0))
            features["peak_DNI"] = float(item.get("peak_DNI") or 0.0)
            features["goes_subclass"] = float(item.get("goes_subclass") or 0.0)
            features["confidence"] = float(item.get("confidence") or 0.0)
            features["goes_class_numeric"] = _class_num(item.get("goes_class", ""))
            features["hope_lead_time_s"] = float(item.get("hope_lead_time_sec") or item.get("hope_lead_time_s") or 0.0)
            features["peak_HEL_flux"] = float(item.get("peak_flux_hel1os") or 0.0)

            # parse dates for durations
            start_time = item.get("start_time")
            peak_time = item.get("peak_time")
            end_time = item.get("end_time")

            def _parse_t(t_val):
                if isinstance(t_val, str):
                    try:
                        return datetime.fromisoformat(t_val)
                    except Exception:
                        return None
                if isinstance(t_val, datetime):
                    return t_val
                return None

            t_start = _parse_t(start_time)
            t_peak = _parse_t(peak_time)
            t_end = _parse_t(end_time)

            if t_start and t_end:
                features["flare_duration_s"] = (t_end - t_start).total_seconds()
            if t_start and t_peak:
                features["rise_time_s"] = (t_peak - t_start).total_seconds()
            if t_peak and t_end:
                features["decay_time_s"] = (t_end - t_peak).total_seconds()

        # 2. NowcastEvent object
        elif isinstance(item, NowcastEvent):
            features["goes_class_numeric"] = _class_num(item.goes_class)
            features["goes_subclass"] = float(item.goes_subclass)
            features["confidence"] = float(item.confidence)
            features["hope_lead_time_s"] = float(item.lead_time_sec or 0.0)

        # 3. List of PhysicsFrame objects
        elif isinstance(item, list) and len(item) > 0 and isinstance(item[0], PhysicsFrame):
            frames: List[PhysicsFrame] = item
            T_MK = [f.T_MK for f in frames if f.T_MK is not None]
            EM_cm3 = [f.EM_cm3 for f in frames if f.EM_cm3 is not None]
            dT_dt = [f.dT_dt for f in frames if f.dT_dt is not None]
            dEM_dt = [f.dEM_dt for f in frames if f.dEM_dt is not None]
            SHEI = [f.SHEI for f in frames if f.SHEI is not None]
            dSHEI_dt = [f.dSHEI_dt for f in frames if f.dSHEI_dt is not None]
            DNI = [f.DNI for f in frames if f.DNI is not None]
            tnpsv_mag = [f.tnpsv_magnitude for f in frames if f.tnpsv_magnitude is not None]
            tnpsv_ang = [f.tnpsv_angle_rad for f in frames if f.tnpsv_angle_rad is not None]

            # Peak flux index
            fluxes = [
                (f.solexs_3_6kev or 0.0) + (f.solexs_6_10kev or 0.0)
                for f in frames
            ]
            peak_idx = int(np.argmax(fluxes)) if fluxes else 0

            # compute peak features
            features["peak_T_MK"] = float(max(T_MK)) if T_MK else 0.0
            features["peak_EM_cm3"] = _signed_log(float(max(EM_cm3)) if EM_cm3 else 0.0)
            features["dT_dt_peak"] = float(max(dT_dt)) if dT_dt else 0.0
            features["dEM_dt_peak"] = _signed_log(float(max(dEM_dt)) if dEM_dt else 0.0)
            features["peak_SHEI"] = float(max(SHEI)) if SHEI else 0.0
            features["peak_dSHEI_dt"] = float(max(dSHEI_dt)) if dSHEI_dt else 0.0

            hel_fluxes = [
                (f.hel1os_10_25kev or 0.0) + (f.hel1os_25_50kev or 0.0) + (f.hel1os_50_70kev or 0.0)
                for f in frames
            ]
            features["peak_HEL_flux"] = float(max(hel_fluxes)) if hel_fluxes else 0.0
            features["peak_DNI"] = float(max(DNI)) if DNI else 0.0
            features["peak_TNPSV_mag"] = float(max(tnpsv_mag)) if tnpsv_mag else 0.0
            features["peak_TNPSV_angle"] = float(tnpsv_ang[np.argmax(tnpsv_mag)]) if tnpsv_mag else 0.0

            # Lead time
            hope_lead = 0.0
            for f in frames:
                if f.hope_flag and f.hope_lead_time_sec is not None:
                    hope_lead = float(f.hope_lead_time_sec)
                    break
            features["hope_lead_time_s"] = hope_lead

            # Durations
            t_start = frames[0].timestamp
            t_end = frames[-1].timestamp
            t_peak = frames[peak_idx].timestamp

            features["flare_duration_s"] = (t_end - t_start).total_seconds()
            features["rise_time_s"] = (t_peak - t_start).total_seconds()
            features["decay_time_s"] = (t_end - t_peak).total_seconds()

            # If event is also passed
            if event is not None:
                gcls = getattr(event, "goes_class", getattr(event, "predicted_class", ""))
                features["goes_class_numeric"] = _class_num(gcls)
                features["goes_subclass"] = float(getattr(event, "goes_subclass", 1.0))
                features["confidence"] = float(getattr(event, "confidence", getattr(event, "peak_class_prob", 1.0)))
                # check lead time options
                hope_lead_val = getattr(event, "hope_lead_time_sec", getattr(event, "lead_time_sec", getattr(event, "lead_time_estimate_sec", None)))
                if hope_lead_val is not None:
                    features["hope_lead_time_s"] = float(hope_lead_val)
                else:
                    features["hope_lead_time_s"] = hope_lead
            else:
                # Fallback to estimate from fluxes
                max_flux = fluxes[peak_idx] if fluxes else 0.0
                # GOES-class estimate
                # counts * 1e-8 W/m^2
                flux_wm2 = max_flux * 1.0e-8
                if flux_wm2 >= 1e-4:
                    features["goes_class_numeric"] = 4.0
                    features["goes_subclass"] = flux_wm2 / 1e-4
                elif flux_wm2 >= 1e-5:
                    features["goes_class_numeric"] = 3.0
                    features["goes_subclass"] = flux_wm2 / 1e-5
                elif flux_wm2 >= 1e-6:
                    features["goes_class_numeric"] = 2.0
                    features["goes_subclass"] = flux_wm2 / 1e-6
                elif flux_wm2 >= 1e-7:
                    features["goes_class_numeric"] = 1.0
                    features["goes_subclass"] = flux_wm2 / 1e-7
                else:
                    features["goes_class_numeric"] = 0.0
                    features["goes_subclass"] = flux_wm2 / 1e-8
                features["confidence"] = 1.0

        return np.array([features[k] for k in self.FEATURES], dtype=np.float32)
