from __future__ import annotations

import logging
import time
import json
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np

from aruna.models.forecast import ForecastFrame
from aruna.models.physics import PhysicsFrame
from aruna.l5_analogs.models import AnalogMatch, NO_ANALOG
from aruna.l5_analogs.index import AnalogIndex
from aruna.l5_analogs.dtw import compute_dtw_distance

logger = logging.getLogger(__name__)

class AnalogOrchestrator:
    """
    Orchestrates the retrieval and refinement of historical solar flare analogs (PRD §9).
    Consumes L4 ForecastFrames and the historical sliding window of PhysicsFrames.
    Uses FAISS L2-normalized search to retrieve top candidate events,
    then evaluates them using z-normalized DTW.
    """
    def __init__(
        self,
        db_url: str = "sqlite:///aruna_catalog.db",
        index_path: str = "models/faiss_analog.index",
        top_k_faiss: int = 10,
        top_k_return: int = 3,
        no_analog_sigma: float = 3.0,
        kafka_producer = None,
        kafka_topic: str = "aruna.analogs.results"
    ) -> None:
        self.index = AnalogIndex(db_url=db_url, index_path=index_path)
        self.top_k_faiss = top_k_faiss
        self.top_k_return = top_k_return
        self.no_analog_sigma = no_analog_sigma
        self.kafka_producer = kafka_producer
        self.kafka_topic = kafka_topic

    def register_timeseries(self, event_id: str, ts: np.ndarray) -> None:
        """Register timeseries helper for testing or catalog ingestion."""
        self.index.register_timeseries(event_id, ts)

    def process(
        self,
        forecast_frame: ForecastFrame,
        physics_frames: List[PhysicsFrame]
    ) -> List[AnalogMatch]:
        """
        Retrieves analogs for the current event sequence.
        
        Parameters:
            forecast_frame : The latest L4 ForecastFrame.
            physics_frames : List of PhysicsFrames representing the current event timeline.
            
        Returns:
            List[AnalogMatch]: The top refined analogs, or a list with NO_ANALOG sentinel.
        """
        t_start = time.monotonic()
        
        # Cold start check: if catalog is empty, return NO_ANALOG
        if len(self.index.event_ids) == 0:
            logger.info("Cold start: empty catalog. Returning NO_ANALOG.")
            results = [NO_ANALOG]
            self._publish(results)
            return results

        # 1. Extract 17-dimensional query features from physics_frames
        raw_query = self.index.embedder.extract_raw(physics_frames)
        
        # 2. Query FAISS index for top K candidates
        candidates = self.index.search(raw_query, top_k=self.top_k_faiss)
        
        if not candidates:
            results = [NO_ANALOG]
            self._publish(results)
            return results

        # 3. Construct 4-D query timeseries from physics_frames
        # Features: ["T_MK", "EM_cm3", "DNI", "SHEI"]
        query_ts = np.zeros((len(physics_frames), 4), dtype=np.float32)
        for idx, f in enumerate(physics_frames):
            query_ts[idx, 0] = f.T_MK if f.T_MK is not None else 0.0
            query_ts[idx, 1] = f.EM_cm3 if f.EM_cm3 is not None else 0.0
            query_ts[idx, 2] = f.DNI if f.DNI is not None else 0.0
            query_ts[idx, 3] = f.SHEI if f.SHEI is not None else 0.0

        # 4. Refine candidates using DTW
        dtw_scores = []
        for eid, goes_class, dist, raw_v in candidates:
            cand_ts = self.index.library_timeseries.get(eid)
            if cand_ts is None:
                dtw_dist = float('inf')
            else:
                dtw_dist = compute_dtw_distance(query_ts, cand_ts)
            
            dtw_scores.append((dtw_dist, eid, goes_class, raw_v))

        # Sort by DTW distance ascending
        dtw_scores.sort(key=lambda x: x[0])

        # 5. Check statistical threshold for NO_ANALOG fallback
        nearest_dist = dtw_scores[0][0]
        threshold = self.index.library_mean_dtw + self.no_analog_sigma * self.index.library_std_dtw
        
        # If nearest neighbor distance exceeds threshold, trigger NO_ANALOG fallback
        no_analog_triggered = (nearest_dist > threshold) or (nearest_dist == float('inf'))
        
        if no_analog_triggered:
            logger.warning(
                "Nearest analog distance %.2f exceeds statistical threshold %.2f. "
                "Triggering NO_ANALOG fallback.",
                nearest_dist, threshold
            )
            results = [NO_ANALOG]
        else:
            results = []
            for dist, eid, goes_class, raw_v in dtw_scores[:self.top_k_return]:
                similarity = 1.0 / (1.0 + dist) if dist != float('inf') else 0.0
                results.append(AnalogMatch(
                    event_id=eid,
                    goes_class=goes_class,
                    dtw_distance=dist,
                    similarity=similarity,
                    feature_vector=raw_v,
                    no_analog_flag=False
                ))

        # Publish results to Kafka
        self._publish(results)
        
        elapsed = time.monotonic() - t_start
        logger.info("AnalogOrchestrator: processed in %.3fs. Retrieved %d matches.", elapsed, len(results))
        return results

    def _publish(self, matches: List[AnalogMatch]) -> None:
        """Publishes the analog retrieval results to the message bus."""
        if self.kafka_producer is None:
            return
        try:
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "matches": [
                    {
                        "event_id": m.event_id,
                        "goes_class": m.goes_class,
                        "dtw_distance": m.dtw_distance,
                        "similarity": m.similarity,
                        "no_analog_flag": m.no_analog_flag
                    } for m in matches
                ]
            }
            self.kafka_producer.send(
                self.kafka_topic,
                key=b"analogs",
                value=json.dumps(payload).encode()
            )
        except Exception as e:
            logger.error("Failed to publish analog results to Kafka: %s", e)
