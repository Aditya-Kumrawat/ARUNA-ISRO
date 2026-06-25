import os
import sqlite3
import pickle
import logging
import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler
from aruna.l5_analogs.embedder import PhysicsFeatureEmbedder

logger = logging.getLogger(__name__)

# index.py lives at python/aruna/l5_analogs/index.py → project root is 4 levels up
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_DB_URL = f"sqlite:///{_PROJECT_ROOT / 'aruna_catalog.db'}"
_DEFAULT_INDEX_PATH = str(_PROJECT_ROOT / "models" / "faiss_analog.index")


class AnalogIndex:
    """
    Manages a FAISS index, fits scaling statistics, and handles automatic rebuilding
    after new master catalog entries.
    """
    def __init__(self, db_url: str = _DEFAULT_DB_URL, index_path: str = _DEFAULT_INDEX_PATH) -> None:
        self.db_url = db_url
        self.index_path = index_path
        self.scaler: Optional[StandardScaler] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.event_ids: List[str] = []
        self.event_goes_classes: List[str] = []
        self.raw_features: np.ndarray = np.empty((0, 17), dtype=np.float32)
        self.library_timeseries: Dict[str, np.ndarray] = {}
        self.last_count = -1
        self.library_mean_dtw = 0.0
        self.library_std_dtw = 1.0
        self.embedder = PhysicsFeatureEmbedder()
        self.load()

    def _get_connection(self):
        """Returns a raw sqlite3 connection based on db_url."""
        path = self.db_url.replace("sqlite:///", "").replace("sqlite://", "")
        if not path or path == ":memory:":
            path = ":memory:"
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        return conn

    def get_db_count(self) -> int:
        """Gets count of rows in master_flare_catalog."""
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='master_flare_catalog'")
            if not cur.fetchone():
                return 0
            cur.execute("SELECT COUNT(*) FROM master_flare_catalog")
            return cur.fetchone()[0]
        except Exception as e:
            logger.error("Failed to get DB count: %s", e)
            return 0
        finally:
            conn.close()

    def register_timeseries(self, event_id: str, ts: np.ndarray) -> None:
        """Registers a timeseries sequence for an event, enabling DTW comparison."""
        self.library_timeseries[event_id] = ts
        self.library_mean_dtw = None
        self.library_std_dtw = None
        # Save metadata immediately to ensure persistence
        self._save_metadata()

    def _save_metadata(self) -> None:
        """Saves scaler and event metadata/timeseries alongside index."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True)
            meta_path = self.index_path + ".meta"
            with open(meta_path, "wb") as f:
                pickle.dump({
                    "scaler": self.scaler,
                    "event_ids": self.event_ids,
                    "event_goes_classes": self.event_goes_classes,
                    "raw_features": self.raw_features,
                    "library_timeseries": self.library_timeseries
                }, f)
        except Exception as e:
            logger.error("Failed to save FAISS metadata: %s", e)

    def rebuild(self) -> None:
        """Queries all events from the database and rebuilds the FAISS index."""
        # Retrieve existing timeseries from meta if disk file exists to avoid losing them
        meta_path = self.index_path + ".meta"
        if os.path.exists(meta_path) and not self.library_timeseries:
            try:
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                    self.library_timeseries = meta.get("library_timeseries", {})
            except Exception:
                pass

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='master_flare_catalog'")
            if not cur.fetchone():
                self.index = None
                self.scaler = None
                self.event_ids = []
                self.event_goes_classes = []
                self.raw_features = np.empty((0, 17), dtype=np.float32)
                self.last_count = 0
                return

            cur.execute("SELECT * FROM master_flare_catalog")
            columns = [d[0] for d in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        except Exception as e:
            logger.error("Failed to query catalog for rebuilding index: %s", e)
            rows = []
        finally:
            conn.close()

        count = len(rows)
        self.last_count = count

        if count == 0:
            self.index = None
            self.scaler = None
            self.event_ids = []
            self.event_goes_classes = []
            self.raw_features = np.empty((0, 17), dtype=np.float32)
            return

        self.event_ids = []
        self.event_goes_classes = []
        raw_features = []
        for r in rows:
            raw_vec = self.embedder.extract_raw(r)
            raw_features.append(raw_vec)
            self.event_ids.append(r["event_id"])
            self.event_goes_classes.append(r.get("goes_class", "UNKNOWN"))

        X = np.array(raw_features, dtype=np.float32)
        self.raw_features = X

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
        X_norm = np.where(norms > 0, X_scaled / norms, X_scaled)

        self.index = faiss.IndexFlatL2(17)
        self.index.add(X_norm.astype(np.float32))

        self.embedder.scaler = self.scaler

        # Save index to disk
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True)
            self.library_mean_dtw = None
            self.library_std_dtw = None
            faiss.write_index(self.index, self.index_path)
            self._save_metadata()
        except Exception as e:
            logger.error("Failed to save FAISS index: %s", e)

    def load(self) -> bool:
        """Loads index and scaler from disk if they exist."""
        meta_path = self.index_path + ".meta"
        if not os.path.exists(self.index_path) or not os.path.exists(meta_path):
            return False

        try:
            self.index = faiss.read_index(self.index_path)
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
                self.scaler = meta["scaler"]
                self.event_ids = meta["event_ids"]
                self.event_goes_classes = meta["event_goes_classes"]
                self.raw_features = meta.get("raw_features", np.empty((0, 17), dtype=np.float32))
                self.library_timeseries = meta.get("library_timeseries", {})
            self.embedder.scaler = self.scaler
            self.last_count = self.get_db_count()
            self.library_mean_dtw = None
            self.library_std_dtw = None
            return True
        except Exception as e:
            logger.error("Failed to load FAISS index or metadata: %s", e)
            return False

    def search(self, raw_query: np.ndarray, top_k: int = 10) -> List[Tuple[str, str, float, np.ndarray]]:
        """
        Searches the FAISS index for the raw query vector.
        Rebuilds the index automatically if the database count has changed.
        
        Returns:
            List of Tuples: (event_id, goes_class, distance, raw_feature_vector)
        """
        db_cnt = self.get_db_count()
        if db_cnt != self.last_count or self.index is None:
            self.rebuild()

        if self.index is None or len(self.event_ids) == 0:
            return []

        # Encode query using embedder
        query_norm = self.embedder.encode(raw_query)

        k = min(top_k, len(self.event_ids))
        distances, indices = self.index.search(query_norm.reshape(1, -1), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.event_ids):
                continue
            eid = self.event_ids[idx]
            gcls = self.event_goes_classes[idx]
            
            # Fetch raw vector from cached features in memory
            raw_v = self.raw_features[idx] if idx < len(self.raw_features) else np.zeros(17)
            results.append((eid, gcls, float(dist), raw_v))

        return results

    def update_dtw_statistics(self) -> None:
        """Computes the mean and standard deviation of pairwise DTW distances in the library."""
        keys = list(self.library_timeseries.keys())
        if len(keys) < 2:
            self.library_mean_dtw = 0.0
            self.library_std_dtw = 1.0
            return

        distances = []
        from aruna.l5_analogs.dtw import compute_dtw_distance
        
        # Limit calculation to at most 100 random pairs if library is large
        rng = np.random.default_rng(42)
        n_pairs = min(100, len(keys) * (len(keys) - 1) // 2)
        
        pairs = set()
        while len(pairs) < n_pairs:
            i, j = rng.choice(len(keys), size=2, replace=False)
            pair = tuple(sorted((keys[i], keys[j])))
            pairs.add(pair)
            
        _MAX_DTW_LEN = 100
        for k1, k2 in pairs:
            ts1 = self.library_timeseries[k1]
            ts2 = self.library_timeseries[k2]
            if len(ts1) > _MAX_DTW_LEN:
                ts1 = ts1[::len(ts1)//_MAX_DTW_LEN][:_MAX_DTW_LEN]
            if len(ts2) > _MAX_DTW_LEN:
                ts2 = ts2[::len(ts2)//_MAX_DTW_LEN][:_MAX_DTW_LEN]
            dist = compute_dtw_distance(ts1, ts2)
            if dist != float('inf') and not np.isnan(dist):
                distances.append(dist)
                
        if distances:
            self.library_mean_dtw = float(np.mean(distances))
            self.library_std_dtw = float(np.std(distances))
        else:
            self.library_mean_dtw = 0.0
            self.library_std_dtw = 1.0
