from datetime import datetime
from aruna.models.frame_header import FrameHeader


def test_frame_header_defaults():
    h = FrameHeader(timestamp=datetime.utcnow(), source="LIVE")
    assert h.frame_id  # uuid generated
    assert h.version == "2.0.0"
    assert h.quality == "GOOD"
    assert isinstance(h.latency_ms, int)


def test_frame_header_source_validation():
    import pytest
    with pytest.raises(ValueError):
        FrameHeader(timestamp=datetime.utcnow(), source="INVALID")
