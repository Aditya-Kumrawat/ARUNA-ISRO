from aruna.models.nowcast import NowcastEvent
from .models import ThreatOutput, SafingEntry
from .rules import (
    HF_BLACKOUT,
    GAGANYAAN,
    RADIATION_RISK,
    SAFING_REGISTRY,
)
from .navic import compute_navic_risk, parse_goes_band
from .bulletin import get_dos_donts, format_bulletin

class ThreatOrchestrator:
    def __init__(self, config=None, kafka_producer=None) -> None:
        self.config = config
        self.pub = kafka_producer

    def process(self, event: NowcastEvent) -> ThreatOutput:
        """
        Process a NowcastEvent and produce a ThreatOutput report.
        """
        band = parse_goes_band(event.goes_class, event.goes_subclass)
        
        # 1. Compute NavIC risk and probability
        navic_risk, navic_prob = compute_navic_risk(event)
        
        # 2. Lookup HF, radiation, and Gaganyaan advisories
        hf_risk, hf_duration_min = HF_BLACKOUT.get(band, ("NONE", 0))
        radiation_risk = RADIATION_RISK.get(band, "NOMINAL")
        gaganyaan_advisory = GAGANYAAN.get(band, "NOMINAL")
        
        # 3. Sort Safing Queue by vulnerability descending
        def get_vulnerability(sat: SafingEntry) -> float:
            return sat.vulnerability.get(band, 0.0)
            
        safing_queue = sorted(SAFING_REGISTRY, key=get_vulnerability, reverse=True)
        
        # 4. Generate DOs and DONTs
        dos, donts = get_dos_donts(event.goes_class, event.goes_subclass)
        
        # 5. Build ThreatOutput skeleton
        threat = ThreatOutput(
            navic_risk=navic_risk,
            navic_prob=navic_prob,
            hf_risk=hf_risk,
            hf_duration_min=hf_duration_min,
            radiation_risk=radiation_risk,
            gaganyaan_advisory=gaganyaan_advisory,
            safing_queue=safing_queue,
            dos=dos,
            donts=donts,
            bulletin_text="",
            source_event_id=event.event_id,
        )
        
        # 6. Format the bulletin with the populated threat fields
        threat.bulletin_text = format_bulletin(event, threat)
        
        # 7. Publish to Kafka if publisher is available
        if self.pub is not None:
            self._publish(threat)
            
        return threat

    def _publish(self, threat: ThreatOutput) -> None:
        # Placeholder for publishing to Kafka topic 'aruna.threat.output'
        pass
