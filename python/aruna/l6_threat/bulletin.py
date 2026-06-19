from aruna.models.nowcast import NowcastEvent
from .models import ThreatOutput

def get_dos_donts(goes_class: str, goes_subclass: float) -> tuple[list[str], list[str]]:
    """Generates lists of DOs and DONTs based on the flare classification."""
    is_major = goes_class == "X" or (goes_class == "M" and goes_subclass >= 5.0)
    
    if is_major:
        dos = [
            "Power grid operators: Monitor geomagnetic induced currents (GIC).",
            "Aviation: Reroute polar flights to lower latitudes.",
            "Satellite operators: Initiate solar panel safing procedures for vulnerable payloads.",
            "Gaganyaan: Advise crew to retreat to the radiation shelter.",
        ]
        donts = [
            "Do not initiate extravehicular activities (EVA).",
            "Do not rely on HF communication for critical operations.",
            "Do not perform orbital adjustment maneuvers unless critical.",
        ]
    else:
        dos = [
            "Routine monitoring of ionospheric parameters.",
            "Standard satellite operations continue.",
        ]
        donts = [
            "No immediate restrictions required.",
        ]
        
    return dos, donts

def format_bulletin(event: NowcastEvent, threat: ThreatOutput) -> str:
    """Formats the Space Weather Advisory Bulletin following SWMC templates."""
    bulletin = []
    bulletin.append("=====================================================================")
    bulletin.append("                    SPACE WEATHER ADVISORY BULLETIN                  ")
    bulletin.append("=====================================================================")
    bulletin.append(f"Event ID: {event.event_id}")
    bulletin.append(f"Issued: {event.timestamp.isoformat()} UTC")
    bulletin.append(f"GOES Class: {event.goes_label}")
    bulletin.append(f"Confidence: {event.confidence:.2f}")
    bulletin.append(f"Source Event ID: {threat.source_event_id}")
    bulletin.append("---------------------------------------------------------------------")
    bulletin.append("OPERATIONAL THREAT SUMMARY")
    bulletin.append("---------------------------------------------------------------------")
    bulletin.append("NavIC Region Risk:")
    for region, risk in threat.navic_risk.items():
        prob = threat.navic_prob.get(region, 0.0)
        bulletin.append(f"  * {region.replace('_', ' ').title()}: {risk} (Probability: {prob:.2f})")
    bulletin.append(f"HF Communication: {threat.hf_risk} blackout (Estimated Duration: {threat.hf_duration_min} min)")
    bulletin.append(f"Radiation Risk: {threat.radiation_risk}")
    bulletin.append(f"Gaganyaan Advisory: {threat.gaganyaan_advisory}")
    bulletin.append("---------------------------------------------------------------------")
    bulletin.append("ACTION PLAN (DOs & DONTs)")
    bulletin.append("---------------------------------------------------------------------")
    bulletin.append("DO:")
    for item in threat.dos:
        bulletin.append(f"  - {item}")
    bulletin.append("DON'T:")
    for item in threat.donts:
        bulletin.append(f"  - {item}")
    bulletin.append("=====================================================================")
    return "\n".join(bulletin)
