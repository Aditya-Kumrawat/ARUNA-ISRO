"""
ARUNA-XL — Extension modules sitting above the core L0–L7 pipeline.

These modules consume L2–L4 outputs and add multi-payload intelligence:
  XL-2: MPCI (Multi-Payload Coherence Index)     — mpci.py
  XL-3: Physics Tokens (21-token feature vector) — physics_tokens.py
"""
from .mpci import MPCIEvaluator, MPCIResult, compute_MPCI
from .physics_tokens import PhysicsTokens, extract_tokens, TOKEN_NAMES, N_TOKENS

__all__ = [
    "MPCIEvaluator",
    "MPCIResult",
    "compute_MPCI",
    "PhysicsTokens",
    "extract_tokens",
    "TOKEN_NAMES",
    "N_TOKENS",
]
