"""Domain Gate module for GeoPol Bot."""

from backend.app.domain_gate.schemas import (
    DomainLabel,
    DomainStructuredClassification,
    DomainGateRequest,
    DomainGateResponse,
)
from backend.app.domain_gate.classifier import DomainGateClassifier
from backend.app.domain_gate.policy import (
    DOMAIN_POLICY_SYSTEM_PROMPT,
    GEOPOLITICAL_CATEGORIES,
    NON_GEOPOLITICAL_CATEGORIES,
)

__all__ = [
    "DomainLabel",
    "DomainStructuredClassification",
    "DomainGateRequest",
    "DomainGateResponse",
    "DomainGateClassifier",
    "DOMAIN_POLICY_SYSTEM_PROMPT",
    "GEOPOLITICAL_CATEGORIES",
    "NON_GEOPOLITICAL_CATEGORIES",
]
