"""Target resolution layer for structured semantic navigation."""

from .resolver import TargetResolver
from .types import (
    ClarificationRequest,
    Evidence,
    NavigableAnchor,
    RequiredNextStep,
    ResolutionDiagnostics,
    ResolutionResult,
    TargetDraft,
    TargetRef,
    build_ambiguous_result,
    build_clarification_request,
)

__all__ = [
    "ClarificationRequest",
    "Evidence",
    "NavigableAnchor",
    "RequiredNextStep",
    "ResolutionDiagnostics",
    "ResolutionResult",
    "TargetDraft",
    "TargetRef",
    "TargetResolver",
    "build_ambiguous_result",
    "build_clarification_request",
]
