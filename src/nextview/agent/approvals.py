"""Human-approval gates (spec 2.7): kinds, allowed decisions and the approval card.

A model change or an export never happens without an approval record
``{id, kind, decision, reason, ts}`` in the state and in the trace (``loop.apply_approval``).
"""

from __future__ import annotations

import math
from enum import Enum

__all__ = ["ApprovalKind", "DECISIONS", "allowed_decisions", "make_card", "check_decision", "ApprovalError"]


class ApprovalKind(str, Enum):
    SWITCH_MODEL = "switch_model"  # R8
    USE_RO = "use_ro"  # R9
    DRIFT_RESTART = "drift_restart"  # R6
    ACCEPT_WITH_WARNINGS = "accept_with_warnings"  # R12
    EXPORT = "export"  # R11
    OUT_OF_SCOPE = "out_of_scope"  # R10


DECISIONS = {
    ApprovalKind.SWITCH_MODEL: ("approve", "deny"),
    ApprovalKind.USE_RO: ("approve", "deny"),
    ApprovalKind.DRIFT_RESTART: ("approve", "deny"),
    ApprovalKind.ACCEPT_WITH_WARNINGS: ("approve", "deny"),
    ApprovalKind.EXPORT: ("approve", "deny"),
    ApprovalKind.OUT_OF_SCOPE: ("acknowledge",),
}

OPTION_LABELS = {
    ApprovalKind.SWITCH_MODEL: {"approve": "Switch to rational model", "deny": "Keep k5 (pin model)"},
    ApprovalKind.USE_RO: {"approve": "Refit with release-object model", "deny": "Keep model (pin)"},
    ApprovalKind.DRIFT_RESTART: {"approve": "Lock focus and restart", "deny": "Continue (flagged)"},
    ApprovalKind.ACCEPT_WITH_WARNINGS: {"approve": "Accept with warnings", "deny": "Discard"},
    ApprovalKind.EXPORT: {"approve": "Export calibration", "deny": "Keep capturing"},
    ApprovalKind.OUT_OF_SCOPE: {"acknowledge": "Acknowledge (ends session)"},
}


class ApprovalError(ValueError):
    """Bad approval request (unknown id, no pending approval, decision not allowed): BAD_REQUEST."""

    code = "BAD_REQUEST"


def _fmt(x, nd=2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:.{nd}f}" if math.isfinite(x) else "n/a"


def _question(kind: ApprovalKind, ev: dict) -> str:
    if kind is ApprovalKind.SWITCH_MODEL:
        cv = ev.get("cv_rms", {})
        p = ev.get("radial_trend_p")
        ptxt = "n/a" if p is None else f"{float(p):.2g}"
        return (f"Radial residual trend p={ptxt}; held-out RMS k5 {_fmt(cv.get('k5'))} px vs rational "
                f"{_fmt(cv.get('rational'))} px. Switch to rational model?")
    if kind is ApprovalKind.USE_RO:
        g = ev.get("cv_gain")
        gtxt = "n/a" if g is None else f"{100 * float(g):.0f}%"
        return (f"Board looks warped (warp score {_fmt(ev.get('warp_score'))}); the release-object model lowers "
                f"held-out RMS by {gtxt}. Refit with calibrateCameraROExtended?")
    if kind is ApprovalKind.DRIFT_RESTART:
        return "Focus/zoom may have changed — lock focus and restart, or continue?"
    if kind is ApprovalKind.ACCEPT_WITH_WARNINGS and not _exportable(ev):
        return (f"Budget used ({ev.get('views')} views, {ev.get('attempts')} attempts): fewer than 3 views, so there is "
                f"no calibration to export. Discard the session?")
    if kind is ApprovalKind.ACCEPT_WITH_WARNINGS:
        return (f"Budget used ({ev.get('views')} views, {ev.get('attempts')} attempts): U95 {_fmt(ev.get('U95_px'))} px, "
                f"P(field p95 > {_fmt(ev.get('tau_px'))} px) = {_fmt(ev.get('p_exceed'))} — {ev.get('label')}. "
                f"Accept with warnings or discard?")
    if kind is ApprovalKind.EXPORT:
        return (f"Stop: P(field p95 > {_fmt(ev.get('tau_px'))} px) = {_fmt(ev.get('p_exceed'))}, U95 {_fmt(ev.get('U95_px'))} px "
                f"over {ev.get('views')} views. Export the calibration?")
    if kind is ApprovalKind.OUT_OF_SCOPE:
        return (f"Lens appears to be fisheye (estimated HFOV {_fmt(ev.get('hfov_deg'), 0)}° > {_fmt(ev.get('hfov_limit_deg'), 0)}°); "
                f"this tool fits pinhole/rational models only — use cv2.fisheye")
    raise ValueError(kind)


def _exportable(ev: dict) -> bool:
    """An export needs a calibration, so at least 3 views (unknown counts are not blocked here)."""
    v = ev.get("views")
    return v is None or int(v) >= 3


def allowed_decisions(kind: ApprovalKind, evidence: dict) -> tuple[str, ...]:
    """The decisions a card offers: an accept-with-warnings card over fewer than 3 views offers only "deny" (Discard)."""
    if kind is ApprovalKind.ACCEPT_WITH_WARNINGS and not _exportable(evidence):
        return ("deny",)
    return DECISIONS[kind]


def make_card(kind: ApprovalKind | str, approval_id: str, evidence: dict, rule: str, step: int) -> dict:
    """The pending-approval card stored in the state and returned as ``approval`` (spec 7.3)."""
    kind = ApprovalKind(kind)
    return {
        "id": approval_id,
        "kind": kind.value,
        "rule": rule,
        "step": int(step),
        "question": _question(kind, evidence),
        "options": [{"decision": d, "label": OPTION_LABELS[kind][d]} for d in allowed_decisions(kind, evidence)],
        "evidence": evidence,
    }


def check_decision(card: dict | None, approval_id: str, decision: str) -> ApprovalKind:
    if card is None:
        raise ApprovalError("there is no pending approval")
    if approval_id != card["id"]:
        raise ApprovalError(f"approval id {approval_id!r} does not match the pending {card['id']!r}")
    kind = ApprovalKind(card["kind"])
    allowed = [o["decision"] for o in card.get("options") or ()] or list(DECISIONS[kind])
    if decision not in allowed or decision not in DECISIONS[kind]:
        raise ApprovalError(f"decision {decision!r} not allowed for {kind.value}; allowed: {allowed}")
    return kind
