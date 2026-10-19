"""Metadata-only authorization for FAVOR+ research GPU entry points.

These records document human approval and external allocation; they cannot
create either. No CUDA import, availability check or device query occurs here.
"""
from __future__ import annotations

import datetime as dt
import uuid

CUDA_VALIDATION = "favor_plus_cuda_validation"
REAL_PILOT = "favor_plus_assist2017_five_step_pilot"
# User-protected LightSAKT2 resource. A full-parent assignment would include it.
PROTECTED_RESOURCES = frozenset({
    "MIG-d2fcfd79-720e-5218-82c9-0a132be68920",
    "GPU-5a568e3c-e96c-169f-b063-f4478340ff51",
})


def require_gpu_authorization(record: dict, visible: str | None, action: str,
                              *, now: dt.datetime | None = None) -> str:
    """Fail closed before any CUDA API; also recheck at execution boundaries."""
    if not isinstance(record, dict) or action not in (CUDA_VALIDATION, REAL_PILOT):
        raise ValueError("a supported action and external allocation record are required")
    if record.get("exclusive") is not True or record.get("independent") is not True:
        raise ValueError("independent exclusive GPU/MIG allocation must be confirmed")
    for key in ("owner", "job_id", "external_allocation_evidence"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"missing external allocation field: {key}")
    resource = record.get("resource_uuid", "")
    if not isinstance(resource, str):
        raise ValueError("a canonical GPU/MIG UUID is required")
    if resource in PROTECTED_RESOURCES:
        raise ValueError("protected LightSAKT2 resource must not be used")
    try:
        prefix, identifier = resource.split("-", 1)
        if prefix not in ("GPU", "MIG") or str(uuid.UUID(identifier)) != identifier:
            raise ValueError()
    except (AttributeError, ValueError):
        raise ValueError("a canonical GPU/MIG UUID is required") from None
    if visible != resource:
        raise ValueError("CUDA_VISIBLE_DEVICES must identify only the assigned UUID")
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        start = dt.datetime.fromisoformat(record["starts_utc"].replace("Z", "+00:00"))
        end = dt.datetime.fromisoformat(record["expires_utc"].replace("Z", "+00:00"))
        if start.tzinfo is None or end.tzinfo is None or not start <= now < end:
            raise ValueError()
    except (KeyError, TypeError, AttributeError, ValueError):
        raise ValueError("allocation requires a currently valid timezone-aware window") from None
    approval = record.get("user_approval", {})
    if (not isinstance(approval, dict) or approval.get("approved") is not True
            or not isinstance(approval.get("evidence"), str) or not approval["evidence"].strip()
            or not isinstance(approval.get("actions"), list) or action not in approval["actions"]):
        raise ValueError("actual user approval for this GPU action is required")
    return resource
