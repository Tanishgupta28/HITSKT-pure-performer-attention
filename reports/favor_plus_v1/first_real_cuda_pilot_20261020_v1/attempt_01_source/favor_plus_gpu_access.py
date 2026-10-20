"""Metadata-only authorization for FAVOR+ research GPU entry points.

Records document external allocation or an explicit allocation-owner transfer;
they cannot create permission. Boundary checks use OS files only, never CUDA.
"""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import uuid

CUDA_VALIDATION = "favor_plus_cuda_validation"
REAL_PILOT = "favor_plus_assist2017_five_step_pilot"
# User-protected LightSAKT2 resource. A full-parent assignment would include it.
PROTECTED_RESOURCES = frozenset({
    "MIG-d2fcfd79-720e-5218-82c9-0a132be68920",
    "GPU-5a568e3c-e96c-169f-b063-f4478340ff51",
})
TRANSFERRED_MIG = "MIG-d2fcfd79-720e-5218-82c9-0a132be68920"
OWNER_TRANSFER = "owner_confirmed_existing_research_allocation"
LIGHT_HOLD = Path("/workspace/LightSAKT2/experiments/USER_TRAINING_HOLD.json")
DEVICE_EXPOSURE = Path("/var/run/nvidia-container-devices")


def require_gpu_authorization(record: dict, visible: str | None, action: str,
                              *, now: dt.datetime | None = None) -> str:
    """Fail closed before any CUDA API; also recheck at execution boundaries."""
    if not isinstance(record, dict) or action not in (CUDA_VALIDATION, REAL_PILOT):
        raise ValueError("a supported action and external allocation record are required")
    owner_transfer = record.get("allocation_kind") == OWNER_TRANSFER
    if record.get("exclusive") is not True or (not owner_transfer and record.get("independent") is not True):
        raise ValueError("independent exclusive GPU/MIG allocation must be confirmed")
    evidence_key = "owner_confirmation_evidence" if owner_transfer else "external_allocation_evidence"
    for key in ("owner", "job_id", evidence_key):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"missing external allocation field: {key}")
    resource = record.get("resource_uuid", "")
    if not isinstance(resource, str):
        raise ValueError("a canonical GPU/MIG UUID is required")
    transfer = record.get("resource_transfer", {})
    if owner_transfer and (resource != TRANSFERRED_MIG
            or record.get("owner_confirms_existing_allocation") is not True
            or not isinstance(transfer, dict) or transfer.get("from_project") != "LightSAKT2"
            or transfer.get("to_model") != "hitskt_favor_plus_v1"
            or transfer.get("authorized") is not True):
        raise ValueError("explicit owner-confirmed transfer of the exact historical MIG is required")
    if resource in PROTECTED_RESOURCES and not owner_transfer:
        raise ValueError("protected LightSAKT2 resource must not be used")
    if record.get("known_conflicting_reservation") or record.get("known_explicit_restriction"):
        raise ValueError("conflicting reservation or explicit restriction forbids execution")
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
        if owner_transfer and (end-start).total_seconds() > 7200:
            raise ValueError()
    except (KeyError, TypeError, AttributeError, ValueError):
        raise ValueError("allocation requires a currently valid timezone-aware window") from None
    approval = record.get("user_approval", {})
    if (not isinstance(approval, dict) or approval.get("approved") is not True
            or not isinstance(approval.get("evidence"), str) or not approval["evidence"].strip()
            or not isinstance(approval.get("actions"), list) or action not in approval["actions"]):
        raise ValueError("actual user approval for this GPU action is required")
    return resource


def conflicting_device_processes(proc_root: Path = Path("/proc"), *, own_pid: int | None = None) -> list[int]:
    """Conservative conflict check for the previously inspected GPU0/MIG nodes.

    No arbitrary argv/environment is read or logged. Other parent-GPU users are
    conservatively treated as conflicts. This is a boundary snapshot, not a
    watcher or proof about invisible scheduler reservations.
    """
    own_pid = os.getpid() if own_pid is None else own_pid
    conflicts = []
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name) == own_pid:
            continue
        try:
            descriptors = list((proc/"fd").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise RuntimeError("cannot inspect device conflicts in this process namespace") from exc
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError:
                continue
            if target in ("/dev/nvidia0", "/dev/nvidia-caps/nvidia-cap21", "/dev/nvidia-caps/nvidia-cap22"):
                conflicts.append(int(proc.name))
                break
    return sorted(conflicts)


def require_execution_access(record: dict, visible: str | None, action: str) -> str:
    """Metadata plus actual release/exposure/conflict checks before CUDA/stages."""
    resource = require_gpu_authorization(record, visible, action)
    if record.get("allocation_kind") == OWNER_TRANSFER:
        if not LIGHT_HOLD.is_file() or not (DEVICE_EXPOSURE/resource).exists():
            raise RuntimeError("LightSAKT2 hold or exact transferred device exposure missing")
        conflicts = conflicting_device_processes()
        if conflicts:
            raise RuntimeError(f"conflicting GPU handle owners at execution boundary: {conflicts}")
    return resource
