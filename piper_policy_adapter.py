"""Canonical Piper policy/data boundary, independent of hardware and model weights.

The Dream SDK supplies the operational wire adapter and LeRobot recorder.  This module pins the
Piper-specific names layered above them so a fine-tune sees stable camera/state/action fields and
an explicit record of which arm actually executed a command.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "piper-policy-adapter-v1"
CAMERA_MAP = {
    "overhead": "scene",
    "wrist_l": "wrist_l",
    "wrist_r": "wrist_r",
}
ARMS = ("left", "right")
ARM_JOINTS = {"left": 6, "right": 6}
ACTION_HORIZON = 48
ACTION_HZ = 20.0
# LeRobot's recorder uses left joints then left gripper, then the same order on the right.
STATE_NAMES = [
    name
    for arm in ARMS
    for name in [
        *(f"{arm}_joint_{index}" for index in range(ARM_JOINTS[arm])),
        f"{arm}_gripper",
    ]
]
REQUIRED_FEATURES = (
    "observation.state",
    "action",
    *(f"observation.images.{camera}" for camera in CAMERA_MAP.values()),
)


def policy_adapter_manifest(
    *,
    checkpoint: str,
    embodiment: str,
    active_arms: tuple[str, ...],
    execute: bool,
    action_space: str = "eef",
    horizon: int = ACTION_HORIZON,
    action_hz: float = ACTION_HZ,
) -> dict[str, Any]:
    """Describe model fields, recorder fields and the exact authority of each recorded arm."""
    unknown = set(active_arms).difference(ARMS)
    if unknown:
        raise ValueError(f"Unknown active arms: {sorted(unknown)}")
    if horizon <= 0 or action_hz <= 0:
        raise ValueError("horizon and action_hz must be positive")
    executed = {arm: execute and arm in active_arms for arm in ARMS}
    return {
        "schema": SCHEMA_VERSION,
        "checkpoint": checkpoint,
        "embodiment": embodiment,
        "model_boundary": {
            "camera_roles": dict(CAMERA_MAP),
            "state_parts": {
                f"{arm}_{part}": width
                for arm in ARMS
                for part, width in (("arm", ARM_JOINTS[arm]), ("gripper", 1), ("grip", 1))
            },
            "derived_state_parts": [
                f"{arm}_{part}"
                for arm in ARMS
                for part in ("eef_xyz", "eef_rot6")
            ],
            "action_space": action_space,
            "action_semantics": "model-native; Dream client converts tool poses to absolute joints",
            "horizon": horizon,
            "action_hz": action_hz,
        },
        "dataset_boundary": {
            "format": "LeRobot v2.1",
            "state_names": list(STATE_NAMES),
            "action_names": list(STATE_NAMES),
            "joint_units": "radians",
            "gripper_units": "normalized opening in [0, 1], 0 closed and 1 open",
            "videos": {role: camera for role, camera in CAMERA_MAP.items()},
            "raw_replies": "chunks/reply_*.npz",
            "executed_actions": "data/chunk-000/episode_000000.parquet:action",
        },
        "authority": {
            "observe_arms": list(ARMS),
            "active_arms": list(active_arms),
            "executed_arms": [arm for arm in ARMS if executed[arm]],
            "action_valid": executed,
            "note": "An action row is training evidence only for arms whose action_valid flag is true.",
        },
        "training": {
            "task_success_label_required": True,
            "default_eligibility": "candidate_unverified",
            "reason": "A complete recording is not necessarily a successful demonstration.",
        },
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_episode(path: Path) -> dict[str, Any]:
    """Validate one recorder episode without loading model, robot or video codecs."""
    rollout_path = path / "rollout.json"
    info_path = path / "meta" / "info.json"
    errors: list[str] = []
    rollout: dict[str, Any] = {}
    info: dict[str, Any] = {}
    if not rollout_path.is_file():
        errors.append("missing rollout.json")
    else:
        rollout = _read_json(rollout_path)
    if not info_path.is_file():
        errors.append("missing meta/info.json")
    else:
        info = _read_json(info_path)

    status = rollout.get("status")
    frames = int(rollout.get("frames") or 0)
    stop_reason = rollout.get("stop_reason")
    if status != "complete":
        errors.append(f"recording status is {status!r}, expected 'complete'")
    if frames <= 0:
        errors.append("recording has no frames")
    if stop_reason != "stopped":
        errors.append(f"run stopped with {stop_reason!r}, expected operator stop")
    if rollout.get("error") is not None:
        errors.append(f"run recorded error: {rollout['error']}")

    features = info.get("features", {})
    missing_features = [name for name in REQUIRED_FEATURES if name not in features]
    errors.extend(f"missing feature {name}" for name in missing_features)
    for feature in ("observation.state", "action"):
        names = features.get(feature, {}).get("names")
        if names != STATE_NAMES:
            errors.append(f"{feature} names differ from the Piper adapter schema")

    data_file = next(path.glob("data/chunk-*/episode_*.parquet"), None)
    if data_file is None:
        errors.append("missing LeRobot parquet data")
    video_files = {
        camera: path / "videos" / "chunk-000" / f"observation.images.{camera}" / "episode_000000.mp4"
        for camera in CAMERA_MAP.values()
    }
    for camera, video in video_files.items():
        if not video.is_file():
            errors.append(f"missing video {camera}")

    adapter_path = path / "policy_adapter.json"
    policy_adapter = _read_json(adapter_path) if adapter_path.is_file() else None
    if policy_adapter is not None and not isinstance(policy_adapter, dict):
        errors.append("policy_adapter.json must contain an object")
        policy_adapter = None
    if policy_adapter is not None:
        if policy_adapter.get("schema") != SCHEMA_VERSION:
            errors.append(f"policy adapter schema is {policy_adapter.get('schema')!r}")
        if policy_adapter.get("checkpoint") != rollout.get("checkpoint"):
            errors.append("policy adapter checkpoint differs from rollout checkpoint")
        boundary = policy_adapter.get("dataset_boundary", {})
        if boundary.get("state_names") != STATE_NAMES:
            errors.append("policy adapter state names differ from the dataset contract")
        if boundary.get("action_names") != STATE_NAMES:
            errors.append("policy adapter action names differ from the dataset contract")
        if boundary.get("videos") != CAMERA_MAP:
            errors.append("policy adapter camera mapping differs from the dataset contract")
        action_valid = policy_adapter.get("authority", {}).get("action_valid")
        if (
            not isinstance(action_valid, dict)
            or set(action_valid) != set(ARMS)
            or any(type(value) is not bool for value in action_valid.values())
        ):
            errors.append("policy adapter action_valid must map left/right to booleans")
        elif not any(action_valid.values()):
            errors.append("policy adapter has no executed arm")
    label = rollout.get("task_success")
    ready = not errors and label == "success" and policy_adapter is not None
    if ready:
        eligibility = "training_ready"
    elif not errors:
        eligibility = "candidate_needs_task_success_label"
    else:
        eligibility = "rejected"
    return {
        "episode": path.name,
        "task": rollout.get("task"),
        "checkpoint": rollout.get("checkpoint"),
        "status": status,
        "frames": frames,
        "duration_s": rollout.get("duration_s"),
        "stop_reason": rollout.get("stop_reason"),
        "task_success": label,
        "policy_adapter": None if policy_adapter is None else policy_adapter.get("schema"),
        "action_valid": None if policy_adapter is None else policy_adapter.get("authority", {}).get("action_valid"),
        "eligible_for_training": ready,
        "eligibility": eligibility,
        "errors": errors,
    }


def build_index(root: Path) -> dict[str, Any]:
    episodes = [
        audit_episode(path)
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "rollout.json").is_file()
    ]
    counts: dict[str, int] = {}
    for episode in episodes:
        key = episode["eligibility"]
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "root": str(root.resolve()),
        "counts": counts,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/rollouts"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    index = build_index(args.root)
    if args.output:
        write_json(args.output, index)
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
