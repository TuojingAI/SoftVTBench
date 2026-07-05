#!/usr/bin/env python3
"""Resolve object-soft target positions from a small JSON config.

The script prints exactly one "x y z" line by default, so shell wrappers can
capture it directly. Use --explain for a human-readable debug record.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


BUILTIN_LEFT = (-0.11934115, -0.23984468, 0.045)
BUILTIN_RIGHT = (0.04998322, -0.10013359, 0.045)
BUILTIN_ANCHORS = {
    "left_back": BUILTIN_LEFT,
    "right_middle": BUILTIN_RIGHT,
    "libero_left": BUILTIN_LEFT,
    "libero_right": BUILTIN_RIGHT,
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_pos(value: Any, *, default_z: float | None = None) -> tuple[float, float, float]:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ValueError(f"position must be a string or list, got {type(value).__name__}")

    if len(parts) not in (2, 3):
        raise ValueError(f"position must have 2 or 3 values, got {len(parts)}")
    x = float(parts[0])
    y = float(parts[1])
    if len(parts) == 3:
        z = float(parts[2])
    elif default_z is not None:
        z = float(default_z)
    else:
        raise ValueError("2D position needs a default z")
    return (x, y, z)


def load_config(path: str) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("top-level config must be a JSON object")
    return data


def named_anchor_map(config: dict[str, Any], fallback_z: float) -> dict[str, tuple[float, float, float]]:
    anchors = dict(BUILTIN_ANCHORS)
    raw = config.get("anchors", {})
    if raw is None:
        return anchors
    if not isinstance(raw, dict):
        raise ValueError("anchors must be a JSON object mapping name to [x, y, z]")
    for name, pos in raw.items():
        anchors[str(name)] = parse_pos(pos, default_z=fallback_z)
    return anchors


def resolve_anchor_list(
    settings: dict[str, Any],
    anchors: dict[str, tuple[float, float, float]],
    fallback_z: float,
) -> list[tuple[str, tuple[float, float, float]]]:
    raw_positions = settings.get("positions")
    if raw_positions is not None:
        if not isinstance(raw_positions, list) or not raw_positions:
            raise ValueError("positions must be a non-empty list")
        resolved = []
        for idx, item in enumerate(raw_positions):
            if isinstance(item, dict):
                name = str(item.get("name", f"pos{idx}"))
                pos = item.get("pos", item.get("position"))
            else:
                name = f"pos{idx}"
                pos = item
            resolved.append((name, parse_pos(pos, default_z=fallback_z)))
        return resolved

    raw_anchor_names = settings.get("anchor_names")
    if raw_anchor_names is None:
        raw_anchor_names = settings.get("anchors")
    if raw_anchor_names is None:
        raw_anchor_names = ["left_back", "right_middle"]
    if isinstance(raw_anchor_names, str):
        raw_anchor_names = [x.strip() for x in raw_anchor_names.split(",") if x.strip()]
    if not isinstance(raw_anchor_names, list) or not raw_anchor_names:
        raise ValueError("anchor_names must be a non-empty list")

    resolved = []
    for idx, item in enumerate(raw_anchor_names):
        if isinstance(item, str):
            if item not in anchors:
                raise ValueError(f"unknown anchor name: {item}")
            resolved.append((item, anchors[item]))
        elif isinstance(item, dict):
            name = str(item.get("name", f"anchor{idx}"))
            pos = item.get("pos", item.get("position"))
            resolved.append((name, parse_pos(pos, default_z=fallback_z)))
        else:
            resolved.append((f"anchor{idx}", parse_pos(item, default_z=fallback_z)))
    return resolved


def stable_rng(asset: str, demo_id: int, salt: str) -> random.Random:
    seed_text = f"{asset}:{demo_id}:{salt}"
    seed = int(hashlib.sha1(seed_text.encode()).hexdigest()[:8], 16)
    return random.Random(seed)


def env_nonempty(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return value.strip()


def apply_env_overrides(settings: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(settings)
    env_map = {
        "OBJECT_SOFT_POS_SELECTION": "selection",
        "OBJECT_SOFT_XY_JITTER": "xy_jitter",
        "OBJECT_SOFT_X_JITTER": "x_jitter",
        "OBJECT_SOFT_Y_JITTER": "y_jitter",
        "OBJECT_SOFT_POS_Z": "z",
        "OBJECT_SOFT_ANCHOR_NAMES": "anchor_names",
    }
    for env_name, key in env_map.items():
        value = env_nonempty(env_name)
        if value is None:
            continue
        if key in {"xy_jitter", "x_jitter", "y_jitter", "z"}:
            result[key] = float(value)
        elif key == "anchor_names":
            result[key] = [x.strip() for x in value.split(",") if x.strip()]
        else:
            result[key] = value
    return result


def select_anchor(
    items: list[tuple[str, tuple[float, float, float]]],
    *,
    selection: str,
    mode: str,
    asset: str,
    demo_id: int,
    rng: random.Random,
    anchor_index: int | None,
) -> tuple[str, tuple[float, float, float]]:
    forced = mode.lower()
    if forced in {"left", "libero_left", "left_back"}:
        for name, pos in items:
            if "left" in name:
                return name, pos
        return ("libero_left", BUILTIN_LEFT)
    if forced in {"right", "libero_right", "right_middle"}:
        for name, pos in items:
            if "right" in name:
                return name, pos
        return ("libero_right", BUILTIN_RIGHT)

    sel = selection.lower()
    if sel in {"alternate", "demo_parity", "cycle"}:
        return items[demo_id % len(items)]
    if sel in {"fixed", "first"}:
        idx = int(anchor_index or 0)
        return items[idx % len(items)]
    if sel in {"random", "stable_random"}:
        return items[rng.randrange(len(items))]
    if sel == "asset_map":
        left_assets = {"pastry001", "pastry005", "pastry007", "pastry008"}
        want_left = asset in left_assets
        for name, pos in items:
            if want_left and "left" in name:
                return name, pos
            if not want_left and "right" in name:
                return name, pos
        return items[0]
    raise ValueError(f"unknown selection mode: {selection}")


def clip_value(value: float, low: Any, high: Any) -> float:
    if low is not None:
        value = max(value, float(low))
    if high is not None:
        value = min(value, float(high))
    return value


def resolve_position(args: argparse.Namespace) -> tuple[tuple[float, float, float], dict[str, Any]]:
    fallback = parse_pos(args.fallback_pos)
    fallback_z = fallback[2]
    config = load_config(args.config)
    anchors = named_anchor_map(config, fallback_z)

    default_settings = config.get("default", {})
    if default_settings is None:
        default_settings = {}
    if not isinstance(default_settings, dict):
        raise ValueError("default must be a JSON object")

    asset_settings = {}
    for section in ("assets", "asset_overrides"):
        section_value = config.get(section, {})
        if section_value is None:
            continue
        if not isinstance(section_value, dict):
            raise ValueError(f"{section} must be a JSON object")
        value = section_value.get(args.asset)
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(f"{section}.{args.asset} must be a JSON object")
            asset_settings = deep_merge(asset_settings, value)

    settings = deep_merge(default_settings, asset_settings)
    if settings.get("enabled", True) is False:
        settings = {
            "selection": os.environ.get("OBJECT_SOFT_POS_ANCHOR_MODE", "alternate"),
            "anchor_names": ["left_back", "right_middle"],
            "xy_jitter": os.environ.get("OBJECT_SOFT_XY_JITTER", "0.020"),
        }
    settings = apply_env_overrides(settings)

    mode = args.mode or os.environ.get("OBJECT_SOFT_POS_MODE", "config")
    anchor_items = resolve_anchor_list(settings, anchors, fallback_z)
    selection = str(settings.get("selection", os.environ.get("OBJECT_SOFT_POS_ANCHOR_MODE", "alternate")))
    seed_salt = str(settings.get("seed_salt", "object-soft-pos-config-v1"))
    rng = stable_rng(args.asset, args.demo_id, seed_salt)
    anchor_index = settings.get("anchor_index")
    name, anchor = select_anchor(
        anchor_items,
        selection=selection,
        mode=mode,
        asset=args.asset,
        demo_id=args.demo_id,
        rng=rng,
        anchor_index=int(anchor_index) if anchor_index is not None else None,
    )

    xy_jitter = float(settings.get("xy_jitter", 0.020) or 0.0)
    x_jitter = float(settings.get("x_jitter", xy_jitter) or 0.0)
    y_jitter = float(settings.get("y_jitter", xy_jitter) or 0.0)
    x = anchor[0] + rng.uniform(-x_jitter, x_jitter)
    y = anchor[1] + rng.uniform(-y_jitter, y_jitter)
    z_setting = settings.get("z", None)
    z = anchor[2] if z_setting is None else float(z_setting)

    clip = settings.get("clip", {}) or {}
    if not isinstance(clip, dict):
        raise ValueError("clip must be a JSON object")
    x = clip_value(x, clip.get("x_min"), clip.get("x_max"))
    y = clip_value(y, clip.get("y_min"), clip.get("y_max"))
    z = clip_value(z, clip.get("z_min"), clip.get("z_max"))

    meta = {
        "asset": args.asset,
        "demo_id": args.demo_id,
        "config": args.config,
        "anchor": name,
        "selection": selection,
        "mode": mode,
        "x_jitter": x_jitter,
        "y_jitter": y_jitter,
        "settings": settings,
    }
    return (x, y, z), meta


def distractor_layout_from_config(config: dict[str, Any]) -> str:
    settings = config.get("distractors", config.get("mixed_spread_distractors", {}))
    if settings in (None, {}):
        return ""
    if not isinstance(settings, dict):
        raise ValueError("distractors must be a JSON object")
    if settings.get("enabled", True) is False:
        return ""
    layout = settings.get("layout", settings.get("positions"))
    if layout in (None, ""):
        return ""

    if isinstance(layout, dict):
        cleaned = {}
        for name, value in layout.items():
            pos = parse_pos(value, default_z=0.0)
            cleaned[str(name)] = [round(pos[0], 8), round(pos[1], 8)]
        return json.dumps(cleaned, separators=(",", ":"), sort_keys=True)

    if isinstance(layout, list):
        cleaned_list = []
        for idx, value in enumerate(layout):
            if isinstance(value, dict):
                pos = value.get("xy", value.get("pos", value.get("position")))
                if pos is None:
                    raise ValueError(f"distractors.layout[{idx}] needs xy/pos/position")
            else:
                pos = value
            parsed = parse_pos(pos, default_z=0.0)
            cleaned_list.append([round(parsed[0], 8), round(parsed[1], 8)])
        return json.dumps(cleaned_list, separators=(",", ":"))

    raise ValueError("distractors.layout must be a JSON list or object")


def distractors_enabled_from_config(config: dict[str, Any]) -> bool:
    settings = config.get("distractors", config.get("mixed_spread_distractors", {}))
    if settings in (None, {}):
        return True
    if not isinstance(settings, dict):
        raise ValueError("distractors must be a JSON object")
    return bool(settings.get("enabled", True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset")
    parser.add_argument("--demo-id", type=int)
    parser.add_argument("--fallback-pos")
    parser.add_argument("--config", default=os.environ.get("OBJECT_SOFT_POS_CONFIG", ""))
    parser.add_argument("--mode", default=os.environ.get("OBJECT_SOFT_POS_MODE", "config"))
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--print-distractor-layout", action="store_true")
    parser.add_argument("--print-distractor-enabled", action="store_true")
    args = parser.parse_args()

    try:
        if args.print_distractor_enabled:
            print("1" if distractors_enabled_from_config(load_config(args.config)) else "0")
            return 0
        if args.print_distractor_layout:
            print(distractor_layout_from_config(load_config(args.config)))
            return 0
        missing = [name for name in ("asset", "demo_id", "fallback_pos") if getattr(args, name) in (None, "")]
        if missing:
            raise ValueError(f"missing required argument(s): {', '.join(missing)}")
        pos, meta = resolve_position(args)
    except Exception as exc:
        print(f"[object-soft-position-config] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.explain:
        print(
            json.dumps(
                {
                    "position": [round(pos[0], 8), round(pos[1], 8), round(pos[2], 8)],
                    "meta": meta,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
