#!/usr/bin/env python3
"""Write a top-down SVG preview for object-soft target and distractor layout."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace

import object_soft_position_config as poscfg


def parse_demo_ids(raw: str | None, start: int, count: int) -> list[int]:
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return list(range(start, start + count))


def svg_escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def parse_distractors(config: dict) -> list[tuple[str, float, float]]:
    raw = poscfg.distractor_layout_from_config(config)
    if not raw:
        return []
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        return [(str(name), float(xy[0]), float(xy[1])) for name, xy in parsed.items()]
    return [(f"d{i}", float(xy[0]), float(xy[1])) for i, xy in enumerate(parsed)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--asset", default="pastry001")
    parser.add_argument("--demo-start", type=int, default=990000)
    parser.add_argument("--demo-count", type=int, default=10)
    parser.add_argument("--demo-ids")
    parser.add_argument("--fallback-pos", default="-0.11934115 -0.02000000 0.045")
    parser.add_argument("--mode", default="config")
    parser.add_argument("--xlim", default="-0.26,0.26")
    parser.add_argument("--ylim", default="-0.34,0.18")
    parser.add_argument("--basket-pos", default="-0.11934115 0.13000000 0.020")
    args = parser.parse_args()

    config = poscfg.load_config(args.config)
    fallback = poscfg.parse_pos(args.fallback_pos)
    anchors = poscfg.named_anchor_map(config, fallback[2])
    demo_ids = parse_demo_ids(args.demo_ids, args.demo_start, args.demo_count)
    targets = []
    for demo_id in demo_ids:
        pos, meta = poscfg.resolve_position(
            SimpleNamespace(
                asset=args.asset,
                demo_id=demo_id,
                fallback_pos=args.fallback_pos,
                config=args.config,
                mode=args.mode,
            )
        )
        targets.append((demo_id, meta["anchor"], pos[0], pos[1], pos[2]))

    distractors = parse_distractors(config)
    basket = poscfg.parse_pos(args.basket_pos)
    xmin, xmax = [float(x) for x in args.xlim.split(",")]
    ymin, ymax = [float(y) for y in args.ylim.split(",")]
    width, height = 1100, 900
    pad = 80

    def sx(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;font-size:18px;fill:#17202a}",
        ".small{font-size:14px;fill:#34495e}.grid{stroke:#d6dbdf;stroke-width:1}.axis{stroke:#839192;stroke-width:2}",
        ".anchor{fill:#ffffff;stroke:#c0392b;stroke-width:3}.target{fill:#e74c3c;stroke:#7b241c;stroke-width:2}",
        ".distractor{fill:#95a5a6;stroke:#34495e;stroke-width:2}.basket{fill:#2ecc71;stroke:#145a32;stroke-width:3}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fbfcfc"/>',
        f'<text x="{pad}" y="42">object-soft layout preview: {svg_escape(args.asset)} demos {svg_escape(demo_ids[0])}-{svg_escape(demo_ids[-1])}</text>',
        f'<rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" fill="#ffffff" stroke="#566573" stroke-width="2"/>',
    ]

    gx = xmin
    while gx <= xmax + 1e-9:
        x = sx(gx)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height-pad}"/>')
        lines.append(f'<text class="small" x="{x-24:.1f}" y="{height-pad+28}">{gx:.2f}</text>')
        gx += 0.05
    gy = ymin
    while gy <= ymax + 1e-9:
        y = sy(gy)
        lines.append(f'<line class="grid" x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}"/>')
        lines.append(f'<text class="small" x="{pad-64}" y="{y+5:.1f}">{gy:.2f}</text>')
        gy += 0.05

    if xmin <= 0 <= xmax:
        lines.append(f'<line class="axis" x1="{sx(0):.1f}" y1="{pad}" x2="{sx(0):.1f}" y2="{height-pad}"/>')
    if ymin <= 0 <= ymax:
        lines.append(f'<line class="axis" x1="{pad}" y1="{sy(0):.1f}" x2="{width-pad}" y2="{sy(0):.1f}"/>')

    bx, by = sx(basket[0]), sy(basket[1])
    lines.append(f'<rect class="basket" x="{bx-18:.1f}" y="{by-18:.1f}" width="36" height="36" rx="4"/>')
    lines.append(f'<text class="small" x="{bx+22:.1f}" y="{by+5:.1f}">basket</text>')

    for name, pos in anchors.items():
        x, y = sx(pos[0]), sy(pos[1])
        lines.append(f'<circle class="anchor" cx="{x:.1f}" cy="{y:.1f}" r="11"/>')
        lines.append(f'<text class="small" x="{x+15:.1f}" y="{y-12:.1f}">{svg_escape(name)}</text>')

    for name, x_raw, y_raw in distractors:
        x, y = sx(x_raw), sy(y_raw)
        lines.append(f'<circle class="distractor" cx="{x:.1f}" cy="{y:.1f}" r="13"/>')
        lines.append(f'<text class="small" x="{x+16:.1f}" y="{y+5:.1f}">{svg_escape(name)}</text>')

    for demo_id, anchor, x_raw, y_raw, _z in targets:
        x, y = sx(x_raw), sy(y_raw)
        lines.append(f'<circle class="target" cx="{x:.1f}" cy="{y:.1f}" r="8"/>')
        lines.append(f'<text class="small" x="{x+10:.1f}" y="{y-8:.1f}">{svg_escape(demo_id)} {svg_escape(anchor)}</text>')

    legend_x = width - 330
    lines.extend(
        [
            f'<circle class="target" cx="{legend_x}" cy="48" r="8"/><text class="small" x="{legend_x+18}" y="53">resolved target after jitter</text>',
            f'<circle class="anchor" cx="{legend_x}" cy="76" r="8"/><text class="small" x="{legend_x+18}" y="81">configured target anchor</text>',
            f'<circle class="distractor" cx="{legend_x}" cy="104" r="8"/><text class="small" x="{legend_x+18}" y="109">mixed-spread distractor XY</text>',
            f'<rect class="basket" x="{legend_x-8}" y="122" width="16" height="16"/><text class="small" x="{legend_x+18}" y="137">basket</text>',
        ]
    )
    lines.append("</svg>")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
