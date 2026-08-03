"""Dependency-light repository quality checks shared by release tooling."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from typing import Iterable


IGNORED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist"}
LEGAL_NAMES = {"LICENSE", "LICENCE", "LICENSE_GEMMA.txt"}
GENERATED_SUFFIXES = {".pyc", ".pyo", ".bak", ".orig", ".rej", ".tmp"}
PRIVATE_PATH = re.compile(
    r"/vepfs|/Users/|/home/(?!\$USER)|/mnt/|/data/Projects|root@\d{1,3}(?:\.\d{1,3}){3}|anonymous_run"
    # A bare IPv4 literal reachable as a host. Anchored on a host-like keyword so
    # dotted version strings (e.g. a "10.3.7.77" wheel version in a lock file) do
    # not trip the check; the wildcard bind and loopback are explicitly allowed.
    r"|(?i:(?<![A-Za-z])(?:host|hostname|server|ssh|scp|addr|address|endpoint|ip)(?![A-Za-z]))"
    r"[^\n]{0,24}?\b(?!0\.0\.0\.0\b)(?!127\.)\d{1,3}(?:\.\d{1,3}){3}\b"
)


def files_under(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(
            part in IGNORED_DIRS or part.endswith(".egg-info")
            for part in path.relative_to(root).parts
        )
    )


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def duplicate_groups(root: Path) -> list[list[Path]]:
    groups: dict[tuple[int, str], list[Path]] = {}
    for path in files_under(root):
        size = path.stat().st_size
        if not size or path.name in LEGAL_NAMES:
            continue
        groups.setdefault((size, digest(path)), []).append(path)
    return [paths for paths in groups.values() if len(paths) > 1]


def cross_repository_duplicates(left: Path, right: Path) -> list[tuple[list[Path], list[Path]]]:
    def index(root: Path) -> dict[tuple[int, str], list[Path]]:
        result: dict[tuple[int, str], list[Path]] = {}
        for path in files_under(root):
            size = path.stat().st_size
            if not size or path.name in LEGAL_NAMES:
                continue
            result.setdefault((size, digest(path)), []).append(path)
        return result

    left_index, right_index = index(left), index(right)
    return [(left_index[key], right_index[key]) for key in left_index.keys() & right_index.keys()]


def syntax_errors(root: Path) -> list[str]:
    errors: list[str] = []
    for path in files_under(root):
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".toml":
                tomllib.loads(path.read_text(encoding="utf-8"))
            elif path.suffix in {".yaml", ".yml", ".cff"}:
                import yaml

                yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    for path in (item for item in files_under(root) if item.suffix == ".sh"):
        result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True, check=False)
        if result.returncode:
            errors.append(f"{path}: {result.stderr.strip()}")
    return errors


def generated_artifacts(root: Path) -> list[Path]:
    return [
        path
        for path in files_under(root)
        if path.name == ".DS_Store"
        or path.suffix in GENERATED_SUFFIXES
    ]


def private_path_hits(paths: Iterable[Path]) -> list[str]:
    hits: list[str] = []
    for path in paths:
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in {".py", ".sh", ".yaml", ".yml", ".json", ".toml", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if PRIVATE_PATH.search(line):
                hits.append(f"{path}:{line_number}: {line.strip()}")
    return hits
