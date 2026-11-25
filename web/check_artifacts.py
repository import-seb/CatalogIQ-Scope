"""Verify the pinned serving bundle without loading models or training data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

DEFAULT_PINS = Path(__file__).resolve().parents[1] / "src/catalogiq/resources/inference/frozen_routes.json"


def required_files(pins):
    """Return unique paths and hashes, rejecting ambiguous or unsafe pins."""
    files = {}
    routes = pins.get("routes")
    if not isinstance(routes, dict) or not routes:
        raise ValueError("The route registry must contain model routes.")
    for target, route in routes.items():
        hashes = route.get("files_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError(f"No serving files declared for {target}.")
        for name, expected in hashes.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not path.parts:
                raise ValueError(f"Unsafe serving path: {name}")
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"Invalid SHA-256 for {name}")
            if name in files and files[name]["sha256"] != expected:
                raise ValueError(f"Conflicting hashes for {name}")
            entry = files.setdefault(name, {"path": name, "sha256": expected, "targets": []})
            entry["targets"].append(target)
    return list(files.values())


def verify_bundle(directory, pins):
    root = Path(directory).resolve()
    results = []
    for item in required_files(pins):
        entry = {**item}
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            entry["status"] = "outside_bundle"
        elif not path.is_file():
            entry["status"] = "missing"
        else:
            try:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                entry["actual_sha256"] = digest.hexdigest()
                entry["status"] = "ok" if entry["actual_sha256"] == item["sha256"] else "mismatch"
            except OSError:
                entry["status"] = "unreadable"
        results.append(entry)
    return {
        "registry_version": pins.get("version"),
        "required_count": len(results),
        "verified_count": sum(item["status"] == "ok" for item in results),
        "ready": all(item["status"] == "ok" for item in results),
        "files": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, help="Root containing the original relative serving paths")
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    parser.add_argument("--list", action="store_true", help="List required files and hashes without reading artifacts")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable result")
    args = parser.parse_args()
    try:
        pins = json.loads(args.pins.read_text(encoding="utf-8"))
        if args.list:
            files = required_files(pins)
            print(json.dumps({"registry_version": pins.get("version"), "required_count": len(files), "files": files}, indent=2))
            return 0
        if args.artifacts is None:
            parser.error("provide --artifacts or use --list")
        report = verify_bundle(args.artifacts, pins)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Serving files verified: {report['verified_count']}/{report['required_count']}")
        for item in report["files"]:
            if item["status"] != "ok":
                print(f"{item['status']}: {item['path']}")
        print("File checks passed. Real inference still needs a smoke test." if report["ready"] else "Bundle is incomplete or differs from the pinned version. Do not start live inference yet.")
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
