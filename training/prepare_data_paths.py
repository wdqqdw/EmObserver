#!/usr/bin/env python3
"""Materialize portable mvei-data:// image references as local absolute paths."""

import argparse
import json
from pathlib import Path


PREFIX = "mvei-data://"


def materialize(value, data_root: Path, missing: set):
    if isinstance(value, dict):
        return {key: materialize(item, data_root, missing) for key, item in value.items()}
    if isinstance(value, list):
        return [materialize(item, data_root, missing) for item in value]
    if isinstance(value, str) and value.startswith(PREFIX):
        relative = Path(value[len(PREFIX):])
        target = (data_root / relative).resolve()
        target.relative_to(data_root)
        if not target.is_file():
            missing.add(str(target))
        return str(target)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    missing = set()
    rows = 0

    with args.input.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            row = materialize(json.loads(line), data_root, missing)
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

    if missing:
        temporary.unlink(missing_ok=True)
        sample = "\n".join(f"  - {path}" for path in sorted(missing)[:10])
        raise FileNotFoundError(
            f"{len(missing)} referenced data files are missing. First entries:\n{sample}"
        )
    temporary.replace(output)
    print(f"Prepared {rows} records: {output}")


if __name__ == "__main__":
    main()

