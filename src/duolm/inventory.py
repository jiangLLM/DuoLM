"""Freeze Books item membership from the official 5-core CSV's ID column.

This preserves the original experiment's catalog filter: every distinct
parent_asin in Books.csv.gz. No rating, timestamp, or user-to-item association is
used to choose items. The upstream file is itself a 5-core interaction dataset;
this is not the set of all Books metadata records.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import tempfile

from .build_next_item_assets import _atomic_json
from .prepare_next_item import _verify_inputs_unchanged, file_hash


def build_inventory(source, output):
    """Read only parent_asin values; write the canonical sorted ID-only array."""
    source, output = Path(source), Path(output)
    source_hash = file_hash(source)
    identity = {"schema": "public-catalog-id-inventory-v1",
                "uses_ratings": False, "uses_timestamps": False,
                "uses_user_association": False, "source_column": "parent_asin",
                "source_sha256": source_hash, "code_sha256": file_hash(__file__)}
    manifest_path = output / "manifest.json"
    ids_path = output / "catalog-item-ids.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (any(previous.get(key) != value for key, value in identity.items())
                or previous.get("item_ids_sha256") != file_hash(ids_path)):
            raise ValueError("existing inventory differs from the requested source or code")
        return previous
    if output.exists() and any(output.iterdir()):
        raise ValueError("untracked inventory output is not overwritten")
    ids, count = set(), 0
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if header.count("parent_asin") != 1:
            raise ValueError("public dataset requires one parent_asin column")
        column = header.index("parent_asin")
        for row in reader:
            if len(row) != len(header):
                raise ValueError("public CSV row does not match its header")
            item = row[column]
            if not item or item.strip() != item:
                raise ValueError("invalid public catalog item ID")
            ids.add(item)
            count += 1
    if not ids:
        raise ValueError("public catalog item inventory is empty")
    _verify_inputs_unchanged({"source": source}, {"source": source_hash})
    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output,
                                         prefix=".item-ids-", delete=False) as stream:
            temporary = Path(stream.name)
            # Preserve the original generator's exact JSON serialization.
            stream.write(json.dumps(sorted(ids), separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, ids_path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    report = {**identity, "item_ids_sha256": file_hash(ids_path),
              "item_count": len(ids), "source_rows": count}
    _atomic_json(manifest_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="official Books 5-core CSV or .csv.gz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_inventory(args.source, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
