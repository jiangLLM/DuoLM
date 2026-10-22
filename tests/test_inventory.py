import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from duolm.inventory import build_inventory


class InventoryTests(unittest.TestCase):
    def test_original_filter_and_bytes_ignore_nonidentity_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix, reverse in (("first", False), ("second", True)):
                path = root / (suffix + ".csv.gz")
                rows = [["private-user", "item-b", "not-a-rating", "not-a-time"],
                        ["different-user", "item-a", "invalid", "invalid"],
                        ["private-user", "item-b", "1", "0"]]
                with gzip.open(path, "wt", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["user_id", "parent_asin", "rating", "timestamp"])
                    writer.writerows(reversed(rows) if reverse else rows)
                report = build_inventory(path, root / suffix)
                self.assertEqual(report["source_rows"], 3)
                self.assertEqual(report["item_count"], 2)
                self.assertTrue(all(report[key] is False for key in
                                    ("uses_ratings", "uses_timestamps", "uses_user_association")))
                self.assertEqual((root / suffix / "catalog-item-ids.json").read_bytes(),
                                 b'["item-a","item-b"]\n')
                self.assertNotIn("private-user", (root / suffix / "manifest.json").read_text())
                self.assertEqual(report, build_inventory(path, root / suffix))
            self.assertEqual(json.loads((root / "first/manifest.json").read_text())["item_ids_sha256"],
                             json.loads((root / "second/manifest.json").read_text())["item_ids_sha256"])

    def test_changed_source_or_output_is_never_silently_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.csv"
            source.write_text("parent_asin\nitem-a\n")
            build_inventory(source, root / "out")
            source.write_text("parent_asin\nitem-b\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                build_inventory(source, root / "out")
            source.write_text("parent_asin\nitem-a\n")
            (root / "out/catalog-item-ids.json").write_text('["corrupt"]')
            with self.assertRaisesRegex(ValueError, "differs"):
                build_inventory(source, root / "out")

    def test_invalid_ids_and_columns_are_rejected_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, content in enumerate(("parent_asin\n padded\n", "other\na\n", "parent_asin\n")):
                source, output = root / f"source-{index}.csv", root / f"out-{index}"
                source.write_text(content)
                with self.assertRaises(ValueError):
                    build_inventory(source, output)
                self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
