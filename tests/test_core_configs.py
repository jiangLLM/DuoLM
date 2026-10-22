"""Scientific recipe boundaries independent of any private checkpoints."""
import json
from pathlib import Path
import unittest

from duolm.train_next_item import validate_config


ROOT = Path(__file__).resolve().parents[1] / "configs"


def recipe(name):
    return json.loads((ROOT / name).read_text())


class PublicConfigTests(unittest.TestCase):
    def test_all_public_recipes_validate_before_optional_teacher_binding(self):
        for directory in ("books", "games", "ablations"):
            for path in (ROOT / directory).glob("*.json"):
                with self.subTest(path=path):
                    config = json.loads(path.read_text())
                    validate_config(config, require_teacher_cache=False)
                    self.assertFalse(any("backup" in key for key in config["training"]))
                    if config["loss"]["recipe"] == "ce_kd":
                        with self.assertRaisesRegex(ValueError, "teacher cache"):
                            validate_config(config)

    def test_games_final_teacher_cache_preserves_bf16_tree_profile(self):
        cfg = recipe("games/teacher-cache.json")
        self.assertEqual(cfg["evaluation"]["compute_dtype"], "bfloat16")
        self.assertEqual(cfg["evaluation"]["scoring_backend"], "tree")
        self.assertEqual(cfg["evaluation"]["input_assembly"], "legacy")

    def test_main_teacher_and_student_are_mixed8_and_fast_readouts_are_separate(self):
        teacher, main, fast = [recipe(name) for name in
            ("books/teacher.json", "books/student.json", "ablations/default.json")]
        self.assertEqual(teacher["schema_version"], "books-next-item-v1")
        self.assertEqual(teacher["model"]["variant"], "T")
        self.assertEqual(main["data"]["train_candidates"], 8)
        self.assertEqual(main["data"]["train_policy"], "mixed-history")
        self.assertEqual(main["training"]["microbatch_queries"], 1)
        self.assertEqual(main["model"].get("input_assembly", "legacy"), "legacy")
        self.assertEqual(fast["training"]["microbatch_queries"], 8)
        self.assertEqual(fast["model"]["input_assembly"], "batched")
        for filename, readout in (("default", "last_token"), ("mem1", "memory"), ("perceiver1", "perceiver")):
            cfg = recipe(f"ablations/{filename}.json")
            self.assertEqual(cfg["data"], fast["data"])
            self.assertEqual(cfg["loss"], fast["loss"])
            self.assertEqual(cfg["training"], fast["training"])
            self.assertEqual(cfg["model"]["a_readout"], readout)
            self.assertEqual(cfg["model"]["m"], 1)
        self.assertEqual(recipe("ablations/v1.json")["model"]["variant"], "V1")
        self.assertEqual(recipe("ablations/ce-only.json")["loss"]["recipe"], "ce")
        self.assertEqual(recipe("ablations/beta09.json")["loss"]["beta_final"], .9)
        self.assertEqual(recipe("ablations/tau2.json")["loss"]["temperature"], 2)


if __name__ == "__main__":
    unittest.main()
