import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "batch", Path(__file__).resolve().parents[1] / "scripts/batch_retarget.py"
)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / "data/processed/demo_001"
        self.folder.mkdir(parents=True)
        self.events = {
            "events": {
                name: {"frame": 30 * (i + 1), "time_s": float(i + 1)}
                for i, name in enumerate(batch.EVENTS)
            }
        }
        batch.write(self.folder / "human_events.json", self.events)
        batch.write(
            self.folder / "tracking_report.json",
            dict(fps=30, decoded_frames=240, timestamps="frame_index_div_fps"),
        )
        (self.folder / "marker_smoothed.csv").write_text(
            "seconds,x_m,y_m,height_m\n0,0,0,0\n7,1,1,1\n"
        )

    def validate(self):
        return batch.validate_events(
            self.folder / "human_events.json",
            self.folder / "marker_smoothed.csv",
            self.folder / "tracking_report.json",
        )

    def test_valid_events(self):
        self.assertEqual(self.validate(), self.events)

    def test_reject_missing_annotation(self):
        self.events["events"]["closure_start"]["time_s"] = None
        batch.write(self.folder / "human_events.json", self.events)
        with self.assertRaises(ValueError):
            self.validate()

    def test_reject_out_of_order(self):
        self.events["events"]["release_complete"] = {"frame": 30, "time_s": 1}
        batch.write(self.folder / "human_events.json", self.events)
        with self.assertRaises(ValueError):
            self.validate()

    def test_reject_outside_tracking(self):
        (self.folder / "marker_smoothed.csv").write_text(
            "seconds,x_m,y_m,height_m\n3,0,0,0\n7,1,1,1\n"
        )
        with self.assertRaises(ValueError):
            self.validate()

    def test_changed_inputs_invalidate_tracking(self):
        item = {
            "id": "demo_001",
            "video": "video.mov",
            "calibration": "cal.npz",
            "config": "cfg.json",
        }
        for key in ("video", "calibration", "config"):
            (self.root / item[key]).write_text("original")
        item["tracking_inputs"] = batch.input_hashes(self.root, item)
        item["tracking_outputs"] = {
            "marker_smoothed.csv": batch.digest(self.folder / "marker_smoothed.csv")
        }
        self.assertTrue(batch.tracking_current(self.root, item))
        (self.root / "video.mov").write_text("changed")
        self.assertFalse(batch.tracking_current(self.root, item))

    def test_changed_output_invalidates_tracking(self):
        item = {
            "id": "demo_001",
            "video": "video.mov",
            "calibration": "cal.npz",
            "config": "cfg.json",
        }
        for key in ("video", "calibration", "config"):
            (self.root / item[key]).write_text("original")
        item["tracking_inputs"] = batch.input_hashes(self.root, item)
        item["tracking_outputs"] = {
            "marker_smoothed.csv": batch.digest(self.folder / "marker_smoothed.csv")
        }
        (self.folder / "marker_smoothed.csv").write_text("changed")
        self.assertFalse(batch.tracking_current(self.root, item))

    def test_reject_path_id(self):
        with self.assertRaises(ValueError):
            batch.safe_id("../pilot_01")

    def test_script_hash_ignores_formatting_but_detects_code_changes(self):
        import ast
        import hashlib

        scripts = self.root / "scripts"
        scripts.mkdir()
        source = scripts / "example.py"
        source.write_text("value = 1\n")
        original = batch.digest(source)
        syntax_hash = hashlib.sha256(
            ast.dump(ast.parse(source.read_text()), include_attributes=False).encode()
        ).hexdigest()
        batch.write(
            self.root / "configs/script_revisions.json",
            {"example.py": {"source_sha256": original, "ast_sha256": syntax_hash}},
        )
        previous = batch.SCRIPTS
        try:
            batch.SCRIPTS = scripts
            source.write_text("# A comment\nvalue=1\n")
            self.assertEqual(batch.script_digest(source), original)
            source.write_text("value = 2\n")
            self.assertNotEqual(batch.script_digest(source), original)
        finally:
            batch.SCRIPTS = previous


if __name__ == "__main__":
    unittest.main()
