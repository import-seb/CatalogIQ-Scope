import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "check_artifacts.py"
spec = importlib.util.spec_from_file_location("artifact_check", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class ServingBundleChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pins = {"version": "synthetic-v1", "routes": {"Segment": {"files_sha256": {
            "model/checkpoint.pt": hashlib.sha256(b"model").hexdigest(),
            "model/policy.json": hashlib.sha256(b"{}\n").hexdigest(),
        }}}}

    def test_missing_and_changed_files_are_reported_without_loading_a_model(self):
        (self.root / "model").mkdir()
        (self.root / "model/checkpoint.pt").write_bytes(b"changed")
        result = check.verify_bundle(self.root, self.pins)
        self.assertFalse(result["ready"])
        self.assertEqual([r["status"] for r in result["files"]], ["mismatch", "missing"])
        (self.root / "model/checkpoint.pt").write_bytes(b"model")
        (self.root / "model/policy.json").write_bytes(b"{}\n")
        result = check.verify_bundle(self.root, self.pins)
        self.assertTrue(result["ready"])
        self.assertEqual(result["verified_count"], 2)
        # Artifact hashes are exact bytes; unlike source hashes, line endings matter.
        (self.root / "model/policy.json").write_bytes(b"{}\r\n")
        self.assertFalse(check.verify_bundle(self.root, self.pins)["ready"])

    def test_unsafe_paths_bad_hashes_and_conflicting_pins_are_rejected(self):
        for name in ("../outside.pt", "/outside.pt", "C:/outside.pt", "model\\outside.pt"):
            pins = {"routes": {"Segment": {"files_sha256": {name: "0" * 64}}}}
            with self.assertRaises(ValueError):
                check.required_files(pins)
        with self.assertRaises(ValueError):
            check.required_files({"routes": {"Segment": {"files_sha256": {"model.pt": "bad"}}}})
        self.pins["routes"]["Brand"] = {"files_sha256": {"model/checkpoint.pt": "0" * 64}}
        with self.assertRaises(ValueError):
            check.required_files(self.pins)

    def test_shared_files_are_counted_once(self):
        self.pins["routes"]["Brand"] = self.pins["routes"]["Segment"]
        files = check.required_files(self.pins)
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0]["targets"], ["Segment", "Brand"])

    def test_cli_lists_requirements_and_fails_when_bundle_is_missing(self):
        pins = self.root / "pins.json"
        pins.write_text(json.dumps(self.pins), encoding="utf-8")
        result = subprocess.run([sys.executable, str(SCRIPT), "--pins", str(pins), "--list"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(result.stdout)["required_count"], 2)
        result = subprocess.run([sys.executable, str(SCRIPT), "--pins", str(pins), "--artifacts", str(self.root), "--json"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)["ready"])

    def test_live_server_stops_before_model_import_when_bundle_is_missing(self):
        result = subprocess.run([sys.executable, str(SCRIPT.with_name("server.py")), "--live", "--artifacts", str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("serving files failed verification", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
