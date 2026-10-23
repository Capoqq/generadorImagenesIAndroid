import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.package_model import REQUIRED, package


class PackageModelTest(unittest.TestCase):
    def test_manifest_hashes_match_zip_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in REQUIRED:
                (root / name).write_bytes(name.encode())
            output = root / "model.zip"
            package(root, output, "sd15-test", "creativeml-openrail-m", "https://example.org/model")
            with zipfile.ZipFile(output) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(set(REQUIRED), set(manifest["sha256"]))
                self.assertEqual(set(REQUIRED) | {"manifest.json"}, set(archive.namelist()))
                import hashlib
                for name in REQUIRED:
                    self.assertEqual(manifest["sha256"][name], hashlib.sha256(archive.read(name)).hexdigest())

    def test_missing_component_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "Missing or empty"):
                package(root, root / "model.zip", "sd15-test", "license", "https://example.org")


if __name__ == "__main__":
    unittest.main()
