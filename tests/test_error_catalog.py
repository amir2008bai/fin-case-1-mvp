"""The worked examples in docs/error_catalog.md must keep matching the committed data and truth."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ErrorCatalogTests(unittest.TestCase):
    def test_catalog_examples_recompute_to_the_truth(self):
        result = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "catalog_check.py")], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-2000:])
        self.assertIn("10/10 matched", result.stdout)


if __name__ == "__main__":
    unittest.main()
