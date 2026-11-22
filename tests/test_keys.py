"""The secret key file: created private and complete, never accepted short or empty."""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from audit_env import keys


class SecretKeyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "nested" / "secret.key"
        environment = {key: value for key, value in os.environ.items() if key != keys.ENV_VAR}
        environment[keys.KEY_FILE_VAR] = str(self.path)
        patches = (mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(keys, "_process_key", None))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.temp.cleanup)

    def secret(self):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            return keys.secret_key(), stderr.getvalue()

    def test_creates_a_private_key_once(self):
        first, _ = self.secret()
        self.assertEqual(len(first), 64)
        self.assertEqual(self.path.read_bytes(), first)
        self.assertEqual(self.secret()[0], first)
        if os.name == "posix":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_short_or_empty_key_file_is_never_the_key(self):
        self.path.parent.mkdir(parents=True)
        for content in (b"", b"short"):
            self.path.write_bytes(content)
            keys._process_key = None
            key, warning = self.secret()
            self.assertGreaterEqual(len(key), keys.MIN_KEY_BYTES)
            self.assertNotEqual(key, content)
            self.assertIn("fewer than", warning)



class DefaultKeyTests(unittest.TestCase):
    def test_without_configuration_the_key_stays_in_memory(self):
        """A default key file in the home folder could be read by an agent run as a subprocess."""
        import io
        import os
        import tempfile
        from contextlib import redirect_stderr
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as home:
            clean = {k: v for k, v in os.environ.items() if k not in (keys.ENV_VAR, keys.KEY_FILE_VAR)}
            clean["HOME"] = clean["USERPROFILE"] = home
            with mock.patch.dict(os.environ, clean, clear=True), mock.patch.object(keys, "_process_key", None), \
                    redirect_stderr(io.StringIO()):
                first, second = keys.secret_key(), keys.secret_key()
            self.assertEqual(first, second)
            self.assertGreaterEqual(len(first), keys.MIN_KEY_BYTES)
            self.assertEqual(list(Path(home).rglob("*")), [])


if __name__ == "__main__":
    unittest.main()
