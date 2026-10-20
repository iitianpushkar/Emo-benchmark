"""Guard against broken command routes after moving script groups."""
import ast
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._paths import SCRIPT_PATHS, resolve_script


class RepositoryLayoutTests(unittest.TestCase):
    def test_registry_targets_and_local_imports_exist(self):
        for name in SCRIPT_PATHS:
            path = resolve_script(name)
            self.assertTrue(path.is_file(), str(path))
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('scripts.'):
                    self.assertIsNotNone(importlib.util.find_spec(node.module), node.module)

    def test_dispatcher_and_direct_cli(self):
        result = subprocess.run([sys.executable, '-m', 'scripts', 'summarize_information_horizon', '--help'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        # Subprocess helpers must work even if the caller is outside the repo.
        result = subprocess.run([sys.executable, str(resolve_script('run_information_horizon.py')), '--help'], cwd='/tmp', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unknown_command_fails(self):
        result = subprocess.run([sys.executable, '-m', 'scripts', 'not_a_command'], cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unknown command', result.stderr)


if __name__ == '__main__':
    unittest.main()
