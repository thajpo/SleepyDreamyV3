import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_console_scripts_reference_importable_callables():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = project["project"]["scripts"]

    assert set(scripts) == {"dreamer-train", "dreamer-inspect"}
    for target in scripts.values():
        module_name, attribute = target.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attribute))


@pytest.mark.parametrize("module_name", ["dreamer.main", "dreamer.inspect"])
def test_supported_cli_help_exits_successfully(module_name):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "help" in result.stdout.lower()


@pytest.mark.parametrize("command", ["dreamer-train", "dreamer-inspect"])
def test_installed_console_script_help_exits_successfully(command):
    executable = shutil.which(command)
    assert executable is not None, f"{command} was not installed by the project"

    result = subprocess.run(
        [executable, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "help" in result.stdout.lower()
