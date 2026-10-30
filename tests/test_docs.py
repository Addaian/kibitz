"""
The README is the plan, so its instructions have to be executable.

This exists because Appendix B once documented `python3 -m kibitz.parser_all`,
a module that never existed — the parse step had only ever been run from a
scratch script. A command nobody can run is worse than no command.
"""

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
README = (ROOT / "README.md").read_text()


def _appendix_b_commands():
    block = re.search(r"## Appendix B.*?```bash\n(.*?)```", README, re.S)
    assert block, "Appendix B lost its command block"
    for line in block.group(1).splitlines():
        line = line.split("#")[0].strip()
        if line.startswith("python3"):
            yield line


COMMANDS = list(_appendix_b_commands())


def test_appendix_b_documents_the_whole_pipeline():
    assert len(COMMANDS) >= 4, COMMANDS


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: c.split()[-1])
def test_every_documented_command_resolves(command):
    parts = command.split()
    if parts[1] == "-m":
        module = parts[2]
        assert importlib.util.find_spec(module), f"no module named {module}"
    else:
        script = ROOT / parts[1]
        assert script.exists(), f"{script} does not exist"


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: c.split()[-1])
def test_every_documented_entry_point_is_runnable(command):
    """A module needs a main(); a script needs an if __name__ block."""
    parts = command.split()
    if parts[1] == "-m":
        path = ROOT / (parts[2].replace(".", "/") + ".py")
    else:
        path = ROOT / parts[1]
    tree = ast.parse(path.read_text())
    has_main = any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body)
    has_guard = any(
        isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"
        for n in tree.body)
    assert has_main and has_guard, f"{path.name} is not runnable as documented"


def test_the_decision_log_exists_and_is_populated():
    """Rule in Appendix A: a deviation is not done until it is recorded there."""
    block = re.search(r"## Appendix A.*?(?=\n## |\Z)", README, re.S)
    assert block, "decision log missing"
    rows = re.findall(r"^\| \d+ \|", block.group(0), re.M)
    assert len(rows) >= 10, f"only {len(rows)} decisions recorded"


@pytest.mark.parametrize("label", ["brilliant", "sacrifice", "forcing"])
def test_dropped_classifications_are_explained_not_just_removed(label):
    block = re.search(r"## Appendix A.*?(?=\n## |\Z)", README, re.S).group(0)
    assert label in block, f"{label} was removed from the enum without a recorded reason"


# --------------------------------------------------------------------------- #
# import hygiene
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", sorted((ROOT / "scripts").glob("*.py")),
                         ids=lambda p: p.name)
def test_no_script_shadows_a_stdlib_module(path):
    """
    Running `python3 scripts/foo.py` puts scripts/ on sys.path[0], so a file
    named like a standard-library module hijacks it for every later import.
    scripts/inspect.py did exactly that and broke huggingface_hub's import of
    `inspect` deep inside mlx-lm.
    """
    assert path.stem not in sys.stdlib_module_names, (
        f"scripts/{path.name} shadows the stdlib module '{path.stem}'")


@pytest.mark.parametrize("path", sorted((ROOT / "kibitz").glob("*.py")),
                         ids=lambda p: p.name)
def test_no_package_module_shadows_a_stdlib_module(path):
    if path.stem == "__init__":
        return
    assert path.stem not in sys.stdlib_module_names, (
        f"kibitz/{path.name} shadows the stdlib module '{path.stem}'")
