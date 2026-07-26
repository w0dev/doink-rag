"""The playbooks repo deploys a wheel pinned by git tag. If the packaged
version and the tag disagree, the deployed artifact cannot be traced back to a
commit."""
import subprocess
import tomllib
from pathlib import Path


def _pyproject_version() -> str:
    data = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    return data["project"]["version"]


def test_package_version_matches_the_latest_git_tag():
    tag = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True, text=True, cwd=Path(__file__).parent.parent,
    ).stdout.strip()
    assert tag, "no git tag found"
    assert tag == f"v{_pyproject_version()}", (
        f"pyproject version {_pyproject_version()!r} does not match tag {tag!r}. "
        f"Bump pyproject before tagging, or the built wheel is misnamed."
    )
