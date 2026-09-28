"""Record the versions of the running interpreter and key packages in artifacts/environment.json.

Every value is read from the running environment; nothing is hard-coded.
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import version

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.data import utc_now_iso, write_json

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib")


def _os_name() -> str | None:
    try:
        return platform.freedesktop_os_release().get("PRETTY_NAME")
    except OSError:
        return None


def main() -> None:
    manifest = {
        "recorded_at_utc": utc_now_iso(),
        "os": _os_name(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in PACKAGES},
    }
    path = config.ARTIFACTS_DIR / "environment.json"
    write_json(manifest, path)
    print(f"OS {manifest['os']}")
    print(f"Python {manifest['python']}")
    for name, value in manifest["packages"].items():
        print(f"{name} {value}")
    print(f"Wrote {path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
