"""Record the interpreter, packages, CPU and BLAS/threading setup in artifacts/environment.json.

Every value is read from the running environment; nothing is hard-coded.
"""

from __future__ import annotations

import io
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from importlib.metadata import version
from typing import Any

import numpy as np
import scipy.linalg  # noqa: F401  (loads SciPy's BLAS so threadpoolctl can see it)
import sklearn.linear_model  # noqa: F401  (loads scikit-learn's OpenMP runtime)
from threadpoolctl import threadpool_info

from taiwan_soft_classifiers import config
from taiwan_soft_classifiers.data import utc_now_iso, write_json

PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")
THREAD_VARIABLES = (
    "PYTHONHASHSEED",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENBLAS_CORETYPE",
)
LSCPU_FIELDS = (
    "Architecture",
    "CPU(s)",
    "Vendor ID",
    "Model name",
    "Thread(s) per core",
    "Core(s) per socket",
    "Socket(s)",
    "Hypervisor vendor",
    "L2 cache",
    "L3 cache",
)
CPU_FLAGS = ("sse4_2", "avx", "avx2", "fma", "avx512f", "avx512dq", "avx512_vnni")


def _os_name() -> str | None:
    try:
        return platform.freedesktop_os_release().get("PRETTY_NAME")
    except OSError:
        return None


def _lscpu_summary() -> dict[str, Any] | None:
    if shutil.which("lscpu") is None:
        return None
    output = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
    fields = {}
    for line in output.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    flags = set(fields.get("Flags", "").split())
    summary: dict[str, Any] = {k: fields[k] for k in LSCPU_FIELDS if k in fields}
    summary["flags"] = {flag: flag in flags for flag in CPU_FLAGS}
    return summary


def _numpy_config() -> Any:
    try:
        config_dict = np.show_config(mode="dicts")
    except TypeError:  # older numpy without ``mode``
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            np.show_config()
        return buffer.getvalue()
    return json.loads(json.dumps(config_dict, default=str))


def main() -> None:
    manifest = {
        "recorded_at_utc": utc_now_iso(),
        "os": _os_name(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _lscpu_summary(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "packages": {name: version(name) for name in PACKAGES},
        "threadpool_info": json.loads(json.dumps(threadpool_info(), default=str)),
        "numpy_config": _numpy_config(),
    }
    path = config.ARTIFACTS_DIR / "environment.json"
    write_json(manifest, path)
    print(f"OS {manifest['os']}")
    print(f"Python {manifest['python']} ({manifest['machine']})")
    for name, value in manifest["packages"].items():
        print(f"{name} {value}")
    cpu = manifest["cpu"] or {}
    print(f"CPU {cpu.get('Model name')} | vendor {cpu.get('Vendor ID')} | CPUs {cpu.get('CPU(s)')}")
    print(f"CPU flags {cpu.get('flags')}")
    for name, value in manifest["thread_environment"].items():
        print(f"{name}={value}")
    for pool in manifest["threadpool_info"]:
        print(
            "threadpool "
            + " | ".join(
                f"{key}={pool.get(key)}"
                for key in ("user_api", "internal_api", "version", "num_threads", "architecture")
            )
        )
    print(f"Wrote {path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
