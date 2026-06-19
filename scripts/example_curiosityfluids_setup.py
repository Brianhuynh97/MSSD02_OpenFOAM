#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

from matplotlib import pyplot as plt

from openfoam_project.airfoil import NACA4Spec, generate_naca4_airfoil
from openfoam_project.case import (
    DEFAULT_IMAGE,
    FlowConfig,
    _docker_command,
    _find_project_root,
    setup_case_from_example,
)


PROJECT_DIR = ROOT.resolve()
OPENFOAM_IMAGE = DEFAULT_IMAGE


def foam_cmd(*args: str, workdir: str = "/work") -> list[str]:
    return _docker_command(PROJECT_DIR, OPENFOAM_IMAGE, *args)


def main() -> None:
    case_dir = PROJECT_DIR / "openfoam_naca"

    airfoil_code = "4412"
    chord_length = 1.0
    n_surface_points = 32

    spec = NACA4Spec(
        max_camber=int(airfoil_code[0]),
        camber_position=int(airfoil_code[1]),
        thickness=int(airfoil_code[2:]),
    )
    discretized_surface_points = generate_naca4_airfoil(spec, chord=chord_length, points=n_surface_points)

    ax = plt.subplots(1, 1, figsize=(10, 5))[1]
    ax.plot(discretized_surface_points[:, 0], discretized_surface_points[:, 1], "o-", label=f"NACA {airfoil_code}")
    ax.set_aspect("equal")
    ax.legend()
    plot_path = PROJECT_DIR / "results" / "single_case" / f"airfoil_naca{airfoil_code}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(discretized_surface_points.shape)

    rho_freestream = 1.225
    u_freestream = 30.0
    flow = FlowConfig(
        velocity=u_freestream,
        density=rho_freestream,
        viscosity=1.5e-5,
        end_time=0.6,
        write_interval=0.05,
    )

    setup_case_from_example(case_dir, spec, n_surface_points, flow)

    project_root = _find_project_root(case_dir)
    case_name = case_dir.resolve().relative_to(project_root).as_posix()

    subprocess.run(foam_cmd("blockMesh", "-case", case_name), check=True)
    subprocess.run(foam_cmd("checkMesh", "-case", case_name), check=True)
    subprocess.run(foam_cmd("foamRun", "-case", case_name), check=True)


if __name__ == "__main__":
    main()
