# OpenFOAM NACA Airfoil Study

This repository contains Python scripts to prepare and run OpenFOAM simulations of flow over NACA 4-digit airfoils, with a focus on the three study steps required by the assignment:

- `scripts/run_point_convergence.py`: for one selected airfoil, runs
  1. a convergence study with respect to the number of surface points,
  2. a runtime-convergence study to determine a suitable simulation end time,
  3. exports plots, CSV tables, and a Markdown summary with recommended settings.
- `scripts/run_camber_study.py`: using the chosen discretization and end time, runs the camber sweep for `M = 0..8` at zero angle of attack and exports the lift/drag versus camber plot.

The Python package in `openfoam_project/` contains the reusable pieces:

- NACA 4-digit geometry generation
- `blockMeshDict` generation adapted from `curiosityFluidsAirfoilMesher.py`
- OpenFOAM case file generation
- one shared case-setup helper that follows the notebook/example sequence directly
- optional Docker-based OpenFOAM execution
- force/residual post-processing and plot export

## Setup

Create an environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For actually running CFD cases, you also need one of:

- Docker with the image `microfluidica/openfoam:13`
- a local OpenFOAM 13 installation with `blockMesh`, `checkMesh`, and `foamRun` on `PATH`

## Usage

Prepare the convergence-study case directories without running OpenFOAM:

```bash
python3 scripts/run_point_convergence.py --prepare-only
```

Run the full convergence study through Docker:

```bash
python3 scripts/run_point_convergence.py --runner docker
```

The convergence script writes:

- `results/convergence/point_convergence.csv`
- `results/convergence/runtime_convergence.csv`
- `results/convergence/point_convergence.png`
- `results/convergence/runtime_convergence.png`
- `results/convergence/convergence_summary.md`
- `results/convergence/recommended_settings.json`

The summary file reports the recommended number of surface points and the recommended simulation end time based on coefficient and residual tolerances.

