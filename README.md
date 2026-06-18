# OpenFOAM NACA Airfoil Study

This repository is a command-line project scaffold for MSSD Project 02, based on the exercise notes at `02-openfoam-notes.html`.

It is structured around the two requested study scripts:

- `scripts/run_point_convergence.py`: prepares or runs a surface-point convergence study and a simulation-time convergence study for one airfoil.
- `scripts/run_camber_study.py`: prepares or runs the camber sweep for `M = 0..8` at zero angle of attack.

The Python package in `openfoam_project/` contains the reusable pieces:

- NACA 4-digit geometry generation
- `blockMeshDict` generation adapted from `curiosityFluidsAirfoilMesher.py`
- OpenFOAM case file generation
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

Run the convergence study through Docker:

```bash
python3 scripts/run_point_convergence.py --runner docker
```

Prepare the camber-study case directories:

```bash
python3 scripts/run_camber_study.py --prepare-only
```

Run the camber study:

```bash
python3 scripts/run_camber_study.py --runner docker
```

Generated case folders go under `cases/`. Generated CSV summaries and plots go under `results/`.
