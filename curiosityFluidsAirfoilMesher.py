#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openfoam_project.mesher import curiosityFluidsAirfoilMesher as _write_block_mesh_dict


def curiosityFluidsAirfoilMesher(
    airfoil_points: np.ndarray,
    path_to_blockdict: Path | str = "openfoam_naca/system/blockMeshDict",
) -> Path:
    points = np.asarray(airfoil_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("airfoil_points must be an array of shape (n_points, 2)")
    return _write_block_mesh_dict(points, Path(path_to_blockdict))


def load_airfoil_points(path: Path | str, *, skiprows: int = 1) -> np.ndarray:
    """Load a legacy two-column airfoil coordinate file."""
    return np.loadtxt(path, skiprows=skiprows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an OpenFOAM blockMeshDict from either a legacy airfoil "
            "coordinate file or, when imported, a NumPy array of surface points."
        )
    )
    parser.add_argument("airfoil_file", help="Path to a two-column airfoil coordinate file")
    parser.add_argument(
        "-o",
        "--output",
        default="system/blockMeshDict",
        help="Path to the blockMeshDict output file",
    )
    parser.add_argument(
        "--skiprows",
        type=int,
        default=1,
        help="Number of header rows to skip when reading the coordinate file",
    )
    args = parser.parse_args()

    airfoil_points = load_airfoil_points(args.airfoil_file, skiprows=args.skiprows)
    output_path = curiosityFluidsAirfoilMesher(airfoil_points, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
