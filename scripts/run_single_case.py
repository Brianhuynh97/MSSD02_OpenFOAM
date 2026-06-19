#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set up and run one OpenFOAM airfoil case: generate a discretized surface, "
            "write a blockMeshDict using the curiosityFluids mesher logic, create the "
            "OpenFOAM case files, and optionally run the solver."
        )
    )
    parser.add_argument("--camber", type=int, default=4)
    parser.add_argument("--camber-position", type=int, default=4)
    parser.add_argument("--thickness", type=int, default=12)
    parser.add_argument("--surface-points", type=int, default=32)
    parser.add_argument("--end-time", type=float, default=0.8)
    parser.add_argument("--velocity", type=float, default=30.0)
    parser.add_argument("--density", type=float, default=1.225)
    parser.add_argument("--viscosity", type=float, default=1.5e-5)
    parser.add_argument("--write-interval", type=float, default=0.05)
    parser.add_argument("--case-dir", type=Path, default=Path("openfoam_naca"))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "single_case")
    parser.add_argument("--runner", choices=["docker", "local"], default="docker")
    parser.add_argument("--image", default="microfluidica/openfoam:13")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from openfoam_project.airfoil import NACA4Spec
    from openfoam_project.case import FlowConfig, run_case, setup_case_from_example

    spec = NACA4Spec(args.camber, args.camber_position, args.thickness)
    flow = FlowConfig(
        velocity=args.velocity,
        density=args.density,
        viscosity=args.viscosity,
        end_time=args.end_time,
        write_interval=args.write_interval,
    )
    case_dir = args.case_dir
    print(f"[single-case] preparing {case_dir}", flush=True)
    airfoil_points = setup_case_from_example(case_dir, spec, args.surface_points, flow)

    from openfoam_project.postprocess import save_airfoil_plot

    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_airfoil_plot(
        airfoil_points,
        args.results_dir / f"airfoil_naca{spec.digits}.png",
        title=f"NACA {spec.digits} discretization",
    )
    npy_path = args.results_dir / f"airfoil_naca{spec.digits}_points.csv"
    import numpy as np

    np.savetxt(npy_path, airfoil_points, delimiter=",", header="x,y", comments="")

    if args.prepare_only:
        return

    from openfoam_project.postprocess import (
        read_force_coeffs,
        read_residuals,
        save_force_plot,
        save_residual_plot,
        summarize_tail_force_coeffs,
    )

    print(f"[single-case] running {case_dir}", flush=True)
    run_case(case_dir, image=args.image, runner=args.runner)

    force_df = read_force_coeffs(case_dir)
    residual_df = read_residuals(case_dir)
    summary = summarize_tail_force_coeffs(force_df)
    save_force_plot(force_df, args.results_dir / f"forces_naca{spec.digits}.png")
    save_residual_plot(residual_df, args.results_dir / f"residuals_naca{spec.digits}.png")

    summary_path = args.results_dir / f"summary_naca{spec.digits}.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# Single Case Summary: NACA {spec.digits}",
                "",
                f"- Case directory: `{case_dir}`",
                f"- Surface points: `{args.surface_points}`",
                f"- End time: `{args.end_time}` s",
                f"- Tail-averaged drag coefficient: `{summary.cd:.6f}`",
                f"- Tail-averaged lift coefficient: `{summary.cl:.6f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as exc:
        print(f"[single-case] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(130)
