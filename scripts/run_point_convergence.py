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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the surface-point and runtime convergence study.")
    parser.add_argument("--camber", type=int, default=4)
    parser.add_argument("--camber-position", type=int, default=4)
    parser.add_argument("--thickness", type=int, default=12)
    parser.add_argument("--surface-points", nargs="+", type=int, default=[12, 16, 20, 24, 32, 48, 64])
    parser.add_argument("--time-study-end-times", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--time-study-points", type=int, default=32)
    parser.add_argument("--velocity", type=float, default=30.0)
    parser.add_argument("--density", type=float, default=1.225)
    parser.add_argument("--viscosity", type=float, default=1.5e-5)
    parser.add_argument("--write-interval", type=float, default=0.05)
    parser.add_argument("--cases-dir", type=Path, default=Path("cases"))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "convergence")
    parser.add_argument("--runner", choices=["docker", "local"], default="docker")
    parser.add_argument("--image", default="microfluidica/openfoam:13")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def plot_point_convergence(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(df["points"], df["cd"], marker="o", label="Cd")
    plt.plot(df["points"], df["cl"], marker="o", label="Cl")
    plt.xlabel("Airfoil surface points")
    plt.ylabel("Tail-averaged coefficient")
    plt.title("Surface-point convergence")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_runtime_convergence(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(df["end_time"], df["cd"], marker="o", label="Cd")
    plt.plot(df["end_time"], df["cl"], marker="o", label="Cl")
    plt.xlabel("Simulation end time (s)")
    plt.ylabel("Tail-averaged coefficient")
    plt.title("Runtime convergence")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    from openfoam_project.airfoil import NACA4Spec
    from openfoam_project.case import FlowConfig, prepare_case, run_case

    spec = NACA4Spec(args.camber, args.camber_position, args.thickness)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    point_rows = []
    total_point_cases = len(args.surface_points)
    for index, points in enumerate(args.surface_points, start=1):
        case_dir = args.cases_dir / "convergence_points" / f"naca{spec.digits}_pts{points:03d}"
        print(f"[study] point convergence {index}/{total_point_cases}: preparing {case_dir}", flush=True)
        flow = FlowConfig(
            velocity=args.velocity,
            density=args.density,
            viscosity=args.viscosity,
            end_time=max(args.time_study_end_times),
            write_interval=args.write_interval,
        )
        prepare_case(case_dir, spec, points, flow)
        if args.prepare_only:
            continue
        import pandas as pd

        from openfoam_project.postprocess import (
            read_force_coeffs,
            read_residuals,
            save_force_plot,
            save_residual_plot,
            summarize_tail_force_coeffs,
        )

        print(f"[study] point convergence {index}/{total_point_cases}: running {case_dir}", flush=True)
        run_case(case_dir, image=args.image, runner=args.runner)
        force_df = read_force_coeffs(case_dir)
        residual_df = read_residuals(case_dir)
        summary = summarize_tail_force_coeffs(force_df)
        save_force_plot(force_df, args.results_dir / f"forces_pts{points:03d}.png")
        save_residual_plot(residual_df, args.results_dir / f"residuals_pts{points:03d}.png")
        point_rows.append({"points": points, "cd": summary.cd, "cl": summary.cl})

    if point_rows:
        import pandas as pd

        point_df = pd.DataFrame(point_rows).sort_values("points")
        point_df.to_csv(args.results_dir / "point_convergence.csv", index=False)
        plot_point_convergence(point_df, args.results_dir / "point_convergence.png")

    time_rows = []
    total_time_cases = len(args.time_study_end_times)
    for index, end_time in enumerate(args.time_study_end_times, start=1):
        case_dir = args.cases_dir / "convergence_time" / f"naca{spec.digits}_pts{args.time_study_points:03d}_t{end_time:.2f}"
        print(f"[study] runtime convergence {index}/{total_time_cases}: preparing {case_dir}", flush=True)
        flow = FlowConfig(
            velocity=args.velocity,
            density=args.density,
            viscosity=args.viscosity,
            end_time=end_time,
            write_interval=args.write_interval,
        )
        prepare_case(case_dir, spec, args.time_study_points, flow)
        if args.prepare_only:
            continue
        import pandas as pd

        from openfoam_project.postprocess import (
            read_force_coeffs,
            read_residuals,
            save_force_plot,
            save_residual_plot,
            summarize_tail_force_coeffs,
        )

        print(f"[study] runtime convergence {index}/{total_time_cases}: running {case_dir}", flush=True)
        run_case(case_dir, image=args.image, runner=args.runner)
        force_df = read_force_coeffs(case_dir)
        residual_df = read_residuals(case_dir)
        summary = summarize_tail_force_coeffs(force_df)
        save_force_plot(force_df, args.results_dir / f"time_forces_{end_time:.2f}.png")
        save_residual_plot(residual_df, args.results_dir / f"time_residuals_{end_time:.2f}.png")
        time_rows.append({"end_time": end_time, "cd": summary.cd, "cl": summary.cl})

    if time_rows:
        import pandas as pd

        time_df = pd.DataFrame(time_rows).sort_values("end_time")
        time_df.to_csv(args.results_dir / "runtime_convergence.csv", index=False)
        plot_runtime_convergence(time_df, args.results_dir / "runtime_convergence.png")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as exc:
        print(f"[study] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(130)
