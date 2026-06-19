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
    parser = argparse.ArgumentParser(description="Run the surface-point and runtime convergence study.")
    parser.add_argument("--camber", type=int, default=4)
    parser.add_argument("--camber-position", type=int, default=4)
    parser.add_argument("--thickness", type=int, default=12)
    parser.add_argument("--surface-points", nargs="+", type=int, default=[12, 16, 20, 24, 32, 48, 64])
    parser.add_argument("--time-study-end-times", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument(
        "--time-study-points",
        type=int,
        default=None,
        help="Surface points for runtime study. Defaults to the recommended point count from the first study.",
    )
    parser.add_argument("--velocity", type=float, default=30.0)
    parser.add_argument("--density", type=float, default=1.225)
    parser.add_argument("--viscosity", type=float, default=1.5e-5)
    parser.add_argument("--write-interval", type=float, default=0.05)
    parser.add_argument("--cases-dir", type=Path, default=Path("cases"))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "convergence")
    parser.add_argument("--runner", choices=["docker", "local"], default="docker")
    parser.add_argument("--image", default="microfluidica/openfoam:13")
    parser.add_argument("--points-cd-tol", type=float, default=5e-4)
    parser.add_argument("--points-cl-tol", type=float, default=5e-3)
    parser.add_argument("--runtime-cd-tol", type=float, default=5e-4)
    parser.add_argument("--runtime-cl-tol", type=float, default=5e-3)
    parser.add_argument("--runtime-residual-tol", type=float, default=1e-2)
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


def build_point_recommendation(
    df: pd.DataFrame,
    *,
    cd_tol: float,
    cl_tol: float,
) -> tuple[int, pd.DataFrame]:
    reference = df.sort_values("points").iloc[-1]
    enriched = df.copy()
    enriched["delta_cd_to_finest"] = (enriched["cd"] - reference["cd"]).abs()
    enriched["delta_cl_to_finest"] = (enriched["cl"] - reference["cl"]).abs()
    eligible = enriched[
        (enriched["delta_cd_to_finest"] <= cd_tol)
        & (enriched["delta_cl_to_finest"] <= cl_tol)
    ].sort_values("points")
    recommended_points = int(eligible.iloc[0]["points"]) if not eligible.empty else int(reference["points"])
    return recommended_points, enriched.sort_values("points")


def build_runtime_recommendation(
    df: pd.DataFrame,
    *,
    cd_tol: float,
    cl_tol: float,
    residual_tol: float,
) -> tuple[float, pd.DataFrame]:
    enriched = df.sort_values("end_time").reset_index(drop=True).copy()
    enriched["delta_cd_from_previous"] = enriched["cd"].diff().abs()
    enriched["delta_cl_from_previous"] = enriched["cl"].diff().abs()
    eligible = enriched[
        enriched["delta_cd_from_previous"].le(cd_tol)
        & enriched["delta_cl_from_previous"].le(cl_tol)
        & enriched["ux_residual"].le(residual_tol)
        & enriched["uy_residual"].le(residual_tol)
        & enriched["p_residual"].le(residual_tol)
    ]
    recommended_end_time = float(eligible.iloc[0]["end_time"]) if not eligible.empty else float(enriched.iloc[-1]["end_time"])
    return recommended_end_time, enriched


def write_summary(
    output_path: Path,
    *,
    spec: "NACA4Spec",
    recommended_points: int | None,
    runtime_study_points: int,
    recommended_end_time: float | None,
    point_failures: list[dict[str, object]],
    time_failures: list[dict[str, object]],
) -> None:
    lines = [
        "# Convergence Study Summary",
        "",
        f"- Airfoil: `NACA {spec.digits}`",
    ]
    if recommended_points is not None:
        lines.append(f"- Recommended surface discretization: `{recommended_points}` points")
    lines.append(f"- Surface discretization used for runtime study: `{runtime_study_points}` points")
    if recommended_end_time is not None:
        lines.append(f"- Recommended simulation end time: `{recommended_end_time:.2f}` s")
    if point_failures:
        lines.extend(["", "## Point-study failures", ""])
        lines.extend(f"- `{entry['case_dir']}`: {entry['error']}" for entry in point_failures)
    if time_failures:
        lines.extend(["", "## Runtime-study failures", ""])
        lines.extend(f"- `{entry['case_dir']}`: {entry['error']}" for entry in time_failures)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    from openfoam_project.airfoil import NACA4Spec
    from openfoam_project.case import FlowConfig, run_case, setup_case_from_example

    spec = NACA4Spec(args.camber, args.camber_position, args.thickness)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    point_rows = []
    point_failures = []
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
        setup_case_from_example(case_dir, spec, points, flow)
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
        try:
            run_case(case_dir, image=args.image, runner=args.runner)
            force_df = read_force_coeffs(case_dir)
            residual_df = read_residuals(case_dir)
            summary = summarize_tail_force_coeffs(force_df)
            save_force_plot(force_df, args.results_dir / f"forces_pts{points:03d}.png")
            save_residual_plot(residual_df, args.results_dir / f"residuals_pts{points:03d}.png")
            point_rows.append({"points": points, "cd": summary.cd, "cl": summary.cl})
        except Exception as exc:
            print(f"[study] point convergence {index}/{total_point_cases}: failed for {case_dir}: {exc}", file=sys.stderr, flush=True)
            point_failures.append({"points": points, "case_dir": str(case_dir), "error": str(exc)})

    recommended_points = None
    if point_rows:
        import pandas as pd

        point_df = pd.DataFrame(point_rows).sort_values("points")
        recommended_points, enriched_point_df = build_point_recommendation(
            point_df,
            cd_tol=args.points_cd_tol,
            cl_tol=args.points_cl_tol,
        )
        enriched_point_df.to_csv(args.results_dir / "point_convergence.csv", index=False)
        plot_point_convergence(enriched_point_df, args.results_dir / "point_convergence.png")
    if point_failures:
        import pandas as pd

        pd.DataFrame(point_failures).to_csv(args.results_dir / "point_convergence_failures.csv", index=False)

    if args.time_study_points is not None:
        time_study_points = args.time_study_points
    elif recommended_points is not None:
        time_study_points = recommended_points
    else:
        time_study_points = max(args.surface_points)

    time_rows = []
    time_failures = []
    total_time_cases = len(args.time_study_end_times)
    for index, end_time in enumerate(args.time_study_end_times, start=1):
        case_dir = args.cases_dir / "convergence_time" / f"naca{spec.digits}_pts{time_study_points:03d}_t{end_time:.2f}"
        print(f"[study] runtime convergence {index}/{total_time_cases}: preparing {case_dir}", flush=True)
        flow = FlowConfig(
            velocity=args.velocity,
            density=args.density,
            viscosity=args.viscosity,
            end_time=end_time,
            write_interval=args.write_interval,
        )
        setup_case_from_example(case_dir, spec, time_study_points, flow)
        if args.prepare_only:
            continue
        import pandas as pd

        from openfoam_project.postprocess import (
            read_force_coeffs,
            read_residuals,
            save_force_plot,
            save_residual_plot,
            summarize_final_residuals,
            summarize_tail_force_coeffs,
        )

        print(f"[study] runtime convergence {index}/{total_time_cases}: running {case_dir}", flush=True)
        try:
            run_case(case_dir, image=args.image, runner=args.runner)
            force_df = read_force_coeffs(case_dir)
            residual_df = read_residuals(case_dir)
            summary = summarize_tail_force_coeffs(force_df)
            residual_summary = summarize_final_residuals(residual_df)
            save_force_plot(force_df, args.results_dir / f"time_forces_{end_time:.2f}.png")
            save_residual_plot(residual_df, args.results_dir / f"time_residuals_{end_time:.2f}.png")
            time_rows.append(
                {
                    "end_time": end_time,
                    "cd": summary.cd,
                    "cl": summary.cl,
                    "ux_residual": residual_summary.ux,
                    "uy_residual": residual_summary.uy,
                    "p_residual": residual_summary.p,
                }
            )
        except Exception as exc:
            print(f"[study] runtime convergence {index}/{total_time_cases}: failed for {case_dir}: {exc}", file=sys.stderr, flush=True)
            time_failures.append({"end_time": end_time, "case_dir": str(case_dir), "error": str(exc)})

    recommended_end_time = None
    if time_rows:
        import pandas as pd

        time_df = pd.DataFrame(time_rows).sort_values("end_time")
        recommended_end_time, enriched_time_df = build_runtime_recommendation(
            time_df,
            cd_tol=args.runtime_cd_tol,
            cl_tol=args.runtime_cl_tol,
            residual_tol=args.runtime_residual_tol,
        )
        enriched_time_df.to_csv(args.results_dir / "runtime_convergence.csv", index=False)
        plot_runtime_convergence(enriched_time_df, args.results_dir / "runtime_convergence.png")
    if time_failures:
        import pandas as pd

        pd.DataFrame(time_failures).to_csv(args.results_dir / "runtime_convergence_failures.csv", index=False)

    if not args.prepare_only:
        write_summary(
            args.results_dir / "convergence_summary.md",
            spec=spec,
            recommended_points=recommended_points,
            runtime_study_points=time_study_points,
            recommended_end_time=recommended_end_time,
            point_failures=point_failures,
            time_failures=time_failures,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as exc:
        print(f"[study] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(130)
