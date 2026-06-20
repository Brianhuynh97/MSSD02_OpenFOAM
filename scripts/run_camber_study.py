#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the camber sweep for NACA airfoils.")
    parser.add_argument("--camber-values", nargs="+", type=int, default=list(range(0, 9)))
    parser.add_argument("--camber-position", type=int, default=4)
    parser.add_argument("--thickness", type=int, default=12)
    parser.add_argument("--surface-points", type=int, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--velocity", type=float, default=30.0)
    parser.add_argument("--density", type=float, default=1.225)
    parser.add_argument("--viscosity", type=float, default=1.5e-5)
    parser.add_argument("--write-interval", type=float, default=0.05)
    parser.add_argument("--cases-dir", type=Path, default=Path("cases"))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "camber")
    parser.add_argument(
        "--convergence-results-dir",
        type=Path,
        default=Path("results") / "convergence",
        help="Directory containing recommended_settings.json from the convergence study.",
    )
    parser.add_argument(
        "--retry-surface-points",
        nargs="*",
        type=int,
        default=[32, 40, 48, 64, 80, 96],
        help="Additional surface-point counts to try if checkMesh fails for a camber case.",
    )
    parser.add_argument("--runner", choices=["docker", "local"], default="docker")
    parser.add_argument("--image", default="microfluidica/openfoam:13")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def plot_camber_sweep(df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(df["camber"], df["cd"], marker="o", label="Cd")
    plt.plot(df["camber"], df["cl"], marker="o", label="Cl")
    plt.xlabel('Camber digit "M"')
    plt.ylabel("Tail-averaged coefficient")
    plt.title("Lift and drag vs. camber")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def load_recommended_settings(results_dir: Path) -> dict[str, object]:
    path = results_dir / "recommended_settings.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing convergence recommendations: {path}. Run scripts/run_point_convergence.py first, "
            "or pass --surface-points and --end-time explicitly."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_study_settings(args: argparse.Namespace) -> tuple[int, float]:
    surface_points = args.surface_points
    end_time = args.end_time
    if surface_points is not None and end_time is not None:
        return surface_points, end_time

    settings = load_recommended_settings(args.convergence_results_dir)
    if surface_points is None:
        value = settings.get("recommended_points")
        if value is None:
            raise RuntimeError("convergence recommendations do not contain recommended_points")
        surface_points = int(value)
    if end_time is None:
        value = settings.get("recommended_end_time")
        if value is None:
            raise RuntimeError("convergence recommendations do not contain recommended_end_time")
        end_time = float(value)
    return surface_points, end_time


def build_retry_schedule(base_surface_points: int, retry_surface_points: list[int]) -> list[int]:
    schedule: list[int] = []
    for points in [base_surface_points, *retry_surface_points]:
        if points >= base_surface_points and points not in schedule:
            schedule.append(points)
    return schedule


def write_summary(
    output_path: Path,
    *,
    requested_surface_points: int,
    end_time: float,
    rows: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    lines = [
        "# Camber Study Summary",
        "",
        f"- Requested baseline surface discretization: `{requested_surface_points}` points",
        f"- Simulation end time: `{end_time:.2f}` s",
        f"- Successful camber cases: `{len(rows)}`",
    ]
    retried_rows = [row for row in rows if int(row["surface_points"]) != requested_surface_points]
    if retried_rows:
        lines.extend(["", "## Cases requiring additional surface points", ""])
        lines.extend(
            f"- `NACA {row['airfoil']}`: used `{row['surface_points']}` points instead of `{requested_surface_points}`"
            for row in retried_rows
        )
    if failures:
        lines.extend(["", "## Failed camber cases", ""])
        lines.extend(f"- `NACA {row['airfoil']}`: {row['error']}" for row in failures)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    from openfoam_project.airfoil import NACA4Spec
    from openfoam_project.case import FlowConfig, run_case, setup_case_from_example

    args.results_dir.mkdir(parents=True, exist_ok=True)
    requested_surface_points, end_time = resolve_study_settings(args)
    retry_schedule = build_retry_schedule(requested_surface_points, args.retry_surface_points)
    rows = []
    failures = []
    attempt_log = []
    total_cases = len(args.camber_values)
    for index, camber in enumerate(args.camber_values, start=1):
        spec = NACA4Spec(camber, args.camber_position, args.thickness)
        flow = FlowConfig(
            velocity=args.velocity,
            density=args.density,
            viscosity=args.viscosity,
            end_time=end_time,
            write_interval=args.write_interval,
        )

        if args.prepare_only:
            case_dir = args.cases_dir / "camber" / f"naca{spec.digits}_pts{requested_surface_points:03d}"
            print(f"[study] camber sweep {index}/{total_cases}: preparing {case_dir}", flush=True)
            setup_case_from_example(case_dir, spec, requested_surface_points, flow)
            continue

        import pandas as pd

        from openfoam_project.postprocess import (
            read_force_coeffs,
            read_residuals,
            save_force_plot,
            save_residual_plot,
            summarize_tail_force_coeffs,
        )

        case_completed = False
        last_error: Exception | None = None
        for attempt_index, surface_points in enumerate(retry_schedule, start=1):
            case_dir = args.cases_dir / "camber" / f"naca{spec.digits}_pts{surface_points:03d}"
            print(
                f"[study] camber sweep {index}/{total_cases}: preparing {case_dir} "
                f"(attempt {attempt_index}/{len(retry_schedule)})",
                flush=True,
            )
            setup_case_from_example(case_dir, spec, surface_points, flow)

            print(
                f"[study] camber sweep {index}/{total_cases}: running {case_dir} "
                f"(attempt {attempt_index}/{len(retry_schedule)})",
                flush=True,
            )
            try:
                run_case(case_dir, image=args.image, runner=args.runner)
                force_df = read_force_coeffs(case_dir)
                residual_df = read_residuals(case_dir)
                summary = summarize_tail_force_coeffs(force_df)
                save_force_plot(force_df, args.results_dir / f"forces_naca{spec.digits}.png")
                save_residual_plot(residual_df, args.results_dir / f"residuals_naca{spec.digits}.png")
                rows.append(
                    {
                        "camber": camber,
                        "airfoil": spec.digits,
                        "surface_points": surface_points,
                        "cd": summary.cd,
                        "cl": summary.cl,
                    }
                )
                attempt_log.append(
                    {
                        "camber": camber,
                        "airfoil": spec.digits,
                        "surface_points": surface_points,
                        "status": "success",
                        "case_dir": str(case_dir),
                        "error": "",
                    }
                )
                case_completed = True
                break
            except Exception as exc:
                last_error = exc
                error_text = str(exc)
                print(
                    f"[study] camber sweep {index}/{total_cases}: failed for {case_dir}: {error_text}",
                    file=sys.stderr,
                    flush=True,
                )
                attempt_log.append(
                    {
                        "camber": camber,
                        "airfoil": spec.digits,
                        "surface_points": surface_points,
                        "status": "failed",
                        "case_dir": str(case_dir),
                        "error": error_text,
                    }
                )
                if "invalid mesh" not in error_text:
                    break

        if not case_completed:
            failures.append(
                {
                    "camber": camber,
                    "airfoil": spec.digits,
                    "case_dir": str(case_dir),
                    "error": str(last_error) if last_error is not None else "case failed",
                }
            )

    if rows:
        import pandas as pd

        df = pd.DataFrame(rows).sort_values("camber")
        df.to_csv(args.results_dir / "camber_sweep.csv", index=False)
        plot_camber_sweep(df, args.results_dir / "camber_sweep.png")
    if attempt_log:
        import pandas as pd

        pd.DataFrame(attempt_log).to_csv(args.results_dir / "camber_attempt_log.csv", index=False)
    failure_path = args.results_dir / "camber_failures.csv"
    if failures:
        import pandas as pd

        pd.DataFrame(failures).to_csv(failure_path, index=False)
    elif failure_path.exists():
        failure_path.unlink()
    write_summary(
        args.results_dir / "camber_summary.md",
        requested_surface_points=requested_surface_points,
        end_time=end_time,
        rows=rows,
        failures=failures,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as exc:
        print(f"[study] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(130)
