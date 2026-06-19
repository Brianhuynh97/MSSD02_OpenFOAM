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
    parser = argparse.ArgumentParser(description="Run the camber sweep for NACA airfoils.")
    parser.add_argument("--camber-values", nargs="+", type=int, default=list(range(0, 9)))
    parser.add_argument("--camber-position", type=int, default=4)
    parser.add_argument("--thickness", type=int, default=12)
    parser.add_argument("--surface-points", type=int, default=32)
    parser.add_argument("--end-time", type=float, default=0.8)
    parser.add_argument("--velocity", type=float, default=30.0)
    parser.add_argument("--density", type=float, default=1.225)
    parser.add_argument("--viscosity", type=float, default=1.5e-5)
    parser.add_argument("--write-interval", type=float, default=0.05)
    parser.add_argument("--cases-dir", type=Path, default=Path("cases"))
    parser.add_argument("--results-dir", type=Path, default=Path("results") / "camber")
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


def main() -> None:
    args = parse_args()

    from openfoam_project.airfoil import NACA4Spec
    from openfoam_project.case import FlowConfig, run_case, setup_case_from_example

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    total_cases = len(args.camber_values)
    for index, camber in enumerate(args.camber_values, start=1):
        spec = NACA4Spec(camber, args.camber_position, args.thickness)
        case_dir = args.cases_dir / "camber" / f"naca{spec.digits}_pts{args.surface_points:03d}"
        print(f"[study] camber sweep {index}/{total_cases}: preparing {case_dir}", flush=True)
        flow = FlowConfig(
            velocity=args.velocity,
            density=args.density,
            viscosity=args.viscosity,
            end_time=args.end_time,
            write_interval=args.write_interval,
        )
        setup_case_from_example(case_dir, spec, args.surface_points, flow)
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

        print(f"[study] camber sweep {index}/{total_cases}: running {case_dir}", flush=True)
        try:
            run_case(case_dir, image=args.image, runner=args.runner)
            force_df = read_force_coeffs(case_dir)
            residual_df = read_residuals(case_dir)
            summary = summarize_tail_force_coeffs(force_df)
            save_force_plot(force_df, args.results_dir / f"forces_naca{spec.digits}.png")
            save_residual_plot(residual_df, args.results_dir / f"residuals_naca{spec.digits}.png")
            rows.append({"camber": camber, "airfoil": spec.digits, "cd": summary.cd, "cl": summary.cl})
        except Exception as exc:
            print(f"[study] camber sweep {index}/{total_cases}: failed for {case_dir}: {exc}", file=sys.stderr, flush=True)
            failures.append({"camber": camber, "airfoil": spec.digits, "case_dir": str(case_dir), "error": str(exc)})

    if rows:
        import pandas as pd

        df = pd.DataFrame(rows).sort_values("camber")
        df.to_csv(args.results_dir / "camber_sweep.csv", index=False)
        plot_camber_sweep(df, args.results_dir / "camber_sweep.png")
    if failures:
        import pandas as pd

        pd.DataFrame(failures).to_csv(args.results_dir / "camber_failures.csv", index=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as exc:
        print(f"[study] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(130)
