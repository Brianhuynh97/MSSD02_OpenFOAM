from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ForceSummary:
    cd: float
    cl: float


@dataclass(frozen=True)
class ResidualSummary:
    ux: float
    uy: float
    p: float


def read_force_coeffs(case_dir: Path) -> pd.DataFrame:
    path = case_dir / "postProcessing" / "computeLiftDrag" / "0" / "forceCoeffs.dat"
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    df.columns = ["time", "cm", "cd", "cl", "clf", "clr", "cdf", "cdr"][: len(df.columns)]
    return df


def read_residuals(case_dir: Path) -> pd.DataFrame:
    path = case_dir / "postProcessing" / "residuals" / "0" / "residuals.dat"
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    df.columns = ["time", "Ux", "Uy", "p"][: len(df.columns)]
    return df


def summarize_tail_force_coeffs(force_df: pd.DataFrame, tail_fraction: float = 0.1) -> ForceSummary:
    tail_count = max(1, int(len(force_df) * tail_fraction))
    tail = force_df.tail(tail_count)
    return ForceSummary(cd=float(tail["cd"].mean()), cl=float(tail["cl"].mean()))


def summarize_final_residuals(residual_df: pd.DataFrame) -> ResidualSummary:
    last_row = residual_df.tail(1).iloc[0]
    return ResidualSummary(ux=float(last_row["Ux"]), uy=float(last_row["Uy"]), p=float(last_row["p"]))


def save_force_plot(force_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(force_df["time"], force_df["cd"], label="Cd", color="red", linewidth=2)
    plt.plot(force_df["time"], force_df["cl"], label="Cl", color="blue", linewidth=2)
    plt.title("Aerodynamic Force Coefficients Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Coefficient Value")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_residual_plot(residual_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(residual_df["time"], residual_df["p"], label="Pressure", color="black", alpha=0.8)
    plt.plot(residual_df["time"], residual_df["Ux"], label="Velocity Ux", color="blue", alpha=0.8)
    plt.plot(residual_df["time"], residual_df["Uy"], label="Velocity Uy", color="green", alpha=0.8)
    plt.yscale("log")
    plt.title("Solver Residuals Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Residual")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_airfoil_plot(airfoil_points: np.ndarray, output_path: Path, *, title: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(airfoil_points[:, 0], airfoil_points[:, 1], "o-", linewidth=1.5, markersize=3)
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
