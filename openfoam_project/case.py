from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .airfoil import NACA4Spec, generate_naca4_airfoil
from .mesher import curiosityFluidsAirfoilMesher


DEFAULT_IMAGE = "microfluidica/openfoam:13"
MACOS_DOCKER_CLI = Path("/Applications/Docker.app/Contents/Resources/bin/docker")


@dataclass(frozen=True)
class FlowConfig:
    velocity: float = 30.0
    density: float = 1.225
    viscosity: float = 1.5e-5
    end_time: float = 0.6
    write_interval: float = 0.05
    z_length: float = 1.0


def foam_banner(*, class_name: str, object_name: str, location: str | None = None, version: str = "2.0") -> str:
    location_line = f'    location    "{location}";\n' if location is not None else ""
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     | Website:  https://openfoam.org\n"
        "    \\\\  /    A nd           | Version:  13\n"
        "     \\\\/     M anipulation  |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n"
        "{\n"
        f"    version     {version};\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"{location_line}"
        f"    object      {object_name};\n"
        "}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )


def write_case_stub(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(exist_ok=True)
    (case_dir / "system").mkdir(exist_ok=True)
    (case_dir / "0").mkdir(exist_ok=True)
    (case_dir / "case.foam").write_text("", encoding="utf-8")


def create_control_dict(case_dir: Path, flow: FlowConfig) -> Path:
    area = 1.0 * flow.z_length
    content = f"""{foam_banner(class_name="dictionary", object_name="controlDict", location="system")}

application     foamRun;
solver          incompressibleFluid;

startFrom       startTime;
startTime       0;

stopAt          endTime;
endTime         {flow.end_time};

deltaT          0.0005;

writeControl    runTime;
writeInterval   {flow.write_interval};

purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    computeLiftDrag
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (airfoil);
        p               p;
        U               U;
        rho             rhoInf;
        rhoInf          {flow.density};
        CofR            (0 0 0);
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        magUInf         {flow.velocity};
        lRef            1.0;
        Aref            {area};
    }}

    residuals
    {{
        type            residuals;
        libs            ("libutilityFunctionObjects.so");
        writeControl    timeStep;
        writeInterval   1;
        fields          (U p);
    }}
}}

// ************************************************************************* //
"""
    path = case_dir / "system" / "controlDict"
    path.write_text(content, encoding="utf-8")
    return path


def create_momentum_transport_sa(case_dir: Path) -> Path:
    content = f"""{foam_banner(class_name="dictionary", object_name="momentumTransport", location="constant")}

simulationType  RAS;

RAS
{{
    model           SpalartAllmaras;
    turbulence      on;
    printCoeffs     on;
}}

// ************************************************************************* //
"""
    path = case_dir / "constant" / "momentumTransport"
    path.write_text(content, encoding="utf-8")
    return path


def generate_openfoam_fvschemes_fvsolution(case_dir: Path, steady_state: bool = True) -> None:
    ddt_scheme = "steadyState" if steady_state else "Euler"
    n_outer_correctors = 1 if steady_state else 2
    relaxation_factors = """relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        nuTilda         0.7;
    }
}""" if steady_state else ""

    fv_schemes = f"""{foam_banner(class_name="dictionary", object_name="fvSchemes")}

ddtSchemes
{{
    default         {ddt_scheme};
}}

gradSchemes
{{
    default         Gauss linear;
    grad(U)         Gauss linear;
}}

divSchemes
{{
    default         none;
    div(phi,U)      Gauss linearUpwind grad(U);
    div(phi,nuTilda) Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}

wallDist
{{
    method          meshWave;
}}
"""
    fv_solution = f"""{foam_banner(class_name="dictionary", object_name="fvSolution")}

solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.01;
        smoother        GaussSeidel;
    }}

    pFinal
    {{
        $p;
        tolerance       1e-06;
        relTol          0;
    }}

    "(U|nuTilda)"
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-05;
        relTol          0.1;
    }}

    "(U|nuTilda)Final"
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-05;
        relTol          0;
    }}
}}

PIMPLE
{{
    nOuterCorrectors {n_outer_correctors};
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
}}

{relaxation_factors}
"""
    (case_dir / "system" / "fvSchemes").write_text(fv_schemes, encoding="utf-8")
    (case_dir / "system" / "fvSolution").write_text(fv_solution, encoding="utf-8")


def generate_physical_properties(case_dir: Path, nu_value: float) -> Path:
    content = f"""{foam_banner(class_name="dictionary", object_name="physicalProperties", location="constant")}

viscosityModel  constant;
nu              [0 2 -1 0 0 0 0] {nu_value};
"""
    path = case_dir / "constant" / "physicalProperties"
    path.write_text(content, encoding="utf-8")
    return path


def generate_0_directory(case_dir: Path, flow: FlowConfig) -> None:
    nu_tilda = 3.0 * flow.viscosity
    cv1 = 7.1
    chi = nu_tilda / flow.viscosity
    fv1 = (chi**3) / (chi**3 + cv1**3)
    nut = nu_tilda * fv1

    fields = {
        "U": f"""{foam_banner(class_name="volVectorField", object_name="U", location="0")}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({flow.velocity} 0 0);
boundaryField
{{
    farfield {{ type freestream; freestreamValue uniform ({flow.velocity} 0 0); }}
    airfoil {{ type noSlip; }}
    frontAndBack {{ type empty; }}
}}
""",
        "p": f"""{foam_banner(class_name="volScalarField", object_name="p", location="0")}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    farfield {{ type freestreamPressure; freestreamValue uniform 0; }}
    airfoil {{ type zeroGradient; }}
    frontAndBack {{ type empty; }}
}}
""",
        "nuTilda": f"""{foam_banner(class_name="volScalarField", object_name="nuTilda", location="0")}
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform {nu_tilda};
boundaryField
{{
    farfield {{ type freestream; freestreamValue uniform {nu_tilda}; }}
    airfoil {{ type fixedValue; value uniform 0; }}
    frontAndBack {{ type empty; }}
}}
""",
        "nut": f"""{foam_banner(class_name="volScalarField", object_name="nut", location="0")}
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform {nut};
boundaryField
{{
    farfield {{ type calculated; value uniform {nut}; }}
    airfoil {{ type nutUSpaldingWallFunction; value uniform 0; }}
    frontAndBack {{ type empty; }}
}}
""",
    }
    for name, content in fields.items():
        (case_dir / "0" / name).write_text(content, encoding="utf-8")


def setup_case_from_example(
    case_dir: Path,
    spec: NACA4Spec,
    points: int,
    flow: FlowConfig,
    *,
    chord: float = 1.0,
    overwrite: bool = True,
) -> np.ndarray:
    """Replicate the notebook example workflow and return the surface points."""
    if overwrite and case_dir.exists():
        shutil.rmtree(case_dir)

    write_case_stub(case_dir)
    airfoil_points = generate_naca4_airfoil(spec, chord=chord, points=points)
    curiosityFluidsAirfoilMesher(airfoil_points, case_dir / "system" / "blockMeshDict")
    create_control_dict(case_dir, flow)
    create_momentum_transport_sa(case_dir)
    generate_openfoam_fvschemes_fvsolution(case_dir)
    generate_physical_properties(case_dir, flow.viscosity)
    generate_0_directory(case_dir, flow)
    return airfoil_points


def prepare_case(case_dir: Path, spec: NACA4Spec, points: int, flow: FlowConfig, *, overwrite: bool = True) -> Path:
    setup_case_from_example(case_dir, spec, points, flow, overwrite=overwrite)
    return case_dir


def _docker_command(project_root: Path, image: str, *args: str) -> list[str]:
    docker_cli = _find_docker_cli()
    return [docker_cli, "run", "--rm", "-v", f"{project_root}:/work", "-w", "/work", image, *args]


def _find_project_root(case_dir: Path) -> Path:
    resolved = case_dir.resolve()
    for parent in [resolved, *resolved.parents]:
        if (parent / "pyproject.toml").exists() or (parent / "README.md").exists():
            return parent
    raise RuntimeError(f"could not determine project root for case directory: {case_dir}")


def _find_docker_cli() -> str:
    docker_cli = shutil.which("docker")
    if docker_cli is not None:
        return docker_cli
    if MACOS_DOCKER_CLI.exists():
        return str(MACOS_DOCKER_CLI)
    raise RuntimeError("docker is not installed; use --runner local on a machine with OpenFOAM commands available.")


def _ensure_docker_ready() -> None:
    docker_cli = _find_docker_cli()

    probe = subprocess.run(
        [docker_cli, "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "docker is installed but the daemon is not running. Start Docker Desktop and wait for `docker info` to succeed."
        )


def _run_check_mesh(case_dir: Path, command: Iterable[str], *, image: str, runner: str, project_root: Path) -> None:
    command_list = list(command)
    print(f"[openfoam] {case_dir.name}: {' '.join(command_list)}", flush=True)
    if runner == "docker":
        completed = subprocess.run(
            _docker_command(project_root, image, *command_list),
            check=True,
            text=True,
            capture_output=True,
        )
    else:
        if shutil.which(command_list[0]) is None:
            raise RuntimeError(f"required OpenFOAM command not found: {command_list[0]}")
        completed = subprocess.run(
            command_list,
            check=True,
            cwd=project_root,
            text=True,
            capture_output=True,
        )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if "Mesh OK." not in completed.stdout:
        raise RuntimeError(f"checkMesh reported an invalid mesh for {case_dir}")


def run_openfoam_commands(case_dir: Path, commands: Iterable[Iterable[str]], *, image: str = DEFAULT_IMAGE, runner: str = "docker") -> None:
    project_root = _find_project_root(case_dir)
    if runner == "docker":
        _ensure_docker_ready()
        for command in commands:
            try:
                if next(iter(command)) == "checkMesh":
                    _run_check_mesh(case_dir, command, image=image, runner=runner, project_root=project_root)
                    continue
                print(f"[openfoam] {case_dir.name}: {' '.join(command)}", flush=True)
                subprocess.run(_docker_command(project_root, image, *command), check=True)
            except KeyboardInterrupt as exc:
                raise KeyboardInterrupt(f"interrupted while running {' '.join(command)} for {case_dir}") from exc
        return

    for command in commands:
        try:
            if next(iter(command)) == "checkMesh":
                _run_check_mesh(case_dir, command, image=image, runner=runner, project_root=project_root)
                continue
            if shutil.which(next(iter(command))) is None:
                raise RuntimeError(f"required OpenFOAM command not found: {next(iter(command))}")
            print(f"[openfoam] {case_dir.name}: {' '.join(command)}", flush=True)
            subprocess.run(list(command), check=True, cwd=project_root)
        except KeyboardInterrupt as exc:
            raise KeyboardInterrupt(f"interrupted while running {' '.join(command)} for {case_dir}") from exc


def run_case(case_dir: Path, *, image: str = DEFAULT_IMAGE, runner: str = "docker") -> None:
    project_root = _find_project_root(case_dir)
    case_name = case_dir.resolve().relative_to(project_root).as_posix()
    if runner == "docker":
        commands = [
            ("blockMesh", "-case", case_name),
            ("checkMesh", "-case", case_name),
            ("foamRun", "-case", case_name),
        ]
    else:
        commands = [
            ("blockMesh", "-case", str(case_dir)),
            ("checkMesh", "-case", str(case_dir)),
            ("foamRun", "-case", str(case_dir)),
        ]
    run_openfoam_commands(case_dir, commands, image=image, runner=runner)
