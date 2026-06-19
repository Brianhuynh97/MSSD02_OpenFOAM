from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def curiosityFluidsAirfoilMesher(airfoil_points: np.ndarray, output_path: Path | str = "openfoam_naca/system/blockMeshDict") -> Path:
    """Compatibility wrapper mirroring the example's mesher call shape."""
    return write_block_mesh_dict(airfoil_points, Path(output_path))


def write_block_mesh_dict(airfoil_points: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chord_length = 1
    domain_height = 20
    wake_length = 20
    first_layer_height = 0.001
    growth_rate = 1.05
    max_cell_size = 0.5

    bl_height = 0.1
    leading_edge_grading = 2
    trailing_edge_grading = 0.8
    inlet_grading_factor = 0.5
    trailing_block_angle = 5

    nbl = int(np.rint(math.log(1 - (bl_height / first_layer_height * (1 - growth_rate))) / math.log(growth_rate)))
    max_layer_thickness = first_layer_height * growth_rate**nbl
    bl_grading = max_layer_thickness / first_layer_height
    lff = (domain_height / 2 - bl_height)
    nff = int(
        np.rint(
            math.log(max_cell_size / max_layer_thickness)
            / math.log(1 - max_layer_thickness / lff + max_cell_size / lff)
        )
    )
    ff_grading = max_cell_size / max_layer_thickness
    nffa = int(np.rint(3.14159 / 2 * domain_height / 2 / max_cell_size))

    x = airfoil_points[:, 0]
    y = airfoil_points[:, 1]
    numpoint = x.size
    x_top = np.array([])
    y_top = np.array([])
    x_bottom = np.array([])
    y_bottom = np.array([])
    top_count = 0
    bottom_count = 0
    for i in range(numpoint):
        if x[i] - x[i - 1] < 0 and i != 0 and i != numpoint - 1:
            x_top = np.append(x_top, x[i])
            y_top = np.append(y_top, y[i])
            top_count += 1
        elif x[i] - x[i - 1] > 0 and i != 0 and i != numpoint - 1:
            x_bottom = np.append(x_bottom, x[i])
            y_bottom = np.append(y_bottom, y[i])
            bottom_count += 1

    swx = x_bottom[x_bottom < 0.25]
    swy = y_bottom[x_bottom < 0.25]
    lsw = sum(math.sqrt((swx[i + 1] - swx[i]) ** 2 + (swy[i + 1] - swy[i]) ** 2) for i in range(len(swx) - 1))
    rsw = leading_edge_grading ** (1 / nffa)
    dx2sw = leading_edge_grading * lsw * ((1 - rsw**nffa) / (1 - rsw)) ** (-1) if leading_edge_grading != 1 else lsw / nffa

    nwx = x_top[x_top < 0.25]
    nwy = y_top[x_top < 0.25]
    lnw = sum(math.sqrt((nwx[i + 1] - nwx[i]) ** 2 + (nwy[i + 1] - nwy[i]) ** 2) for i in range(len(nwx) - 1))
    rnw = leading_edge_grading ** (1 / nffa)
    dx2 = leading_edge_grading * lnw * ((1 - rnw**nffa) / (1 - rnw)) ** (-1) if leading_edge_grading != 1 else lnw / nffa

    nex = x_top[x_top > 0.25]
    ney = y_top[x_top > 0.25]
    lne = sum(math.sqrt((nex[i + 1] - nex[i]) ** 2 + (ney[i + 1] - ney[i]) ** 2) for i in range(len(nex) - 1))
    nne = int(np.rint(math.log(trailing_edge_grading) / math.log(1 - dx2 * (1 / lne - trailing_edge_grading / lne))))

    sex = x_bottom[x_bottom > 0.25]
    sey = y_bottom[x_bottom > 0.25]
    lse = sum(math.sqrt((sex[i + 1] - sex[i]) ** 2 + (sey[i + 1] - sey[i]) ** 2) for i in range(len(sex) - 1))
    nse = int(np.rint(math.log(trailing_edge_grading) / math.log(1 - dx2sw * (1 / lse - trailing_edge_grading / lse))))
    dxset = trailing_edge_grading * dx2sw

    nwake = int(np.rint(math.log(max_cell_size / dxset) / math.log(1 - dxset / wake_length + max_cell_size / wake_length)))
    wake_grading = max_cell_size / dxset

    nx_top = np.zeros(top_count)
    ny_top = np.zeros(top_count)
    for i in range(top_count):
        if i == 0:
            denom = ((x_top[i + 1] - x_top[i]) ** 2 + (y_top[i + 1] - y_top[i]) ** 2) ** 0.5
            ny_top[i] = -(x_top[i + 1] - x_top[i]) / denom
            nx_top[i] = (y_top[i + 1] - y_top[i]) / denom
        elif i == top_count - 1:
            denom = ((x_top[i] - x_top[i - 1]) ** 2 + (y_top[i] - y_top[i - 1]) ** 2) ** 0.5
            ny_top[i] = -(x_top[i] - x_top[i - 1]) / denom
            nx_top[i] = (y_top[i] - y_top[i - 1]) / denom
        else:
            denom = ((x_top[i + 1] - x_top[i - 1]) ** 2 + (y_top[i + 1] - y_top[i - 1]) ** 2) ** 0.5
            ny_top[i] = -(x_top[i + 1] - x_top[i - 1]) / denom
            nx_top[i] = (y_top[i + 1] - y_top[i - 1]) / denom

    nx_bottom = np.zeros(bottom_count)
    ny_bottom = np.zeros(bottom_count)
    for i in range(bottom_count):
        if i == 0:
            denom = ((x_bottom[i + 1] - x_bottom[i]) ** 2 + (y_bottom[i + 1] - y_bottom[i]) ** 2) ** 0.5
            ny_bottom[i] = -(x_bottom[i + 1] - x_bottom[i]) / denom
            nx_bottom[i] = (y_bottom[i + 1] - y_bottom[i]) / denom
        elif i == bottom_count - 1:
            denom = ((x_bottom[i] - x_bottom[i - 1]) ** 2 + (y_bottom[i] - y_bottom[i - 1]) ** 2) ** 0.5
            ny_bottom[i] = -(x_bottom[i] - x_bottom[i - 1]) / denom
            nx_bottom[i] = (y_bottom[i] - y_bottom[i - 1]) / denom
        else:
            denom = ((x_bottom[i + 1] - x_bottom[i - 1]) ** 2 + (y_bottom[i + 1] - y_bottom[i - 1]) ** 2) ** 0.5
            ny_bottom[i] = -(x_bottom[i + 1] - x_bottom[i - 1]) / denom
            nx_bottom[i] = (y_bottom[i + 1] - y_bottom[i - 1]) / denom

    xt = (x[1] + x[numpoint - 2]) / 2
    yt = (y[1] + y[numpoint - 2]) / 2
    nxt = -(0 - yt) / ((1 - xt) ** 2 + (0 - yt) ** 2) ** 0.5
    nyt = (1 - xt) / ((1 - xt) ** 2 + (0 - yt) ** 2) ** 0.5
    thetawake = math.atan(nxt / nyt)

    inlet_grading = leading_edge_grading * domain_height / inlet_grading_factor
    rinlet_grading = inlet_grading ** (1 / nffa)
    dx_inlet_grading = (3.14159 * domain_height / 4) * ((1 - rinlet_grading**nffa) / (1 - rinlet_grading)) ** (-1)
    ltop = 1 + math.tan(trailing_block_angle * 3.14159 / 180) * domain_height / 2

    px = np.zeros(19)
    py = np.zeros(19)
    px[0], py[0] = 0, 0
    px[1], py[1] = 1 + wake_length, -domain_height / 2
    px[2], py[2] = 1 + math.tan(trailing_block_angle * 3.14159 / 180) * domain_height / 2, -domain_height / 2
    px[3], py[3] = 0, -domain_height / 2
    px[4], py[4] = -domain_height / 2, 0
    px[5], py[5] = 0, domain_height / 2
    px[6], py[6] = 1 + math.tan(trailing_block_angle * 3.14159 / 180) * domain_height / 2, domain_height / 2
    px[7], py[7] = 1 + wake_length, domain_height / 2
    px[9], py[9] = 1 + wake_length, 0 - math.tan(thetawake) * wake_length
    px[8], py[8] = 1 + wake_length, py[9] + bl_height
    px[10], py[10] = 1 + wake_length, py[9] - bl_height
    px[12], py[12] = 1, 0
    px[11] = 1 + bl_height * (y_top[1] - y_top[0]) / ((x_top[1] - x_top[0]) ** 2 + (y_top[1] - y_top[0]) ** 2) ** 0.5
    py[11] = 0 - bl_height * (x_top[1] - x_top[0]) / ((x_top[1] - x_top[0]) ** 2 + (y_top[1] - y_top[0]) ** 2) ** 0.5
    px[13] = 1 + bl_height * (y_bottom[bottom_count - 1] - y_bottom[bottom_count - 2]) / (
        (x_bottom[bottom_count - 1] - x_bottom[bottom_count - 2]) ** 2
        + (y_bottom[bottom_count - 1] - y_bottom[bottom_count - 2]) ** 2
    ) ** 0.5
    py[13] = 0 - bl_height * (x_bottom[bottom_count - 1] - x_bottom[bottom_count - 2]) / (
        (x_bottom[bottom_count - 1] - x_bottom[bottom_count - 2]) ** 2
        + (y_bottom[bottom_count - 1] - y_bottom[bottom_count - 2]) ** 2
    ) ** 0.5
    px[15], py[15] = 0.25, np.interp(0.25, x_top[::-1], y_top[::-1])
    ny15 = 0.001 / ((0.001) ** 2 + (np.interp(0.251, x_top[::-1], y_top[::-1]) - np.interp(0.25, x_top[::-1], y_top[::-1])) ** 2) ** 0.5
    nx15 = -(np.interp(0.251, x_top[::-1], y_top[::-1]) - np.interp(0.25, x_top[::-1], y_top[::-1])) / (
        (0.001) ** 2 + (np.interp(0.251, x_top[::-1], y_top[::-1]) - np.interp(0.25, x_top[::-1], y_top[::-1])) ** 2
    ) ** 0.5
    px[14], py[14] = 0.25 + nx15 * bl_height, py[15] + ny15 * bl_height
    px[16], py[16] = 0.25, np.interp(0.25, x_bottom, y_bottom)
    px[17], py[17] = 0.25 - nx15 * bl_height, py[16] - ny15 * bl_height
    px[18], py[18] = -bl_height, 0

    cpx = np.array([-domain_height / 2 * math.cos(3.14159 / 4)] * 2)
    cpy = np.array([domain_height / 2 * math.sin(3.14159 / 4), -domain_height / 2 * math.sin(3.14159 / 4)])

    nwcpx = np.array([nwx[i] + np.interp(nwx[i], x_top[::-1], nx_top[::-1]) * bl_height for i in range(len(nwx))])
    nwcpy = np.array([nwy[i] + np.interp(nwx[i], x_top[::-1], ny_top[::-1]) * bl_height for i in range(len(nwx))])
    lnwbl = sum(((nwcpx[i + 1] - nwcpx[i]) ** 2 + (nwcpy[i + 1] - nwcpy[i]) ** 2) ** 0.5 for i in range(len(nwx) - 1))

    swcpx = np.array([swx[i] + np.interp(swx[i], x_bottom, nx_bottom) * bl_height for i in range(len(swx))])
    swcpy = np.array([swy[i] + np.interp(swx[i], x_bottom, ny_bottom) * bl_height for i in range(len(swx))])
    lswbl = sum(((swcpx[i + 1] - swcpx[i]) ** 2 + (swcpy[i + 1] - swcpy[i]) ** 2) ** 0.5 for i in range(len(swx) - 1))

    necpx = np.array([nex[i] + np.interp(nex[i], x_top[::-1], nx_top[::-1]) * bl_height for i in range(len(nex))])
    necpy = np.array([ney[i] + np.interp(nex[i], x_top[::-1], ny_top[::-1]) * bl_height for i in range(len(nex))])
    lnebl = sum(((necpx[i + 1] - necpx[i]) ** 2 + (necpy[i + 1] - necpy[i]) ** 2) ** 0.5 for i in range(len(nex) - 1))
    dxnebl = lnebl * ((1 - rnw**nffa) / (1 - rnw)) ** (-1) if leading_edge_grading != 1 else lnebl / nffa

    secpx = np.array([sex[i] + np.interp(sex[i], x_bottom, nx_bottom) * bl_height for i in range(len(sex))])
    secpy = np.array([sey[i] + np.interp(sex[i], x_bottom, ny_bottom) * bl_height for i in range(len(sex))])
    lsebl = sum(((secpx[i + 1] - secpx[i]) ** 2 + (secpy[i + 1] - secpy[i]) ** 2) ** 0.5 for i in range(len(sex) - 1))
    dxsebl = lswbl * ((1 - rsw**nffa) / (1 - rsw)) ** (-1) if leading_edge_grading != 1 else lswbl / nffa

    def solve_ratio(start_dx: float, count: int, length: float, *, label: str) -> float:
        err = 100.0
        ratio = 1.5
        dr = 0.00001
        iterations = 0
        max_iterations = 10000
        tolerance = 1e-7
        while err > tolerance:
            old = ratio
            f_val = start_dx * (1 - ratio**count) / (1 - ratio) - length
            ru = ratio + dr
            rl = ratio - dr
            fu = start_dx * (1 - ru**count) / (1 - ru) - length
            fl = start_dx * (1 - rl**count) / (1 - rl) - length
            deriv = (fu - fl) / (2 * dr)
            if not math.isfinite(deriv) or abs(deriv) < 1e-14:
                raise RuntimeError(f"mesh grading solve failed for {label}: derivative became singular")
            ratio = old - f_val / deriv
            if not math.isfinite(ratio):
                raise RuntimeError(f"mesh grading solve failed for {label}: Newton iteration diverged")
            err = abs(ratio - old)
            iterations += 1
            if iterations >= max_iterations:
                raise RuntimeError(
                    f"mesh grading solve did not converge for {label} after {max_iterations} iterations"
                )
        return ratio

    top_grading = solve_ratio(dx_inlet_grading, nne, ltop, label="top_grading") ** nne
    bottom_grading = top_grading
    small_cell_top = dx_inlet_grading * top_grading
    ltop_wake = wake_length - math.tan(thetawake) * domain_height / 2
    top_wake_grading = solve_ratio(small_cell_top, nwake, ltop_wake, label="top_wake_grading") ** nwake
    nebl_edge_grading = solve_ratio(dxnebl, nne, lnebl, label="nebl_edge_grading") ** nne
    sebl_edge_grading = solve_ratio(dxsebl, nse, lsebl, label="sebl_edge_grading") ** nse

    l14_5 = math.sqrt((px[14] - px[5]) ** 2 + (py[14] - py[5]) ** 2)
    grading14_5 = solve_ratio(max_layer_thickness, nff, l14_5, label="grading14_5") ** nff
    l17_3 = math.sqrt((px[17] - px[3]) ** 2 + (py[17] - py[3]) ** 2)
    grading17_3 = solve_ratio(max_layer_thickness, nff, l17_3, label="grading17_3") ** nff

    with output_path.open("w", encoding="utf-8") as f:
        f.write("/*--------------------------------*- C++ -*----------------------------------*\\\n")
        f.write("  =========                 |\n")
        f.write("  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n")
        f.write("   \\\\    /   O peration     | Website:  https://openfoam.org\n")
        f.write("    \\\\  /    A nd           | Version:  13\n")
        f.write("     \\\\/     M anipulation  |\n")
        f.write("\\*---------------------------------------------------------------------------*/\n")
        f.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    object      blockMeshDict;\n}\n")
        f.write("// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")
        f.write(f"convertToMeters {chord_length};\n\nvertices\n(\n")
        for i in range(19):
            f.write(f"    ({px[i]} {py[i]} 0)\n")
        for i in range(19):
            f.write(f"    ({px[i]} {py[i]} 1)\n")
        f.write(");\n\nblocks\n(\n")
        f.write(
            f"    hex (2 1 10 13 21 20 29 32) ({nwake} {nff} 1) simpleGrading "
            f"({top_wake_grading} {wake_grading} {wake_grading} {top_wake_grading} {1/ff_grading} {1/ff_grading} {1/ff_grading} {1/ff_grading} 1 1 1 1)\n"
        )
        f.write(
            f"    hex (3 2 13 17 22 21 32 36) ({nse} {nff} 1) simpleGrading "
            f"({bottom_grading} {sebl_edge_grading} {sebl_edge_grading} {bottom_grading} {1/grading17_3} {1/ff_grading} {1/ff_grading} {1/grading17_3} 1 1 1 1)\n"
        )
        f.write(
            f"    hex (4 3 17 18 23 22 36 37) ({nffa} {nff} 1) edgeGrading "
            f"({1/inlet_grading} {1/leading_edge_grading} {1/leading_edge_grading} {1/inlet_grading} {1/ff_grading} {1/grading17_3} {1/grading17_3} {1/ff_grading} 1 1 1 1)\n"
        )
        f.write(
            f"    hex (4 18 14 5 23 37 33 24) ({nff} {nffa} 1) edgeGrading "
            f"({1/ff_grading} {1/grading14_5} {1/grading14_5} {1/ff_grading} {1/inlet_grading} {1/leading_edge_grading} {1/leading_edge_grading} {1/inlet_grading} 1 1 1 1)\n"
        )
        f.write(
            f"    hex (14 11 6 5 33 30 25 24) ({nne} {nff} 1) edgeGrading "
            f"({nebl_edge_grading} {top_grading} {top_grading} {nebl_edge_grading} {grading14_5} {ff_grading} {ff_grading} {grading14_5} 1 1 1 1)\n"
        )
        f.write(
            f"    hex (11 8 7 6 30 27 26 25) ({nwake} {nff} 1) simpleGrading "
            f"({wake_grading} {top_wake_grading} {top_wake_grading} {wake_grading} {ff_grading} {ff_grading} {ff_grading} {ff_grading} 1 1 1 1)\n"
        )
        f.write(f"    hex (12 9 8 11 31 28 27 30) ({nwake} {nbl} 1) simpleGrading ({wake_grading} {bl_grading} 1)\n")
        f.write(
            f"    hex (15 12 11 14 34 31 30 33) ({nne} {nbl} 1) edgeGrading "
            f"({trailing_edge_grading} {nebl_edge_grading} {nebl_edge_grading} {trailing_edge_grading} {bl_grading} {bl_grading} {bl_grading} {bl_grading} 1 1 1 1)\n"
        )
        f.write(
            f"    hex (0 15 14 18 19 34 33 37) ({nffa} {nbl} 1) edgeGrading "
            f"({leading_edge_grading} {1/leading_edge_grading} {1/leading_edge_grading} {leading_edge_grading} {bl_grading} {bl_grading} {bl_grading} {bl_grading} 1 1 1 1)\n"
        )
        f.write(
            f"    hex (18 17 16 0 37 36 35 19) ({nffa} {nbl} 1) edgeGrading "
            f"({1/leading_edge_grading} {leading_edge_grading} {leading_edge_grading} {1/leading_edge_grading} {1/bl_grading} {1/bl_grading} {1/bl_grading} {1/bl_grading} 1 1 1 1)\n"
        )
        f.write(
            f"    hex (17 13 12 16 36 32 31 35) ({nse} {nbl} 1) edgeGrading "
            f"({sebl_edge_grading} {trailing_edge_grading} {trailing_edge_grading} {sebl_edge_grading} {1/bl_grading} {1/bl_grading} {1/bl_grading} {1/bl_grading} 1 1 1 1)\n"
        )
        f.write(f"    hex (13 10 9 12 32 29 28 31) ({nwake} {nbl} 1) simpleGrading ({wake_grading} {1/bl_grading} 1)\n")
        f.write(");\n\nedges\n(\n")

        def write_polyline(name: str, start: int, end: int, xs: np.ndarray, ys: np.ndarray, z: int) -> None:
            f.write(f"    {name} {start} {end}\n    (\n")
            for i in range(len(xs)):
                f.write(f"        ({xs[i]} {ys[i]} {z})\n")
            f.write("    )\n")

        write_polyline("polyLine", 11, 14, necpx, necpy, 0)
        write_polyline("polyLine", 30, 33, necpx, necpy, 1)
        write_polyline("polyLine", 14, 18, nwcpx, nwcpy, 0)
        write_polyline("polyLine", 33, 37, nwcpx, nwcpy, 1)
        write_polyline("polyLine", 18, 17, swcpx, swcpy, 0)
        write_polyline("polyLine", 37, 36, swcpx, swcpy, 1)
        write_polyline("polyLine", 17, 13, secpx, secpy, 0)
        write_polyline("polyLine", 36, 32, secpx, secpy, 1)
        write_polyline("spline", 12, 15, nex, ney, 0)
        write_polyline("spline", 31, 34, nex, ney, 1)
        f.write(f"    arc 4 5 ({cpx[0]} {cpy[0]} 0)\n")
        f.write(f"    arc 3 4 ({cpx[1]} {cpy[1]} 0)\n")
        f.write(f"    arc 23 24 ({cpx[0]} {cpy[0]} 1)\n")
        f.write(f"    arc 22 23 ({cpx[1]} {cpy[1]} 1)\n")
        write_polyline("spline", 15, 0, nwx, nwy, 0)
        write_polyline("spline", 34, 19, nwx, nwy, 1)
        write_polyline("spline", 0, 16, swx, swy, 0)
        write_polyline("spline", 19, 35, swx, swy, 1)
        write_polyline("spline", 16, 12, sex, sey, 0)
        write_polyline("spline", 35, 31, sex, sey, 1)
        f.write(");\n\n")
        f.write(
            "boundary\n(\n"
            "    frontAndBack\n    {\n        type empty;\n        faces\n        (\n"
            "          (1 2 13 10)\n          (2 3 17 13)\n          (3 4 18 17)\n          (18 4 5 14)\n"
            "          (11 14 5 6)\n          (8 11 6 7)\n          (9 12 11 8)\n          (12 15 14 11)\n"
            "          (0 18 14 15)\n          (17 18 0 16)\n          (13 17 16 12)\n          (10 13 12 9)\n"
            "          (21 20 29 32)\n          (22 21 32 36)\n          (23 22 36 37)\n          (23 37 33 24)\n"
            "          (33 30 25 24)\n          (30 27 26 25)\n          (31 28 27 30)\n          (34 31 30 33)\n"
            "          (37 19 34 33)\n          (37 36 35 19)\n          (36 32 31 35)\n          (32 29 28 31)\n"
            "         );\n     }\n"
            "    farfield\n    {\n        type patch;\n        faces\n        (\n"
            "          (4 23 24 5)\n          (5 24 25 6)\n          (6 25 26 7)\n          (27 8 7 26)\n"
            "          (28 9 8 27)\n          (29 10 9 28)\n          (20 1 10 29)\n          (21 2 1 20)\n"
            "          (22 3 2 21)\n          (23 4 3 22)\n         );\n     }\n"
            "    airfoil\n    {\n        type wall;\n        faces\n        (\n"
            "          (34 15 12 31)\n          (19 0 15 34)\n          (0 19 35 16)\n          (16 35 31 12)\n"
            "         );\n     }\n);\n"
        )

    return output_path
