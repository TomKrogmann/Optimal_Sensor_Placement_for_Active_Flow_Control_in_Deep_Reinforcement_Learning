#!/usr/bin/python3

from os import makedirs
from os.path import join
from io import StringIO
from re import sub
import torch as pt
from pandas import read_csv, DataFrame
from flowtorch.data import FOAMDataloader, mask_box, mask_sphere

###
### function definitions for data processing
###

def parse_surface_field_values(file_path: str) -> DataFrame:
    """Extract time and force coefficients output.

    :param file_path: path to *surfaceFieldValue* output file
    :type file_path: str
    """
    with open(file_path, "r") as ffile:
        data = sub("[()]", "", ffile.read())
        names = ("t", "cx", "cy", "cz")
        data = read_csv(StringIO(data), header=None, names=names, comment="#", delim_whitespace=True)
    return data


def process_force_coeffs(simulation_path: str, output: str):
    """Get lift and drag forces acting on the three cylinders.

    The force coefficients are stored in a `DataFrame` following
    the naming convection:
        cx_a - force in x acting on cylinder a
        cy_a - force in y acting on cylinder a
        ...

    :param simulation_path: path to OpenFOAM simulation
    :type simulation_path: str
    :param output: where to store the coefficients
    :type output: str
    """
    data = DataFrame()
    for cylinder in ("a", "b", "c"):
        path = join(simulation_path, f"postProcessing/field_cylinder_{cylinder}/0/surfaceFieldValue.dat")
        data_i = parse_surface_field_values(path)
        if cylinder == "a":
            data["t"] = data_i.t
        data[f"cx_{cylinder}"] = data_i.cx
        data[f"cy_{cylinder}"] = data_i.cy
        data[f"cz_{cylinder}"] = data_i.cz
    data.to_pickle(join(output, "coeffs.pkl"))


def process_snapshots(path: str, output: str):
    """Load and mask snapshots of p and U.

    All available snapshots of U and p are processed. Masks are
    applied to remove the three cylinders and to reduce the overall
    snapshots size. In addition to the snapshots, time and vertices
    are saved to the output folder.

    :param path: OpenFOAM simulation folder
    :type path: str
    :param output: where to store time, vertices, and snapshots
    :type output: str

    """
    loader = FOAMDataloader(path, distributed=True)
    # write times
    times = loader.write_times[1:]
    times_num = pt.tensor([float(t) for t in times])
    pt.save(times_num, join(output, "snapshot_times.pt"))
    # vertices
    vertices = loader.vertices[:, :2]
    mask = mask_box(vertices, [-2.5, -2.0], [10.0, 2.0])
    pt.save(pt.masked_select(vertices[:, 0], mask), join(output, "x.pt"))
    pt.save(pt.masked_select(vertices[:, 1], mask), join(output, "y.pt"))
    # snapshots
    n_selected = mask.sum().item()
    p = pt.zeros((n_selected, len(times)))
    U = pt.zeros((2*n_selected, len(times)))
    for ti, t in enumerate(times):
        print(f"\rProcessing time {t}", end="")
        pi, Ui = loader.load_snapshot(["p", "U"], t)
        p[:, ti] = pt.masked_select(pi, mask)
        U[:n_selected, ti] = pt.masked_select(Ui[:, 0], mask)
        U[n_selected:, ti] = pt.masked_select(Ui[:, 1], mask)
    pt.save(p, join(output, "p.pt"))
    pt.save(U, join(output, "U.pt"))
    print("\nFinished processing snapshots.")



###
### process simulation data 
###

run = "./run"
output = "./output"
cases = ("pinnball_re170", )

for case in cases:
    # case output directory
    makedirs(join(output, case), exist_ok=True)
    # force coefficients
    process_force_coeffs(join(run, case), join(output, case))
    # process snapshots
    process_snapshots(join(run, case), join(output, case))
