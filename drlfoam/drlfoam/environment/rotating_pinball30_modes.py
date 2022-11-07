
from typing import Tuple
from os import remove
from os.path import join, isfile, isdir
from glob import glob
from re import sub
from io import StringIO
from shutil import rmtree
from pandas import read_csv, DataFrame
import torch as pt
from .environment import Environment
from ..constants import TESTCASE_PATH, DEFAULT_TENSOR_TYPE
from ..utils import (check_pos_int, check_pos_float, replace_line_in_file,
                     get_time_folders, get_latest_time, replace_line_latest)


pt.set_default_tensor_type(DEFAULT_TENSOR_TYPE)


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


def _parse_forces(path: str) -> DataFrame:
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
        times_folder_forces = glob(
            join(path, "postProcessing", f"field_cylinder_{cylinder}", "*"))
        print("Loop of", cylinder, ":", times_folder_forces)
        force_path = join(times_folder_forces[0], "surfaceFieldValue.dat")
        # data_path = join(path, f"postProcessing//0/surfaceFieldValue.dat")
        data_i = parse_surface_field_values(force_path)
        if cylinder == "a":
            data["t"] = data_i.t
        data[f"cx_{cylinder}"] = data_i.cx
        data[f"cy_{cylinder}"] = data_i.cy
        data[f"cz_{cylinder}"] = data_i.cz
    return data

def _parse_probes(path: str, n_probes: int) -> DataFrame:
    with open(path, "r") as pfile:
        pdata = sub("[()]", "", pfile.read())
    names = ["t"] + [f"p{i}" for i in range(n_probes)]
    return read_csv(
        StringIO(pdata), header=None, names=names, comment="#", delim_whitespace=True
    )


def _parse_trajectory(path: str) -> DataFrame:
    names = ["t", "omega_a", "alpha_a", "beta_a", "omega_b", "alpha_b", "beta_b", "omega_c", "alpha_c", "beta_c"]
    tr = read_csv(path, sep=",", header=0, names=names)
    return tr


class RotatingPinball2D30MODES(Environment):
    def __init__(self, r1: float = 0.0, r2: float = 0.2):
        super(RotatingPinball2D30MODES, self).__init__(
            join(TESTCASE_PATH, "rotatingPinball2D/pinball_re30_modes"), "Allrun.pre",
            "Allrun", "Allclean", 2, 7, 3 # MPI_ranks, n_states, n_actions - changed to 3 
        )
        self._r1 = r1
        self._r2 = r2
        self._initialized = False
        self._start_time = 0
        self._end_time = 375
        self._control_interval = 125
        self._train = True
        self._seed = 0
        self._action_bounds = 5.0
        self._policy = "policy.pt"
        self._case = "re30"
    # def _reward(self, cx_a: pt.Tensor, cx_b: pt.Tensor, cx_c: pt.Tensor, cy_a: pt.Tensor, cy_b: pt.Tensor, cy_c: pt.Tensor) -> pt.Tensor:
    #     return self._r1 - ((((cx_a+cx_b+cx_c)/3)) + self._r2 * ((cy_a+cy_b+cy_c)/3).abs())

    def _reward(self, cx_mean: pt.Tensor, cy_mean: pt.Tensor, actions: pt.Tensor):
        
        self._r1 = pt.zeros(actions.shape[0])

        if self._case == "re30":
           #cx_mean[:] = -0.673
            self._r1[:] = 2.01175
        elif self._case == "re100":
           #cx_mean[:] = -0.5121
            self._r1[:] = 1.5368
        elif self._case == "re170":
           #cx_mean[:] = -0.4623
            self._r1[:] = 1.38945
        # Rabault + Holm implementation of reward
        #return ((cx_moving_average + cx_mean.abs()) - self._r2 * cy_moving_average.abs())
        # drlfoam implementation of reward
        return (self._r1 - (cx_mean + self._r2 * cy_mean.abs()))

    @property
    def start_time(self) -> float:
        return self._start_time

    @start_time.setter
    def start_time(self, value: float):
        check_pos_float(value, "start_time", with_zero=True)
        proc = True if self.initialized else False
        new = f"        startTime     {value};"
        replace_line_latest(self.path, "U", "startTime", new, proc)
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "timeStart",
            f"        timeStart       {value};"
        )
        self._start_time = value

    @property
    def end_time(self) -> float:
        return self._end_time

    @end_time.setter
    def end_time(self, value: float):
        check_pos_float(value, "end_time", with_zero=True)
        replace_line_in_file(
            join(self.path, "system", "controlDict"),
            "endTime",
            f"endTime         {value};"
        )
        self._end_time = value

    @property
    def control_interval(self) -> int:
        return self._control_interval

    @control_interval.setter
    def control_interval(self, value: int):
        check_pos_int(value, "control_interval")
        proc = True if self.initialized else False
        new = f"        interval        {value};"
        replace_line_latest(self.path, "U", "interval", new, proc)
        # replace_line_in_file(
        #     join(self.path, "system", "controlDict"),
        #     "executeInterval",
        #     f"        executeInterval {value};",
        # )
        # replace_line_in_file(
        #     join(self.path, "system", "controlDict"),
        #     "writeInterval",
        #     f"        writeInterval   {value};",
        # )
        self._control_interval = value

    @property
    def actions_bounds(self) -> float:
        return self._action_bounds

    @actions_bounds.setter
    def action_bounds(self, value: float):
        proc = True if self.initialized else False
        new = f"        absOmegaMax     {value:2.4f};"
        replace_line_latest(self.path, "U", "absOmegaMax", new, proc)
        self._action_bounds = value

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int):
        check_pos_int(value, "seed", with_zero=True)
        proc = True if self.initialized else False
        new = f"        seed     {value};"
        replace_line_latest(self.path, "U", "seed", new, proc)
        self._seed = value

    @property
    def policy(self) -> str:
        return self._policy

    @policy.setter
    def policy(self, value: str):
        proc = True if self.initialized else False
        new = f"        policy     {value};"
        replace_line_latest(self.path, "U", "policy", new, proc)
        self._policy = value

    @property
    def train(self) -> bool:
        return self._train

    @train.setter
    def train(self, value: bool):
        proc = True if self.initialized else False
        value_cpp = "true" if value else "false"
        new = f"        train           {value_cpp};"
        replace_line_latest(self.path, "U", "train", new, proc)
        self._train = value

    @property
    def observations(self) -> dict:
        obs = {}
        try:
            forces = _parse_forces(self.path)
            tr_path = join(self.path, "trajectory.csv")
            tr = _parse_trajectory(tr_path)
            times_folder_probes = glob(
                join(self.path, "postProcessing", "probes", "*"))
            probes_path = join(times_folder_probes[0], "p")
            probes = _parse_probes(probes_path, self._n_states)
            p_names = ["p{:d}".format(i) for i in range(self._n_states)]
            states = pt.from_numpy(probes[p_names].values)
            if states.shape[0] != 1: # only create states, actions, and rewards if at least 1 action has been sampled, otherwise throws error in update method of agent 
            
                obs["states"] = pt.from_numpy(probes[p_names].values)
                print("states shape:", pt.from_numpy(probes[p_names].values).shape)
                
                for cylinder in ("a", "b", "c"):
                    obs[f"actions_{cylinder}"] = pt.from_numpy(tr[f"omega_{cylinder}"].values)
                    obs[f"cx_{cylinder}"] = pt.from_numpy(forces[f"cx_{cylinder}"].values)
                    obs[f"cy_{cylinder}"] = pt.from_numpy(forces[f"cy_{cylinder}"].values)
                    obs[f"alpha_{cylinder}"] = pt.from_numpy(tr[f"alpha_{cylinder}"].values)
                    obs[f"beta_{cylinder}"] = pt.from_numpy(tr[f"beta_{cylinder}"].values)

                for coeff in ("cx", "cy"):
                    # sum = pt.sum(pt.stack((obs[f"{coeff}_a"],obs[f"{coeff}_b"],obs[f"{coeff}_c"])), dim=0)
                    # split = pt.split(sum,5)
                    # obs[f"{coeff}_moving_average"] = pt.FloatTensor([split[i].mean() for i in range(len(split))])
                    obs[f"{coeff}_mean"] = pt.sum(pt.stack((obs[f"{coeff}_a"],obs[f"{coeff}_b"],obs[f"{coeff}_c"])), dim=0)
                
                obs["actions"] = pt.cat((obs["actions_a"].unsqueeze(1),obs["actions_b"].unsqueeze(1),obs["actions_c"].unsqueeze(1)),1)
                print("actions shape:", obs["actions"].shape)

                obs[f"rewards"] = self._reward(obs["cx_mean"],obs["cy_mean"],obs["actions"])
                print("rewards shape:", obs[f"rewards"].shape)

                if obs["actions"].shape[0] != obs["states"].shape[0]:
                    obs = {}

        except Exception as e:
            print("Could not parse observations: ", e)
        finally:
            return obs

    def reset(self):
        files = ["log.pimpleFoam", "finished.txt", "trajectory.csv"]
        for f in files:
            f_path = join(self.path, f)
            if isfile(f_path):
                remove(f_path)
        post = join(self.path, "postProcessing")
        if isdir(post):
            rmtree(post)
        times = get_time_folders(join(self.path, "processor0"))
        times = [t for t in times if float(t) > self.start_time]
        for p in glob(join(self.path, "processor*")):
            for t in times:
                rmtree(join(p, t))
