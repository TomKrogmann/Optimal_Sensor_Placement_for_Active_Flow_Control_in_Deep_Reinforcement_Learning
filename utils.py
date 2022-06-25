from cmath import sqrt
from random import Random
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from stl import mesh
import torch as pt
import numpy as np
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis import SVD
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import qmc
from scipy import spatial
from typing import Dict, Tuple
import os

def load_input_data(dataset: str, output: str) -> Tuple[pt.Tensor, np.array, np.array]:
    """
    Within the Flowtorch package, load data from OpenFOAM Case and apply mask to specified flow quantity.

    Parameters
    ----------
    dataset : str
        Name of OpenFOAM case in Datasets folder.
    output : str
        Path to save figures
        
    Returns
    -------
    data_matrix : pt.Tensor
        Tensor holding values of the specified flow quantity.
    x : np.array
        array with corresponding x coordinates of the specified flow quantity
    y : np.array
        array with corresponding y coordinates of the specified flow quantity
    """
    path = DATASETS[dataset]
    loader = FOAMDataloader(path)
    times = loader.write_times
    fields = loader.field_names
    print(f"Number of available snapshots: {len(times)}")
    print("First five write times: ", times[:5])
    print(f"Fields available at t={times[-1]}: ", fields[times[-1]])

    # load vertices and discard z-coordinate
    vertices = loader.vertices[:, :2]
    mask = mask_box(vertices, lower=[-2.5, -2], upper=[10, 2])
    # mask = mask_box(vertices, lower=[-2.5, -4], upper=[10, 4])

    print(f"Selected vertices: {mask.sum().item()}/{mask.shape[0]}")

    every = 1  # use only every 4th vertex
    fig, ax = plt.subplots()
    d = 0.5
    ax.scatter(
        vertices[::every, 0] / d, vertices[::every, 1] / d, s=0.5, c=mask[::every]
    )
    ax.set_aspect("equal", "box")
    # ax.set_xlim(0.0, 2.2/d)
    # ax.set_ylim(0.0, 0.41/d)
    ax.set_xlabel(r"$x/d$")
    ax.set_ylabel(r"$y/d$")
    plt.savefig(f"{output}/cylinder_mask.png", bbox_inches="tight")
    plt.close()

    window_times = [t for t in times if 200 <= float(t) <= 300.0]
    # print(f"Window times: {window_times}")

    n_points = mask.sum().item()
    data_matrix = pt.zeros((n_points, len(window_times)))
    for i, t in enumerate(window_times):
        data_matrix[:, i] = pt.masked_select(loader.load_snapshot("p", t), mask)

    x = (pt.masked_select(vertices[:, 0], mask) / d).numpy()
    y = (pt.masked_select(vertices[:, 1], mask) / d).numpy()

    return data_matrix, x, y


def lh_sampling(n_samples: int, x: float, y: float) -> np.array:
    """
    2D Latin Hypercube Sampling.

    Parameters
    ----------
    n_samples : int
        Number of points to sample.
    x : float
        X coordinates to sample from.
    y : float
        Y coordinates to sample from.

    Returns
    -------
    samples : np.array
        Array holding sampled x and y values.
    """
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n_samples)
    lower_bounds = [np.amin(x), np.amin(y)]
    upper_bounds = [np.amax(x), np.amax(y)]
    samples = np.array((qmc.scale(sample, lower_bounds, upper_bounds)))
    return samples


def point_within_circle(
    circle_x: float, circle_y: float, rad: float, x: float, y: float
) -> bool:
    """
    Compute distance to circle center to verify if a sampled point lies on the cylinder's circle area.

    Parameters
    ----------
    circle_x : float
        Number of samples in the dataset.
    circle_x : float
        Number of samples in the dataset.
    rad : float
        Number of samples in the dataset.
    x : float
        Number of samples in the dataset.
    y : float
        Number of samples in the dataset.

    Returns
    -------
    _ : bool
        Flag to check if point lies within circle area.
    """
    if (x - circle_x) * (x - circle_x) + (y - circle_y) * (y - circle_y) <= rad * rad:
        return True
    else:
        return False

def load_target_data(file_path: str, time_start: int, time_end: int, time_step: float) -> Tuple[np.array, np.array]:
    """
    Load target data for the random forrest regressor (lift- and drag coefficient) in chosen time interval from .csv file of an OpenFOAM Case  

    Parameters
    ----------
    file_path : str
        Name of OpenFOAM case in Datasets folder.
    time_start: int
        Start value of considered time interval.
    time_end: int
        End value of considered time interval.
    time_step: float
        Chosen time step of considered time interval.

    Returns
    -------
    time : np.array
        Array holding chosen time interval.
    coeffs : np.array
        Array with lift- and drag coeffients in chosen time interval
    """
    names_coeffs = ["col{:d}".format(i) for i in range(13)]
    names_coeffs[0] = "Time"
    names_coeffs[1] = "Cd"
    names_coeffs[3] = "Cl"
    keep = ["Time", "Cd", "Cl"]

    read_data = pd.read_csv(file_path, sep="\t", names=names_coeffs, usecols=keep, comment="#", iterator=True, chunksize=6000)
    df = pd.concat([chunk[(chunk['Time'] >= time_start) & (chunk['Time'] <= time_end)] for chunk in read_data]).to_numpy()
    
    time_interval = np.arange(time_start, time_end + time_step, time_step) # create array for considered time interval for plotting
       
    Cl, Cd = [], []
    for i, _ in enumerate(time_interval):
        idx = list(df[:,0]).index(time_interval[i])
        Cl.append(df[idx,2])
        Cd.append(df[idx,1])

    coeffs = np.column_stack((np.asarray(Cl),np.asarray(Cd))) # stack Cl and Cd into coeffs array

    return time_interval, coeffs

class MinMaxScalerLecture(object):
        """Class to scale/re-scale data to the range [-1, 1] and back."""
        def __init__(self):
            self.min = None
            self.max = None
            self.trained = False

        def fit(self, data):
            self.min = np.amin(data)
            self.max = np.amax(data)
            self.trained = True

        def scale(self, data):
            assert self.trained
            #assert len(data.shape) == 2
            data_norm = (data - self.min) / (self.max - self.min)
            return 2.0*data_norm - 1.0

        def rescale(self, data_norm):
            assert self.trained
            #assert len(data_norm.shape) == 2
            data = (data_norm + 1.0) * 0.5
            return data * (self.max - self.min) + self.min

def get_n_percent_highest_values(n_percent: int, values: np.array, points: np.array, store_values: np.array, store_points: np.array) -> Tuple[np.array, np.array]:
    """
    Function to extract n_percent highest values and coordinates of Permutation Importance and Feature Importance in every iteration.

    Parameters
    ----------
    n_percent : int
        Percent highest values to extract.
    values : np.array
        Array with values to extract from.
    points: np.array
        Array with corresponding coordinates to values.
    store_values: np.array
        Array to append highest values in every iteration
    store_points: np.array
        Array to append coordinates of highest values in every iteration

    Returns
    -------
    store_values : np.array
        Array holding highest values from all iterations.
    store_points : np.array
        Array holding corresponding coordinates from all iterations 
    """
    keep = round((n_percent/100)*len((values)))
    idx_highest = np.argpartition(values, -keep)[-keep:]
    print(idx_highest)

    store_values = np.hstack([store_values,values[idx_highest]])
    #values_highest = np.hstack([store_values,values[idx_highest]])
    print(store_values)

    store_points = np.append(store_points, points[idx_highest,:], axis=0)
    #coords_highest = np.append(store_points, points[idx_highest,:], axis=0)
    print(store_points)

    #return values_highest, coords_highest
    return store_values, store_points

def initialize_centroids_improved(k: int, data: pt.Tensor) -> pt.Tensor:
    """Randomly select data points as initial centroids.
    """
    n_points = data.shape[0]
    probs = pt.ones(n_points) / n_points
    rows = pt.zeros(k, dtype=pt.int64)
    rows[0] = pt.multinomial(probs, 1)
    distance = pt.zeros((n_points, k-1))
    for i in range(1, k):
        distance[:, i-1] = pt.linalg.norm(data-data[rows[i-1]], dim=1).square()
        min_dist = distance[:, :i].min(dim=1).values
        probs = min_dist / min_dist.sum()
        rows[i] = pt.multinomial(probs, 1)
    return data[rows]

def find_nearest_centroid(centroids: pt.Tensor, data: pt.Tensor) -> pt.Tensor:
    """Find the id of the nearest centroid for each data point.
    """
    n_points = data.shape[0]
    n_centroids = centroids.shape[0]
    labels = pt.zeros(n_points, dtype=pt.int64)
    distance = pt.zeros((n_points, n_centroids))
    for i in range(n_centroids):
        distance[:, i] = pt.linalg.norm(data - centroids[i], dim=1)
    return pt.argmin(distance, dim=1)

def update_centroids(centroids: pt.Tensor, data: pt.Tensor) -> pt.Tensor:
    """Update centroid position based on cluster mean value.
    """
    n_centroids = centroids.shape[0]
    new_centroids = pt.zeros_like(centroids)
    cluster_ids = find_nearest_centroid(centroids, data)
    for i in range(n_centroids):
        new_centroids[i] = data[cluster_ids == i].mean(dim=0)
    return new_centroids

def find_centroids(k: int, data: pt.Tensor, max_iter: int=100,
                   tol: float=1.0e-6, verbose=False):
    centroids = initialize_centroids_improved(k, data)
    for i in range(max_iter):
        old_centroids = centroids[:]
        centroids = update_centroids(centroids, data)
        mean_diff = pt.linalg.norm(centroids-old_centroids, dim=1).mean()
        if mean_diff < tol:
            if verbose:
                print(f"Clustering converged after {i+1} iterations.")
            break
    return centroids, i+1

def k_means_clustering(repeat: int, points: np.array, n_sensors: int) -> np.array:
    """
    K_means ++ clustering   

    Parameters
    ----------
    repeat: int
        Number of clustering repetitions.
    points: np.array
        Array holding coordinates to use for clustering.
    n_sensors: int
        Desired number of centroids (sensors).

    Returns
    -------
    sensors : pt.Tensor
        Tensor holding coordinates of clustered centroids.
    """
    data = pt.from_numpy(points)

    sensors = initialize_centroids_improved(n_sensors, data)

    for _ in range(repeat):
        sensors, _ = find_centroids(n_sensors, data)

    return sensors.numpy()