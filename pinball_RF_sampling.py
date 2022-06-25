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
from utils import *

# increase plot resolution
plt.rcParams["figure.dpi"] = 160

output = "/home/tom/flowtorch/flowtorch/output/test/verify_MinMaxScaler_lecture_feature_globalMinMaxScaling_PI_test_data"

# load data from OpenFOAM Case
data_matrix, x, y = load_input_data("RE_170_1000s", output)

# specifiy number of iterations for feature selection with RF regressor
iterations = 100

# allocate arrays to append Permutation Importance as well as Feature Importance values and coordinates in every loop
store_values_highest_PI = np.empty(0)
store_coords_highest_PI = np.empty((0, 2))

store_values_highest_FI = np.empty(0)
store_coords_highest_FI = np.empty((0, 2))

for iteration_loop in range(iterations):

    # Latin Hypercube sampling
    samples = lh_sampling(n_samples=100, x=x, y=y)

    circles = {
        "circle_x": [-1.3, 0, 0],
        "circle_y": [0, 0.75, -0.75],
        "radius": [0.5, 0.5, 0.5],
    }

    # Delte sampled points on circle areas
    delete = []
    for i, _ in enumerate(circles):
        circle_x, circle_y, radius = [item[i] for item in circles.values()]
        for j, _ in enumerate(samples):
            if point_within_circle(
                circle_x, circle_y, radius, samples[j, 0], samples[j, 1]
            ):
                delete.append(j)
            else:
                pass
        samples = np.delete(samples, delete, axis=0)
        delete.clear()

    # Assign samples to nearest x and y values on the mesh
    coords_mesh = np.column_stack((x, y))
    indices = np.argmin(
        spatial.distance.cdist(coords_mesh, samples, "sqeuclidean"), axis=0
    )
    points = np.zeros((len(indices), 2))

    for i, values in enumerate(indices):
        points[i] = coords_mesh[values]

    # Generate input data for random forrest regressor by extracting corresponding flow quantity values from the mask
    data = np.zeros((len(indices), np.shape(data_matrix)[1]))

    for i, values in enumerate(indices):
        data[i] = data_matrix[values]

    # load target data for random forrest regressor
    time_interval, coeffs = load_target_data(
        file_path="/home/tom/flowtorch/flowtorch/datasets/RE_170_1000s/postProcessing/forces/0/coefficient.dat",
        time_start=200,
        time_end=300,
        time_step=0.5,
    )

    # set-up RF parameters
    model_params = {
        "n_jobs": -1,
        "random_state": 0,
        "bootstrap": True,
        "n_estimators": 100,
        "max_depth": 7,
        "max_features": "auto",
        "max_samples": 0.6,
        "min_samples_leaf": 2,
        "min_samples_split": 2,
    }

    # instanciate object of RandomForrestRegressor
    model = RandomForestRegressor(**model_params)

    # split data into 2/3 train and 1/3 test

    idx_train_test = int((2 / 3) * len(time_interval))
    input_train = data.transpose(1, 0)[:idx_train_test]
    input_test = data.transpose(1, 0)[idx_train_test:]

    target_train = coeffs[:idx_train_test]
    target_test = coeffs[idx_train_test:]

    ## MinMaxScaling Ml-CFD

    scaler_X_train = MinMaxScalerLecture()
    scaler_X_train.fit(input_train)
    X_train_data_norm = scaler_X_train.scale(input_train)
    X_test_data_norm = scaler_X_train.scale(input_test)

    scaler_y_train_cl = MinMaxScalerLecture()
    scaler_y_train_cl.fit(target_train[:, 0])
    cl_train_data_norm = scaler_y_train_cl.scale(target_train[:, 0])
    cl_test_data_norm = scaler_y_train_cl.scale(target_test[:, 0])

    scaler_y_train_cd = MinMaxScalerLecture()
    scaler_y_train_cd.fit(target_train[:, 1])
    cd_train_data_norm = scaler_y_train_cd.scale(target_train[:, 1])
    cd_test_data_norm = scaler_y_train_cd.scale(target_test[:, 1])

    y_train_data_norm = np.column_stack((cl_train_data_norm, cd_train_data_norm))
    y_test_data_norm = np.column_stack((cl_test_data_norm, cd_test_data_norm))

    # fit data to the model
    forest = model.fit(X_train_data_norm, y_train_data_norm)
    prediction_train = model.predict(X_train_data_norm)

    # prediction with test data
    prediction_test = model.predict(X_test_data_norm)

    # Compute permutation importance
    result_PI = permutation_importance(
        forest, input_test, target_test, n_repeats=10, n_jobs=-1, random_state=0
    )

    # Compute feature importance
    result_FI = forest.feature_importances_

    # Extract highest n_percent values of Permutation Importance and Feature Importance and corresponding coordinates in every iteration
    store_values_highest_PI, store_coords_highest_PI = get_n_percent_highest_values(
        n_percent=2,
        values=np.asarray(result_PI.importances_mean),
        points=points,
        store_values=store_values_highest_PI,
        store_points=store_coords_highest_PI,
    )
    store_values_highest_FI, store_coords_highest_FI = get_n_percent_highest_values(
        n_percent=2,
        values=result_FI,
        points=points,
        store_values=store_values_highest_FI,
        store_points=store_coords_highest_FI,
    )

    print(f"Finished iteration {iteration_loop}")

# Validate sensor placement with K-means++ Clustering by prediction

min_sensors = 15
max_sensors = 21
store_mse = np.empty((0, 3))

for i in range(min_sensors, max_sensors):
    sensors_FI_clustering = k_means_clustering(
        repeat=100, points=store_coords_highest_FI, n_sensors=i
    )
    sensors_PI_clustering = k_means_clustering(
        repeat=100, points=store_coords_highest_PI, n_sensors=i
    )
    # fig, ax = plt.subplots()

    distance, indices = spatial.KDTree(coords_mesh).query(sensors_PI_clustering)

    coords_sensors_PI = np.zeros((len(indices), 2))
    values_sensors_PI = np.zeros((len(indices), np.shape(data_matrix)[1]))

    for i, values in enumerate(indices):
        coords_sensors_PI[i] = coords_mesh[values]
        values_sensors_PI[i] = data_matrix[values]

    np.save(
        f"/home/tom/flowtorch/flowtorch/output/test/verify_MinMaxScaler_lecture_feature_globalMinMaxScaling_PI_test_data/coords_{i}_sensors_PI.npy",
        coords_sensors_PI,
    )

    model = RandomForestRegressor(**model_params)

    input_train = values_sensors_PI.transpose(1, 0)[:idx_train_test]
    input_test = values_sensors_PI.transpose(1, 0)[idx_train_test:]

    scaler_X_train = MinMaxScalerLecture()
    scaler_X_train.fit(input_train)
    X_train_data_norm = scaler_X_train.scale(input_train)
    X_test_data_norm = scaler_X_train.scale(input_test)

    forest = model.fit(X_train_data_norm, y_train_data_norm)
    prediction_test = model.predict(X_test_data_norm)

    rescale_prediction_cl_test = scaler_y_train_cl.rescale(prediction_test[:, 0])
    rescale_prediction_cd_test = scaler_y_train_cd.rescale(prediction_test[:, 1])

    mse_cl = mean_squared_error(target_test[:, 0], rescale_prediction_cl_test)
    mse_cd = mean_squared_error(target_test[:, 1], rescale_prediction_cd_test)

    mse_store = np.vstack(np.array([i, mse_cl, mse_cd]))

np.save(
    f"/home/tom/flowtorch/flowtorch/output/test/verify_MinMaxScaler_lecture_feature_globalMinMaxScaling_PI_test_data/mse_prediction.npy",
    mse_store,
)
