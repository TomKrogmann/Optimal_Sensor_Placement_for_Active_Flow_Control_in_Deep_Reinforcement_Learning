# Optimal sensor placement for active flow control with deep reinforcement learning 

## Introduction

Test.

## Dependencies

The OpenFOAM simulations require a Singularity image with OpenFOAM-v2106 named *of_2106.sif* to be located in the repository's top-level folder. To build a suitable image, run:
```
sudo singularity build of_2106.sif docker://andreweiner/of_pytorch:of2106-py1.9.0-cpu
```

## Simulations

### Pinnball base simulation

To execute the basic pinnball simulation at different characteristic Reynolds numbers, execute the `run_base_simulations` script:
```
./run_base_simulations
```
The `process_base_simulations.py` processes the force coefficients and snapshots from all base simulations and stores the processed data as pickle files under *output/case/*:
```
python3 process_base_simulations.py
```
Compute singular value decomposition of pressure and velocity snapshots:
```
python3 svd_base_simulations.py
```

## Optimal Sensor Placement

- *sensor_placement_random_forest.ipynb*: sensor placement via random forest feature importance and permutation importance
- *sensor_placement_reconstruction.ipynb*: sensor placement based on QR pivoting

To execute the notebooks, start jupyter via `jupyter-lab` and selected the notebooks in the left menu:

## 