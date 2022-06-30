import matplotlib.pyplot as plt
import numpy as np

output = "/home/tom/flowtorch/flowtorch/output/test/Re_170_check_seed_2"


# load data
pointcloud_PI = np.load(f"{output}/coords_highest_PI.npy")
pointcloud_values_PI = np.load(f"{output}/values_highest_PI.npy")
target_test = np.load(f"{output}/targets_test.npy")
time_interval_test = np.load(f"{output}/time_interval_test.npy")
mse_predictions = np.load(f"{output}/mse_prediction.npy")

for i,_ in enumerate(mse_predictions[:,0]): 
    initial_coords = np.load(f"{output}/initial_coords_{i+1}_sensors_PI.npy")
    final_coords = np.load(f"{output}/final_coords_{i+1}_sensors_PI.npy")
    prediction = np.load(f"{output}/prediction_{i+1}_sensors_PI.npy")

    plt.figure(1)
    circle_1_plot = plt.Circle((-1.3, 0), 0.5, color='k')
    circle_2_plot = plt.Circle((0, 0.75), 0.5, color='k')
    circle_3_plot = plt.Circle((0, -0.75), 0.5, color='k')

    fig, ax = plt.subplots()
    plt.scatter(pointcloud_PI[:,0],pointcloud_PI[:,1],c=pointcloud_values_PI)
    cb = plt.colorbar()
    cb.set_label('Permutation Importance')
    #plt.scatter(initial_coords[:, 0], initial_coords[:, 1], marker="*", edgecolors="C3", s=200, facecolors="none", label="initial centroids")
    plt.scatter(final_coords[:, 0], final_coords[:, 1], c="r", marker="*", s=200, label="Final Sensors")
    #for j in range(initial_coords.shape[0]):
        #plt.annotate("", xy=final_coords[j], xytext=initial_coords[j], arrowprops=dict(arrowstyle="->"))
    ax.add_patch(circle_1_plot)
    ax.add_patch(circle_2_plot)
    ax.add_patch(circle_3_plot)
    plt.xlabel(r"$x/d$")
    plt.ylabel(r"$y/d$")
    plt.legend()
    plt.savefig(f"{output}/Pointcloud_with_{i+1}_sensors.png", bbox_inches="tight")
    plt.close()

    plt.figure(2)
    plt.plot(time_interval_test, prediction[:,0], label='prediction') 
    plt.plot(time_interval_test, target_test[:,0], label='test_data') 
    plt.xlabel("time[s]")
    plt.ylabel(r"$c_l$")
    plt.legend()
    plt.savefig(f"{output}/RF_Cl_prediction_{i+1}_sensor_placement.png", bbox_inches="tight")
    plt.close()

    plt.figure(3)
    plt.plot(time_interval_test, prediction[:,1], label='prediction') 
    plt.plot(time_interval_test, target_test[:,1], label='test_data') 
    plt.xlabel("time[s]")
    plt.ylabel(r"$c_d$")
    #plt.ylim(0,1.5)
    plt.legend()
    plt.savefig(f"{output}/RF_Cd_prediction_sensor{i+1}_placement.png", bbox_inches="tight")
    plt.close()

plt.figure(4)

plt.plot(mse_predictions[:,0],mse_predictions[:,1],label="Cl")
plt.plot(mse_predictions[:,0],mse_predictions[:,2],label="Cd")
plt.xlabel("numbers of sensors")
plt.ylabel("Mean squared error")
plt.legend()
plt.savefig(f"{output}/MSE_Cl_Cd_RF.png", bbox_inches="tight")
plt.close()
