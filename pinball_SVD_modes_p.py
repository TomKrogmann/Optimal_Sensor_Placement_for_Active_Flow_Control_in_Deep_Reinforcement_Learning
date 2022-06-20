import matplotlib.pyplot as plt
import matplotlib.tri as tri
from stl import mesh
import torch as pt
import numpy as np
from flowtorch import DATASETS
from flowtorch.data import FOAMDataloader, mask_box
from flowtorch.analysis import SVD
import pysensors as ps
from scipy import spatial

#from flowtorch.data import FOAMCase

# increase plot resolution
plt.rcParams["figure.dpi"] = 160

#output = "/home/tom/flowtorch/flowtorch/output/pinball_POD_p_Re170_1000s"
output = "/home/tom/flowtorch/flowtorch/output/test"

path = DATASETS["RE_170_1000s"]
loader = FOAMDataloader(path)
times = loader.write_times
fields = loader.field_names
print(f"Number of available snapshots: {len(times)}")
print("First five write times: ", times[:5])
print(f"Fields available at t={times[-1]}: ", fields[times[-1]])

# load vertices and discard z-coordinate
vertices = loader.vertices[:, :2]
#mask = mask_box(vertices, lower=[-2.5, -4], upper=[10, 4])
mask = mask_box(vertices, lower=[-2.5, -2], upper=[10, 2])

print(f"Selected vertices: {mask.sum().item()}/{mask.shape[0]}")

every = 1 # use only every 4th vertex
fig, ax = plt.subplots()
d = 0.5
ax.scatter(vertices[::every, 0]/d, vertices[::every, 1]/d, s=0.5, c=mask[::every])
ax.set_aspect("equal", 'box')
#ax.set_xlim(0.0, 2.2/d)
#ax.set_ylim(0.0, 0.41/d)
ax.set_xlabel(r"$x/d$")
ax.set_ylabel(r"$y/d$")
plt.savefig(f"{output}/cylinder_mask.png", bbox_inches="tight")
plt.close()

window_times = [t for t in times if 200 <= float(t) <= 300.0]
#print(f"Window times: {window_times}")

n_points = mask.sum().item()
data_matrix = pt.zeros((n_points, len(window_times)))
for i,t in enumerate(window_times):
    data_matrix[:,i] = pt.masked_select(loader.load_snapshot("p",t), mask)

x = pt.masked_select(vertices[:, 0], mask)/d
y = pt.masked_select(vertices[:, 1], mask)/d

# print(f"U: {U.shape}")
# print(f"V: {V.shape}")
# print(f"U: {U}")
# print(f"V: {V}")

N = data_matrix.shape[1]

data_matrix_plot = data_matrix - data_matrix.mean(dim=1).unsqueeze(-1)

C_snap = data_matrix_plot.T @ data_matrix_plot / (N - 1)
vals_s, vecs_s = pt.linalg.eigh(C_snap)
vals_s, indices = pt.sort(vals_s, descending=True)
vecs_s = vecs_s[:, indices]

plt.bar(range(1, 21), vals_s[:20]/vals_s.sum()*100)
plt.gca().set_xticks(range(1, 21))
plt.xlabel(r"$i$")
plt.ylabel(r"$\lambda_{i,rel} \times 100\%$")
plt.title("individual contribution to TKE")
plt.savefig(f"{output}/P_cylinder_eigvals_snap.png", bbox_inches="tight")
plt.close()

plt.figure(3)
fig, (ax1, ax2) = plt.subplots(2, sharex = True)

cont1 = ax1.tricontourf(x,y, data_matrix.mean(dim=1))
cont2 = ax2.tricontourf(x,y, data_matrix.var(dim=1))
plt.colorbar(cont1, ax=ax1)
plt.colorbar(cont2, ax=ax2)
for ax in (ax1,ax2):
    ax.add_patch(plt.Circle((-1.3, 0), 0.5, color='k'))
    ax.add_patch(plt.Circle((0, 0.75), 0.5, color='k'))
    ax.add_patch(plt.Circle((0, -0.75), 0.5, color='k'))
    ax.set_aspect("equal")
    ax.set_ylabel(r"$y/d$")

ax1.set_title(r"$P$ - mean")
ax1.set_xlabel(r"$x/d$")
ax2.set_title(r"$P$ - variance")
ax2.set_xlabel(r"$x/d$")
plt.savefig(f"{output}/c_p-without_mean_variance.png", bbox_inches="tight")
plt.close()

U,s,VH = pt.linalg.svd(data_matrix - data_matrix.mean(dim=1).unsqueeze(-1), full_matrices = False)
  
# plt.figure(4)
# plt.bar(range(1, 50+1), s[:50])
# plt.xlabel("i")
# plt.ylabel(r"$\sigma_i$")
# plt.savefig(f"{output}/sigma.svg", bbox_inches="tight")
# plt.close()

# plt.figure(5)
# s_rel = [si/s.sum().item() for si in s]
# plt.bar(range(1, 50+1), s_rel[:50])
# plt.xlabel("i")
# plt.ylabel(r"$\sigma_{i, rel}$")
# plt.savefig(f"{output}/sigma_rel.svg", bbox_inches="tight")
# plt.close()

plt.figure(6)
fig,axarr = plt.subplots(4, figsize=(6,7), sharex=True)

for i in range(4):
    cont = axarr[i].tricontourf(x, y, U[:, i])
    plt.colorbar(cont, ax=axarr[i])
    axarr[i].set_aspect("equal", "box")
    axarr[i].set_ylabel(r"$y/d$")
    axarr[i].set_title(f"mode {i+1}")
    axarr[i].add_patch(plt.Circle((-1.3, 0), 0.5, color='k'))
    axarr[i].add_patch(plt.Circle((0, 0.75), 0.5, color='k'))
    axarr[i].add_patch(plt.Circle((0, -0.75), 0.5, color='k'))
    axarr[i].set_aspect("equal", 'box')
axarr[-1].set_xlabel(r"$x_d$")
plt.savefig(f"{output}/modes_tricontour.png", bbox_inches="tight")
plt.close()


########## PySensors Reconstruction with SVD modes

n_basis_modes = 20

data = np.asarray(data_matrix - data_matrix.mean(dim=1).unsqueeze(-1))
print(data.shape)
data = data.transpose(1,0) ######### why transpose?
print(data.shape)

X_train = data[:134]

combined_x_y = np.vstack((x, y)).T

costs = np.zeros(len(combined_x_y))

def withScipy(X,Y):  # faster
    return np.argmin(spatial.distance.cdist(X,Y,'sqeuclidean'),axis=0)

point_x = np.array([0])
point_y = np.array([0])

point = np.vstack((point_x, point_y)).T

index_point = withScipy(combined_x_y,point)

print(index_point)

costs[index_point] = 1

#costs = costs.reshape(-1)

basis = ps.basis.SVD(n_basis_modes=n_basis_modes)
optimizer = ps.optimizers.CCQR(sensor_costs=costs)
model = ps.SSPOR(optimizer=optimizer,basis=basis)
model.fit(X_train)

# Ranked list of sensors
svd_ranked_sensors = model.get_selected_sensors()
#print('Original ranked sensors:', ranked_sensors[:10])
print('SVD ranked sensors:', svd_ranked_sensors[:19])

n_sensors = 20

goal_x_y = np.zeros((n_sensors, 2))

print("First:", goal_x_y)

for i in range(n_sensors):
    goal_x_y[i] = combined_x_y[svd_ranked_sensors[i]]
    #goal_x_y[i][1] = combined_x_y[svd_ranked_sensors[i]]

print("Second:", goal_x_y)


fig, ax = plt.subplots()
plt.plot(goal_x_y[:, 0], goal_x_y[:, 1], "o")
circle1= plt.Circle((-1.3, 0), 0.5, color='k')
circle2= plt.Circle((0, 0.75), 0.5, color='k')
circle3= plt.Circle((0, -0.75), 0.5, color='k')

ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)

ax = plt.gca()
ax.set_xlim([-3, 20])
ax.set_ylim([-4, 4])

plt.xlabel("x")
plt.ylabel("y")
plt.savefig(f"{output}/SVD_modes_sensor_placement.png", bbox_inches="tight")
#plt.show()
plt.close()

sensor_range = np.arange(0, 50, 1)
errors = model.reconstruction_error(X_train, sensor_range=sensor_range)

plt.plot(sensor_range, errors, '-o')
plt.xlabel('Number of sensors')
plt.ylabel('Reconstruction error (MSE)')
plt.title('Reconstruction error for different numbers of sensors');
plt.savefig(f"{output}/Reconstruction error for different numbers of sensors.png", bbox_inches="tight")
plt.close()