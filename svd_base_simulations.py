#!/usr/bin/python3

from os.path import join
import torch as pt

output = "./output"
cases = ("pinnball_re170", )

# snapshots before t_start are ignored
t_start = 100.0

for case in cases:
    t = pt.load(join(output, case, "snapshot_times.pt"))
    start_idx = (t-t_start).abs().argmin()
    p = pt.load(join(output, case, "p.pt"))[:, start_idx:]
    U, s, VH = pt.linalg.svd(p, full_matrices=False)
    pt.save(U, join(output, case, "svd_p_U.pt"))
    pt.save(s, join(output, case, "svd_p_s.pt"))
    pt.save(VH.conj().T, join(output, case, "svd_p_V.pt"))
    del p, U, s, VH
    vel = pt.load(join(output, case, "U.pt"))[:, start_idx:]
    U, s, VH = pt.linalg.svd(vel, full_matrices=False)
    pt.save(U, join(output, case, "svd_vel_U.pt"))
    pt.save(s, join(output, case, "svd_vel_s.pt"))
    pt.save(VH.conj().T, join(output, case, "svd_vel_V.pt"))