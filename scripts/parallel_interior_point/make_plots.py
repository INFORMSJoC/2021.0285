#  ___________________________________________________________________________
#
#  Parapint
#  Copyright (c) 2020
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import matplotlib
import pyomo.environ as pe
import numpy as np


font = {"size": 12}
matplotlib.rc("font", **font)


def estimate(x, y):
    m = pe.ConcreteModel()
    m.slope = pe.Var()
    m.intercept = pe.Var()
    m.data_set = pe.Set(initialize=list(range(len(x))))
    m.y_est = pe.Var(m.data_set)
    m.intercept.fix(0)

    obj_expr = 0
    for ndx in m.data_set:
        obj_expr += (m.y_est[ndx] - y[ndx]) ** 2
    m.obj = pe.Objective(expr=obj_expr)

    m.cons = pe.Constraint(m.data_set)
    for ndx in m.data_set:
        m.cons[ndx] = m.y_est[ndx] == m.slope * x[ndx] + m.intercept

    opt = pe.SolverFactory('ipopt')
    res = opt.solve(m)
    pe.assert_optimal_termination(res)

    return m.slope.value, m.intercept.value


def main():
    fs_file = "serial_results.csv"
    parallel_file = "parallel_results.csv"

    fs_res = dict()
    f = open(fs_file, "r")
    reader = csv.DictReader(f)
    for row in reader:
        end_t = int(row["end_t"])
        nfe_x = int(row["nfe_x"])
        if end_t < 2:
            continue
        solve_time = float(row["IP solve"])
        if nfe_x not in fs_res:
            fs_res[nfe_x] = list()
        fs_res[nfe_x].append((end_t, solve_time))
    f.close()
    fs_end_t = dict()
    fs_solve_time = dict()
    for k, v in fs_res.items():
        if k == 30:
            v.sort(key=lambda x: x[0])
            fs_end_t[k] = [i[0] for i in v]
            fs_solve_time[k] = [i[1] for i in v]
            plt.scatter(
                fs_end_t[k],
                fs_solve_time[k],
                s=2 * rcParams["lines.markersize"] ** 2,
                marker="o",
                label="Full Space, Serial",
            )
            slope, intercept = estimate(fs_end_t[k], fs_solve_time[k])
            x_projected = np.linspace(2, 1024, 1000)
            y_projected = slope * x_projected + intercept
            plt.plot(
                x_projected,
                y_projected,
                label="Full Space, Serial, \nLinear Extrapolation",
            )

    parallel_res = dict()
    f = open(parallel_file, "r")
    reader = csv.DictReader(f)
    for row in reader:
        end_t = int(row["end_t"])
        nfe_x = int(row["nfe_x"])
        solve_time = float(row["IP solve"])
        if nfe_x not in parallel_res:
            parallel_res[nfe_x] = list()
        parallel_res[nfe_x].append((end_t, solve_time))
    f.close()
    parallel_end_t = dict()
    parallel_solve_time = dict()
    for k, v in parallel_res.items():
        if k == 30:
            v.sort(key=lambda x: x[0])
            parallel_end_t[k] = [i[0] for i in v]
            parallel_solve_time[k] = [i[1] for i in v]
            plt.scatter(
                parallel_end_t[k],
                parallel_solve_time[k],
                s=3 * rcParams["lines.markersize"] ** 2,
                marker="+",
                label="Parallel\nSchur-Complement",
            )

    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xticks(
        [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        ["2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"],
    )
    plt.yticks(
        [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        ["2", "4", "8", "16", "32", "64", "128", "256", "512", "1024", "2048", "4096"],
    )
    plt.legend()
    plt.xlabel("Time Horizon/# of Processes")
    plt.ylabel("Solution Time (s)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
