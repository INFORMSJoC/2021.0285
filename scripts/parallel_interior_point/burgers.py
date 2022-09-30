import pyomo.environ as pe
from pyomo import dae
import parapint
import numpy as np
from mpi4py import MPI
import math
import logging
import argparse
from pyomo.common.timing import HierarchicalTimer
import csv
import os


"""
Run this example with, e.g., 

mpirun -np 4 python -m mpi4py burgers.py --nfe_x 30 --end_t 4 --nfe_t_per_t 1600 --nblocks 4

"""

comm: MPI.Comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    logging.basicConfig(level=logging.INFO)


class Args(object):
    def __init__(self):
        self.nfe_x = 50
        self.nfe_t = 200
        self.end_t = 1
        self.nblocks = 4
        self.method = 'psc'

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--nfe_x', type=int, required=True, help='number of finite elements for x')
        parser.add_argument('--end_t', type=int, required=True, help='end time')
        parser.add_argument('--nfe_t_per_t', type=int, required=False, default=1600, help='number of finite elements for t per unit time')
        parser.add_argument('--method', type=str, required=True, help='either "psc" for parallel schur-complement or "fs" for full-space serial')
        parser.add_argument('--nblocks', type=int, required=True, help='number of time blocks for schur complement')
        args = parser.parse_args()
        self.nfe_x = args.nfe_x
        self.end_t = args.end_t
        self.nfe_t = args.nfe_t_per_t * args.end_t
        self.nblocks = args.nblocks
        self.method = args.method

        if self.method not in {'psc', 'fs'}:
            raise ValueError('method should be either "psc" for parallel schur-complement or "fs" for full-space serial')


def build_burgers_model(nfe_x=50, nfe_t=50, start_t=0, end_t=1, add_init_conditions=True):
    dt = (end_t - start_t) / float(nfe_t)

    start_x = 0
    end_x = 1
    dx = (end_x - start_x) / float(nfe_x)

    m = pe.Block(concrete=True)
    m.omega = pe.Param(initialize=0.02)
    m.v = pe.Param(initialize=0.01)
    m.r = pe.Param(initialize=0)

    m.x = dae.ContinuousSet(bounds=(start_x, end_x))
    m.t = dae.ContinuousSet(bounds=(start_t, end_t))

    m.y = pe.Var(m.x, m.t)
    m.dydt = dae.DerivativeVar(m.y, wrt=m.t)
    m.dydx = dae.DerivativeVar(m.y, wrt=m.x)
    m.dydx2 = dae.DerivativeVar(m.y, wrt=(m.x, m.x))

    m.u = pe.Var(m.x, m.t)

    def _y_init_rule(m, x, t):
        if x <= 0.5 * end_x:
            return 1 * round(math.cos(2*math.pi*t))
        return 0

    m.y0 = pe.Param(m.x, m.t, default=_y_init_rule)

    def _upper_x_bound(m, t):
        return m.y[end_x, t] == 0

    m.upper_x_bound = pe.Constraint(m.t, rule=_upper_x_bound)

    def _lower_x_bound(m, t):
        return m.y[start_x, t] == 0

    m.lower_x_bound = pe.Constraint(m.t, rule=_lower_x_bound)

    def _upper_x_ubound(m, t):
        return m.u[end_x, t] == 0

    m.upper_x_ubound = pe.Constraint(m.t, rule=_upper_x_ubound)

    def _lower_x_ubound(m, t):
        return m.u[start_x, t] == 0

    m.lower_x_ubound = pe.Constraint(m.t, rule=_lower_x_ubound)

    def _lower_t_bound(m, x):
        if x == start_x or x == end_x:
            return pe.Constraint.Skip
        return m.y[x, start_t] == m.y0[x, start_t]

    def _lower_t_ubound(m, x):
        if x == start_x or x == end_x:
            return pe.Constraint.Skip
        return m.u[x, start_t] == 0

    if add_init_conditions:
        m.lower_t_bound = pe.Constraint(m.x, rule=_lower_t_bound)
        m.lower_t_ubound = pe.Constraint(m.x, rule=_lower_t_ubound)

    # PDE
    def _pde(m, x, t):
        if t == start_t or x == end_x or x == start_x:
            e = pe.Constraint.Skip
        else:
            last_t = m.t.prev(t)
            e = m.dydt[x, t] - m.v * m.dydx2[x, t] + m.dydx[x, t] * m.y[x, t] == m.r + m.u[x, last_t]
        return e

    m.pde = pe.Constraint(m.x, m.t, rule=_pde)

    # Discretize Model
    disc = pe.TransformationFactory('dae.finite_difference')
    disc.apply_to(m, nfe=nfe_t, wrt=m.t, scheme='BACKWARD')
    disc.apply_to(m, nfe=nfe_x, wrt=m.x, scheme='CENTRAL')

    # Solve control problem using Pyomo.DAE Integrals
    def _intX(m, x, t):
        return (m.y[x, t] - m.y0[x, t]) ** 2 + m.omega * m.u[x, t] ** 2

    m.intX = dae.Integral(m.x, m.t, wrt=m.x, rule=_intX)

    def _intT(m, t):
        return m.intX[t]

    m.intT = dae.Integral(m.t, wrt=m.t, rule=_intT)

    def _obj(m):
        e = 0.5 * m.intT
        for x in sorted(m.x):
            if x == start_x or x == end_x:
                pass
            else:
                e += 0.5 * 0.5 * dx * dt * m.omega * m.u[x, start_t] ** 2
        return e

    m.obj = pe.Objective(rule=_obj)

    return m


class BurgersInterface(parapint.interfaces.MPIDynamicSchurComplementInteriorPointInterface):
    def __init__(self, start_t, end_t, num_time_blocks, nfe_t, nfe_x):
        self.nfe_x = nfe_x
        self.dt = (end_t - start_t) / float(nfe_t)
        super(BurgersInterface, self).__init__(start_t=start_t,
                                               end_t=end_t,
                                               num_time_blocks=num_time_blocks,
                                               comm=comm)

    def build_model_for_time_block(self, ndx, start_t, end_t, add_init_conditions):
        dt = self.dt
        nfe_t = math.ceil((end_t - start_t) / dt)
        m = build_burgers_model(
            nfe_x=self.nfe_x, nfe_t=nfe_t, start_t=start_t, end_t=end_t,
            add_init_conditions=add_init_conditions
        )

        return (m,
                ([m.y[x, start_t] for x in sorted(m.x) if x not in {0, 1}]),
                ([m.y[x, end_t] for x in sorted(m.x) if x not in {0, 1}]))


def write_csv(fname, args, timer):
    if rank == 0:
        if os.path.exists(fname):
            needs_header = False
        else:
            needs_header = True
        f = open(fname, "a")
        writer = csv.writer(f)
        fieldnames = ['end_t', 'nfe_x', 'nfe_t', 'size', 'n_blocks']
        timer_identifiers = timer.get_timers()
        fieldnames.extend(timer_identifiers)
        if needs_header:
            writer.writerow(fieldnames)
        row = [args.end_t, args.nfe_x, args.nfe_t, size, args.nblocks]
        row.extend(timer.get_total_time(name) for name in timer_identifiers)
        writer.writerow(row)
        f.close()


def run_parallel(args, subproblem_solver_class, subproblem_solver_options):
    interface = BurgersInterface(start_t=0,
                                 end_t=args.end_t,
                                 num_time_blocks=args.nblocks,
                                 nfe_t=args.nfe_t,
                                 nfe_x=args.nfe_x)
    linear_solver = parapint.linalg.MPISchurComplementLinearSolver(
        subproblem_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(args.nblocks)},
        schur_complement_solver=subproblem_solver_class(**subproblem_solver_options))
    options = parapint.algorithms.IPOptions()
    options.linalg.solver = linear_solver
    timer = HierarchicalTimer()
    comm.Barrier()
    status = parapint.algorithms.ip_solve(interface=interface, options=options, timer=timer)
    assert status == parapint.algorithms.InteriorPointStatus.optimal
    interface.load_primals_into_pyomo_model()
    fname = "parallel_results.csv"
    write_csv(fname, args, timer)


def run_full_space(args, subproblem_solver_class, subproblem_solver_options):
    m = build_burgers_model(nfe_x=args.nfe_x, nfe_t=args.nfe_t, start_t=0,
                            end_t=args.end_t, add_init_conditions=True)
    interface = parapint.interfaces.InteriorPointInterface(m)
    linear_solver = subproblem_solver_class(**subproblem_solver_options)
    options = parapint.algorithms.IPOptions()
    options.linalg.solver = linear_solver
    timer = HierarchicalTimer()
    status = parapint.algorithms.ip_solve(interface=interface, options=options, timer=timer)
    assert status == parapint.algorithms.InteriorPointStatus.optimal
    interface.load_primals_into_pyomo_model()
    fname = "serial_results.csv"
    write_csv(fname, args, timer)


if __name__ == '__main__':
    args = Args()
    args.parse_arguments()

    # cntl[1] is the MA27 pivot tolerance

    if args.method == 'psc':
        run_parallel(
            args=args,
            subproblem_solver_class=parapint.linalg.InteriorPointMA27Interface,
            subproblem_solver_options={'cntl_options': {1: 1e-6}}
        )
    else:
        run_full_space(
            args=args,
            subproblem_solver_class=parapint.linalg.InteriorPointMA27Interface,
            subproblem_solver_options={'cntl_options': {1: 1e-6}}
        )
