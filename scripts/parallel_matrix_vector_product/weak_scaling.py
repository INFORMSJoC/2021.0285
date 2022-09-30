from mpi4py import MPI
import numpy as np
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
import operator
import csv
from typing import Sequence, Dict
from scipy.sparse import coo_matrix
import time
import os


def distribute_blocks(num_blocks: int, rank: int, size: int) -> Sequence[int]:
    local_blocks = list()
    for ndx in range(num_blocks):
        if ndx % size == rank:
            local_blocks.append(ndx)
    return local_blocks


def get_ownership_map(num_blocks: int, size: int) -> Dict[int, int]:
    ownership_map = dict()
    for ndx in range(num_blocks):
        for rank in range(size):
            if ndx % size == rank:
                ownership_map[ndx] = rank
                break
    return ownership_map


def get_random_coo_matrix(n_rows, n_cols, nnz):
    rows = np.random.randint(low=0, high=n_rows, size=nnz, dtype=np.int64)
    cols = np.random.randint(low=0, high=n_cols, size=nnz, dtype=np.int64)
    data = np.random.normal(size=nnz)
    block = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
    return block


def get_random_mpi_block_bordered_matrix(
    n_block_rows, n_block_cols, row_block_length, col_block_length, sparsity, seed
):
    np.random.seed(seed)
    comm: MPI.Comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ownership_map = get_ownership_map(num_blocks=n_block_cols, size=size)
    rank_owner = np.ones((n_block_rows, n_block_cols), dtype=np.int64) * -1
    for j in range(n_block_cols):
        rank_owner[j, j] = ownership_map[j]
        rank_owner[n_block_rows - 1, j] = ownership_map[j]
    res = MPIBlockMatrix(
        nbrows=n_block_rows,
        nbcols=n_block_cols,
        rank_ownership=rank_owner,
        mpi_comm=comm,
    )
    for ndx in distribute_blocks(num_blocks=n_block_cols, rank=rank, size=size):
        block = get_random_coo_matrix(
            row_block_length,
            col_block_length,
            int(sparsity * row_block_length * col_block_length),
        )
        res.set_block(ndx, ndx, block)
        block = get_random_coo_matrix(
            row_block_length,
            col_block_length,
            int(sparsity * row_block_length * col_block_length),
        )
        res.set_block(n_block_rows - 1, ndx, block)
    return res


def get_random_mpi_block_vector(n_blocks, block_length, seed):
    np.random.seed(seed)
    comm: MPI.Comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    ownership_map = get_ownership_map(num_blocks=n_blocks, size=size)
    rank_owner = [ownership_map[i] for i in range(n_blocks)]
    res = MPIBlockVector(
        nblocks=n_blocks, rank_owner=rank_owner, mpi_comm=MPI.COMM_WORLD
    )
    for i in distribute_blocks(num_blocks=n_blocks, rank=rank, size=size):
        res.set_block(i, np.random.normal(size=block_length))
    return res


def performance_helper_binary(a, b, n_evals, operation):
    cumulative_time = 0
    comm: MPI.Comm = MPI.COMM_WORLD
    for i in range(n_evals):
        t0 = time.time()
        res = operation(a, b)
        t1 = time.time()
        t = t1 - t0
        t = comm.allreduce(t, MPI.MAX)
        cumulative_time += t
    return cumulative_time, res


def run_weak_scaling():
    comm: MPI.Comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_blocks = size
    n_evals = 10
    sparsity = 0.001

    if rank == 0:
        if os.path.exists("mpi_weak_matvec.csv"):
            needs_header = False
        else:
            needs_header = True
        f = open("mpi_weak_matvec.csv", "a")
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(
                [
                    "nblocks",
                    "n_evals",
                    "sparsity",
                    "block_length",
                    "num_procs",
                    "mpi_block_time",
                ]
            )
    block_length = 100000
    mpi_block_A = get_random_mpi_block_bordered_matrix(
        n_block_rows=n_blocks + 1,
        n_block_cols=n_blocks,
        row_block_length=block_length,
        col_block_length=block_length,
        sparsity=sparsity,
        seed=rank,
    )
    mpi_block_x = get_random_mpi_block_vector(
        n_blocks=n_blocks, block_length=block_length, seed=rank
    )
    comm.Barrier()
    mpi_block_time, mpi_block_res = performance_helper_binary(
        mpi_block_A, mpi_block_x, n_evals, operator.mul
    )
    if rank == 0:
        writer.writerow(
            [
                n_blocks,
                n_evals,
                sparsity,
                block_length,
                size,
                mpi_block_time,
            ]
        )
        f.close()


if __name__ == "__main__":
    run_weak_scaling()
