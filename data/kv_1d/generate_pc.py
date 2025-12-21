import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ufl")
import dolfin as dl
import numpy as np
import random
import sys, os
import argparse
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import time
import pickle
sys.path.insert(0, '../../')
from modeling import *

def sample_points(n_points):
    # Sample points in (0,1) on grid excluding the two boundary points
    return [np.sort(np.random.choice(np.linspace(0.0, 1.0, 51)[1:-1], n, replace=False)) for n in n_points]

def sample_values(n_points, bound):
    # Sample values in given bound
    values_list = []
    for n in n_points:
        values_list.append(np.random.uniform(low=bound[0], high=bound[1], size=n))
    return values_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='1d KV PC microstructure data generation')

    parser.add_argument('--process_id',
                        default=0,
                        type=int,
                        help="The processor ID for parallel data generation")

    parser.add_argument('--seed',
                        default=0,
                        type=int,
                        help="The random seed for sample generation")

    parser.add_argument('--n_samples',
                        default=1000,
                        type=int,
                        help="The number of samples")

    parser.add_argument('--n_pieces_max',
                        default=20,
                        type=int,
                        help="The maximum number of pieces")

    parser.add_argument('--n_pieces_min',
                        default=5,
                        type=int,
                        help="The minimum number of pieces")

    parser.add_argument('--E_max',
                        default=1,
                        type=float,
                        help="The maximum E value")

    parser.add_argument('--E_min',
                        default=0.1,
                        type=float,
                        help="The minimum E value")

    parser.add_argument('--nu_max',
                        default=1,
                        type=float,
                        help="The maximum nu value")

    parser.add_argument('--nu_min',
                        default=0.1,
                        type=float,
                        help="The minimum nu value")

    parser.add_argument('--trajectory_interval',
                        default=20,
                        type=int,
                        help="The number of interval for trajectory generation")

    parser.add_argument('--trajectory_bound',
                        default=0.5,
                        type=float,
                        help="The upper and lower bound of the trajectory value")

    parser.add_argument('--output_path',
                        default="./dataset/",
                        type=str,
                        help="The output path for saving")

    parser.add_argument('--file_name',
                        default="data.pkl",
                        type=str,
                        help="The name for the output pickle file")

    dl.parameters["form_compiler"]["optimize"] = True
    dl.parameters["form_compiler"]["cpp_optimize"] = True
    dl.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

    args = parser.parse_args()

    tracer = vars(args)  # store the command line argument values into the dictionary

    if not os.path.exists(tracer["output_path"]):
        os.makedirs(tracer["output_path"], exist_ok=True)

    settings = KelvinVoightSettings()
    settings["n_cells"] = 500
    settings["FE_order"] = 1
    settings["dim"] = 1
    np.random.seed(tracer["seed"])
    random.seed(tracer["seed"])

    from datetime import datetime, timedelta

    padded_string = datetime.now().strftime("%Y-%m-%d") + "_PC1D_process" + str(tracer["process_id"]) + "_"

    cell_problem = KelvinVoightCellProblem(**settings)
    cell_problem.parameters["verbose"] = False
    cell_problem.parameters["nt"] = 5000
    cell_problem.parameters["T"] = 1.0
    cell_problem.parameters[
        "solve_for_viscous_stress"] = False  # whether to use PDE solve for estimating viscous stress, default to False
    tracer = {**tracer, **cell_problem.parameters, **settings}

    Vh = cell_problem.FunctionSpace()
    Sh = cell_problem.FunctionSpace(type="scalar")

    rank = Vh.mesh().mpi_comm().rank

    d = cell_problem.d_sym  # dimension of the symmetric part of strain components
    stress_array = np.zeros((tracer["n_samples"], tracer["nt"] + 1, d))
    strain_array = np.zeros((tracer["n_samples"], tracer["nt"] + 1, d))
    strain_rate_array = np.zeros((tracer["n_samples"], tracer["nt"] + 1, d))
    E_array = np.zeros((tracer["n_samples"], tracer["n_cells"] + 1))
    nu_array = np.zeros((tracer["n_samples"], tracer["n_cells"] + 1))
    nu_prime = np.zeros((tracer["n_samples"], d, d))
    E_prime = np.zeros((tracer["n_samples"], d, d))
    kernel = np.zeros((tracer["n_samples"], tracer["nt"] + 1, d, d))

    for ii in range(tracer["process_id"]):
        n_points = np.random.randint(low=tracer["n_pieces_min"], high=tracer["n_pieces_max"] + 1,  # plus one to
                                     # include the max in randomint
                                     size=tracer["n_samples"])
        _ = sample_points(n_points)
        _ = sample_values(n_points, [tracer["E_min"], tracer["E_max"]])
        _ = sample_values(n_points, [tracer["nu_min"], tracer["nu_max"]])
        for jj in range(tracer["n_samples"]):
            _, _, _ = generate_trajectory(tracer["trajectory_interval"], d,
                                                     low=-tracer["trajectory_bound"],
                                                     high=tracer["trajectory_bound"])

    full_t_list, full_value_list = [], []
    n_points = np.random.randint(low=tracer["n_pieces_min"], high=tracer["n_pieces_max"] + 1, size=tracer["n_samples"])
    points_list = sample_points(n_points)
    E_values_list = sample_values(n_points, [tracer["E_min"], tracer["E_max"]])
    nu_values_list = sample_values(n_points, [tracer["nu_min"], tracer["nu_max"]])

    if rank == 0:
        print("Process %d starts data generation..." % (tracer["process_id"]))
    t0 = time.time()
    progress_every = max(1, tracer["n_samples"] // 20)
    for ii in range(tracer["n_samples"]):
        # sample random trajectories
        strain, t_list, value_list = generate_trajectory(tracer["trajectory_interval"], d,
                                                                           low=-tracer["trajectory_bound"],
                                                                           high=tracer["trajectory_bound"])
        full_t_list.append(t_list)
        full_value_list.append(value_list)
        strain_rate = strain.rate()

        # sample random values for Young's modulus and poisson ratio
        E = PC1DMicrostructure(Sh, points_list[ii], E_values_list[ii])
        nu = PC1DMicrostructure(Sh, points_list[ii], nu_values_list[ii])

        # pass in microstructures for system assembly
        cell_problem.set_microstructure(E, nu)

        # solve the cell problem
        t, stress = cell_problem.solve(strain, strain_rate)
        stress_unit, stress_rate_1_unit, stress_rate_2_unit = cell_problem.extract_memory_form()

        # assign solution
        stress_array[ii] = stress
        nu_prime[ii] = stress_unit[0]
        E_prime[ii] = stress_rate_1_unit[0]
        kernel[ii] = -stress_rate_2_unit
        strain_array[ii] = strain(t)
        strain_rate_array[ii] = strain_rate(t)
        E_array[ii] = E.get_local()
        nu_array[ii] = nu.get_local()
        if (ii + 1) % progress_every == 0 and rank == 0:
            t_ii = time.time()
            per_sample_time = (t_ii - t0) / (ii + 1)
            remaining = (tracer["n_samples"] - ii-1)*per_sample_time
            print("%d samples finished with %1.2fs per sample. Estimated remaining time:" % (
                ii + 1, per_sample_time) + str(timedelta(seconds=remaining)))
            sys.stdout.flush()
    t1 = time.time()
    if rank == 0:
        print("Process " + str(tracer["process_id"]) + " averaged time:", (t1 - t0) / tracer["n_samples"])

    tracer["strain"] = strain_array
    tracer["strain_rate"] = strain_rate_array
    tracer["stress"] = stress_array
    tracer["nu"] = nu_array
    tracer["E"] = E_array
    tracer["nu_prime"] = nu_prime
    tracer["E_prime"] = E_prime
    tracer["kernel"] = kernel
    tracer["time"] = t
    tracer["trajectory_generator_time"] = full_t_list
    tracer["trajectory_generator_strain"] = full_value_list
    tracer["pieces_points"] = points_list
    tracer["E_pieces_values"] = E_values_list
    tracer["nu_pieces_values"] = nu_values_list

    if rank == 0:
        with open(tracer["output_path"] + padded_string + tracer["file_name"], 'wb') as f:
            pickle.dump(tracer, f)
