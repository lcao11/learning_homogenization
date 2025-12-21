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
    parser = argparse.ArgumentParser(description='1d elasto-viscoplastic piecewise constant microstructure data generation')

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
                        default=10,
                        type=int,
                        help="The maximum number of pieces")

    parser.add_argument('--n_pieces_min',
                        default=2,
                        type=int,
                        help="The minimum number of pieces")

    parser.add_argument('--youngs_modulus_min',
                        default=1.0,
                        type=float,
                        help="The minimum E value")

    parser.add_argument('--youngs_modulus_max',
                        default=10,
                        type=float,
                        help="The maximum nu value")

    parser.add_argument('--rate_constant_min',
                        default=0.5,
                        type=float,
                        help="The minimum rate constant value")
    
    parser.add_argument('--rate_constant_max',
                        default=1.5,
                        type=float,
                        help="The minimum rate constant value")
    
    parser.add_argument('--rate_exponent_min',
                        default=1,
                        type=float,
                        help="The minimum rate exponent value")
    
    parser.add_argument('--rate_exponent_max',
                        default=10,
                        type=float,
                        help="The minimum rate exponent value")

    parser.add_argument('--yield_stress_min',
                        default=0.1,
                        type=float,
                        help="The minimum yield stress value")
    
    parser.add_argument('--yield_stress_max',
                        default=1.0,
                        type=float,
                        help="The minimum yield stress value")

    parser.add_argument('--trajectory_interval',
                        default=10,
                        type=int,
                        help="The number of interval for trajectory generation")

    parser.add_argument('--trajectory_bound',
                        default=0.5,
                        type=float,
                        help="The upper and lower bound of the trajectory value")

    parser.add_argument('--output_path',
                        default="./data/",
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

    settings = ElastoViscoplasticity1DSettings()
    settings["n_cells"] = 500
    settings["FE_order"] = 1
    np.random.seed(tracer["seed"])
    random.seed(tracer["seed"])

    from datetime import datetime, timedelta

    padded_string = datetime.now().strftime("%Y-%m-%d") + "_ElastoViscoplastic_PC1D_process" + str(tracer["process_id"]) + "_"

    cell_problem = ElastoViscoplasticity1DCellProblem(**settings)
    cell_problem.parameters["verbose"] = False
    cell_problem.parameters["nt"] = 5000
    cell_problem.parameters["T"] = 1.0
    tracer = {**tracer, **cell_problem.parameters, **settings}

    Vh = cell_problem.FunctionSpace()

    rank = Vh.mesh().mpi_comm().rank

    d = 1
    stress_array = np.zeros((tracer["n_samples"], tracer["nt"] + 1, d))
    strain_array = np.zeros((tracer["n_samples"], tracer["nt"] + 1, d))
    plastic_strain_array = np.zeros((tracer["n_samples"], tracer["nt"] + 1, d))
    youngs_modulus_array = np.zeros((tracer["n_samples"], tracer["n_cells"] + 1))
    rate_exponent_array = np.zeros((tracer["n_samples"], tracer["n_cells"] + 1))
    yield_stress_array = np.zeros((tracer["n_samples"], tracer["n_cells"] + 1))
    rate_constant_array = np.zeros((tracer["n_samples"], tracer["n_cells"] + 1))

    for ii in range(tracer["process_id"]):
        n_points = np.random.randint(low=tracer["n_pieces_min"], high=tracer["n_pieces_max"] + 1,  # plus one to
                                     # include the max in randomint
                                     size=tracer["n_samples"])
        _ = sample_points(n_points)
        _ = sample_values(n_points, [tracer["youngs_modulus_min"], tracer["youngs_modulus_max"]])
        _ = sample_values(n_points, [tracer["rate_constant_min"], tracer["rate_constant_max"]])
        _ = sample_values(n_points, [tracer["yield_stress_min"], tracer["yield_stress_max"]])
        _ = sample_values(n_points, [tracer["rate_exponent_min"], tracer["rate_exponent_max"]])

        for jj in range(tracer["n_samples"]):
            _, _, _ = generate_trajectory(tracer["trajectory_interval"], d,
                                                     low=-tracer["trajectory_bound"],
                                                     high=tracer["trajectory_bound"])

    full_t_list, full_value_list = [], []
    n_points = np.random.randint(low=tracer["n_pieces_min"], high=tracer["n_pieces_max"] + 1, size=tracer["n_samples"])
    points_list = sample_points(n_points)
    youngs_modulus_list = sample_values(n_points, [tracer["youngs_modulus_min"], tracer["youngs_modulus_max"]])
    yield_stress_list = sample_values(n_points, [tracer["yield_stress_min"], tracer["yield_stress_max"]])
    rate_exponent_list = sample_values(n_points, [tracer["rate_exponent_min"], tracer["rate_exponent_max"]])
    rate_constant_list = sample_values(n_points, [tracer["rate_constant_min"], tracer["rate_constant_max"]])

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

        # sample random values for Young's modulus and poisson ratio
        youngs_modulus = PC1DMicrostructure(Vh, points_list[ii], youngs_modulus_list[ii])
        rate_constant = PC1DMicrostructure(Vh, points_list[ii], rate_constant_list[ii])
        yield_stress = PC1DMicrostructure(Vh, points_list[ii], yield_stress_list[ii])
        rate_exponent = PC1DMicrostructure(Vh, points_list[ii], rate_exponent_list[ii])

        # pass in microstructures for system assembly
        cell_problem.set_microstructure(youngs_modulus, rate_constant, yield_stress, rate_exponent)

        # solve the cell problem
        t, stress, plastic_strain = cell_problem.solve(strain)

        # assign solution
        stress_array[ii] = np.expand_dims(stress, axis=1)
        strain_array[ii] = strain(t)
        plastic_strain_array[ii] = np.expand_dims(plastic_strain, axis=1)
        youngs_modulus_array[ii] = youngs_modulus.get_local()
        rate_constant_array[ii] = rate_constant.get_local()
        yield_stress_array[ii] = yield_stress.get_local()
        rate_exponent_array[ii] = rate_exponent.get_local()
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
    tracer["plastic_strain"] = plastic_strain_array
    tracer["stress"] = stress_array
    tracer["youngs_modulus"] = youngs_modulus_array
    tracer["yield_stress"] = yield_stress_array
    tracer["rate_exponent"] = rate_exponent_array
    tracer["rate_constant"] = rate_constant_array
    tracer["time"] = t
    tracer["trajectory_generator_time"] = full_t_list
    tracer["trajectory_generator_strain"] = full_value_list
    tracer["pieces_points"] = points_list
    tracer["youngs_modulus_pieces_values"] = youngs_modulus_list
    tracer["rate_constant_pieces_values"] = rate_constant_list
    tracer["yield_stress_pieces_values"] = yield_stress_list
    tracer["rate_exponent_pieces_values"] = rate_exponent_list

    if rank == 0:
        with open(tracer["output_path"] + padded_string + tracer["file_name"], 'wb') as f:
            pickle.dump(tracer, f)
