import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ufl")
import dolfin as dl  # Import the fenics library. Make sure you can do this!
import numpy as np
import random
import sys, os
import argparse
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import time
import pickle
import hippylib as hp
sys.path.insert(0, '../../')
from scipy.ndimage import gaussian_filter1d
from modeling import *


class microstructure_hierarchical:
    def __init__(self, Sh, settings):
        """
        :param settings: The parsed argument as a dictionary.
        """
        self._settings = settings
        self._Sh = Sh
        measure = PeriodicBiLaplacianPrior(self._Sh, 0.1, 0.1)  # a dummy object used to generate noise vector
        self._noise = dl.Vector()
        measure.init_vector(self._noise, "noise")

    def sample_statistics(self, n_samples):
        microstructures = ["youngs_modulus", "rate_constant" , "yield_stress", "rate_exponent"]
        out = np.zeros((n_samples, len(microstructures), 2))
        for ii, name in enumerate(microstructures):  # loop through E and nu
            # The correlation length is sampled using a reciprocal distribution
            out[:, ii, 0] = np.exp(
                np.random.uniform(low=np.log(self._settings["rho_min"]), high=np.log(self._settings["rho_max"]),
                                  size=n_samples))
            # The pointwise std of GRF is sampled using a uniform distribution
            out[:, ii, 1] = np.random.uniform(low=self._settings["sigma_min"], high=self._settings["sigma_max"],
                                              size=n_samples)

        return out

    def sample_microstructure(self, stats, mean_func):
        microstructures = ["youngs_modulus", "rate_constant" , "yield_stress", "rate_exponent"]
        out = [None for ii in microstructures]
        for ii, name in enumerate(microstructures):
            microstructure = PeriodicRandomMicrostructure(self._Sh, bound=[self._settings[name + "_min"], self._settings[name + "_max"]],
                                              correlation_length=stats[ii, 0],
                                              pointwise_std=stats[ii, 1], mean=mean_func[ii])
            hp.parRandom.normal(1., microstructure._noise)
            out[ii] = microstructure.sample()
        return out

    def consume_random(self, n_samples):
        for ii in range(n_samples):
            # Match sampling path in sample_microstructure:
            # For each of 4 properties: one explicit normal + one inside RandomMicrostructure.sample().
            hp.parRandom.normal(1., self._noise)
            hp.parRandom.normal(1., self._noise)
            hp.parRandom.normal(1., self._noise)
            hp.parRandom.normal(1., self._noise)
            hp.parRandom.normal(1., self._noise)
            hp.parRandom.normal(1., self._noise)
            hp.parRandom.normal(1., self._noise)
            hp.parRandom.normal(1., self._noise)
        # NOTE: Hippylib RNG is independent of NumPy RNG; this is kept only to preserve prior behavior.
        np.random.uniform(size=8 * n_samples)

def sample_points(n_points):
    return [np.sort(np.random.choice(np.linspace(0.0, 1.0, 51)[1:-1], n, replace=False)) for n in n_points]


def sample_values(n_points, bound):
    values_list = []
    for n in n_points:
        values_list.append(np.random.uniform(low=bound[0], high=bound[1], size=n))
    return values_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='1d elasto-viscoplastic random field microstructure data generation')

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
    
    parser.add_argument('--rho_max',
                        default=0.5,
                        type=float,
                        help="The largest correlation length")

    parser.add_argument('--rho_min',
                        default=0.05,
                        type=float,
                        help="The minimum correlation length")

    parser.add_argument('--sigma_max',
                        default=0.3,
                        type=float,
                        help="The maximum pointwise standard deviation")

    parser.add_argument('--sigma_min',
                        default=0.1,
                        type=float,
                        help="the minimum pointwise standard deviation")

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
                        default=2.0,
                        type=float,
                        help="The minimum rate constant value")
    
    parser.add_argument('--rate_exponent_min',
                        default=1,
                        type=float,
                        help="The minimum rate exponent value")
    
    parser.add_argument('--rate_exponent_max',
                        default=20,
                        type=float,
                        help="The minimum rate exponent value")

    parser.add_argument('--yield_stress_min',
                        default=0.1,
                        type=float,
                        help="The minimum yield stress value")
    
    parser.add_argument('--yield_stress_max',
                        default=1.0,
                        type=float,
                        help="The maximum yield stress value")

    parser.add_argument('--trajectory_interval',
                        default=20,
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

    padded_string = datetime.now().strftime("%Y-%m-%d") + "_ElastoViscoplastic_random_field_process" + str(tracer["process_id"]) + "_"

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

    microstructure = microstructure_hierarchical(Vh, tracer)

    for ii in range(tracer["process_id"]):
        n_points = np.random.randint(low=tracer["n_pieces_min"], high=tracer["n_pieces_max"] + 1,  # plus one to
                                     # include the max in randomint
                                     size=tracer["n_samples"])
        _ = sample_points(n_points)
        _ = sample_values(n_points, [tracer["youngs_modulus_min"], tracer["youngs_modulus_max"]])
        _ = sample_values(n_points, [tracer["rate_constant_min"], tracer["rate_constant_max"]])
        _ = sample_values(n_points, [tracer["yield_stress_min"], tracer["yield_stress_max"]])
        _ = sample_values(n_points, [tracer["rate_exponent_min"], tracer["rate_exponent_max"]])
        microstructure.consume_random(tracer["n_samples"])

        for jj in range(tracer["n_samples"]):
            _, _, _ = generate_trajectory(tracer["trajectory_interval"], d,
                                                     low=-tracer["trajectory_bound"],
                                                     high=tracer["trajectory_bound"])

    full_t_list, full_value_list = [], []
    n_points = np.random.randint(low=tracer["n_pieces_min"], high=tracer["n_pieces_max"] + 1, size=tracer["n_samples"])
    points_list = sample_points(n_points)
    youngs_modulus_list = sample_values(n_points, [tracer["youngs_modulus_min"], tracer["youngs_modulus_max"]])
    rate_constant_list = sample_values(n_points, [tracer["rate_constant_min"], tracer["rate_constant_max"]])
    yield_stress_list = sample_values(n_points, [tracer["yield_stress_min"], tracer["yield_stress_max"]])
    rate_exponent_list = sample_values(n_points, [tracer["rate_exponent_min"], tracer["rate_exponent_max"]])
    microstructure_stats = microstructure.sample_statistics(tracer["n_samples"])

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
        youngs_modulus_mean = PC1DMicrostructure(Vh, points_list[ii], youngs_modulus_list[ii])
        rate_constant_mean = PC1DMicrostructure(Vh, points_list[ii], rate_constant_list[ii])
        yield_stress_mean = PC1DMicrostructure(Vh, points_list[ii], yield_stress_list[ii])
        rate_exponent_mean = PC1DMicrostructure(Vh, points_list[ii], rate_exponent_list[ii])
        youngs_modulus_mean.set_local(gaussian_filter1d(youngs_modulus_mean.get_local(), 5, mode="wrap"))
        rate_constant_mean.set_local(gaussian_filter1d(rate_constant_mean.get_local(), 5, mode="wrap"))
        yield_stress_mean.set_local(gaussian_filter1d(yield_stress_mean.get_local(), 5, mode="wrap"))
        rate_exponent_mean.set_local(gaussian_filter1d(rate_exponent_mean.get_local(), 5, mode="wrap"))
        microstructure_list = microstructure.sample_microstructure(microstructure_stats[ii], [hp.vector2Function(microstructure, Vh) for microstructure in [youngs_modulus_mean, rate_constant_mean, yield_stress_mean, rate_exponent_mean]])


        # pass in microstructures for system assembly
        cell_problem.set_microstructure(microstructure_list[0], microstructure_list[1], microstructure_list[2], microstructure_list[3])

        # solve the cell problem
        t, stress, plastic_strain = cell_problem.solve(strain)

        # assign solution
        stress_array[ii] = np.expand_dims(stress, axis=1)
        strain_array[ii] = strain(t)
        plastic_strain_array[ii] = np.expand_dims(plastic_strain, axis=1)
        youngs_modulus_array[ii] = microstructure_list[0].get_local()
        rate_constant_array[ii] = microstructure_list[1].get_local()
        yield_stress_array[ii] = microstructure_list[2].get_local()
        rate_exponent_array[ii] = microstructure_list[3].get_local()
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
