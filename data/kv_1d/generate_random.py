import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ufl")
import dolfin as dl
import numpy as np
import random
import sys, os
import argparse
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
import time
import pickle
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, '../../')
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
        type = ["E", "nu"]
        out = np.zeros((n_samples, 2, 2))
        for ii, name in enumerate(type):  # loop through E and nu
            # The correlation length is sampled using a reciprocal distribution
            out[:, ii, 0] = np.exp(
                np.random.uniform(low=np.log(self._settings["rho_min"]), high=np.log(self._settings["rho_max"]),
                                  size=n_samples))
            # The pointwise std of GRF is sampled using a uniform distribution
            out[:, ii, 1] = np.random.uniform(low=self._settings["sigma_min"], high=self._settings["sigma_max"],
                                              size=n_samples)

        return out

    def sample_microstructure(self, stats, mean_func):
        out = [None for ii in range(2)]
        type = ["E", "nu"]
        for ii, name in enumerate(type):
            microstructure = PeriodicRandomMicrostructure(self._Sh, bound=[self._settings[name + "_min"], self._settings[name + "_max"]],
                                              correlation_length=stats[ii, 0],
                                              pointwise_std=stats[ii, 1], mean=mean_func[ii])
            hp.parRandom.normal(1., microstructure._noise)
            out[ii] = microstructure.sample()
        return out

    def consume_random(self, n_samples):
        for ii in range(n_samples):
            # Match the sampling path in sample_microstructure:
            # for each of (E, nu): one explicit normal + one inside RandomMicrostructure.sample().
            hp.parRandom.normal(1., self._noise)
            hp.parRandom.normal(1., self._noise)
            hp.parRandom.normal(1., self._noise)
            hp.parRandom.normal(1., self._noise)
        np.random.uniform(size=4 * n_samples)

def sample_points(n_points):
    # Sample points in (0,1) on grid excluding the two boundary points
    return [np.sort(np.random.choice(np.linspace(0.0, 1.0, 51)[1:-1], n, replace=False)) for n in n_points]


def sample_values(n_points, bound):
    E_values_list, nu_values_list = [], []
    for n in n_points:
        mixture_idx=[random.choice([-1, 1]) for ii in range(n)]
        values = np.zeros((n, 2))
        for jj, idx in enumerate(mixture_idx):
            mean = bound if idx>0 else bound[::-1]
            while True:
                values[jj] = np.random.multivariate_normal(mean = mean, cov = np.diag([0.06, 0.06]))
                if all([bound[0]<values[jj, ii]<bound[1] for ii in range(2)]):
                    break
        E_values_list.append(values[:, 0])
        nu_values_list.append(values[:, 1])
    return E_values_list, nu_values_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='1d KV random field microstructure data generation')

    parser.add_argument('--process_id',
                        default=0,
                        type=int,
                        help="The processor ID for parallel data generation")

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
                        default=0.3,
                        type=float,
                        help="The largest correlation length")

    parser.add_argument('--rho_min',
                        default=0.01,
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

    parser.add_argument('--E_max',
                        default=1,
                        type=float,
                        help="The maximum modulus value")

    parser.add_argument('--E_min',
                        default=0.1,
                        type=float,
                        help="The minimum modulus value")

    parser.add_argument('--nu_max',
                        default=1,
                        type=float,
                        help="The maximum poisson ratio value")

    parser.add_argument('--nu_min',
                        default=0.1,
                        type=float,
                        help="The minimum poisson ratio value")

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
    np.random.seed(1)
    random.seed(1)

    from datetime import datetime, timedelta

    padded_string = datetime.now().strftime("%Y-%m-%d") + "_mixture_random_field_process" + str(tracer["process_id"]) + "_"

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

    microstructure = microstructure_hierarchical(Sh, tracer)
    for ii in range(tracer["process_id"]):
        microstructure.consume_random(tracer["n_samples"])  # consume random samples
        n_points = np.random.randint(low=tracer["n_pieces_min"], high=tracer["n_pieces_max"] + 1, size=tracer["n_samples"])
        points_list = sample_points(n_points)
        E_values_list, nu_values_list = sample_values(n_points, [-1, 1])
        for jj in range(tracer["n_samples"]):
            _, _, _ = generate_trajectory(tracer["trajectory_interval"], d,
                                                  low=-tracer["trajectory_bound"],
                                                  high=tracer["trajectory_bound"])
    t0 = time.time()
    full_t_list, full_value_list = [], []
    random_field_stats = microstructure.sample_statistics(tracer["n_samples"])
    n_points = np.random.randint(low=tracer["n_pieces_min"], high=tracer["n_pieces_max"] + 1, size=tracer["n_samples"])
    points_list = sample_points(n_points)
    E_values_list, nu_values_list = sample_values(n_points, [-1, 1])

    print("Process %d starts data generation..." % (tracer["process_id"]))
    progress_every = max(1, tracer["n_samples"] // 20)
    for ii in range(tracer["n_samples"]):
        # sample random trajectories
        strain, t_list, value_list= generate_trajectory(tracer["trajectory_interval"], d,
                                                                 low=-tracer["trajectory_bound"],
                                                                 high=tracer["trajectory_bound"])
        full_t_list.append(t_list)
        full_value_list.append(value_list)
        strain_rate = strain.rate()

        # sample random values for Young's modulus and poisson ratio
        E = PC1DMicrostructure(Sh, points_list[ii], E_values_list[ii])
        nu = PC1DMicrostructure(Sh, points_list[ii], nu_values_list[ii])
        E.set_local(gaussian_filter1d(E.get_local(), 5, mode="wrap"))
        nu.set_local(gaussian_filter1d(nu.get_local(), 5, mode="wrap"))
        microstructure_list = microstructure.sample_microstructure(random_field_stats[ii], [hp.vector2Function(m, Sh) for m in [E, nu]])

        # pass in microstructures for system assembly
        cell_problem.set_microstructure(microstructure_list[0], microstructure_list[1])

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
        E_array[ii] = microstructure_list[0].get_local()
        nu_array[ii] = microstructure_list[1].get_local()
        if (ii + 1) % progress_every == 0 and rank == 0:
            t_ii = time.time()
            per_sample_time = (t_ii - t0) / (ii + 1)
            remaining = (tracer["n_samples"] - ii - 1) * per_sample_time
            print("%d samples finished with %1.2fs per sample. Estimated remaining time:" % (
                ii + 1, per_sample_time) + str(timedelta(seconds=remaining)))
            sys.stdout.flush()
    t1 = time.time()
    if rank == 0:
        print("Process " + str(tracer["process_id"]) + " averaged time:", (t1 - t0) / tracer["n_samples"])

    tracer["strain"] = strain_array
    tracer["strain_rate"] = strain_rate_array
    tracer["stress"] = stress_array
    tracer["E"] = E_array
    tracer["nu"] = nu_array
    tracer["nu_prime"] = nu_prime
    tracer["E_prime"] = E_prime
    tracer["kernel"] = kernel
    tracer["time"] = t
    tracer["trajectory_generator_time"] = full_t_list
    tracer["trajectory_generator_strain"] = full_value_list
    material_name = ["E", "nu"]
    for ii, name in enumerate(material_name):
        tracer[name + "_correlation"] = random_field_stats[:, ii, 0]
        tracer[name + "_pointwise_std"] = random_field_stats[:, ii, 1]
    tracer["pieces_points"] = points_list
    tracer["E_pieces_values"] = E_values_list
    tracer["nu_pieces_values"] = nu_values_list

    with open(tracer["output_path"] + padded_string + tracer["file_name"], 'wb') as f:
        pickle.dump(tracer, f)
