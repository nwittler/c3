"""Object that deals with the sensitivity test."""

import os
import shutil
import pickle
import itertools
import numpy as np
import tensorflow as tf
from c3.optimizers.optimizer import Optimizer
from c3.utils.utils import log_setup


class SET(Optimizer):
    """Object that deals with the sensitivity test.

    Parameters
    ----------
    dir_path : str
        Filepath to save results
    fom : callable
        Figure of merit
    sampling : str
        Sampling method from the sampling library
    batch_sizes : list
        Number of points to select from each dataset
    sweep_map : list
        Identifiers to be swept
    state_labels : list
        Identifiers for the qubit subspaces
    algorithm : callable
        From the algorithm library
    options : dict
        Options to be passed to the algorithm
    same_dyn : boolean
        ?
    run_name : str
        User specified name for the run, will be used as root folder
    """

    def __init__(
        self,
        dir_path,
        fom,
        estimator_list,
        sampling,
        batch_sizes,
        pmap,
        state_labels=None,
        sweep_map=None,
        sweep_bounds=None,
        algorithm=None,
        run_name=None,
        same_dyn=False,
        options={},
    ):
        """Initiliase."""
        super().__init__(pmap=pmap, algorithm=algorithm)
        self.fom = fom
        self.estimator_list = estimator_list
        self.sampling = sampling
        self.batch_sizes = batch_sizes
        self.state_labels = state_labels
        self.sweep_map = sweep_map
        self.pmap = pmap
        self.sweep_bounds = sweep_bounds
        self.options = options
        self.inverse = False
        self.learn_data = {}
        self.same_dyn = same_dyn
        self.__dir_path = dir_path
        self.__run_name = run_name

    def log_setup(self) -> None:
        """
        Create the folders to store data.
        """
        dir_path = os.path.abspath(self.__dir_path)
        run_name = self.__run_name
        if run_name is None:
            run_name = "_".join(["Sensitivity", self.algorithm.__name__])
        self.logdir = log_setup(dir_path, run_name)
        self.logname = "sensitivity.log"
        if isinstance(self.exp.created_by, str):
            shutil.copy2(self.exp.created_by, self.logdir)
        if isinstance(self.created_by, str):
            shutil.copy2(self.created_by, self.logdir)

    def read_data(self, datafiles):
        # TODO move common methods of sensitivity and c3 to super class
        """
        Open data files and read in experiment results.

        Parameters
        ----------
        datafiles : list of str
            List of paths for files that contain learning data.
        """
        for target, datafile in datafiles.items():
            with open(datafile, "rb+") as file:
                self.learn_data[target] = pickle.load(file)

    def select_from_data(self, batch_size):
        """
        Select a subset of each dataset to compute the goal function on.

        Parameters
        ----------
        batch_size : int
            Number of points to select

        Returns
        -------
        list
            Indeces of the selected data points.
        """
        learn_from = self.learn_from
        sampling = self.sampling
        indeces = sampling(learn_from, batch_size)
        if self.inverse:
            return list(set(all) - set(indeces))
        else:
            return indeces

    def sensitivity(self):
        """
        Run the sensitivity analysis.

        """

        print("Initial parameters:")
        print(self.exp.pmap.print_parameters())
        for ii in range(len(self.sweep_map)):
            self.dfname = "data.dat"
            self.opt_map = [self.sweep_map[ii]]
            self.options["bounds"] = [self.sweep_bounds[ii]]
            print(f"C3:STATUS:Sweeping {self.opt_map}: {self.sweep_bounds[ii]}")
            self.log_setup()
            self.start_log()
            print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")
            x_init = self.exp.pmap.get_parameters()
            try:
                self.algorithm(
                    x_init,
                    fun=self.fct_to_min,
                    fun_grad=self.fct_to_min_autograd,
                    grad_lookup=self.lookup_gradient,
                    options=self.options,
                )
            except KeyboardInterrupt:
                pass
            self.exp.pmap.set_parameters(x_init)

    def goal_run(self, current_params):
        """
        Evaluate the figure of merit for the current model parameters.

        Parameters
        ----------
        val : tf.Tensor
            Current model parameters

        Returns
        -------
        tf.float64
            Figure of merit

        """
        exp_values = []
        exp_stds = []
        sim_values = []
        exp_shots = []
        goals = []
        seq_weigths = []
        count = 0
        # TODO: seq per point is not constant. Remove.

        for target, data in self.learn_data.items():

            self.learn_from = data["seqs_grouped_by_param_set"]
            self.gateset_opt_map = data["opt_map"]
            indeces = self.select_from_data(self.batch_sizes[target])

            for ipar in indeces:
                # if count % 100 == 0:
                #     print("count: " + str(count))

                count += 1
                m = self.learn_from[ipar]
                gateset_params = m["params"]
                gateset_opt_map = self.gateset_opt_map
                m_vals = m["results"]
                m_stds = m["results_std"]
                m_shots = m["shots"]
                sequences = m["seqs"]
                num_seqs = len(sequences)
                if target == "all":
                    num_seqs = len(sequences) * 3

                self.pmap.set_parameters(current_params, [self.sweep_map])
                self.pmap.model.update_model()

                self.pmap.set_parameters(gateset_params, gateset_opt_map)
                # We find the unique gates used in the sequence and compute
                # only them.
                self.exp.opt_gates = list(set(itertools.chain.from_iterable(sequences)))
                self.exp.compute_propagators()
                pops = self.exp.evaluate(sequences)
                sim_vals, _ = self.exp.process(
                    labels=self.state_labels[target], populations=pops
                )

                exp_stds.extend(m_stds)
                exp_shots.extend(m_shots)

                goal = self.fom(
                    m_vals,
                    tf.stack(sim_vals),
                    tf.constant(m_stds, dtype=tf.float64),
                    tf.constant(m_shots, dtype=tf.float64),
                )
                goals.append(goal.numpy())
                seq_weigths.append(num_seqs)
                sim_values.extend(sim_vals)
                exp_values.extend(m_vals)

                with open(self.logdir + self.logname, "a") as logfile:
                    logfile.write(
                        f"\n  Parameterset {ipar + 1}, #{count} of {len(indeces)}:\n"
                        f"{self.pmap.str_parameters(gateset_opt_map)}\n"
                    )
                    logfile.write(
                        "Sequence    Simulation  Experiment  Std           Shots"
                        "    Diff\n"
                    )

                for iseq in range(len(sequences)):
                    m_val = np.reshape(m_vals[iseq], (-1,))
                    m_std = np.reshape(m_stds[iseq], (-1,))
                    shots = np.reshape(m_shots[iseq], (-1,))
                    sim_val = sim_vals[iseq].numpy()
                    with open(self.logdir + self.logname, "a") as logfile:
                        for ii in range(len(sim_val)):
                            logfile.write(
                                f"{iseq + 1:8}    "
                                f"{float(sim_val[ii]):8.6f}    "
                                f"{float(m_val[ii]):8.6f}    "
                                f"{float(m_std[ii]):8.6f}    "
                                f"{float(shots[0]):8}    "
                                f"{float(m_val[ii]-sim_val[ii]):8.6f}\n"
                            )
                        logfile.flush()

        goal = tf.reduce_mean(goals).numpy()

        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write("\nFinished batch with ")
            logfile.write("{}: {}\n".format(self.fom.__name__, goal))
            print("{}: {}".format(self.fom.__name__, goal))
            for est in self.estimator_list:
                val = float(est(exp_values, sim_values, exp_stds, exp_shots).numpy())
                logfile.write("{}: {}\n".format(est.__name__, val))
                # print("{}: {}".format(est.__name__, val))
            print("")
            logfile.flush()

        self.optim_status["params"] = [
            par.numpy().tolist() for par in self.exp.pmap.get_parameters()
        ]
        self.optim_status["goal"] = goal
        self.evaluation += 1
        return goal
