"""Object that deals with experiment design."""

import os
import shutil
import time

import tensorflow as tf

from c3.optimizers.optimizer import Optimizer
from c3.utils.utils import log_setup
from c3.parametermap import ParameterMap

class ExperimentDesign(Optimizer):
    """
    Derive the optimal control parameters to improve knowledge of a model parameter.

    Parameters
    ----------
    dir_path : str
        Filepath to save results
    fid_func : callable
        infidelity function to be minimized
    fid_subspace : list
        Indices identifying the subspace to be compared
    pmap : ParameterMap
        Identifiers for the parameter vector
    callback_fids : list of callable
        Additional fidelity function to be evaluated and stored for reference
    algorithm : callable
        From the algorithm library
        Save plots of control signals
    store_unitaries : boolean
        Store propagators as text and pickle
    options : dict
        Options to be passed to the algorithm
    run_name : str
        User specified name for the run, will be used as root folder
    """

    def __init__(
            self,
            dir_path,
            pmap: ParameterMap,
            model_params,
            algorithm=None,
            store_unitaries=False,
            options={},
            run_name=None,
            interactive=True,
            gateset_opt_map=None,
            opt_gates=None,
            num_control_sets=1
    ) -> None:
        super().__init__(
            pmap=pmap,
            algorithm=algorithm,
            store_unitaries=store_unitaries,
        )
        if gateset_opt_map:
            pmap.set_opt_map([[tuple(par) for par in pset] for pset in gateset_opt_map])

        self.control_dim = len(gateset_opt_map)
        self.num_control_sets = num_control_sets
        self.options = options
        self.__dir_path = dir_path
        self.__run_name = run_name
        self.interactive = interactive
        for model_param in model_params:
            if not "-".join(model_param) in pmap.get_full_params():
                raise Exception(f"C3:ERROR:{model_params} not defined the model.")
        self.model_param_map = [model_params]
        self.opt_gates = opt_gates

    def log_setup(self) -> None:
        """
        Create the folders to store data.
        """
        dir_path = os.path.abspath(self.__dir_path)
        run_name = self.__run_name
        if run_name is None:
            run_name = "_".join(["ED", "Fisher", self.algorithm.__name__])
        self.logdir = log_setup(dir_path, run_name)
        self.logname = "open_loop.log"
        if isinstance(self.exp.created_by, str):
            shutil.copy2(self.exp.created_by, self.logdir)
        if isinstance(self.created_by, str):
            shutil.copy2(self.created_by, self.logdir)

    def load_model_parameters(self, adjust_exp: str) -> None:
        self.pmap.load_values(adjust_exp)
        self.pmap.model.update_model()
        shutil.copy(adjust_exp, os.path.join(self.logdir, "adjust_exp.log"))

    def optimize_controls(self, setup_log: bool = True) -> None:
        """
        Apply a search algorithm to your gateset given a fidelity function.
        """
        if setup_log:
            self.log_setup()
        self.start_log()
        self.exp.set_enable_store_unitaries(self.store_unitaries, self.logdir)
        print(f"C3:STATUS:Saving as: {os.path.abspath(self.logdir + self.logname)}")

        # The first set of control parameters
        x_init_0 = self.pmap.get_parameters_scaled()
        # We duplicate the initial control set the required number of times.
        # TODO: Provide an option for how to initialize all controls.
        x_init = tf.stack([x_init_0] * self.num_control_sets)

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
        self.end_log()

    def goal_run(self, current_params: tf.Tensor) -> tf.float64:
        """
        Evaluate the goal function for current parameters.

        Parameters
        ----------
        current_params : tf.Tensor
            Vector representing the current parameter values.

        Returns
        -------
        tf.float64
            Value of the goal function
        """

        init_state = [0] * self.pmap.model.tot_dim
        init_state[0] = 1
        ground_state = tf.transpose(tf.constant([init_state], dtype=tf.complex128))

        fisher_info = 0
        # Current params is a flatten vector multiple control parameter sets,
        # so we reshape and iterate.
        # TODO: Vectorize loop
        for controls in tf.reshape(current_params, (-1, self.control_dim)):
            with tf.GradientTape() as t1:
                with tf.GradientTape() as t2:
                    self.pmap.set_parameters_scaled(controls)
                    # Here we need to have another loop that samples over a model parameter distribution
                    model_par = self.pmap.get_parameters_scaled(self.model_param_map)[0]
                    model_par_value = tf.constant(model_par)
                    t2.watch(model_par_value)
                    t1.watch(model_par_value)
                    self.pmap.set_parameters_scaled([model_par_value], self.model_param_map)
                    print("Calculating propagator...")
                    propagators = self.exp.compute_propagators()
                    propagator = propagators[self.opt_gates[0]]
                    measurement = tf.abs(tf.linalg.adjoint(ground_state) @ propagator @ ground_state)
                    print(f"Measured {measurement}")
                print("Calculating first order gradient...")
                d_measurement = t2.gradient(measurement, model_par_value)
            print("Calculating second order gradient...")
            d2_measurement = t1.gradient(d_measurement, model_par_value)
            fisher_info -= measurement * d2_measurement

        goal = 1 / fisher_info

        with open(self.logdir + self.logname, "a") as logfile:
            logfile.write(f"\nEvaluation {self.evaluation + 1} returned:\n")
            logfile.write(f"goal: {self.fid_func.__name__}: {float(goal)}\n")
            logfile.flush()

        self.optim_status["params"] = [
            par.numpy().tolist() for par in self.pmap.get_parameters()
        ]
        self.optim_status["goal"] = float(goal)
        self.optim_status["time"] = time.asctime()
        self.evaluation += 1
        return goal
