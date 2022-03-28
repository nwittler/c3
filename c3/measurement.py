import tensorflow as tf

from typing import List

from c3.c3objs import Quantity
from c3.parametermap import ParameterMap
from c3.propagation import PWC
from c3.signal.gates import Instruction
from c3.utils.tf_utils import tf_unitary_overlap


class UnitaryInfid:
    def __init__(self, pmap: ParameterMap, subspace):
        self.prop = PWC(pmap)
        self.pmap = pmap
        index = []
        for name in subspace:
            index.append(pmap.model.names.index(name))  # type: ignore
        self.index = index
        dims = pmap.model.dims  # type: ignore
        self.dims = dims

    def measure(self, parameters, instructions: List[Instruction]):
        self.pmap.set_parameters_scaled(parameters)
        infid = 0
        for instr in instructions:
            actual = self.prop.compute_unitary(instr)
            ideal = instr.get_ideal_gate(self.dims)
            infid += 1 - tf_unitary_overlap(actual, ideal)
        return infid / len(instructions)


class RabiExperiment:
    def __init__(self, qubit_freq) -> None:
        self.qubit_freq: Quantity = qubit_freq
        self.pmap = ParameterMap()
        self.pmap._pars = {
            "amp": Quantity(100e6, unit="Hz 2pi"),
            "freq": Quantity(5e9, unit="Hz 2pi"),
            "time": Quantity(10e-9, unit="s"),
        }
        self.pmap.opt_map = [[("amp")], [("freq")], [("time")]]

    def measure(self, parameters):
        self.pmap.set_parameters_scaled(parameters)
        amp = self.pmap.get_parameter(["amp"]).get_value()
        freq = self.pmap.get_parameter(["freq"]).get_value()
        t = self.pmap.get_parameter(["time"]).get_value()
        q_freq = self.qubit_freq.get_value()
        diff_sq = (q_freq - freq) ** 2
        return tf.cos(tf.sqrt(diff_sq + amp**2) / 2 * t) / tf.sqrt(
            1 + diff_sq / (amp**2)
        )
