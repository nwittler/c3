from typing import List

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
