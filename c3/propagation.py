import tensorflow as tf

from typing import Callable, Dict, List

from c3.parametermap import ParameterMap
from c3.signal.gates import Instruction
from c3.utils.tf_utils import tf_matmul_left


def Propagation(method="pwc"):
    methods = {"pwc": PWC}
    return methods[method]()


class PWC:
    def __init__(self, pmap: ParameterMap):
        self.dt = tf.constant(0.01e-9, dtype=tf.complex128)
        self.folding_stack: Dict[int, List[Callable]] = {}
        self.gen = pmap.generator.generate_signals  # type: ignore
        self.h_of_t = pmap.model.get_Hamiltonian  # type: ignore
        self.get_FR = pmap.model.get_Frame_Rotation  # type: ignore
        # self.precompute(pmap.instructions)

    # def _compute_folding_stack(self , n_steps) -> None:
    #     stack = []
    #     while n_steps > 1:
    #         if not n_steps % 2:  # is divisable by 2
    #             stack.append(_tf_matmul_n_even)
    #         else:
    #             stack.append(_tf_matmul_n_odd)
    #         n_steps = np.ceil(n_steps / 2)
    #     self.folding_stack[n_steps] = stack

    # def precompute(self, instructions: Dict[str, Instruction]) -> None:
    #     for name, instr in instructions.items():
    #         n_steps = int(instr.get_duration() / self.dt)
    #         if n_steps not in self.folding_stack:
    #             self._compute_folding_stack(n_steps)

    def compute_dus(self, instr: Instruction) -> tf.Tensor:
        signal = self.gen(instr)
        hamiltonians = self.h_of_t(signal)
        return tf.linalg.expm(-1.0j * hamiltonians * self.dt)

    def compute_unitary(self, instr: Instruction) -> tf.Tensor:
        dUs = self.compute_dus(instr)
        self.compute_FR(instr)
        return self.FR @ tf_matmul_left(dUs)

    def compute_FR(self, instr: Instruction):
        # TODO change LO freq to at the level of a line
        freqs = {}
        framechanges = {}
        for line, ctrls in instr.comps.items():
            # TODO calculate properly the average frequency that each qubit sees
            offset = 0.0
            for ctrl in ctrls.values():
                if "freq_offset" in ctrl.params.keys():
                    if ctrl.params["amp"] != 0.0:
                        offset = ctrl.params["freq_offset"].get_value()
            freqs[line] = tf.cast(
                ctrls["carrier"].params["freq"].get_value() + offset,
                tf.complex128,
            )
            framechanges[line] = tf.cast(
                ctrls["carrier"].params["framechange"].get_value(),
                tf.complex128,
            )
        t_final = tf.constant(instr.t_end - instr.t_start, dtype=tf.complex128)
        self.FR = self.get_FR(t_final, freqs, framechanges)
