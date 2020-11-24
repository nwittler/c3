"""
integration testing module for C1 optimization through two-qubits example
"""

import copy
import numpy as np

# Main C3 objects
from c3.c3objs import Quantity as Qty
from c3.c3objs import ParameterMap as Pmap
from c3.experiment import Experiment as Exp
from c3.system.model import Model as Mdl
from c3.generator.generator import Generator as Gnr

# Building blocks
import c3.generator.devices as devices
import c3.system.chip as chip
import c3.signal.pulse as pulse
import c3.signal.gates as gates
import c3.system.tasks as tasks

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.fidelities as fidelities
import c3.libraries.envelopes as envelopes

from c3.optimizers.c1 import C1


qubit_lvls = 3
freq_q1 = 5e9 * 2 * np.pi
anhar_q1 = -210e6 * 2 * np.pi
t1_q1 = 27e-6
t2star_q1 = 39e-6
qubit_temp = 50e-3

q1 = chip.Qubit(
    name="Q1",
    desc="Qubit 1",
    freq=Qty(
        value=freq_q1,
        min=4.995e9 * 2 * np.pi,
        max=5.005e9 * 2 * np.pi,
        unit='Hz 2pi'
    ),
    anhar=Qty(
        value=anhar_q1,
        min=-380e6 * 2 * np.pi,
        max=-120e6 * 2 * np.pi,
        unit='Hz 2pi'
    ),
    hilbert_dim=qubit_lvls,
    t1=Qty(
        value=t1_q1,
        min=1e-6,
        max=90e-6,
        unit='s'
    ),
    t2star=Qty(
        value=t2star_q1,
        min=10e-6,
        max=90e-3,
        unit='s'
    ),
    temp=Qty(
        value=qubit_temp,
        min=0.0,
        max=0.12,
        unit='K'
    )
)

freq_q2 = 5.6e9 * 2 * np.pi
anhar_q2 = -240e6 * 2 * np.pi
t1_q2 = 23e-6
t2star_q2 = 31e-6
q2 = chip.Qubit(
    name="Q2",
    desc="Qubit 2",
    freq=Qty(
        value=freq_q2,
        min=5.595e9 * 2 * np.pi,
        max=5.605e9 * 2 * np.pi,
        unit='Hz 2pi'
    ),
    anhar=Qty(
        value=anhar_q2,
        min=-380e6 * 2 * np.pi,
        max=-120e6 * 2 * np.pi,
        unit='Hz 2pi'
    ),
    hilbert_dim=qubit_lvls,
    t1=Qty(
        value=t1_q2,
        min=1e-6,
        max=90e-6,
        unit='s'
    ),
    t2star=Qty(
        value=t2star_q2,
        min=10e-6,
        max=90e-6,
        unit='s'
    ),
    temp=Qty(
        value=qubit_temp,
        min=0.0,
        max=0.12,
        unit='K'
    )
)

coupling_strength = 20e6 * 2 * np.pi
q1q2 = chip.Coupling(
    name="Q1-Q2",
    desc="coupling",
    comment="Coupling qubit 1 to qubit 2",
    connected=["Q1", "Q2"],
    strength=Qty(
        value=coupling_strength,
        min=-1 * 1e3 * 2 * np.pi,
        max=200e6 * 2 * np.pi,
        unit='Hz 2pi'
    ),
    hamiltonian_func=hamiltonians.int_XX
)


drive = chip.Drive(
    name="d1",
    desc="Drive 1",
    comment="Drive line 1 on qubit 1",
    connected=["Q1"],
    hamiltonian_func=hamiltonians.x_drive
)
drive2 = chip.Drive(
    name="d2",
    desc="Drive 2",
    comment="Drive line 2 on qubit 2",
    connected=["Q2"],
    hamiltonian_func=hamiltonians.x_drive
)

m00_q1 = 0.97  # Prop to read qubit 1 state 0 as 0
m01_q1 = 0.04  # Prop to read qubit 1 state 0 as 1
m00_q2 = 0.96  # Prop to read qubit 2 state 0 as 0
m01_q2 = 0.05  # Prop to read qubit 2 state 0 as 1
one_zeros = np.array([0] * qubit_lvls)
zero_ones = np.array([1] * qubit_lvls)
one_zeros[0] = 1
zero_ones[0] = 0
val1 = one_zeros * m00_q1 + zero_ones * m01_q1
val2 = one_zeros * m00_q2 + zero_ones * m01_q2
min = one_zeros * 0.8 + zero_ones * 0.0
max = one_zeros * 1.0 + zero_ones * 0.2
confusion_row1 = Qty(value=val1, min=min, max=max, unit="")
confusion_row2 = Qty(value=val2, min=min, max=max, unit="")
conf_matrix = tasks.ConfusionMatrix(Q1=confusion_row1, Q2=confusion_row2)

init_temp = 50e-3
init_ground = tasks.InitialiseGround(
    init_temp=Qty(
        value=init_temp,
        min=-0.001,
        max=0.22,
        unit='K'
    )
)

model = Mdl(
    [q1, q2], # Individual, self-contained components
    [drive, drive2, q1q2],  # Interactions between components
    [conf_matrix, init_ground] # SPAM processing
)

model.set_lindbladian(False)
model.set_dressed(True)

sim_res = 100e9 # Resolution for numerical simulation
awg_res = 2e9 # Realistic, limited resolution of an AWG
lo = devices.LO(name='lo', resolution=sim_res)
awg = devices.AWG(name='awg', resolution=awg_res)
mixer = devices.Mixer(name='mixer')

resp = devices.Response(
    name='resp',
    rise_time=Qty(
        value=0.3e-9,
        min=0.05e-9,
        max=0.6e-9,
        unit='s'
    ),
    resolution=sim_res
)

dig_to_an = devices.Digital_to_Analog(
    name="dac",
    resolution=sim_res
)

v2hz = 1e9
v_to_hz = devices.Volts_to_Hertz(
    name='v_to_hz',
    V_to_Hz=Qty(
        value=v2hz,
        min=0.9e9,
        max=1.1e9,
        unit='Hz 2pi/V'
    )
)

generator = Gnr([lo, awg, mixer, v_to_hz, dig_to_an, resp])

t_final = 7e-9   # Time for single qubit gates
sideband = 50e6 * 2 * np.pi
gauss_params_single = {
    'amp': Qty(
        value=0.5,
        min=0.4,
        max=0.6,
        unit="V"
    ),
    't_final': Qty(
        value=t_final,
        min=0.5 * t_final,
        max=1.5 * t_final,
        unit="s"
    ),
    'sigma': Qty(
        value=t_final / 4,
        min=t_final / 8,
        max=t_final / 2,
        unit="s"
    ),
    'xy_angle': Qty(
        value=0.0,
        min=-0.5 * np.pi,
        max=2.5 * np.pi,
        unit='rad'
    ),
    'freq_offset': Qty(
        value=-sideband - 3e6 * 2 * np.pi,
        min=-56 * 1e6 * 2 * np.pi,
        max=-52 * 1e6 * 2 * np.pi,
        unit='Hz 2pi'
    ),
    'delta': Qty(
        value=-1,
        min=-5,
        max=3,
        unit=""
    )
}


gauss_env_single = pulse.Envelope(
    name="gauss",
    desc="Gaussian comp for single-qubit gates",
    params=gauss_params_single,
    shape=envelopes.gaussian_nonorm
)

nodrive_env = pulse.Envelope(
    name="no_drive",
    params={
        't_final': Qty(
            value=t_final,
            min=0.5 * t_final,
            max=1.5 * t_final,
            unit="s"
        )
    },
    shape=envelopes.no_drive
)

lo_freq_q1 = 5e9 * 2 * np.pi + sideband
carrier_parameters = {
    'freq': Qty(
        value=lo_freq_q1,
        min=4.5e9 * 2 * np.pi,
        max=6e9 * 2 * np.pi,
        unit='Hz 2pi'
    ),
    'framechange': Qty(
        value=0.0,
        min= -np.pi,
        max= 3 * np.pi,
        unit='rad'
    )
}
carr = pulse.Carrier(
    name="carrier",
    desc="Frequency of the local oscillator",
    params=carrier_parameters
)

lo_freq_q2 = 5.6e9 * 2 * np.pi + sideband
carr_2 = copy.deepcopy(carr)
carr_2.params['freq'].set_value(lo_freq_q2)

X90p_q1 = gates.Instruction(
    name="X90p",
    t_start=0.0,
    t_end=t_final,
    channels=["d1"]
)
X90p_q2 = gates.Instruction(
    name="X90p",
    t_start=0.0,
    t_end=t_final,
    channels=["d2"]
)
QId_q1 = gates.Instruction(
    name="Id",
    t_start=0.0,
    t_end=t_final,
    channels=["d1"]
)
QId_q2 = gates.Instruction(
    name="Id",
    t_start=0.0,
    t_end=t_final,
    channels=["d2"]
)

X90p_q1.add_component(gauss_env_single, "d1")
X90p_q1.add_component(carr, "d1")
QId_q1.add_component(nodrive_env, "d1")
QId_q1.add_component(copy.deepcopy(carr), "d1")

X90p_q2.add_component(copy.deepcopy(gauss_env_single), "d2")
X90p_q2.add_component(carr_2, "d2")
QId_q2.add_component(copy.deepcopy(nodrive_env), "d2")
QId_q2.add_component(copy.deepcopy(carr_2), "d2")

QId_q1.comps['d1']['carrier'].params['framechange'].set_value(
    (-sideband * t_final) % (2*np.pi)
)
QId_q2.comps['d2']['carrier'].params['framechange'].set_value(
    (-sideband * t_final) % (2*np.pi)
)

Y90p_q1 = copy.deepcopy(X90p_q1)
Y90p_q1.name = "Y90p"
X90m_q1 = copy.deepcopy(X90p_q1)
X90m_q1.name = "X90m"
Y90m_q1 = copy.deepcopy(X90p_q1)
Y90m_q1.name = "Y90m"
Y90p_q1.comps['d1']['gauss'].params['xy_angle'].set_value(0.5 * np.pi)
X90m_q1.comps['d1']['gauss'].params['xy_angle'].set_value(np.pi)
Y90m_q1.comps['d1']['gauss'].params['xy_angle'].set_value(1.5 * np.pi)
Q1_gates = [QId_q1, X90p_q1, Y90p_q1, X90m_q1, Y90m_q1]


Y90p_q2 = copy.deepcopy(X90p_q2)
Y90p_q2.name = "Y90p"
X90m_q2 = copy.deepcopy(X90p_q2)
X90m_q2.name = "X90m"
Y90m_q2 = copy.deepcopy(X90p_q2)
Y90m_q2.name = "Y90m"
Y90p_q2.comps['d2']['gauss'].params['xy_angle'].set_value(0.5 * np.pi)
X90m_q2.comps['d2']['gauss'].params['xy_angle'].set_value(np.pi)
Y90m_q2.comps['d2']['gauss'].params['xy_angle'].set_value(1.5 * np.pi)
Q2_gates = [QId_q2, X90p_q2, Y90p_q2, X90m_q2, Y90m_q2]

all_1q_gates_comb = []
for g1 in Q1_gates:
    for g2 in Q2_gates:
        g = gates.Instruction(
            name="NONE",
            t_start=0.0,
            t_end=t_final,
            channels=[]
        )
        g.name = g1.name + ":" + g2.name
        channels = []
        channels.extend(g1.comps.keys())
        channels.extend(g2.comps.keys())
        for chan in channels:
            g.comps[chan] = {}
            if chan in g1.comps:
                g.comps[chan].update(g1.comps[chan])
            if chan in g2.comps:
                g.comps[chan].update(g2.comps[chan])
        all_1q_gates_comb.append(g)

pmap = Pmap(all_1q_gates_comb, generator, model)

exp = Exp(pmap)

generator.devices['awg'].enable_drag_2()

exp.set_opt_gates(["X90p:Id"])

gateset_opt_map = [
    [
        ("X90p:Id", "d1", "gauss", "amp"),
    ],
    [
        ("X90p:Id", "d1", "gauss", "freq_offset"),
    ],
    [
        ("X90p:Id", "d1", "gauss", "xy_angle"),
    ],
    [
        ("X90p:Id", "d1", "gauss", "delta"),
    ]
]

pmap.set_opt_map(gateset_opt_map)

opt = C1(
    dir_path="/tmp/c3log/",
    fid_func=fidelities.average_infid_set,
    fid_subspace=["Q1", "Q2"],
    pmap=pmap,
    algorithm=algorithms.lbfgs,
    options={"maxfun" : 2},
    run_name="better_X90"
)

opt.set_exp(exp)


def run_optim() -> float:
    """
    Perform the optimization run
    """
    opt.optimize_controls()
    return (opt.current_best_goal)


def test_two_qubits() -> None:
    """
    check if optimization result is below 1e-2
    """
    assert run_optim() < 0.01


def test_signals() -> None:
    """
    Test if the generated signals are correct at the first and last sample and at sample #12.
    """
    gen_signal, ts = generator.generate_signals(X90p_q1)
    assert np.max(gen_signal["d1"]["values"]) == 442864262.3882865
    assert gen_signal["d1"]["values"][0] == 0
    assert gen_signal["d1"]["values"][12] == -18838613.771749068
    assert gen_signal["d1"]["values"][-1] == 47702706.02427947
    assert ts[0] == 5e-12
    assert ts[12] == 1.25e-10
    assert ts[-1] == 6.995e-09


def test_hamiltonian_of_t() -> None:
    signal, ts = generator.generate_signals(X90p_q1)
    cflds_t = signal["d1"]["values"][16]
    hdrift, hks = model.get_Hamiltonians()
    hamiltonian = hdrift.numpy() + cflds_t.numpy() * hks["d1"].numpy()
    precomp_hamiltonian = [
        [-2.37105674e+05+0.j, -4.39488626e+05+0.j,  6.42557156e-09+0.j,
        -1.39929598e+07+0.j,  4.80420449e-08+0.j, -5.12932000e+01+0.j,
        -9.18583639e-10+0.j, -7.52868617e+02+0.j, -4.60493681e-10+0.j,],
        [-4.39488626e+05+0.j,  3.51895367e+10+0.j, -1.05404930e+06+0.j,
        -4.21166702e-06+0.j, -1.39659734e+07+0.j, -3.38001933e-07+0.j,
        -1.70280246e+05+0.j, -3.42845091e-07+0.j, -1.12262365e+03+0.j,],
        [ 8.21276864e-09+0.j, -1.05404930e+06+0.j,  6.88776096e+10+0.j,
        -2.87659692e+01+0.j, -4.35040944e-06+0.j, -1.40273221e+07+0.j,
        6.11788641e-06+0.j, -5.67966729e+05+0.j, -5.92691374e-06+0.j,],
        [-1.39929598e+07+0.j, -4.06274727e-06+0.j, -2.87659692e+01+0.j,
        3.14112585e+10+0.j, -1.97916311e+05+0.j, -2.68973468e-05+0.j,
        -1.97926130e+07+0.j, -5.91563241e-06+0.j, -1.02890811e+02+0.j,],
        [ 5.04042505e-08+0.j, -1.39659734e+07+0.j, -1.84797479e-06+0.j,
        -1.97916311e+05+0.j,  6.65933163e+10+0.j, -2.45595202e+05+0.j,
        1.03870858e-05+0.j, -1.97587093e+07+0.j,  1.03713136e-05+0.j,],
        [-5.12932000e+01+0.j, -4.57618676e-07+0.j, -1.40273221e+07+0.j,
        -2.68926902e-05+0.j, -2.45595202e+05+0.j,  1.00297674e+11+0.j,
        3.15118982e+04+0.j,  7.62930650e-06+0.j, -1.97559869e+07+0.j,],
        [-1.21110840e-09+0.j, -1.70280246e+05+0.j,  6.09999393e-06+0.j,
        -1.97926130e+07+0.j,  1.07057873e-05+0.j,  3.15118982e+04+0.j,
        6.15061801e+10+0.j,  6.36236697e+05+0.j, -2.25524516e-05+0.j,],
        [-7.52868617e+02+0.j, -3.44707736e-07+0.j, -5.67966729e+05+0.j,
        -5.93053357e-06+0.j, -1.97587093e+07+0.j,  8.58298081e-06+0.j,
        6.36236697e+05+0.j,  9.66811548e+10+0.j,  1.30140463e+06+0.j,],
        [-5.01826729e-10+0.j, -1.12262365e+03+0.j, -5.92318845e-06+0.j,
        -1.02890811e+02+0.j,  1.04309182e-05+0.j, -1.97559869e+07+0.j,
        -2.25571082e-05+0.j,  1.30140463e+06+0.j,  1.30377086e+11+0.j,],
    ]

    assert (hamiltonian - precomp_hamiltonian < 1e-15).any()


def test_propagation() -> None:
    signal, ts = generator.generate_signals(X90p_q1)
    propagator = exp.propagation(signal, ts, "X90p:Id")
    precomputed = np.array(
        [
            [-5.42470209e-02 - 7.55118401e-01j, 1.31475007e-04 - 1.66893081e-04j,
             5.81482026e-08 + 4.56344704e-08j, 6.51945328e-01 - 3.21728860e-02j,
             1.20669716e-04 + 5.09331324e-05j, 3.19775048e-08 + 1.04614907e-07j,
             -9.41458634e-03 - 2.63990622e-02j, 3.55609615e-05 - 2.18958182e-04j,
             -6.42952047e-07 - 1.07156481e-06j]
        ]
     )

    assert ((propagator.numpy()[3] - precomputed) < 1e-8).all()