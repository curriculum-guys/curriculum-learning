import numpy as np
import itertools

DPOLE_STATES_LENGTH = 6
DPOLE_STATE_RANGES = [
    (-1.944, 1.944),
    (-1.215, 1.215),
    (-0.10472, 0.10472),
    (-0.135088, 0.135088),
    (-0.10472, 0.10472),
    (-0.135088, 0.135088)
]

BIDEDAL_STATES_LENGTH = 200
BIPEDAL_STATE_RANGES = [(0,3) for _ in range(BIDEDAL_STATES_LENGTH)]

def get_experiment_info(experiment):
    lengths = dict(
        xbipedal=BIDEDAL_STATES_LENGTH,
        xdpole=DPOLE_STATES_LENGTH
    )
    ranges = dict(
        xbipedal=BIPEDAL_STATE_RANGES,
        xdpole=DPOLE_STATE_RANGES
    )
    return lengths.get(experiment), ranges.get(experiment)

def generate_states(ranges, n_states):
    states = []
    for state_range in ranges:
        start, stop = state_range
        possible_states = np.linspace(start, stop, n_states)
        states.append(possible_states)
    return states

def generate_grid(experiment, n_states=3):
    _, ranges = get_experiment_info(experiment)
    states = generate_states(ranges, n_states)

    conditions = []
    for element in itertools.product(*states):
        conditions.append(element)
    return conditions
