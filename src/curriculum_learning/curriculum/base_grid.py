import numpy as np
import itertools

STATE_RANGES = [
    1.944,
    1.215,
    0.10472,
    0.135088,
    0.10472,
    0.135088,
]

STATIC_CONDITIONS = list(np.zeros(len(STATE_RANGES)))


def generate_base(interval, negative):
    values = []

    for i in range(1, interval+1):
        value = []
        for base_state in STATE_RANGES:
            state = -base_state/i if negative else base_state/i
            value.append(state)

        values.append(value)
    return values

def generate_base_values(base_interval=3):
    interval = int((base_interval - 1)/2)
    values = []

    values += generate_base(interval=interval, negative=True)
    values.append(STATIC_CONDITIONS)
    values += generate_base(interval=interval, negative=False)

    return sorted(values)

def generate_grid():
    base_values = generate_base_values()
    columns_values = np.transpose(base_values)

    conditions = []
    for element in itertools.product(*columns_values):
        conditions.append(element)

    return conditions
