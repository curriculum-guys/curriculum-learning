import pandas as pd

class SpecialistData:
    def __init__(self, generation_trials) -> None:
        self.generation_trials = generation_trials
        self.clear()

    @property
    def len(self):
        return len(self.__data)

    @property
    def generations(self):
        return int(self.len / self.generation_trials)

    def set_tuple(self, X, y):
        self.X = X
        self.y = y

    def add(self, data):
        self.__data += data

    def get(self):
        return self.__data

    def clear(self):
        self.__data = []
        self.set_tuple([], [])

    def save(self):
        data = []
        X = self.X
        y = self.y
        for i in range(len(X)):
            data.append(X[i] + [y[i]])
        self.add(data)
        return data

    def transform(self, limit):
        data = pd.DataFrame(self.__data)
        limit *= self.generation_trials
        data = data[-limit:]
        x_cols = data.columns[:-1]
        y_col = data.columns[-1]
        X = data[x_cols]
        y = data[y_col]
        return X, y
