import time
import numpy as np
from curriculum_learning.curriculum.logs import CurriculumLogs

class CurriculumManager:
    def __init__(self, name, enabled, specialist, reset_function, trials, proportion, margin=1000):
        self.name = name
        self.enabled = enabled
        self.specialist = specialist
        self.reset_function = reset_function
        self.trials = trials
        self.margin = margin
        self.generation = 1
        self.creation_time = None
        self.proportion = proportion
        self.logger = CurriculumLogs()
        self.counter = 0
        self.create_log(name)

    def create_log(self, name):
        self.logger.create(name)

    @property
    def __generation_log(self):
        return {
            'creation_time': self.creation_time,
            'proportion': self.proportion,
            'hard_tasks': len(self.hard_tasks),
            'easy_tasks': len(self.easy_tasks),
        }

    def update_log(self):
        self.logger.update(
            self.generation,
            self.__generation_log
        )

    @property
    def active(self):
        init_generation = self.specialist.start_generation + self.specialist.fit_batch_size + self.specialist.score_batch_size + 1
        return (self.generation > init_generation) and self.specialist.qualified

    def predict(self, tasks):
        X = [task for task in tasks] # First position is reserved for seed values
        y = self.specialist.predict(X)
        return X, y

    def episodes(self, n, seed):
        return [self.reset_function(seed + i) for i in range(n)]

    def resample(self, X, y):
        easy, hard = [], []
        for i in range(len(y)):
            if y[i] == 'good':
                easy.append(X[i])
            else:
                hard.append(X[i])
        return easy, hard

    def curriculum(self, seed):
        if self.enabled and self.active:
            start_time = time.time()
            n = self.trials * self.margin
            tasks = self.episodes(n, seed)
            X, y = self.predict(tasks)
            self.easy_tasks, self.hard_tasks = self.resample(X, y)
            self.counter = self.margin
            end_time = time.time()

            self.creation_time = end_time - start_time
            self.update_log()

    def select_trials(self, seed):
        if self.counter > 0:
            np.random.seed(seed)
            trials = []
            for i in range(self.trials):
                if i < (self.trials*self.proportion):
                    e = np.random.randint(0, len(self.easy_tasks))
                    trials.append(self.easy_tasks[e])
                else:
                    e = np.random.randint(0, len(self.hard_tasks))
                    trials.append(self.hard_tasks[e])
            return trials
        else:
            return self.episodes(self.trials, seed)
