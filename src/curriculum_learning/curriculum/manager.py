import time
from random import random
from curriculum_learning.curriculum.logs import CurriculumLogs

class CurriculumManager:
    def __init__(self, name, specialist, reset_function, trials):
        self.name = name
        self.specialist = specialist
        self.reset_function = reset_function
        self.trials = trials
        self.generation = 0
        self.creation_time = None
        self.proportion = None
        self.logger = CurriculumLogs()
        self.create_log(name)

    def create_log(self, name):
        self.logger.create(name)

    @property
    def __generation_log(self):
        return {
            'creation_time': self.creation_time,
            'proportion': self.proportion,
            'actual_curriculum': self.actual_curriculum
        }

    def update_log(self):
        self.logger.update(
            self.generation,
            self.__generation_log
        )

    @property
    def active(self):
        init_generation = self.specialist.start_generation + self.specialist.fit_batch_size + self.specialist.score_batch_size + 1
        return (self.generation >= init_generation) and self.specialist.qualified

    def generate_conditions(self, n_conditions, random_conditions=False):
        r = random.randint(1, n_conditions) if random_conditions else 1
        return [self.reset_function(i * r) for i in range(n_conditions)]

    def predict_conditions(self, margin=10):
        raw = self.generate_conditions(self.trials * margin)
        predicted = self.specialist.predict(raw)
        easy, hard = [], []
        for i in range(len(raw)):
            if predicted[i] == 'bad':
                hard.append(raw[i])
            elif predicted[i] == 'good':
                easy.append(raw[i])
        return easy, hard

    def fill_gaps(self, conditions, expected):
        current = len(conditions)

        if current > 0:
            i = 0
            while (current + i) < expected:
                conditions.append(conditions[i % current])
                i += 1
            return conditions[:expected]

    def process_conditions(self, proportion):
        easy, hard = self.predict_conditions()

        expected_easy = int(proportion * self.trials)
        easy_conditions = self.fill_gaps(easy, expected_easy)
        expected_hard = int((1-proportion) * self.trials)
        hard_conditions = self.fill_gaps(hard, expected_hard)

        if hard_conditions and easy_conditions:
            self.actual_curriculum = list(easy_conditions) + list(hard_conditions)
        else:
            self.actual_curriculum = None

    def create_curriculum(self, proportion):
        if self.active:
            start_time = time.time()
            self.process_conditions(proportion)
            end_time = time.time()

            self.proportion = proportion
            self.creation_time = end_time - start_time
            self.update_log()
            return self.actual_curriculum
