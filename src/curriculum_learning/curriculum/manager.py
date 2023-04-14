import time
from random import random
from curriculum_learning.curriculum.logs import CurriculumLogs

class CurriculumManager:
    def __init__(self, name, specialist, reset_function, trials):
        self.name = name
        self.specialist = specialist
        self.reset_function = reset_function
        self.trials = trials
        self.generation = 1
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
            'actual_curriculum': self.actual_curriculum,
            'auto_fill': f'EASY: {self.easy_fill} - HARD: {self.hard_fill}'
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

    def predict_conditions(self, margin=100):
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
            return conditions[:expected], i
        return None, -1

    def process_conditions(self, proportion):
        easy, hard = self.predict_conditions()

        expected_easy = int(proportion * self.trials)
        expected_hard = int(self.trials - expected_easy)

        easy_conditions, self.easy_fill = self.fill_gaps(easy, expected_easy)
        hard_conditions, self.hard_fill = self.fill_gaps(hard, expected_hard)

        if len(hard_conditions) == expected_hard and len(easy_conditions) == expected_easy:
            self.actual_curriculum = list(easy_conditions) + list(hard_conditions)
        else:
            self.actual_curriculum = None

    def create_curriculum(self, proportion):
        if self.active:
            self.proportion = proportion

            start_time = time.time()
            self.process_conditions(proportion)
            end_time = time.time()

            self.creation_time = end_time - start_time
            self.update_log()
            return self.actual_curriculum