from curriculum_learning.specialist.interface import Specialist
from data_interfaces.stats.specialist import SpecialistStats
from data_interfaces.conditions.base import BaseConditions
from sklearn.exceptions import NotFittedError

class SpecialistManager:
    def __init__(self, name, environment, seed, systematic_conditions=None) -> None:
        self.generation = 0
        self.name = name
        self.environment = environment
        self.seed = seed
        self.specialists = {}
        self.integrated_interfaces = {}
        self.systematic_interfaces = {}
        self.systematic_conditions = systematic_conditions
        self.reset_data()

    def read_config(self, config):
        for specialist_name, specialist_config in config.items():
            self.add_specialist(specialist_name, specialist_config)

    def process_specialist_stats(self, specialist):
        return [
            specialist.actual_score,
            'fit' if specialist.fit_start else 'score',
            specialist.fit_predicted_labels_proportion,
            specialist.score_predicted_labels_proportion,
            specialist.actual_score_metrics.get('specialist_tn'),
            specialist.actual_score_metrics.get('specialist_fp'),
            specialist.actual_score_metrics.get('specialist_fn'),
            specialist.actual_score_metrics.get('specialist_tp'),
        ]

    def is_time_to_start(self, specialist):
        return self.generation >= specialist.start_generation

    def is_fitted(self, specialist):
        return specialist.model.fitted

    def add_integrated_interface(self, name):
        save_dir = f'/{self.name}_manager/{name}_stats'
        integrated_interface = SpecialistStats(self.environment, self.seed, save_dir)
        self.integrated_interfaces[name] = integrated_interface

    def add_systematic_interface(self, name):
        save_dir = f'/{self.name}_manager/{name}_base_conditions'
        systematic_interface = BaseConditions(self.environment, self.seed, len(self.systematic_conditions), save_dir)
        self.systematic_interfaces[name] = systematic_interface

    def add_specialist(self, name, config):
        specialist = Specialist(**config)
        specialist.create_log(name)
        self.specialists[name] = specialist
        self.add_integrated_interface(name)
        if self.systematic_conditions:
            self.add_systematic_interface(name)

    def get_specialist(self, name):
        return self.specialists.get(name)

    def remove_specialist(self, name):
        self.specialists.pop(name)

    def save_integrated_metrics(self, name):
        specialist = self.get_specialist(name)
        data_interface = self.integrated_interfaces.get(name)
        if self.is_time_to_start(specialist):
            data = self.process_specialist_stats(specialist)
            data_interface.save_stg(
                data,
                stage=self.generation
            )

    def save_systematic_metrics(self, name):
        process_data = lambda predicted: [0 if p == 'bad' else 1 for p in predicted]
        specialist = self.get_specialist(name)
        data_interface = self.systematic_interfaces.get(name)
        if self.is_fitted(specialist):
            try:
                predicted = specialist.predict(self.systematic_conditions)
                data_interface.save_stg(
                    process_data(predicted),
                    stage=self.generation
                )
            except NotFittedError:
                print(f'[{name}] Specialist Model not fitted yet, skipping evaluation...')

    def save_stg(self):
        for specialist_name in self.specialists.keys():
            self.save_integrated_metrics(specialist_name)
            if self.systematic_conditions:
                self.save_systematic_metrics(specialist_name)

    def save(self):
        for specialist_name in self.specialists.keys():
            self.integrated_interfaces[specialist_name].save()
            if self.systematic_conditions:
                self.systematic_interfaces[specialist_name].save()

    def process_data(self, data):
        X, y = [], []
        for env in data:
            X.append(env[:-1])
            y.append(env[-1])
        return X, y

    def reset_data(self):
        self.update_fit_data(None)
        self.update_score_data(None)

    def update_fit_data(self, data):
        self.fit_data = data

    def update_score_data(self, data):
        self.score_data = data
    
    def update_data(self, data):
        self.update_fit_data(data)
        self.update_score_data(data)

    def __process_generation_data(self, specialist):
        if specialist.fit_start:
            return self.process_data(self.fit_data)
        elif specialist.score_start:
            return self.process_data(self.score_data)

    def process_generation(self):
        for specialist_name in self.specialists.keys():
            specialist = self.get_specialist(specialist_name)
            X, y = self.__process_generation_data(specialist)
            if self.is_time_to_start(specialist):
                specialist.process_generation(X, y)
