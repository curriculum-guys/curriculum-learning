import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from curriculum_learning.specialist.logs import SpecialistLogs
from curriculum_learning.specialist.model import SpecialistModel
from curriculum_learning.specialist.metrics import SpecialistMetrics
from curriculum_learning.specialist.data import SpecialistData
from curriculum_learning.specialist.sampling import SpecialistSamplingControl

class Specialist:
    def __init__(
        self,
        expected_score=0.8,
        fit_batch_size=1,
        score_batch_size=1,
        start_generation=1,
        generation_trials=10,
        fit_historical_data=True
    ):
        # Internal Interfaces
        self.scaler = StandardScaler()
        self.model = SpecialistModel()
        self.data = SpecialistData(generation_trials)
        self.metrics = SpecialistMetrics(self.model.labels, 4)
        self.sampling = SpecialistSamplingControl()
        self.logger = SpecialistLogs()

        # Evolution Control
        self.start_generation = start_generation
        self.generation = start_generation
        self.expected_score = expected_score
        self.fit_batch_size = fit_batch_size
        self.score_batch_size = score_batch_size
        self.fit_historical_data = fit_historical_data

        # Fit Start
        self.__fit_counter_reset()

        # Analytics
        self.actual_score = None
        self.fit_predicted_labels_proportion = None
        self.score_predicted_labels_proportion = None
        self.actual_score_metrics = self.metrics.null_matrix('specialist')

    ### Logging Methods
    def create_log(self, name):
        self.logger.create(name)

    @property
    def __generation_log(self):
        return {
            'score': str(self.actual_score),
            'metrics': str(self.actual_score_metrics.items()),
            'predicted labels': f'fit_proportion: {self.fit_predicted_labels_proportion} | score_proportion: {self.score_predicted_labels_proportion}',
            'cycling': f' fit_start: {self.fit_start} | score_start: {self.score_start}',
            'data': f'length: {self.data.len}'
        }
    ### Logging Methods

    ### Evolution Control Methods
    @property
    def non_historical_qualified(self):
        if self.fit_start:
            return (self.generation+1 - self.fit_start) == self.fit_batch_size

    @property
    def qualified(self):
        return self.actual_score >= self.expected_score and not self.fit_start

    @property
    def fit_batch_qualified(self):
        if self.fit_start:
            return self.data.generations >= self.fit_batch_size

    @property
    def score_batch_qualified(self):
        if self.score_start:
            return self.data.generations >= self.score_batch_size
    ### Evolution Control Methods

    ### Couter Methods
    def __fit_counter_reset(self):
        self.fit_start = self.generation
        self.score_start = None

    def __score_counter_reset(self):
        self.fit_start = None
        self.score_start = self.generation
        self.data.clear()
    ### Couter Methods

    ### Data Transformation Methods
    def transform_data(self, X, y):
        return self.sampling.resample(X, y)

    def normalize_data(self, data):
        self.scaler = self.scaler.partial_fit(data)
        return self.scaler.transform(data)

    def get_labels(self, y):
        median = np.median(y)
        return ['good' if p > median else 'bad' for p in y]

    def get_prediction_proportion(self, X, normalize=True):
        y_pred = list(self.predict(X, normalize=normalize))
        y_good = sum([1 if g == 'good' else 0 for g in y_pred])
        y_bad = sum([1 if b == 'bad' else 0 for b in y_pred])
        return y_good / (y_good + y_bad)

    def get_metrics(self, X, y, target='specialist'):
        y_pred = list(self.predict(X))
        y_true = list(self.get_labels(y))
        return self.metrics.build_matrix(target, y_true, y_pred)

    def set_data(self, X, y):
        if not self.fit_historical_data and self.fit_start:
            if self.non_historical_qualified:
                self.data.set_tuple(X, y)
        else:
            self.data.set_tuple(X, y)
    ### Data Transformation Methods

    ### Model Methods
    def score(self, X, y):
        labels = self.get_labels(y)
        self.actual_score = accuracy_score(labels, self.predict(X))
        self.actual_score_metrics = self.get_metrics(X, y)
        self.score_predicted_labels_proportion = self.get_prediction_proportion(X)

    def fit(self, X, y):
        normalized_X = self.normalize_data(X)
        X_resampled, y_resampled = self.transform_data(normalized_X, y)
        labels = self.get_labels(y_resampled)
        self.model.fit(X_resampled, labels)
        self.fit_predicted_labels_proportion = self.get_prediction_proportion(normalized_X, normalize=False)

    def predict(self, X, normalize=True):
        if normalize:
            normalized_X = self.normalize_data(X)
            return self.model.predict(normalized_X)
        else:
            return self.model.predict(X)
    ### Model Methods
    
    ### Generation Processing
    def process_generation(self, X, y):
        self.set_data(X, y)
        self.logger.update(self.generation, self.__generation_log)

        if self.score_batch_qualified:
            X, y = self.data.transform(limit=self.score_batch_size)
            self.score(X, y)
            if not self.qualified:
                self.__fit_counter_reset()
        elif self.fit_batch_qualified:
            X, y = self.data.transform(limit=self.fit_batch_size)
            self.fit(X, y)
            self.__score_counter_reset()
        self.generation += 1

        data = self.data.save()
        return data
    ### Generation Processing
