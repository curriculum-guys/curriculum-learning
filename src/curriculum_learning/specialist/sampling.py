from imblearn.under_sampling import ClusterCentroids

class SpecialistSamplingControl:
    def __init__(self) -> None:
        self.model = ClusterCentroids(random_state=42)
    
    def resample(self, X, y):
        rX, ry = self.model.fit_resample(X, y)
        return rX, ry
