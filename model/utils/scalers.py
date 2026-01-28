import numpy as np

class NpStandardScaler:
    def __init__(self, axis=0):
        self.axis = axis
        self.mean = None
        self.std = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=self.axis)
        self.std = np.std(X, axis=self.axis) + 1e-5 
    
    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("The scaler has not been fitted yet. Call 'fit' before 'transform'.")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
        