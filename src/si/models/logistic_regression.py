import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function

class LogisticRegression(Model):
    """
    logistic regression is a logistic model using the L2 regularization
    this model solves the logistic regression problem using an adapted Gradient Descent Technique
    """
    def __init__(self, l2_penalty = 0.0, alpha = 0.01, max_iter = 1000, patience = 10, scale=True):
        
        super().__init__
        self.l2_penalty = self.l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale
        
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = cost_history {}
        self.dataset = dataset
        