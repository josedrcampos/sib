import numpy as np

class RidgeRegression(Model):
    def __init__(self, l2_penalty=1.0, alpha=0.01, max_iter=1000, patience=10, scale=True):
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale
        
        # Estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}
        
    def cost(self, y_true, y_pred):
        # Calculates cost function with L2 regulation
        m = len(y_true)
        mse_cost = (1/(2*m)) * np.sum((y_true - y_pred) ** 2)
        l2_cost = (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)
        return mse_cost  l2_cost
    
    def _fit(self, x, y):
        # estimates the theta and theta_zero coefficients, mean, std and cost_history
        x = np.array(x, dtype = float)
        y = np.array(y, dtype = float).reshape(-1,1)
        m, n = X.shape

        # Scaling
        
        if self.scale:
            self.mean = np.mean(x, axis = 0)
            self.std = np.std(x, axis = 0)
            self.std[self.std == 0] = 1.0
            x_scaled = (x -self.mean) / self.std
        else:
            self.mean = np.zeros(n)
            self.std = np.ones(n)
            x_scaled = x.copy()
            
        self.theta = np.zeros