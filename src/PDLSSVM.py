class PDLSSVM:
    def __init__(self, rho, c, c1, c2):
        self.rho = rho
        self.c   = c
        self.c1  = c1
        self.c2  = c2
        
        self.MAX_ITER = 500
        self.ABSTOL   = 1e-4
        self.RELTOL   = 1e-2
        
        
    def fit(self, train_X, train_y_, verbose=False):
        import numpy as np

        self.X = train_X
        y_     = train_y_
        
        start_time = PDLSSVM.get_time()
        
        [m, n] = self.X.shape
        e = np.ones((m, 1))

        alpha = np.random.rand(m, 1)
        beta  = np.random.rand(m, 1)
        z     = np.random.rand(n, 1)
        w     = np.zeros((n, 1))
        u1    = np.zeros((n, 1))
        u2    = np.zeros((m, 1))
        u3    = np.zeros((n, 1))
        t     = 0
        eps1  = 0.001

        y  = np.zeros((m, m))
        np.fill_diagonal(y, y_)

        H  = self.X @ self.X.T
        XY = self.X.T @ y.T
        B  = np.dot(self.X.T, y)
        I  = np.eye(m)

        Iw = np.eye(n)
        Hw = B @ B.T
        ew = np.ones((n, 1))

        Q_inv = np.linalg.inv((1 + 2 * self.rho) * Iw + self.c * Hw)
        # P_inv = np.linalg.inv(y @ H @ y.T + self.rho * (B.T @ B) + (1 / self.c + self.rho) * I)
        P_inv = np.linalg.inv(XY.T @ XY + self.rho * (B.T @ B) + (1 / self.c + self.rho) * I)

        while(t <= self.MAX_ITER) and max([np.linalg.norm(z - B @ beta, 2), np.linalg.norm(w - z, 2), np.linalg.norm(beta - alpha)]) >= eps1:
            theta = 1 / 2 * (z - u1 + B @ beta - u3)
            w     = PDLSSVM.shrinkage(self.c1 / (2 * self.rho) * ew, theta)
            z     = Q_inv @ (self.c * B @ e + self.rho * w + self.rho * u1 + self.rho * B @ beta - self.rho * u3)

            alpha = PDLSSVM.shrinkage(self.c2 / self.rho * e, beta - u2)
            beta  = P_inv * self.rho @ (alpha + u2 + B.T @ z + B.T @ u3 + 1 / self.rho * e)

            u1    = u1 + (w - z)
            u2    = u2 + (alpha - beta)
            u3    = u3 + (z - B @ beta)

            t += 1
    
        end_time = PDLSSVM.get_time()
        total_time = PDLSSVM.time_diff(start_time, end_time)
        
        if verbose:
            print(f"Training Done\nTotal Running Time: {total_time}")
        
        self.y = y
        self.e = e
        self.I = I
        self.w = w
        self.alpha = alpha
        self.z = z
        self.beta = beta
    
        
    def predict(self, test_X, b_flag=True, sign_flag=True, verbose=False):
        import numpy as np

        start_time = PDLSSVM.get_time()
        
        if not sign_flag:
            if self.w == 0:
                pred = np.sign(test_X @ self.z)
            else:
                pred = np.sign(test_X @ self.w)
        else:
            if b_flag:
                XY     = self.X.T @ self.y.T
                ye_inv = np.linalg.pinv(self.y @ self.e)
                b      = ye_inv @ self.e - ye_inv @ (XY.T @ XY + 1 / self.c * self.I) @ self.alpha
                if self.beta.all() == 0:
                    self.w = self.alpha.T @ self.y @ self.X
                    pred   = np.sign(test_X @ self.w.T + b)
                else:
                    self.w = self.beta.T @ self.y @ self.X
                    pred   = np.sign(test_X @ self.w.T + b)
            # without b (intercept)
            else:
                if self.beta.all() == 0:
                    self.w = self.alpha.T @ self.y @ self.X
                    pred   = np.sign(test_X @ self.w.T)
                else:
                    self.w = self.beta.T @ self.y @ self.X
                    pred   = np.sign(test_X @ self.w.T)
                
        sparse_primal = sum(self.w == 0)
        sparse_dual   = sum(self.alpha == 0)
        
        end_time = PDLSSVM.get_time()
        total_time = PDLSSVM.time_diff(start_time, end_time)
        
        if verbose:
            print(f"Prediction Done\nTotal Running Time: {total_time}")
        
        return pred, sparse_primal, sparse_dual
    
    
    @staticmethod
    def get_time():
        import time
        return time.time()
    
    
    @staticmethod
    def time_diff(start_time, end_time):
        return round(end_time - start_time, 4)
    
    
    @staticmethod
    def shrinkage(X, kappa):
        import numpy as np
        return np.maximum(0, np.subtract(X, kappa)) - np.maximum(0, np.subtract(-X, kappa))
    