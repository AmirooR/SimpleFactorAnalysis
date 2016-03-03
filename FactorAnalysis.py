import numpy as np
from numpy.linalg import inv

class FactorAnalyzer:
    """ Factor Analysis implementation """
    def __init__(self, n_components, num_iterations=10, sigma_mul = 5):
        self.n_components = n_components
        self.num_iterations = num_iterations
        self.sigma_mul = sigma_mul

    def fit(self, X):
        self.D = X.shape[0]
        self.I = X.shape[1]
        self.X = np.matrix(X) #TODO: matrix necessary?
        self.M = np.matrix(np.mean(X, axis=1)).T
        self.MExtended = np.matrix( np.dstack([np.asarray(self.M) for i in range(self.I)]) )
        self.X_minus_M = self.X - self.MExtended
        self.Phi = np.matrix(np.random.random_sample((self.D,self.n_components))) #TODO: initialize
        self.Sigma = np.matrix( self.sigma_mul*np.diag(np.random.random_sample( (self.D,))) ) #TODO: check this too!
        for i in range(self.num_iterations):
            print 'Iteration %d\n' % i
            self.iterate()


    def iterate(self):
        # E-step
        sigma_inv = np.diag(np.diag(self.Sigma)**-1)
        A = inv( self.Phi.T * sigma_inv * self.Phi + np.eye(self.n_components))
        EH = A * self.Phi.T*sigma_inv*(self.X_minus_M)
        EHHT = np.dstack(np.asarray(A + EH[:,i]*EH[:,i].T) for i in range(self.I))

        # M-step
        self.Phi =  (self.X_minus_M * EH.T)  * inv( np.sum( EHHT, axis=2) )
        self.Sigma = np.diag(np.diag( self.X_minus_M * self.X_minus_M.T - self.Phi * EH * self.X_minus_M.T))/self.I # TODO: check this too!


