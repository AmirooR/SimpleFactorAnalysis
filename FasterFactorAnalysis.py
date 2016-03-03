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
        self.Sigma = np.random.random_sample((self.D,))
        for i in range(self.num_iterations):
            print 'Iteration %d' % i
            self.iterate()


    def iterate(self):
        # E-step
        sigma_invD = 1/self.Sigma
        PhiT_SigmaInv = np.matrix(
                np.dstack([np.asarray(self.Phi.T[i,:])*np.asarray(sigma_invD) for i in range(self.Phi.shape[1])] ) ).T
        A = inv( PhiT_SigmaInv * self.Phi + np.eye(self.n_components))
        EH = A * PhiT_SigmaInv*(self.X_minus_M)
        EHHT = np.dstack(np.asarray(A + EH[:,i]*EH[:,i].T) for i in range(self.I))

        # M-step
        self.Phi =  (self.X_minus_M * EH.T)  * inv( np.sum( EHHT, axis=2) )
        A1 = np.asarray( [np.dot(self.X_minus_M[i,:],self.X_minus_M[i,:].T) for i in
                range(self.D)] )
        B = EH * self.X_minus_M.T
        A2 = np.asarray( [np.dot(self.Phi[i,:], B[:,i]) for i in
            range(self.D)] )
        self.Sigma = np.matrix( (A1 - A2)/self.I)


