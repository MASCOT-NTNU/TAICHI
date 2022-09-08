import numpy as np
from scipy import sparse
from sksparse.cholmod import cholesky
import os
FILEPATH = os.getcwd() + "/GMRF/"
DEFAULT_NUM_SAMPLES = 250


class spde:
    def __init__(self):
        """Initialize model

        Args:
            model (int, optional): Doesn't do anything. Defaults to 2.
            reduce (bool, optional): Reduced grid size used if set to True. Defaults to False.
            method (int, optional): If model should contain fixed effects on the SINMOD mean. Defaults to 1.
            prev (bool, optional): Loading previous model (used to clear memory)
        """
        self.M = 50
        self.N = 55
        self.P = 6
        self.n = self.M*self.N*self.P
        self.mu = np.load(FILEPATH + 'models/prior.npy')
        self.Stot = sparse.eye(self.n).tocsc()

        tmp = np.load(FILEPATH + 'models/Q.npy')
        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n))
        
        self.Q_fac = self.cholesky(self.Q)
        self.sigma = np.load(FILEPATH + 'models/sigma.npy') # measurement noise robot [0] and SINMOD fitted noise [1]
        tmp = np.load(FILEPATH + 'models/grid.npy') # loading grid data
        self.lats = tmp[:,2] # min & max latitudes
        self.lons = tmp[:,3] # min & max longitudes
        self.x = tmp[:,0] # min & max x grid location
        self.y = tmp[:,1] # min & max y grid locations
        self.setThreshold()
    
    def sample(self,n = 1):
        """Samples the GMRF. Only used to test.

        Args:
            n (int, optional): Number of realizations. Defaults to 1.
        """
        z = np.random.normal(size = self.n*n).reshape(self.n,n)
        data = self.mu[:, np.newaxis] + self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n*n).reshape(self.n,n)*self.sigma
        return(data)

    def cholesky(self,Q):
        """A function calculating the cholesky decoposition of a positive definite precision matrix of the GMRF. Uses the c++ package Cholmod

        Args:
            Q ([N,N] sparse csc matrix): Sparse matrix from scipy.sparse
        """
        try:
            Q_fac = cholesky(Q)
        except:
            print("Supernodal or negative definite precision matrix... continue")
            return(-1)
        else:
            return(Q_fac)

    def candidate(self,ks,n=DEFAULT_NUM_SAMPLES):
        """Returns the marginal variance of all location given that a location (ks) in the GMRF has been measured.
        Uses Monte Carlo samples to calculate the marginal variance for all locations.

        Args:
            ks (integer): Array of indicies of possible candidate locations
            n (int, optional): Number of samples used in the Monte Carlo estimate. Defaults to 40.
        """
        z1 = np.random.normal(size = self.n*n).reshape(self.n,n)
        x = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z1,use_LDLt_decomposition=False))
        res = np.zeros((self.n,ks.size))
        z2 = np.random.normal(size = n).reshape(1,n)*self.sigma
        for i in range(ks.size):
            S = self.Stot[ks[i],:]
            V = self.Q_fac.solve_A(S.transpose().tocsc())
            W = (S@V)[0,0] + self.sigma**2
            U = V.transpose()/W
            c = S@x - z2
            x_ = x -  U.transpose()@c 
            res[:,i] = x_.var(axis=1)
        return(res)

    def update(self, rel, ks):
        """Update mean and precision of the GMRF given some measurements in the field.

        Args:
            rel ([k,1]-array): k number of measurements of the GMRF. (k>0).
            ks ([k,]-array): k number of indicies describing the index of the measurment in the field.
        """
        mu = self.mu.reshape(-1,1)
        if ks.size>0:
            S = self.Stot[ks,:]
            self.Q = self.Q + S.transpose()@S*1/self.sigma**2
            self.Q_fac.cholesky_inplace(self.Q)

            mu = mu - self.Q_fac.solve_A(S.transpose().tocsc())@(S@mu - rel)*1/self.sigma**2
        self.mu = mu.reshape(-1)

    def mvar(self,Q_fac = None, n=DEFAULT_NUM_SAMPLES):
        """Monte Carlo Estimate of the marginal variance of a GMRF.

        Args:
            Q_fac (Cholmod object, optional): Cholmod cholesky object. Defaults to None.
            n (int, optional): Number of samples used in the Monte Varlo estimate. Defaults to 40.
        """
        if Q_fac is None:
            Q_fac = self.Q_fac
        z = np.random.normal(size = self.n*n).reshape(self.n,n)
        data = Q_fac.apply_Pt(Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
        return(data.var(axis = 1) + self.sigma**2)

    def setThreshold(self):
        """Set threshold for Excursion set
        """
        ind = np.load(FILEPATH + 'models/boundary.npy')
        self.threshold = self.mu[ind].mean()
        print('Treshold is set to %.2f'%(self.threshold))

    def getThreshold(self) -> np.ndarray:
        """ Get updated threshold """
        ind = np.load(FILEPATH + 'models/boundary.npy')
        threshold = self.mu[ind].mean()
        return threshold

