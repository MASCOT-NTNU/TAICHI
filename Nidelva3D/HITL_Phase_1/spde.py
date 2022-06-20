import numpy as np
from scipy import sparse
from sksparse.cholmod import cholesky
from Config.Config import FILEPATH
DEFAULT_NUM_SAMPLES = 250 # 150 is too much

class spde:
    def __init__(self, model = 2, reduce = False, method = 1, prev = False):
        """Initialize model

        Args:
            model (int, optional): Doesn't do anything. Defaults to 2.
            reduce (bool, optional): Reduced grid size used if set to True. Defaults to False.
            method (int, optional): If model should contain fixed effects on the SINMOD mean. Defaults to 1.
            prev (bool, optional): Loading previous model (used to clear memory)
        """
        self.M = 45
        self.N = 45
        self.P = 11
        if prev:
            self.loadModel()
        else:
            self.model = model
            self.method = method
            self.reduced = reduce
            if self.reduced:
                self.P = 7
            self.n = self.M*self.N*self.P
                
        
        if self.method == 1:
            if self.reduced:
                self.mu3 = np.load(FILEPATH + 'models/prior_small.npy')
                tmp = np.load(FILEPATH + 'models/S1r.npy')
                self.Stot = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n)) 
            else: 
                self.mu3 = np.load(FILEPATH + 'models/prior.npy')
                tmp = np.load(FILEPATH + 'models/S1.npy')
                self.Stot = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n)) 
            if not prev:
                self.mu = self.mu3
        else:
            if self.reduced:
                self.mu3 = np.load(FILEPATH + 'models/prior_small.npy')
                tmp = np.load(FILEPATH + 'models/S2r.npy')
                self.Stot = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n+2)) 
            else: 
                self.mu3 = np.load(FILEPATH + 'models/prior.npy')
                tmp = np.load(FILEPATH + 'models/S2.npy')
                self.Stot = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n+2)) 
            if not prev:
                self.mu = self.mu3
                self.mu2 = np.hstack([np.zeros(self.n),0,1]).reshape(-1,1) # Mean of random effect and betas
                self.beta0 = 0
                self.beta1 = 1
                
        if not prev:
            if self.method == 1:
                if self.model == 1:
                    if self.reduced:
                        tmp = np.load(FILEPATH + 'models/SA1r.npy') 
                        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n))
                    else:
                        tmp = np.load(FILEPATH + 'models/SA1.npy') 
                        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n)) 
                else:
                    if self.reduced:
                        tmp = np.load(FILEPATH + 'models/NA1r.npy')
                        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n))
                    else:
                        tmp = np.load(FILEPATH + 'models/NA1.npy')
                        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n)) 
            else:
                if self.model == 1:
                    if self.reduced:
                        tmp = np.load(FILEPATH + 'models/SA2r.npy')
                        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n+2,self.n+2))
                    else:
                        tmp = np.load(FILEPATH + 'models/SA2.npy')
                        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n+2,self.n+2)) 
                else:
                    if self.reduced:
                        tmp = np.load(FILEPATH + 'models/NA2r.npy')
                        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n+2,self.n+2))
                    else:
                        tmp = np.load(FILEPATH + 'models/NA2.npy')
                        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n+2,self.n+2)) 
        
        self.Q_fac = cholesky(self.Q)
        self.sigma = np.load(FILEPATH + 'models/sigma.npy') # measurement noise robot [0] and SINMOD fitted noise [1]
        tmp = np.load(FILEPATH + 'models/grid.npy') # loading grid data
        self.lats = tmp[:,2] # min & max latitudes
        self.lons = tmp[:,3] # min & max longitudes
        self.x = tmp[:,0] # min & max x grid location
        self.y = tmp[:,1] # min & max y grid locations
        self.threshold = 27
        

    def sample(self,n = 1):
        """Samples the GMRF. Only used to test.

        Args:
            n (int, optional): Number of realizations. Defaults to 1.
        """
        if self.method == 2:
            z = np.random.normal(size = (self.n+2)*n).reshape((self.n+2),n)
            data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
            data = data[:self.n,:] + self.mu3.reshape(-1,1) + np.random.normal(size = self.n*n).reshape(self.n,n)*self.sigma[1]
        else:
            z = np.random.normal(size = self.n*n).reshape(self.n,n)
            data = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) + np.random.normal(size = self.n)*self.sigma[1] + self.mu3
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

    def save(self):
        """Saves conditional mean and marginal variance
        """
        np.save(FILEPATH + 'models/mucond.npy', self.mu)
        np.save(FILEPATH + 'models/smvarcond.npy', self.mvar())
        if self.method == 2:
            np.save(FILEPATH + 'models/mu2cond.npy',self.mu2)

    def candidate(self,ks,n=DEFAULT_NUM_SAMPLES):
        """Returns the marginal variance of all location given that a location (ks) in the GMRF has been measured.
        Uses Monte Carlo samples to calculate the marginal variance for all locations.

        Args:
            ks (integer): Array of indicies of possible candidate locations
            n (int, optional): Number of samples used in the Monte Carlo estimate. Defaults to 40.
        """
        if self.method == 2:
            z1 = np.random.normal(size = (self.n+2)*n).reshape((self.n+2),n) 
        else:
            z1 = np.random.normal(size = (self.n)*n).reshape((self.n),n) 
        x = self.Q_fac.apply_Pt(self.Q_fac.solve_Lt(z1,use_LDLt_decomposition=False)) 
        res = np.zeros((self.n,ks.size))
        z2 = np.random.normal(size = n).reshape(1,n)*self.sigma[0]
        for i in range(ks.size):
            S = self.Stot[ks[i],:]
            V = self.Q_fac.solve_A(S.transpose().tocsc())
            W = (S@V)[0,0] + self.sigma[0]**2
            U = V.transpose()/W
            c = S@x - z2
            x_ = x -  U.transpose()@c
            if self.method == 2:
                x_ = x_[:self.n,:] + x_[self.n,:] + self.mu3[:,np.newaxis]*x_[self.n+1,:][np.newaxis,:]
            res[:,i] = x_.var(axis=1)
        return(res)

    def update(self, rel, ks):
        """Update mean and precision of the GMRF given some measurements in the field.

        Args:
            rel ([k,1]-array): k number of measurements of the GMRF. (k>0).
            ks ([k,]-array): k number of indicies describing the index of the measurment in the field. 
        """
        if ks.size>0:
            if self.method == 2:
                S = self.Stot[ks,:]
                self.Q = self.Q + S.transpose()@S*1/self.sigma[0]**2
                self.Q_fac.cholesky_inplace(self.Q)
                self.mu2 = self.mu2 - self.Q_fac.solve_A(S.transpose().tocsc())@(S@self.mu2 - rel)*1/self.sigma[0]**2
                self.mu = self.mu2[:self.n,0] + self.mu2[self.n,0] + self.mu3*self.mu2[self.n+1,0]
                self.beta0 = self.mu2[self.n]
                self.beta1 = self.mu2[self.n+1]
            else:
                mu = self.mu.reshape(-1,1)
                S = self.Stot[ks,:]
                self.Q[ks,ks] = self.Q[ks,ks] + 1/self.sigma[0]**2
                self.Q_fac.cholesky_inplace(self.Q)
                mu = mu - self.Q_fac.solve_A(S.transpose()@(S@mu-rel)*1/self.sigma[0]**2)
                self.mu = mu.flatten()

    def mvar(self,Q_fac = None, n=DEFAULT_NUM_SAMPLES):
        """Monte Carlo Estimate of the marginal variance of a GMRF.

        Args:
            Q_fac (Cholmod object, optional): Cholmod cholesky object. Defaults to None.
            n (int, optional): Number of samples used in the Monte Varlo estimate. Defaults to 40.
        """
        if Q_fac is None:
            Q_fac = self.Q_fac
        if self.method == 2:
            z = np.random.normal(size = (self.n+2)*n).reshape((self.n+2),n)
            data = Q_fac.apply_Pt(Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
            data = data[:self.n,:] + data[self.n,:] + self.mu3[:,np.newaxis]*data[self.n+1,:][np.newaxis,:]
        else:
            z = np.random.normal(size = self.n*n).reshape(self.n,n)
            data = Q_fac.apply_Pt(Q_fac.solve_Lt(z,use_LDLt_decomposition=False)) 
        return(data.var(axis = 1))

    def resetQ(self):
        """Resets Q to initial values for the same grid
        """
        self.M = 45
        self.N = 45
        self.P = 11
        self.n = self.M*self.N*self.P
        tmp = np.load(FILEPATH + 'models/SINMOD-NA.npy') # load fitted precision matrix Q
        self.Q = sparse.csc_matrix((np.array(tmp[:,2],dtype = "float32"), (tmp[:,0].astype('int32'), tmp[:,1].astype('int32'))), shape=(self.n,self.n))
        self.Q_fac = cholesky(self.Q)
        if self.reduced:
            self.reduce()
        self.Stot = sparse.eye(self.n).tocsc()
        if self.method == 2:
            self.Q.resize((self.n + 2, self.n + 2))
            self.Q = self.Q.tolil()
            self.Q[self.n, self.n] = 0.02
            self.Q[self.n + 1, self.n + 1] = 10
            self.Q = self.Q.tocsc()
            self.Q_fac = cholesky(self.Q)
            
            self.Stot.resize((self.n,self.n+2))
            self.Stot = self.Stot.tolil()
            self.Stot[:,self.n] = np.ones(self.n)
            self.Stot[:,self.n+1] = self.mu3
            self.Stot = self.Stot.tocsc()
            self.mu = self.mu2[:self.n,0] + self.mu2[self.n,0] + self.mu3*self.mu2[self.n+1,0]
        print("Q: ", self.Q.shape)

    def setThreshold(self):
        """Set threshold for Excursion set
        """
        if self.reduced:
            ind = np.load(FILEPATH + 'models/boundary_reduced.npy')
        else:
            ind = np.load(FILEPATH + 'models/boundary.npy')
        self.threshold = self.mu[ind].mean()
        print('Treshold is set to %.2f'%(self.threshold))
        np.save("models/threshold.npy", self.threshold)

    def setCoefLM(self):
        """Find fixed effects in simple linear model linking SINMOD to predicted GMRF field
        """
        np.save(FILEPATH + 'models/Google_coef.npy',np.polyfit(self.mu3,self.mu,1))
        print("Saved google coefficients.")

    def loadPrev(self):
        if self.method == 2:
            self.mu2 = np.load(FILEPATH + 'models/mu2cond.npy')
            self.mu = self.mu2[:self.n,0] + self.mu2[self.n,0] + self.mu3*self.mu2[self.n+1,0]
            print("Successfully loaded previous model!")
        else:
            print('Wrong method... Nothing is updated.')

    def postProcessing(self):
        self.setThreshold()
        self.save()
        self.resetQ()
        self.setCoefLM()
        print("Post processing is successfuilly!")
        
    def saveModel(self):
        """Method to save the current model for future loading at current update
        """
        Q = self.Q.tocoo()
        r = Q.row
        c = Q.col
        v = Q.data
        if self.method == 1:
            np.savez(FILEPATH + "models/currentModel.npz", model = self.model, method = self.method, reduce = self.reduced, Qr = r, Qc = c, Qv = v, mu = self.mu)
        else:
            np.savez(FILEPATH + "models/currentModel.npz", model = self.model, method = self.method, reduce = self.reduced, Qr = r, Qc = c, Qv = v, mu = self.mu, mu2 = self.mu2)
        
    def loadModel(self):
        tmp = np.load(FILEPATH + 'models/currentModel.npz')
        self.model = tmp['model']*1
        self.method = tmp['method']*1
        self.reduced = (tmp['reduce']*1).astype('bool')
        self.P = 11
        if self.reduced:
            self.P = 7
        self.n = self.M*self.N*self.P
        self.mu = tmp['mu']*1
        if self.method ==2:
            self.mu2 = tmp['mu2']*1
            self.beta0 = self.mu2[self.n]
            self.beta1 = self.mu2[self.n+1]
            self.Q = sparse.csc_matrix((np.array(tmp['Qv']*1,dtype = "float32"), ((tmp['Qr']*1).astype('int32'), (tmp['Qc']*1).astype('int32'))), shape=(self.n+2,self.n+2))
        else:
            self.Q = sparse.csc_matrix((np.array(tmp['Qv']*1,dtype = "float32"), ((tmp['Qr']*1).astype('int32'), (tmp['Qc']*1).astype('int32'))), shape=(self.n,self.n))



t = spde()
#%%
import time
print("start")
t1 = time.time()
t.update(rel = np.random.uniform(25, 30, 10000), ks=np.random.randint(0, len(t.mu), 10000))

t2 = time.time()
print("time consunmed; ", t2 - t1)


