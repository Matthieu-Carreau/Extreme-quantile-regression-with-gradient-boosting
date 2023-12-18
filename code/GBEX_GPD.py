# Gradient boosting for Generalized Pareto Distribution parameters estimation

import numpy as np

from scipy.stats import genpareto

from ModifiedTree import ModifiedTree

class GBEX_GPD():
    """
    Objects of these class are used to estimate the parameters simga(x) and gamma(x)
    of a GPD modelling the tail of the distribution of Y conditionally to X
    """
    def __init__(self, B=100, D_sig=2, D_gam=2, lamb_scale=0.01, lamb_ratio=10, s=0.5, L_sig=10, L_gam=10):
        # Number of trees for each parameters (sigma end gamma)
        self.B = B 

        # Maximum depth of the trees
        self.D_sig = D_sig
        self.D_gam = D_gam

        # Minimum number of datapoints per leaf
        self.L_sig = L_sig
        self.L_gam = L_gam

        # Learning rates
        self.lamb_sig = lamb_scale
        self.lamb_gam = lamb_scale / lamb_ratio

        # Fraction of the dataset used at each iteration
        self.s = s

        # Lists of regression trees
        self.sig_trees = []
        self.gam_trees = []

        # First constant estimates of the parameters
        self.sigma_0 = None
        self.gamma_0 = None

        # Boolean to verify if nan values were encountered during training
        self.corrected_nans = False

    def predict(self, X):
        """Use the lists of trees to predict the parameters for the inputs X, whose shape is (n, d)"""
        sig_pred = np.zeros(X.shape[0]) + self.sigma_0
        gam_pred = np.zeros(X.shape[0]) + self.gamma_0
        for s_tree, g_tree in zip(self.sig_trees, self.gam_trees):
            sig_pred += self.lamb_sig*s_tree.predict(X)
            gam_pred += self.lamb_gam*g_tree.predict(X)

        return sig_pred, gam_pred


    def fit(self, X, Z, sigma_0=None, gamma_0=None):
        """Perform the algorithm 1 to get estimators of sigma(x) and gamma(x)"""
        
        # Initial parameters
        if sigma_0 is None or gamma_0 is None:
            self.gamma_0, _, self.sigma_0 = genpareto.fit(Z, floc=0)
            print("Initialization with sigma={} and gamma={}".format(self.sigma_0, self.gamma_0))
            
        else:
            self.sigma_0 = sigma_0
            self.gamma_0 = gamma_0

        # Reset the estimators
        self.sig_trees = []
        self.gam_trees = []

        n = X.shape[0]
        for b in range(self.B):
            # Step 1: draw a subsample
            S_b = np.random.choice(np.arange(n), int(self.s*n), replace=False)
            X_b = X[S_b]
            Z_b = Z[S_b]

            # Steps 2 and 3: fit regression trees on the residuals and add them to the lists
            sig_tree, gam_tree = self.get_trees(X_b, Z_b)
            
            # Step 4: add the new trees to the lists
            self.sig_trees.append(sig_tree)
            self.gam_trees.append(gam_tree)
        
        if self.corrected_nans:
            print("WARNING: nan values were encountered during training")


    def get_trees(self, X, Z):
        """Perform steps 2, 3 and 4 of the algorithm 1"""
        # Step 2 of algo 1: compute the deviance derivatives

        assert all(Z > 0)

        sig_pred, gam_pred = self.predict(X)
        r_sig = (1 - (1+gam_pred)*Z / (sig_pred+gam_pred*Z)) / sig_pred
        r_gam = -np.log(1+gam_pred*Z/sig_pred) / gam_pred**2 + (1+1/gam_pred)*Z / (sig_pred+gam_pred*Z)

        nan_idx = np.isnan(r_gam)
        if any(nan_idx):
            self.corrected_nans = True
            r_gam[nan_idx] = 1

        # Step 3 of algo 1: fit regression trees
        sig_tree = ModifiedTree(max_depth=self.D_sig, min_samples_leaf=self.L_sig)
        gam_tree = ModifiedTree(max_depth=self.D_gam, min_samples_leaf=self.L_gam)

        sig_tree.fit(X, r_sig)
        gam_tree.fit(X, r_gam)
    
        # Second derivatives
        d2_sig = (Z/sig_pred + (Z - sig_pred)/(sig_pred + gam_pred*Z)) / (sig_pred*(sig_pred + gam_pred*Z))

        d2_gam = 2/gam_pred**3 * np.log(1+gam_pred*Z/sig_pred) 
        d2_gam -= 2*Z / (gam_pred**2*(sig_pred+gam_pred*Z)) 
        d2_gam -= (1+1/gam_pred)*Z**2 / (sig_pred+gam_pred*Z)**2

        nan_idx = np.isnan(d2_gam)
        if any(nan_idx):
            self.corrected_nans = True
            d2_gam[nan_idx] = 1

        # Update the values of the leaves    
        sig_tree.updateLeaves(X, r_sig, d2_sig)
        gam_tree.updateLeaves(X, r_gam, d2_gam)

        return sig_tree, gam_tree