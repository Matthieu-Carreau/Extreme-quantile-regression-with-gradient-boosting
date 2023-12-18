import numpy as np

from GBEX_GPD import GBEX_GPD

import quantile_forest as qf

class Extreme_quantile_regressor():
    """Implements algorithm 2"""
    def __init__(self, tau_0, B=100, D_sig=2, D_gam=2, lamb_scale=0.01, lamb_ratio=10, s=0.5, L_sig=10, L_gam=10):
        # Level of the intermediate quantile
        self.tau_0 = tau_0 

        # Estimator of the intermediate quantile
        self.intermediate_Q_est = qf.RandomForestQuantileRegressor(n_estimators=100, default_quantiles=tau_0, min_samples_leaf=20, max_depth=3)

        # Estimator of the GPD parameters for the tail distribution
        self.GPD_est = GBEX_GPD(B=B, D_sig=D_sig, D_gam=D_gam, lamb_scale=lamb_scale, lamb_ratio=lamb_ratio, s=s, L_sig=L_sig, L_gam=L_gam)
    
    def fit(self, X, Y):
        # Estimation of the intermediate quantiles for each x
        self.intermediate_Q_est.fit(X, Y)
        inter_q_pred = self.intermediate_Q_est.predict(X, self.tau_0)

        # Keep the exceedances
        ind_exceed = Y > inter_q_pred
        X_exc = X[ind_exceed]
        Z = Y[ind_exceed] - inter_q_pred[ind_exceed]

        # Fit the GPD model
        self.GPD_est.fit(X_exc, Z)

    def predict(self, X, tau):
        """
        Parameters:
        X: array of covariates of shape (n, d)
        tau: real parameter such that tau_0 < tau < 1
        or array of shape (n) containing one threshold for
        each datapoint
        """
        inter_q_pred = self.intermediate_Q_est.predict(X)
        sig_pred, gamma_pred = self.GPD_est.predict(X)

        # Compute the extreme quantile estimation 
        # using equation (5)
        numerator = ((1-tau)/(1-self.tau_0))**(-gamma_pred) - 1
        fraction = sig_pred * numerator / gamma_pred
        Q_est = inter_q_pred + fraction

        return Q_est
