import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm
from tqdm import tqdm

def get_GPD_dataset(n, dim=2, sigma_function=lambda x:1, gamma_function=lambda x:1e-3, plot=False):
    """
    Generates a synthetic dataset (X_i, Y_i)

    Arguments:
    -n: number of datapoints
    -dim: dimension of the space for the values of X
    -sigma_function: function that maps x to sigma(x)
    -gamma_function: function that maps x to gamma(x)
    -plot: boolean paramter to plot the datapoints
    """
    # Generate the covariates
    X = (np.random.random((n, dim))-0.5)*8

    # Generate the parameters 
    sigma = np.array([sigma_function(x) for x in X])
    gamma = np.array([gamma_function(x) for x in X])

    unif = np.random.random(n) # Uniform random variables
    # Apply the inverse cdf of the GPD to get Y 
    Y = sigma/gamma*(unif**(-gamma)-1) 

    if plot:
        plt.scatter(X[:, 0], X[:, 1], c=sigma)
        plt.title("True values of sigma")
        plt.colorbar()
        plt.show()
        
        plt.scatter(X[:, 0], X[:, 1], c=gamma)
        plt.title("True values of gamma")
        plt.colorbar()
        plt.show()

        plt.scatter(X[:, 0], X[:, 1], c=Y, marker=".")
        plt.title("Y")
        plt.colorbar()
        plt.show()

    return X, Y

def get_gaussian_dataset(n, dim=2, mu_function=lambda x:0, sigma_function=lambda x:1, plot=False):
    """
    Generates a synthetic dataset (X_i, Y_i)

    Arguments:
    -n: number of datapoints
    -dim: dimension of the space for the values of X
    -mu_function: function that maps x to mu(x)
    -sigma_function: function that maps x to sigma(x)
    -plot: boolean paramter to plot the datapoints
    """
    # Generate the covariates
    X = (np.random.random((n, dim))-0.5)*8

    # Generate the parameters 
    mu = np.array([mu_function(x) for x in X])
    sigma = np.array([sigma_function(x) for x in X])

    # Apply the inverse cdf of the GPD to get Y 
    Y = np.array([m + s*np.random.randn() for m, s in zip(mu, sigma)])

    if plot:
        plt.scatter(X[:, 0], X[:, 1], c=mu)
        plt.title("True values of mu")
        plt.colorbar()
        plt.show()

        plt.scatter(X[:, 0], X[:, 1], c=sigma)
        plt.title("True values of sigma")
        plt.colorbar()
        plt.show()

        plt.scatter(X[:, 0], X[:, 1], c=Y, marker=".")
        plt.title("Y")
        plt.colorbar()
        plt.show()

    return X, Y

def visualize_predictions(gbex, x1_lim=(-4, 4), x2_lim=(-4, 4), n_pts_x1=100, n_pts_x2=100):
    """
    Only use when dim=2, visualize the predictions of sigma and gamma
    made by the estimator 'bgex' over the plane
    """
    # Create the grid of values
    x1_val = np.linspace(x1_lim[0], x1_lim[1], n_pts_x1)
    x2_val = np.linspace(x2_lim[0], x2_lim[1], n_pts_x2)

    x1, x2 = np.meshgrid(x1_val, x2_val)
    xx = np.concatenate((np.expand_dims(x1, axis=2), np.expand_dims(x2, axis=2)), axis=2)

    # Make the predictions
    sig_pred, gam_pred = gbex.predict(xx.reshape(-1, 2))
    sig_pred = sig_pred.reshape(n_pts_x1, n_pts_x2)
    gam_pred = gam_pred.reshape(n_pts_x1, n_pts_x2)

    # Plot the predictions
    x1_labels = np.round(np.linspace(x1_lim[0], x1_lim[1], 10), 1)
    x1_ticks = np.linspace(0, n_pts_x1, 10, dtype='int')
    x2_labels = np.round(np.linspace(x2_lim[0], x2_lim[1], 10), 1)
    x2_ticks = np.linspace(0, n_pts_x2, 10, dtype='int')

    # For sigma
    plt.imshow(sig_pred, origin='lower')
    plt.xticks(ticks=x1_ticks, labels=x1_labels)
    plt.yticks(ticks=x2_ticks, labels=x2_labels)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Prediction of sigma")
    plt.colorbar()
    plt.show()
    
    # For gamma
    plt.imshow(gam_pred, origin='lower')
    plt.xticks(ticks=x1_ticks, labels=x1_labels)
    plt.yticks(ticks=x2_ticks, labels=x2_labels)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Prediction of gamma")
    plt.colorbar()
    plt.show()

def get_gaussian_dataset(n, dim=2, mu_function=lambda x:0, sigma_function=lambda x:1, plot=False):
    """
    Generates a synthetic dataset (X_i, Y_i)

    Arguments:
    -n: number of datapoints
    -dim: dimension of the space for the values of X
    -mu_function: function that maps x to mu(x)
    -sigma_function: function that maps x to sigma(x)
    -plot: boolean paramter to plot the datapoints
    """
    # Generate the covariates
    X = (np.random.random((n, dim))-0.5)*8

    # Generate the parameters 
    mu = np.array([mu_function(x) for x in X])
    sigma = np.array([sigma_function(x) for x in X])

    # Apply the inverse cdf of the GPD to get Y 
    Y = np.array([m + s*np.random.randn() for m, s in zip(mu, sigma)])

    if plot:
        plt.scatter(X[:, 0], X[:, 1], c=mu)
        plt.title("True values of mu")
        plt.colorbar()
        plt.show()

        plt.scatter(X[:, 0], X[:, 1], c=sigma)
        plt.title("True values of sigma")
        plt.colorbar()
        plt.show()

        plt.scatter(X[:, 0], X[:, 1], c=Y, marker=".")
        plt.title("Y")
        plt.colorbar()
        plt.show()

    return X, Y

def visualize_quantile_predictions(quantile_estimator, x1_lim=(-4, 4), x2_lim=(-4, 4), n_pts_x1=100, n_pts_x2=100):
    """
    Only use when dim=2, visualize the predictions of the quantiles
    made by the estimator 'bgex' over the plane
    """
    # Create the grid of values
    x1_val = np.linspace(x1_lim[0], x1_lim[1], n_pts_x1)
    x2_val = np.linspace(x2_lim[0], x2_lim[1], n_pts_x2)

    x1, x2 = np.meshgrid(x1_val, x2_val)
    xx = np.concatenate((np.expand_dims(x1, axis=2), np.expand_dims(x2, axis=2)), axis=2)

    # Make the predictions
    quantiles = quantile_estimator.predict(xx.reshape(-1, 2))
    quantiles = quantiles.reshape(n_pts_x1, n_pts_x2)

    # Plot the predictions
    x1_labels = np.round(np.linspace(x1_lim[0], x1_lim[1], 10), 1)
    x1_ticks = np.linspace(0, n_pts_x1, 10, dtype='int')
    x2_labels = np.round(np.linspace(x2_lim[0], x2_lim[1], 10), 1)
    x2_ticks = np.linspace(0, n_pts_x2, 10, dtype='int')

    # For gamma
    plt.imshow(quantiles, origin='lower')
    plt.xticks(ticks=x1_ticks, labels=x1_labels)
    plt.yticks(ticks=x2_ticks, labels=x2_labels)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Prediction of quantiles")
    plt.colorbar()
    plt.show()

def visualize_extreme_quantile_predictions(quantile_estimator, tau, x1_lim=(-4, 4), x2_lim=(-4, 4), n_pts_x1=100, n_pts_x2=100):
    """
    Only use when dim=2, visualize the predictions of the quantiles
    made by the estimator 'bgex' over the plane
    """
    # Create the grid of values
    x1_val = np.linspace(x1_lim[0], x1_lim[1], n_pts_x1)
    x2_val = np.linspace(x2_lim[0], x2_lim[1], n_pts_x2)

    x1, x2 = np.meshgrid(x1_val, x2_val)
    xx = np.concatenate((np.expand_dims(x1, axis=2), np.expand_dims(x2, axis=2)), axis=2)

    # Make the predictions
    quantiles = quantile_estimator.predict(xx.reshape(-1, 2), tau)
    quantiles = quantiles.reshape(n_pts_x1, n_pts_x2)

    # Plot the predictions
    x1_labels = np.round(np.linspace(x1_lim[0], x1_lim[1], 10), 1)
    x1_ticks = np.linspace(0, n_pts_x1, 10, dtype='int')
    x2_labels = np.round(np.linspace(x2_lim[0], x2_lim[1], 10), 1)
    x2_ticks = np.linspace(0, n_pts_x2, 10, dtype='int')

    # For gamma
    plt.imshow(quantiles, origin='lower')
    plt.xticks(ticks=x1_ticks, labels=x1_labels)
    plt.yticks(ticks=x2_ticks, labels=x2_labels)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Prediction of quantiles at level " + str(tau))
    plt.colorbar()
    plt.show()

def visualize_quantile_function(quantile_function, x1_lim=(-4, 4), x2_lim=(-4, 4), n_pts_x1=100, n_pts_x2=100, title=None):
    """
    Only use when dim=2, visualize the quantiles
    given by 'quantile_function'
    """
    # Create the grid of values
    x1_val = np.linspace(x1_lim[0], x1_lim[1], n_pts_x1)
    x2_val = np.linspace(x2_lim[0], x2_lim[1], n_pts_x2)

    x1, x2 = np.meshgrid(x1_val, x2_val)
    xx = np.concatenate((np.expand_dims(x1, axis=2), np.expand_dims(x2, axis=2)), axis=2)

    # Make the predictions
    quantiles = np.array([quantile_function(x) for x in xx.reshape(-1, 2)])
    quantiles = quantiles.reshape(n_pts_x1, n_pts_x2)

    # Plot the predictions
    x1_labels = np.round(np.linspace(x1_lim[0], x1_lim[1], 10), 1)
    x1_ticks = np.linspace(0, n_pts_x1, 10, dtype='int')
    x2_labels = np.round(np.linspace(x2_lim[0], x2_lim[1], 10), 1)
    x2_ticks = np.linspace(0, n_pts_x2, 10, dtype='int')

    # For gamma
    plt.imshow(quantiles, origin='lower')
    plt.xticks(ticks=x1_ticks, labels=x1_labels)
    plt.yticks(ticks=x2_ticks, labels=x2_labels)
    plt.xlabel("X1")
    plt.ylabel("X2")
    if title is not None:
        plt.title(title)
    plt.colorbar()
    plt.show()

def mise_pareto(estimator, gamma_function, sigma_function, dim, q, n_sim=100):
    """"
    Compute MISE of an estimator with our GPD model

    Arguments :
    -estimator : trained model (gbex, quantile-forest,...)
    -gamma_function & sigma_function : functions used to model Y
    -dim : dimension of covariates X, with X_i iid uniform in [-4,4]
    -q : quantile of interest for prediction
    """
    samples_per_sim = 1000
    ises = []
    for _ in tqdm(range(n_sim)) :
        X_test = (np.random.random((samples_per_sim, dim))-0.5)*8
        # true quantiles
        true_q = []
        for x in X_test :
            gamma_x = gamma_function(x)
            sigma_x = sigma_function(x)
            quantile_x = genpareto.ppf(q, c=gamma_x, scale=sigma_x)
            true_q.append(quantile_x)
        # predicted quantiles
        est_pred = estimator.predict(X_test, q)
        # compute ISE
        ise = ((true_q - est_pred)**2).mean()
        ises.append(ise)
    mise = np.array(ises).mean()
    return(mise)

def mise_gaussian(estimator, mu_function, sigma_function, dim, q, n_sim=100):
    """"
    Compute MISE of an estimator with our gaussian model

    Arguments :
    -estimator : trained model (gbex, quantile-forest,...)
    -mu_function & sigma_function : functions used to model Y
    -dim : dimension of covariates X, with X_i iid uniform in [-4,4]
    -q : quantile of interest for prediction
    """
    samples_per_sim = 1000
    ises = []
    for _ in tqdm(range(n_sim)) :
        X_test = (np.random.random((samples_per_sim, dim))-0.5)*8
        # true quantiles
        true_q = []
        for x in X_test :
            mu_x = mu_function(x)
            sigma_x = sigma_function(x)
            quantile_x = norm.ppf(q, loc=mu_x, scale=sigma_x)
            true_q.append(quantile_x)
        # predicted quantiles
        est_pred = estimator.predict(X_test, q)
        # compute ISE
        ise = ((true_q - est_pred)**2).mean()
        ises.append(ise)
    mise = np.array(ises).mean()
    return(mise)