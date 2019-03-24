from __future__ import division, print_function

import numpy as np
import random
import time
from config import get_config
from logger import get_logger
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import scale

config = get_config()
random_samples = config.getint("BAYESIAN_OPTIMIZER", "RANDOM_SAMPLES")
random_seeds = config.getint("BAYESIAN_OPTIMIZER", "RANDOM_SEEDS")

logger = get_logger(__name__, log_level=("BAYESIAN_OPTIMIZER", "LOGLEVEL"))


class UtilityFunction(object):
    """ An object to compute the acquisition functions.
    """

    def __init__(self, kind, rf_objective, rf_constraint=None, constraint_upper=None, xi=0, kappa=5., whitebox=None, ymax=0.0):
        """ If UCB is to be used, a constant kappa is needed.
        """
        self.implementations = ['ucb', 'ei', 'eiwhite', 'cei', 'poi']
        if kind not in self.implementations:
            err = f"The utility function {kind} has not been implemented, " \
                "please choose one of ucb, ei, cei, or poi."
            raise NotImplementedError(err)
        else:
            self.kind = kind

        self.rf_objective = rf_objective
        self.rf_constraint = rf_constraint
        self.xi = xi
        self.kappa = kappa
        self.constraint_upper = constraint_upper
        self.whitebox = whitebox
        self.ymax = ymax

    def utility(self, x, *args):
        rf_objective = self.rf_objective
        rf_constraint = self.rf_constraint
        constraint_upper = self.constraint_upper
        xi, kappa = self.xi, self.kappa

        if self.kind == 'ucb':
            return UtilityFunction._ucb(x, rf_objective, kappa)
        if self.kind == 'ei':
            return UtilityFunction._ei(x, rf_objective, xi, self.ymax)
        if self.kind == 'eiwhite':
            return UtilityFunction._eiwhite(x, rf_objective, xi, self.whitebox)
        if self.kind == 'cei':
            assert rf_constraint is not None, 'gaussian processor for constraint must be provided'
            assert constraint_upper is not None, 'constraint_upper must be provided'
            return UtilityFunction._cei(x, rf_objective, xi, rf_constraint, constraint_upper)
        if self.kind == 'poi':
            return UtilityFunction._poi(x, rf_objective, xi)
    
    @staticmethod
    def rf_std(rf_objective, x, mu):
      B = len(rf_objective.estimators_)
      ss = 0
      for model in rf_objective.estimators_:
        ss = ss + (model.predict(x)-mu) ** 2
      return ss / B

    @staticmethod
    def _ucb(x, rf_objective, kappa):
        mean = rf_objective.predict(x)
        std = UtilityFunction.rf_std(rf_objective, x, mean)
        return mean + kappa * std

    @staticmethod
    def _ei(x, rf_objective, xi, y_max):
        mean = rf_objective.predict(x)
        std = UtilityFunction.rf_std(rf_objective, x, mean)
        #print('acquisition values at: ', x, ' mean: ', mean, ' std: ', std, ' ymax: ', y_max)
        z = (mean - y_max - xi) / std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _eiwhite(x, rf_objective, xi, whitebox):
        '''
        y_max = gp_objective.y_train_.max()
        mean, std = gp_objective.predict(x, return_std=True)
        z = (mean - y_max - xi) / std
        black = (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)
        white = whitebox(x)
        return (1-gamma) * black + gamma * white
        '''
        white = whitebox(x)
        #print('white box values at: ', x, ' U: ', white)
        return white


    @staticmethod
    def _cei(x, rf_objective, xi, rf_constraint, constraint_upper):
        """ Compute the cdf under constraint_upper (i.e. P(c(x) < constraint_upper)) to modulate the ei(x).
            where c(x) is the estimated marginal distribution of the Gaussian process.
        """
        ei = UtilityFunction._ei(x, rf_objective, xi)

        mean, std = rf_constraint.predict(x, return_std=True)
        z = (constraint_upper - mean) / std

        cumulative_probabiliy = norm.cdf(z)
        return cumulative_probabiliy * ei

    @staticmethod
    def _poi(x, rf_objective, xi):
        y_max = rf_objective.y_train_.max()
        mean, std = rf_objective.predict(x, return_std=True)
        z = (mean - y_max - xi) / std
        return norm.cdf(z)


#def get_eligible(values, gamma=0):
    """ Higher values are more likely to get a score of 1
    """
    #quantile = np.percentile(values, gamma*100)
    #fun = lambda x: 1.0 if x>=quantile else 0.0
    #low = np.min(values)
    #high = np.max(values)
    #fun = lambda x: 1.0 if gamma==1 or gamma==0 or x>=np.random.uniform(low, high) else 0.0
    #vfun = np.vectorize(fun)
    # print('eligivle: ', vfun(values))
    #return vfun(values)


def acq_max(utility, bounds, whitebox, gamma, extrabounds):
    """ A function to find the maximum of the acquisition function
        It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
        optimization method: 1) by sampling of "random_samples" number of random points,
        and 2) by running L-BFGS-B from "random_seeds" number of starting points.
    Args:
        ac: The acquisition function object that return its point-wise value.
        gp_objective: A gaussian process fitted to the relevant data.
        y_max: The current maximum known value of the target function.
        bounds: The variables bounds to limit the search of the acq max.
    Returns
        x_max: The arg max of the acquisition function.
    """

    # Warm up using random sampling
    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(random_samples, bounds.shape[0]))
    # add new features
    x_tries_orig = np.copy(x_tries)
    if (gamma==1.0):
      newFeatures = whitebox(x_tries)
      if(extrabounds is not None):
        newFeatures = dividebyzero(newFeatures, extrabounds[:,1])
      x_tries = np.concatenate((x_tries, newFeatures), axis=1)
    if (gamma==0.5):
      x_tries = whitebox(x_tries)
      if(x_tries is not None):
        x_tries = dividebyzero(x_tries, extrabounds[:,1])
    # eligible = get_eligible(whitebox(x_tries), gamma)
    # ys = np.multiply(utility(x_tries), eligible)
    ys = utility(x_tries)
    # print("---Uniform samples utilities:\n ", ys, ", whitebox:\n ", whitebox(x_tries), ", eligible: ", eligible)
    # print('ys: ',ys)
    x_max = x_tries_orig[ys.argmax()]
    max_acq = ys.max()
    print('**max acq found from uniform random: ', max_acq, ' at x:', x_max)

    logger.info(f'nonzeros in the utility function: {np.count_nonzero(ys)}')

    # Explore the parameter space more throughly using L-BFGS-B
    x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(random_seeds, bounds.shape[0]))
    x_seeds_orig = np.copy(x_seeds)
    if (gamma==1.0):
      newFeatures = whitebox(x_seeds)
      x_seeds = np.concatenate((x_seeds, newFeatures), axis=1)
    if (gamma==0.5):
      x_seeds = whitebox(x_seeds)
    
    for x_try in x_seeds:
        if(gamma==0.5):
          break # can't deal with this case
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -utility(x.reshape(1, -1)), x_try.reshape(1, -1),
                       #bounds=bounds,
                       method="L-BFGS-B")
        # print("found seed: ", res.x, " function: ", -res.fun, " length: ", len(bounds))
        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x[range(len(bounds))]
            max_acq = -res.fun[0]

    print('**max acq found: ', max_acq, ' at x:', x_max)
    # Clip output to make sure it lies within the bounds.
    # Due to floating point operations this is not always guaranteed.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


def unique_rows(a):
    """ A functions to trim repeated rows that may appear when optimizing.
        This is necessary to avoid the sklearn GP object from breaking
    Args:
        a(array): array to trim repeated rows from
    Return:
        mask of unique rows
    """
    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


def get_fitted_randomforest_processor(X_train, y_train, constraint_upper, standardize_y=True, **rf_params):
    # Initialize gaussian process regressor
    rf = RandomForestRegressor()
    rf.set_params(**rf_params)
    logger.debug('Instantiated random forest processor for objective function:\n' + f'{rf}')
    logger.debug(f"Fitting random forest processor")

    if standardize_y:
        if constraint_upper is not None:
            y_train = scale(np.hstack((y_train, constraint_upper)))
            scaled_constraint_upper = y_train[-1]
            y_train = y_train[:-1]
        else:
            y_train = scale([i for i in y_train])
            scaled_constraint_upper = None
        rf.constraint_upper = scaled_constraint_upper
    else:
        rf.constraint_upper = constraint_upper

    logger.debug(f'X_train:\n{X_train}')
    logger.debug(f'y_train\n{y_train}')
    #if rf_params is None or gp_params.get('alpha') is None:
        # Find unique rows of X to avoid GP from breaking
        #ur = unique_rows(X_train)
        #rf.fit(X_train[ur], y_train[ur])
    #else:
    rf.fit(X_train, y_train)
    return rf

def dividebyzero(a, b):
    #with np.errstate(divide='ignore', invalid='ignore'):
    #    c = np.true_divide( a, b )
    #    c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    #return c
    #b[b==0]=1
    #print('a: ', a, ' b: ', b)
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def get_candidate(feature_mat, objective_arr, bounds, acq,
                  constraint_arr=None, constraint_upper=None,
                  kappa=5, xi=0.0, standardize_y=True,
                  whitebox=None, gamma=0.0, extrabounds=None, **rf_params):
    """ Compute the next candidate based on Bayesian Optimization
    Args:
        feature_mat(numpy 2d array): feature vectors
        objective_arr(numpy 1d array): objective values
        bounds(array of tuples): the searching boundary of feature space
            i.e. bounds=[(x1_lo, x1_hi), (x2_lo, x2_hi), (x3_lo, x3_hi)]
        acq(str): kind of acquisition function

    Return:
        argmax(vector): argmax of acquisition function
    """
    # TODO: Put these into config file
    #if rf_params is None:
    seed = 6
    rf_params = {"n_estimators": 100, "max_features": 'auto', 
                     "random_state": seed}
    # Set boundary
    bounds = np.asarray(bounds)
    extrabounds = np.asarray(extrabounds)

    # Add new features
    orig_features = np.copy(feature_mat)
    if(gamma==1.0): 
      newFeatures = whitebox(feature_mat)
      if(extrabounds is not None):
        newFeatures = dividebyzero(newFeatures, extrabounds[:,1])
        #print('--feature_mat: ', feature_mat, '  --newFeatures: ', newFeatures)
      feature_mat = np.concatenate((feature_mat, newFeatures), axis=1)
      # print('--New features added: ', newFeatures)
    if(gamma==0.5):
      feature_mat = whitebox(feature_mat)
      if(extrabounds is not None):
        feature_mat = dividebyzero(feature_mat, extrabounds[:,1])

    objective_arr_np = np.array(objective_arr.tolist())    
    #print('--feature mat: ', feature_mat, ', objective arr: ', objective_arr, ', whitebox: ', whitebox(feature_mat))
    start = time.time()
    rf_objective = get_fitted_randomforest_processor(
        feature_mat, objective_arr_np, constraint_upper, standardize_y=standardize_y, **rf_params)
    rf_constraint, constraint_upper = None, None
    print('Time taken for fitting: ', time.time()-start)
    print('Error in fitting: ', rf_objective.score(feature_mat, objective_arr_np), ' predictions: ', rf_objective.predict(feature_mat), ' true values: ', objective_arr_np)
    print('Importance: ', rf_objective.feature_importances_)

    # Initialize utiliy function
    #if gamma == 1:
        #print('Using white box model')
        #acq = 'ucb'
    #else:
        #print('Using black box model')
        #acq = 'ucb' # with probability (1-gamma), use black box model

    util = UtilityFunction(kind=acq, rf_objective=rf_objective, rf_constraint=rf_constraint,
                           constraint_upper=rf_constraint.constraint_upper if rf_constraint else None,
                           xi=xi, kappa=kappa, whitebox=whitebox, ymax=objective_arr_np.max())

    # Finding argmax of the acquisition function.
    logger.debug("Computing argmax of acquisition function")
    start = time.time()
    argmax = acq_max(utility=util.utility, bounds=bounds, whitebox=whitebox, gamma=gamma, extrabounds=extrabounds)
    print('Time taken for acquisition: ', time.time()-start)

    #print('**found argmax: ', argmax, ', value of whitebox: ', whitebox(argmax))
    return argmax
