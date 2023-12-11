import numpy as np

def log_l(params, r, z, sigma_z):
    """
    params : np.array([z0, psi, rtaper, qtaper])
    """
    
    # Define model z
    rcavity, r0 = 0, 1
    eq1 = params[0] * ((r - rcavity)/r0)**(params[1])
    eq2 = np.exp(-((r - rcavity)/params[2])**(params[3]))
    zmodel = eq1 * eq2
    
    # Calculate chi2
    residuals = z - zmodel
    chi2 = (residuals**2)/(sigma_z**2)
    
    log_probability = -np.sum(chi2)/ 2
    
    return log_probability


def log_prior(params):
    """
    params : np.array([z0, psi, rtaper, qtaper])
    """
    
    if (0.0 < params[0] < 0.5) and (0.0 < params[1] < 2) and (0.0 < params[2] < 10.0) and (1.0 < params[3] < 15.):
        return 0
    
    else:
        return -np.inf
    

def log_posterior(params, r, z, sigma_z):
    
    # Calculate priors
    
    l_priors = log_prior(params)
    if not np.isfinite(l_priors):
        return -np.inf
    
    return l_priors + log_l(params, r, z, sigma_z)


