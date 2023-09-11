import numpy as np

# Update via 
#    dX/dt = -\nabla f(x) - \beta \nabla \log(\rho(x))
# where \rho is the score funciton

def sample_V(X_list, V, sigma, n_iter = 100):
    # Returns the expectation around points X_list
    # 
    avg = None

    for i in range(n_iter):
        X_list_perturbed = X_list + sigma * np.random.randn(*X_list.shape)
        sampled = V(X_list_perturbed)
        if avg is None:
            avg = sampled
        else:
            avg = avg + sampled

    return avg/n_iter

def compute_normalizing_constant(y, V, beta, T, n_iter = 25):
    # TODO vectorize
    # Numerically compute
    # \int \exp(-1/(2\beta) (V(z) + ||z-y||^2/(2T)) dz
    # = (4pi T beta)^(d/2) E_{z ~ N(y, 2 T beta I)}[exp(-1/(2beta) V(z))]
    # Takes the form of a Gaussian expectation
    # can accept vectorized
    # assume y is batched as in [N, d], then returns [N, 1]

    dims = y.shape[-1]
    variance = 2 * T * beta
    std = np.sqrt(variance)
    # front_constant = (2*np.pi*variance)**(dims/2)
    front_constant = 1
    avg = None
    for i in range(n_iter):
        # sample z from normal distribution z ~ N(y, 2 T beta I)
        z = y + std * np.random.randn(*y.shape)
        current = np.exp(-V(z)/(2*beta))
        if avg is None:
            avg = current
        else:
            avg = avg + current

    return avg/n_iter * front_constant



def compute_score(W_list, V, dV, beta, T, sample_iters = 25, compute_energy = False, **kwargs):
    # rho_0 = \sum_i delta_{w_i}
    # W_list = [w_i : i = 0,...,N], shape = [N, d]
    # assume V can accept batched input V : [N, d] -> [N, 1]
    # dV : [N, d] -> [N, d]
    batch_num = W_list.shape[0]
    dims = W_list.shape[-1]
    mat_differences = - np.repeat(W_list[None,:,:], batch_num, axis=0) \
                    + np.repeat(W_list[:,None,:], batch_num, axis=1) # of shape [N,N,d]. mat_differences[i,j] = w_i - w_j
    # print("mat_diff", mat_differences.shape)

    squared_differences = np.sum(mat_differences**2, axis=2) # of shape [N,N]
    
    # approximate normalizing constant for K using MC
    normalizing_constants = compute_normalizing_constant(W_list, V, beta, T, n_iter = sample_iters)
    # print(normalizing_constants)
    # print("normalizing", normalizing_constants.shape)
    # compute the V
    V_arr = V(W_list)  # [N, 1]
    # enforce [N, 1]. Required for broadcasting later.
    if len(V_arr.shape) == 1:
        V_arr = V_arr[:, None]
    elif len(V_arr.shape) > 2:
        raise Exception("V is not outputting a scalar?")
    dV_arr = dV(W_list) # [N, d]



    # for computing the exponential
    unscaled_density = np.exp(-1/(2*beta) * (V_arr + squared_differences/(2*T))) # [N,N]
    # print("us", unscaled_density.shape)
    scaled_density = unscaled_density /(normalizing_constants.T) # [N,N]. normalizing constant.T is [1,N] for Z(w_j)
    # matrix of scaled density K(w_i, w_j)
    
    # print("s", scaled_density.shape)
    rho = np.sum(scaled_density, axis=1) # to get integral with rho_0, sum over j

    # for computing the score d\rho
    # compute pre-multiplier
    # print(dV_arr.shape)
    # print(mat_differences.shape)
    pre_multiplier = -1/(2*beta) * (dV_arr[:, None, :] + mat_differences/T) # should be of the shape [N,N,d]

    score_unsummed = pre_multiplier * scaled_density[:,:,None]
    score = np.sum(score_unsummed, axis=1) # technically only \grad \rho(w_i), not the score.

    if compute_energy: # lazy bool
        # compute the energy here
        energy = 0
        # energy = E_rho_T [V(x) + log rho_T(x)] / normalizing_constant
        # perturbed_X shape [sample_iters, N, d]
        perturbed_X = np.repeat(W_list[None,:,:], sample_iters, axis=0) + np.sqrt(2*beta*T) * np.random.randn(sample_iters, *W_list.shape)
        
        # compute
        
        perturbed_shape = perturbed_X.shape
        num_N = perturbed_shape[-2] # N
        # perturbed_X = perturbed_X.reshape((-1,perturbed_shape[-1]))

        # compute log rho(x) for the perturbed _X
        # now it is between perturbed_X and W_list
        # want the eventual shape [sample_iters, N, d]

        # create another channel at the front to represent inner sum
        # first term: [N, sample_iters, N, d]
        mat_differences_perturbed = - np.repeat(perturbed_X[None,:,:,:], num_N, axis=0) \
                    + np.repeat(W_list[:,None,None,:], sample_iters, axis=1) 
                    
        # ^ of shape [N,sample_iters, N,d]. mat_differences[p,q,...] = w_p - x_q (with additional outer sum w_i information)


        squared_differences_perturbed = np.sum(mat_differences_perturbed**2, axis=3) # of shape [N,sample_iters,N]
        perturbed_X = perturbed_X.reshape((-1,perturbed_shape[-1]))
        perturbed_V = V(perturbed_X) 
        perturbed_V = perturbed_V.reshape(perturbed_shape[0:2]) # [sample_iters, N]
        
        # replacing previous unused variables
        # care if using xx_density again
        unscaled_density = np.exp(-1/(2*beta) * (perturbed_V[None,:] + squared_differences_perturbed/(2*T))) # [N,sample_iters,N]
        scaled_density = unscaled_density/normalizing_constants[:,None,None] # [N, sample_iters, N]
        
        rho_perturbed = np.sum(scaled_density, axis=0) # sum over N to get log rho_T(x)
        log_rho = np.log(rho_perturbed) # [sample_iters, N]

        # # compute V
        # perturbed_X = perturbed_X.reshape((-1,perturbed_shape[-1]))
        # perturbed_V = V(perturbed_X) 
        # perturbed_V = perturbed_V.reshape(perturbed_shape[0:2]) # [sample_iters, N]

        term_1 = np.exp(-perturbed_V/(2*beta))
        term_2 = perturbed_V + log_rho*beta
        # expectation over x
        numerator = np.mean(term_1*term_2, axis=0) # [N]

        # denominator is simply the normalizing constant
        denominator = normalizing_constants

        energy = np.sum(numerator/denominator) # technically should be mean but this would require changing other things
        return rho, score, energy
    else:
        return rho, score # rho: [N]. score: [N,d]

def compute_dPhi(X_list, V, dV, beta, T, sample_iters = 25):
    # perturbed_X shape [sample_iters, N, d]
    perturbed_X = np.repeat(X_list[None,:,:], sample_iters, axis=0) + np.sqrt(2*beta*T) * np.random.randn(sample_iters, *X_list.shape)
    
    # compute
    perturbed_shape = perturbed_X.shape
    perturbed_X = perturbed_X.reshape((-1,perturbed_shape[-1]))

    # E_{z ~ N(x, 2Tbeta)}[exp(-V/2beta)]
    bot = np.exp(-V(perturbed_X)/(2*beta)) # shape [N*sample_iters, ]

    grads = dV(perturbed_X) # shape [N*sample_iters, d]
    if len(bot.shape) == 1:
        top = grads * bot[:,None]
    elif len(bot.shape == 2):
        top = grads * bot
    else:
        raise Exception("bot has too many dims")
    # print(grads.shape, bot.shape)
    # print(top.shape)
    # reshape
    bot = bot.reshape(tuple(perturbed_shape[0:2]+(1,)))
    top = top.reshape(perturbed_shape)
    # average
    bot_avg = np.mean(bot, axis=0)
    top_avg = np.mean(top, axis=0)

    return -top_avg/bot_avg

def update_once(X_list, V, dV, beta, T, stepsize, sample_iters=25, compute_energy=False):
    """_summary_

    Args:
        X_list (list of np.array): List of current points to evolve. Shape [N,d]
        V (function): Potential function
        dV (function): Derivative of potential function
        beta (float): Regularization parameter
        T (float): time to evolve towards
    """
    if len(X_list.shape) == 1: #assume it is 1d.
        X_list = X_list[:, None]
    elif len(X_list.shape) != 2:
        raise Exception("Wrong shape for input")

    if beta <= 0:
        raise Exception("Beta must be positive")
    if T <= 0:
        raise Exception("T must be positive")
    if compute_energy:
        rho, score, energy = compute_score(X_list, V, dV, beta, T, sample_iters = sample_iters, compute_energy=compute_energy) #rho: [N]. score: [N,d]
    else:
        rho, score = compute_score(X_list, V, dV, beta, T, sample_iters = sample_iters, compute_energy=compute_energy) #rho: [N]. score: [N,d]

    # print("rs shape", rho.shape, score.shape)
    grad_V = dV(X_list)

    # last term is d_x log rho
    X_list = X_list - stepsize * grad_V - beta * stepsize * score/(rho[:, None])
    if compute_energy:
        return X_list, energy
    else:
        return X_list

def update_once_Phi(X_list, V, dV, beta, T, stepsize, sample_iters=25):
    """_summary_

    Args:
        X_list (list of np.array): List of current points to evolve. Shape [N,d]
        V (function): Potential function
        dV (function): Derivative of potential function
        beta (float): Regularization parameter
        T (float): time to evolve towards
    """
    if len(X_list.shape) == 1: #assume it is 1d.
        X_list = X_list[:, None]
    elif len(X_list.shape) != 2:
        raise Exception("Wrong shape for input")

    if beta <= 0:
        raise Exception("Beta must be positive")
    if T <= 0:
        raise Exception("T must be positive")

    rho, score = compute_score(X_list, V, dV, beta, T, sample_iters = sample_iters) #rho: [N]. score: [N,d]
    dPhi = compute_dPhi(X_list, V, dV, beta, T, sample_iters = sample_iters)
    # print("rs shape", rho.shape, score.shape)

    # last term is d_x log rho
    # print(X_list.shape, dPhi.shape, score.shape, rho.shape)
    X_list = X_list + stepsize * dPhi - beta * stepsize * score/(rho[:, None])

    return X_list

def update_once_SGD(X_list, dV, beta, stepsize):
    grad_V = dV(X_list)
    X_list = X_list - stepsize*grad_V + np.sqrt(2*beta*stepsize)*np.random.randn(*X_list.shape)
    return X_list

# def compute_logrhoT(X_list, V, beta, T, normalizing_constant, sample_iters = 50):
#     # normalizing_constant shape [N]
#     # 
    
def compute_energy_standalone(W_list, V, beta, T, sample_iters=50):
    # eg for SGD
    # slower since need to recompute normalizing constants
    normalizing_constants = compute_normalizing_constant(W_list, V, beta, T, n_iter = sample_iters)
    perturbed_X = np.repeat(W_list[None,:,:], sample_iters, axis=0) + np.sqrt(2*beta*T) * np.random.randn(sample_iters, *W_list.shape)
    
    # compute
    
    perturbed_shape = perturbed_X.shape
    num_N = perturbed_shape[-2] # N
    # perturbed_X = perturbed_X.reshape((-1,perturbed_shape[-1]))

    # compute log rho(x) for the perturbed _X
    # now it is between perturbed_X and W_list
    # want the eventual shape [sample_iters, N, d]

    # create another channel at the front to represent inner sum
    # first term: [N, sample_iters, N, d]
    mat_differences_perturbed = - np.repeat(perturbed_X[None,:,:,:], num_N, axis=0) \
                + np.repeat(W_list[:,None,None,:], sample_iters, axis=1) 
                
    # ^ of shape [N,sample_iters, N,d]. mat_differences[p,q,...] = w_p - x_q (with additional outer sum w_i information)


    squared_differences_perturbed = np.sum(mat_differences_perturbed**2, axis=3) # of shape [N,sample_iters,N]
    perturbed_X = perturbed_X.reshape((-1,perturbed_shape[-1]))
    perturbed_V = V(perturbed_X) 
    perturbed_V = perturbed_V.reshape(perturbed_shape[0:2]) # [sample_iters, N]
    
    unscaled_density = np.exp(-1/(2*beta) * (perturbed_V[None,:] + squared_differences_perturbed/(2*T))) # [N,sample_iters,N]
    scaled_density = unscaled_density/normalizing_constants[:,None,None] # [N, sample_iters, N]
    
    rho_perturbed = np.sum(scaled_density, axis=0) # sum over N to get log rho_T(x)
    log_rho = np.log(rho_perturbed) # [sample_iters, N]

    # # compute V
    # perturbed_X = perturbed_X.reshape((-1,perturbed_shape[-1]))
    # perturbed_V = V(perturbed_X) 
    # perturbed_V = perturbed_V.reshape(perturbed_shape[0:2]) # [sample_iters, N]

    term_1 = np.exp(-perturbed_V/(2*beta))
    term_2 = perturbed_V + log_rho * beta # added a *beta here
    # expectation over x
    numerator = np.mean(term_1*term_2, axis=0) # [N]

    # denominator is simply the normalizing constant
    denominator = normalizing_constants

    energy = np.sum(numerator/denominator) # technically should be mean but this would require changing other things
    return energy


def update_once_givenscore(X_list, dV, beta, stepsize, score):
    """Compute update given the score of rho_T

    Args:
        X_list (list of np.array): List of current points to evolve. Shape [N,d]
        V (function): Potential function
        dV (function): Derivative of potential function
        beta (float): Regularization parameter
        T (float): time to evolve towards
    """
    if len(X_list.shape) == 1: #assume it is 1d.
        X_list = X_list[:, None]
    elif len(X_list.shape) != 2:
        raise Exception("Wrong shape for input")

    if beta <= 0:
        raise Exception("Beta must be positive")

    # print("rs shape", rho.shape, score.shape)
    grad_V = dV(X_list)

    # last term is d_x log rho
    X_list = X_list - stepsize * grad_V - beta * stepsize * score(X_list)

    return X_list

def compute_energy_standalone_corrected(W_list, V, beta, T, sample_iters=50):
    # eg for SGD
    # slower since need to recompute normalizing constants
    normalizing_constants = compute_normalizing_constant(W_list, V, beta, T, n_iter = sample_iters)
    perturbed_X = np.repeat(W_list[None,:,:], sample_iters, axis=0) + np.sqrt(2*beta*T) * np.random.randn(sample_iters, *W_list.shape)
    
    # compute
    
    perturbed_shape = perturbed_X.shape
    num_N = perturbed_shape[-2] # N
    # perturbed_X = perturbed_X.reshape((-1,perturbed_shape[-1]))

    # compute log rho(x) for the perturbed _X
    # now it is between perturbed_X and W_list
    # want the eventual shape [sample_iters, N, d]

    # create another channel at the front to represent inner sum
    # first term: [N, sample_iters, N, d]
    mat_differences_perturbed = - np.repeat(perturbed_X[None,:,:,:], num_N, axis=0) \
                + np.repeat(W_list[:,None,None,:], sample_iters, axis=1) 
                
    # ^ of shape [N,sample_iters, N,d]. mat_differences[p,q,...] = w_p - x_q (with additional outer sum w_i information)


    squared_differences_perturbed = np.sum(mat_differences_perturbed**2, axis=3) # of shape [N,sample_iters,N]
    perturbed_X = perturbed_X.reshape((-1,perturbed_shape[-1]))
    perturbed_V = V(perturbed_X) 
    perturbed_V = perturbed_V.reshape(perturbed_shape[0:2]) # [sample_iters, N]
    
    unscaled_density = np.exp(-1/(2*beta) * (perturbed_V[None,:] + squared_differences_perturbed/(2*T))) # [N,sample_iters,N]
    scaled_density = unscaled_density/normalizing_constants[:,None,None] # [N, sample_iters, N]
    
    rho_perturbed = np.mean(scaled_density, axis=0) # sum over N to get log rho_T(x)
    log_rho = np.log(rho_perturbed) # [sample_iters, N]

    # # compute V
    # perturbed_X = perturbed_X.reshape((-1,perturbed_shape[-1]))
    # perturbed_V = V(perturbed_X) 
    # perturbed_V = perturbed_V.reshape(perturbed_shape[0:2]) # [sample_iters, N]

    term_1 = np.exp(-perturbed_V/(2*beta))
    term_2 = perturbed_V + log_rho * beta # added a *beta here
    # expectation over x
    numerator = np.mean(term_1*term_2, axis=0) # [N]
    # numerator = E[exp(-V/2beta) (log rho_T(x) + V(x))]
    # denominator is simply the normalizing constant which is expectation * (4pi beta T)^(d/2)
    denominator = normalizing_constants/((4*np.pi*beta*T)**(W_list.shape[-1]/2))

    energy = np.mean(numerator/denominator) # technically should be mean but this would require changing other things
    return energy