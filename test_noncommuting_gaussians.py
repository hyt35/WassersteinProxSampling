import numpy as np
import os
import matplotlib.pyplot as plt
problem_dims = 2

figs_dir = os.path.join("figs", "noncomm_Gaussian_analytic")
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)
#######
# Initialize inputs

eta = 0.001 # Also fine
T = 0.1 # This is fine
beta = 1 #

# Sigma = np.asarray([[2,-1],[-1,5]]) # e.val. 3,1. e.vec. (-1,1), (1,1)
# Sigma_k = np.asarray([[5,0],[0,1]]) # Obviously not commuting

# Sigma = np.asarray([[3,0],[0,1]]) # This one commutes
# Sigma_k = np.asarray([[4,0],[0,4]]) # Obviously not commuting

# Sigma_k = np.asarray([[1.98,-1],[-1,1.98]]) # e.val. 3,1. e.vec. (-1,1), (1,1)
# Sigma_k = np.asarray([[0.1, -0.09],[0.1,-0.09]]) # e.val. 3,1. e.vec. (-1,1), (1,1)
# Sigma = np.asarray([[1.5,0],[0,1]]) # Obviously not commuting
c = 0.5
flag = False
# for i in range(100):
foo = np.random.randn(problem_dims,problem_dims)
bar = np.random.randn(problem_dims,problem_dims)-1
foo = np.random.randn(*foo.shape)*foo
Sigma = np.eye(problem_dims) + 1*bar.T@bar
Sigma_k = np.eye(problem_dims) + 1*foo.T@foo

print(np.max(np.linalg.eigvals(Sigma)))
T = np.min(np.linalg.eigvals(Sigma))*c
print(T)
# min_ev = 1
########
Sigma_inv = np.linalg.inv(Sigma) # Invert covariance for ease
# Expected stationary distribution
Sigma_inf = beta*((np.eye(problem_dims) - T*Sigma_inv) @ Sigma @ (np.eye(problem_dims) + T*Sigma_inv))
# print("Sigma_inv", Sigma_inv)
K = np.eye(problem_dims) + T * Sigma_inv
K_inv = np.linalg.inv(K)
print("K eigvals", np.linalg.eigvals(K))
# print("Sigma_inf", Sigma_inf)

# Delta = (np.sqrt((min_ev+T)/(2*T))+1)/2
# print("Maximum possible stepsize", Delta * min_ev)
# omega_0 = 2*(min_ev-T)/(min_ev+T)
# print("Expected linearity ratio for min component", 1-eta*omega_0/(2*min_ev))

def Lyapunov(mat):
    pass

aux_arr = []
aux2 = []
prev = None
last_cov_T = None
print(np.linalg.eigvals( K_inv @ Sigma_k + Sigma_k @ K_inv  ))
for i in range(1):
    itsig = np.eye(problem_dims) + T * Sigma_inv
    itsig_inv = np.linalg.inv(itsig)
    
    covariance_T = (2*beta*T*itsig_inv) + itsig_inv @ Sigma_k @ itsig_inv

    covariance_T_inv = np.linalg.inv(covariance_T)
    to_mul = np.eye(problem_dims) - eta * Sigma_inv + eta * beta * covariance_T_inv
    Sigma_k = to_mul @ Sigma_k @ to_mul

    # aux = np.sum(np.linalg.eigvals(K_inv @ covariance_T_inv + covariance_T_inv @ K_inv))
    # aux = np.sum(np.linalg.eigvals(K_inv @ covariance_T_inv + covariance_T_inv @ K_inv))
    # aux = np.sum(np.linalg.eigvals(K_inv @ (covariance_T_inv - Sigma_inv)))
    # aux = np.sum(np.linalg.eigvals((covariance_T_inv - Sigma_inv)@ Sigma_k))
    # print(np.abs(covariance_T[0,0] - Sigma[0,0]), np.abs(covariance_T[1,1] -  Sigma[1,1]))
    # aux = np.linalg.norm(covariance_T-Sigma, ord='fro')
    # aux = np.sum(np.abs(np.linalg.eigvals(K @ covariance_T - covariance_T @ K))**2)
    # print(np.linalg.eigvals(Sigma_inv- covariance_T_inv))
    # aux = np.sum(np.linalg.eigvals(Sigma_inv- covariance_T_inv))
    print("K^-1 Sigma_t", np.linalg.eigvals( K_inv @ Sigma_k))
    print("K^-1 Sigma_t + Sigma_T K^-1", np.linalg.eigvals( K_inv @ Sigma_k + Sigma_k @ K_inv  ))
    print("(S^-1 - S^-1) .. (S^-1 - S^-1)", np.linalg.eigvals( (Sigma_inv - covariance_T_inv) @(K_inv @ Sigma_k) @(Sigma_inv - covariance_T_inv)  ))
    print("(S^-1 - S^-1) .. + ..^T (S^-1 - S^-1)", np.linalg.eigvals( (Sigma_inv - covariance_T_inv) @(K_inv @ Sigma_k + Sigma_k @ K_inv) @(Sigma_inv - covariance_T_inv)  ))
    print("K^-1 (S^-1 - S^-1) .. + ..^T (S^-1 - S^-1)", np.linalg.eigvals( K_inv @ (Sigma_inv - covariance_T_inv) @(K_inv @ Sigma_k + Sigma_k @ K_inv) @(Sigma_inv - covariance_T_inv)  ))

    aux = np.sum(np.linalg.eigvals(K_inv@(Sigma_inv - covariance_T_inv) @ K_inv @ Sigma_k @ (Sigma_inv - covariance_T_inv) ))
    print("-d/dt KL",np.linalg.eigvals(K_inv@(Sigma_inv - covariance_T_inv) @ K_inv @ Sigma_k @ (Sigma_inv - covariance_T_inv) ))
    
    # aux = np.sum(np.linalg.eigvals(K_inv @ Sigma_k))
    # print(np.linalg.eigvals(K_inv @ Sigma_k))
    # if any(np.linalg.eigvals(K_inv @ Sigma_k)<0):
    #     flag = True
    
    if any(np.linalg.eigvals(K_inv@(Sigma_inv - covariance_T_inv) @ K_inv @ Sigma_k @ (Sigma_inv - covariance_T_inv))<0):
        flag = True
    print(aux)
    #
    aux_arr.append(aux)
    # print("T", T, np.min(np.linalg.eigvals(K_inv @ Sigma_k)))
    # print(np.linalg.eigvals(K_inv @ covariance_T_inv + covariance_T_inv @ K_inv))
    # print(np.linalg.cond(covariance_T_inv), np.linalg.cond(K_inv))
    # foo1, foo2 = np.linalg.cond(covariance_T_inv), np.linalg.cond(K_inv)
    # print("condition", np.sqrt(foo1*foo2)-np.sqrt(foo1)-np.sqrt(foo2))
    # # aux_arr.append(np.sum(np.linalg.eigvals(covariance_T_inv - Sigma_inv)))
    # print(np.sum(np.linalg.eigvals(covariance_T - Sigma)))
    # aux_arr.append(np.sum(np.linalg.eigvals(covariance_T - Sigma)))
    # aux2.append()
    # aux_arr.append(np.linalg.eigvals())
    # print(np.linalg.eigvals((K - 2*T*covariance_T_inv) @ covariance_T_inv @ K_inv))

fig, ax = plt.subplots()
ax.plot(np.abs(aux_arr))
ax.set_yscale('log')
fig.savefig(os.path.join(figs_dir,"aux1"))


print(flag)