"""参考
https://qiita.com/fujiisoup/items/f2fe3b508763b0cc6832
"""
"""
y = N*1
X = N*K
beta = K*1
L(beta) = (y-X@beta)' @ (y-X@beta) / 2
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tqdm import tqdm
np.set_printoptions(suppress=True)
np.set_printoptions(precision=10)

def diff_L(y,X,beta):
    diff = - X.T @ (y - X @ beta)
    return diff

def update(arg,Lambda,rho):
    alpha = Lambda/rho
    if arg<-alpha:
        result  = arg + alpha
    if arg > alpha:
        result = arg - alpha
    if (- alpha <= arg) and (alpha >= arg):
        result = 0.0
    return result

def updated_par(y,X,beta,Lambda,rho):
    h = beta - diff_L(y,X,beta)/rho
    params = [update(h[i,0],Lambda,rho) for i in range(len(beta))]
    params = np.array(params).reshape(len(beta),1)
    return params

def solve_lasso_by_ista(y,x,Lambda,rho,max_iter=300000,gamma=1e-5):
    beta0 = np.zeros([x.shape[1],1])
    for it in range(max_iter):
        beta_new = updated_par(y,X,beta0,Lambda,rho)
        if (np.abs(beta0 - beta_new) < gamma).all():
            return beta_new
        beta0 = beta_new
    raise ValueError('Not converged.')


def get_likelihood():
    return 0

def get_rho(X):
    return np.max(np.sum(np.abs(X), axis=0))

if __name__ == '__main__':
    np.random.seed(42)
    Lambda = 0.01
    n_samples, n_features = 50, 200
    X = np.random.randn(n_samples, n_features)
    coef = 3 * np.random.randn(n_features)
    inds = np.arange(n_features)
    np.random.shuffle(inds)
    coef[inds[10:]] = 0  # sparsify coef
    coef = coef.reshape(n_features,1)
    X = X.reshape(n_samples,n_features)
    y = X @ coef
    y = y.reshape(n_samples,1)
    y = y + np.random.normal(0,0.1,(n_samples,1))
    rho = get_rho((X.T @ X))
    print(np.ravel(coef))
    estimation = np.ravel(solve_lasso_by_ista(y,X,Lambda,rho))
    print(estimation)
