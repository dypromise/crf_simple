# coding=utf-8

import numpy as np
import crf


def twoloop(s_list, y_list, grad):
    """
    Subroutine of L-BFGS: two-loop recursion.

    :param s_list: delta_x list : s_k = x_k+1 - x_k.
    :param y_list: delta_grad list : y_k = grad(f)_k+1 - grad(f)_k
    :param grad: gradient of this iter.
    :return: H_k * grad_k ( negitive descent direction)
    """

    m = len(s_list)  # length of memerized s_list and y_list.  

    if np.shape(s_list)[0] >= 1:  # h0 is scale, not matrix.think why.
        h0 = 1.0 * np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])  # init h0.
    else:
        h0 = 1

    a = np.empty((m,))
    q = grad.copy()
    rho = np.zeros((1, m))
    for i in range(m - 1, -1, -1):
        rho[i] = 1.0 / np.dot(s_list[i], y_list[i])
        a[i] = rho[i] * np.dot(s_list[i], q)
        q -= a[i] * y_list[i]
    r = h0 * q

    for i in range(m):
        b = rho[i] * np.dot(y_list[i], r)
        r += s_list[i] * (a[i] - b)
    return r


def lbfgs(x, corps,
          featureTS, words2tagids,
          priorfeatureE, m=10, ITER_MAX=100, fun=crf.getnegloglikelihood, gfun=crf.getgradients):
    """
    If you want to know the principle of L-BFGS, please reference:
    "Numerical Optimization(2nd Edition)" chapter7
    Algorithm 7.4 and Algorithm 7.5 in page 178.
    
    :param x: variable to be optimized.
    :param corps: 
    :param featureTS: 
    :param words2tagids: 
    :param priorfeatureE: 
    :param m: length of memerized list.
    :param ITER_MAX: The maximum number of iterations
    :param fun: 
    :param gfun: 
    :return: 
    """

    beta = 0.4  # parameter 'beta' for Backtracking line search method, it control the descent speed of t.
    alpha = 0.4  # another para in Backtracking line search method.
    epsilon = 1e-4  # stop threshold of optimization

    s, y = [], []
    iter = 0
    while iter < ITER_MAX:

        grad = gfun(priorfeatureE, x, corps, featureTS, words2tagids)
        if np.linalg.norm(grad) < epsilon:
            break
        delta_x = -1.0 * twoloop(s, y, grad)

        # Backtracking linear search method
        t = 1.0  # step length
        funcvalue = fun(x, corps, featureTS, words2tagids)
        while (fun(x + t * delta_x, corps, featureTS, words2tagids) > funcvalue + alpha * t * np.dot(
                grad, delta_x)):
            t *= beta
        t *= beta

        x_new = x + t * delta_x

        sk = x_new - x
        yk = gfun(priorfeatureE, x_new, corps, featureTS, words2tagids) - grad

        if np.dot(sk, yk) > 0:  # add new sk, yk to s_list,y_list.
            s.append(sk)
            y.append(yk)
        if np.shape(s)[0] > m:  # discard the oldest sk, yk.
            s.pop(0)
            y.pop(0)

        iter += 1
        x = x_new
        print 'iterations：%d, func_value：%f' % (iter, funcvalue)
    return x, fun(x, corps, featureTS, words2tagids)  
