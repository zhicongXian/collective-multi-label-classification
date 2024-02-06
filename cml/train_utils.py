import numpy as np
from cml.constraints import get_feature_label_constraints


# Set up the function f_k for the constraints


def obj_func(fk_actual, all_possible_fks, sigma, lambdas):
    """

    :param lambdas:
    :type lambdas:
    :param all_possible_fks:
    :type all_possible_fks:
    :param fk_actual:
    :type fk_actual:
    :param q:标签集的维度
    :param DataSets:所有训练样本的特征集
    :param Labels:所有训练样本的标签集
    :param sigma:自己给定的参数值，2**-6,2**-5,2**-4,2**-3,2**-2,2**-1,2**1,2**2,2**3,2**4,2**5,2**6逐个取值，参数寻优
    :return:目标函数，以及待定参数Lambda
    """
    # --TODO still need to improve it until only accepting the parameters lambda and updates
    fk_prob = fk_actual * np.asarray(lambdas)[None, :]
    all_possible_fks_prob = all_possible_fks * np.asarray(lambdas)[None, :, None]  # [nb_samples, nb_constraints]
    z = np.abs(np.sum(np.exp(np.sum(all_possible_fks_prob, 1)), -1))  # [nb_samples]
    temp_sum = np.sum(np.sum(fk_prob, -1) - np.log2(z + np.finfo(np.float64).eps))

    # now parameterized it by lambdas
    # so far reusable are:
    # actual fks
    # list of all the possible fks per_sample
    temp_div = sum((np.array(lambdas)) ** 2 / (2 * sigma ** 2))

    l = temp_sum - temp_div
    return -l  # 求解l的最大值，可以转化为求-l的最小值问题, maximize cross entropy


# use non-gradient based parameter searching:
from zoopt import Dimension, Objective, Parameter, Opt
from scipy.optimize import minimize


def train(k, sigma, train_data, train_target):
    """

    :param train_target:
    :type train_target:
    :param train_data:
    :type train_data:
    :param k: number of constraints nb_features * label_dim + 4 * Combinatorics(label_dim, 2)
    :type k:
    :param sigma:
    :type sigma:
    :return:
    :rtype:
    """

    variables = int(k)  # 变量数目
    init_lam = np.ones([1, int(k)]).tolist()[0]  # 初始点

    fk_actual, all_possible_fks = get_feature_label_constraints(train_data, train_target)

    # use bfgs to test:
    objective = lambda x: obj_func(fk_actual, all_possible_fks, sigma, x)
    # perform the bfgs algorithm search, the achieved result is better than simplex based optimization algorithm
    # such as nelder-mead method.
    result = minimize(objective, x0=init_lam, method='BFGS')
    # summarize the result
    print('Status : %s' % result['message'])
    print('Total Evaluations: %d' % result['nfev'])
    # evaluate solution
    solution = result['x']
    evaluation = objective(solution)
    print('Solution: f(%s) = %.5f' % (solution, evaluation))

    return solution
