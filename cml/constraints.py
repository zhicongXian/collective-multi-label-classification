import numpy as np
from scipy.special import comb
import itertools
from itertools import combinations
import math


def f_k(dataset, labels, d, q):
    """

    :param dataset: 某一个样本的特征集
    :param labels: 某一个样本的标签集
    :param d: 样本的维度，即一个样本含有的特征数
    :param q: 标签的维度，即标签集中标签的个数
    :return: 返回的是fk(x,y)
    """

    f_k_multiplicative_factor = np.repeat(labels[None, :], d, 0)
    f_constraint = np.zeros((int(d * q + 4 * comb(q, 2))))
    f_constraint[:d * q] = (np.repeat(dataset[:, None], q, 1) * f_k_multiplicative_factor).flatten()

    # assign relationship based on the label pairwise
    f_k_label_pairwise = label_pairwise_correlation(labels, q)

    f_constraint[d * q:] = f_k_label_pairwise  # --TODO how does it help with calculating the expectation?

    return f_constraint


def rand_labels(nb_labels):
    """
    #关于这个函数的for循环的嵌套次数，Y标签集中，有几个标签就嵌套几层。（y1,y2,...,yq）
    :return: 返回的是q维的标签集的所有组合情况
    """

    rand_labels_vals = np.array(list(itertools.product([0, 1], repeat=nb_labels)))
    return rand_labels_vals


def get_feature_label_constraints(dataset, labels):
    """

    :param dataset:
    :type dataset:
    :param labels:
    :type labels:
    :return:
    :rtype:
    """
    samples = len(dataset)
    d = len(dataset[0])
    q = len(labels[0])
    list_fks = []
    random_labels = rand_labels(q)

    list_of_label_combis = []
    for j in range(len(random_labels)):
        label_combi = label_pairwise_correlation(random_labels[j], q)
        list_of_label_combis.append(label_combi)

    # list_of_all_the_possible_fks
    all_possible_fks = []
    for i in range(samples):
        fk = f_k(dataset[i], labels[i], d, q)  # create a list of constraints
        fk = np.array(fk)
        list_fks.append(fk)
        all_possible_fks_per_sample = get_z(dataset[i], d, q)
        all_possible_fks_per_sample = np.concatenate(all_possible_fks_per_sample, -1)
        all_possible_fks.append(all_possible_fks_per_sample)
    all_possible_fks = np.stack(all_possible_fks, 0)  # [n_samples,nb_possible_labels, n_constraints ]
    fk_actual = np.stack(list_fks, 0)

    return fk_actual, all_possible_fks


def get_z(data, feature_dim, label_dim):
    """

    :param feature_dim:
    :type feature_dim:
    :param data:
    :type data:
    :param label_dim:
    :type label_dim:
    :return:
    :rtype:
    """
    random_labels = rand_labels(label_dim)
    z = []
    for j in range(len(random_labels)):
        fk_tmp = f_k(data, random_labels[j], feature_dim,
                     label_dim)
        z.append(np.asarray(fk_tmp)[..., None])

    return z


def label_pairwise_correlation(labels, q):
    """

    :param labels:
    :type labels:
    :param q:
    :type q:
    :return:
    :rtype:
    """

    f_k_label_pairwise = []
    for y_j1, y_j2 in combinations(labels, 2):
        if y_j1 == 1 and y_j2 == 1:
            f_k_label_pairwise.append(1)
        else:
            f_k_label_pairwise.append(0)
        if y_j1 == 1 and y_j2 == 0:
            f_k_label_pairwise.append(1)
        else:
            f_k_label_pairwise.append(0)
        if y_j1 == 0 and y_j2 == 1:
            f_k_label_pairwise.append(1)
        else:
            f_k_label_pairwise.append(0)
        if y_j1 == 0 and y_j2 == 0:
            f_k_label_pairwise.append(1)
        else:
            f_k_label_pairwise.append(0)

    return f_k_label_pairwise
