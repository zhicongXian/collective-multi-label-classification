from cml.constraints import rand_labels, get_z, f_k
import numpy as np
import math


def pred(test_data, lambdas, d, q):
    """

    :param test_data:
    :type test_data:
    :param lambdas:
    :type lambdas:
    :param d:
    :type d:
    :param q:
    :type q:
    :return:
    :rtype:
    """
    random_labels = rand_labels(q)

    # calculate the normalization constants z

    z = get_z(test_data, d, q)
    all_possible_fks_prob = np.concatenate(z, -1) * np.asarray(lambdas)[None, :, None]  # [nb_samples, nb_constraints]
    z = np.sum(np.exp(np.sum(all_possible_fks_prob, 1)), -1)  # [nb_samples]

    best_p = -1.0
    best_labels = None

    for i in range(len(random_labels)):
        fk = f_k(test_data, random_labels[i], d, q)
        fk = np.array(fk)
        temp_p = math.exp((lambdas * fk).sum()) / z  # this is the conditional probability distribution for the class
        if temp_p > best_p:
            best_p = temp_p
            best_labels = random_labels[i]

    return best_labels, best_p
