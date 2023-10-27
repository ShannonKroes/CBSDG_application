# -*- coding: utf-8 -*-
"""
These are functions taken from the repository by Vinkenoog et al.
https://zenodo.org/records/6938113
"""
import copy
import os
import pickle
import random
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Sampling import sample_instances

which = lambda lst: list(np.where(lst)[0])


def extract_levels(data):
    levels = np.zeros(data.shape[1])

    for i in range(data.shape[1]):
        levels[i] = len(np.unique(data.T[i]))
    return levels


def single_PoAC_test(i, data, levels, ordered, a_indices):
    """Parallelizable helper function that performs a single test of the
    PoAC_and_proximity_original function
    """
    privacies = []
    aux = data == data[i]
    for j, a_ind in enumerate(a_indices):
        indices = np.all(aux[:, a_ind], axis=1)
        peers_sensitive = data[indices, j]

        if ordered[j] == 1:
            privacy = np.sqrt(np.mean((peers_sensitive - data[i, j]) ** 2))
        else:
            privacy = (np.unique(peers_sensitive).shape[0] - 1) / (levels[j] - 1)
        privacies.append(privacy)

    return privacies


def column_exclusion_indices(n):
    """Create a series of index vectors to exclude each column once
    :param n: number of columns
    Example:
    >>> column_exclusion_indices(3)
    ... array([[1, 2],   # omits 0
    ...        [0, 2],   # omits 1
    ...        [0, 1]])  # omits 2
    """
    return np.array([[x for x in range(n) if x != j] for j in range(n)])


def PoAC_and_proximity_original(data, ordered, no_tests=100):
    """With this function we compute privacy for the original data.
    This function also assess privacy for maximum auxiliary information onl,
    i.e. all variables can be used as background information.
    Optimized by Sander van Rijn <s.j.van.rijn@liacs.leidenuniv.nl> ORCID: 0000-0001-6159-041X
    :param data:      Data for which is want to compute privacy. Order is assumed to be random.
    :param sim:       Simulation specification
    :param no_tests:  Number of tests to perform
    """
    os.chdir("C:/Users/Shannon/Documents/Sanquin/Project 6")

    n, d = data.shape
    levels = extract_levels(data)

    a_indices = column_exclusion_indices(d)
    func = partial(
        single_PoAC_test, data=data, levels=levels, ordered=ordered, a_indices=a_indices
    )

    with Pool(cpu_count()) as p:
        privacy = p.map(func, range(no_tests))

    return np.array(privacy)


def PoAC_and_proximity_mspn(
    data, mspn, ordered, sens="all", p_reps=500, no_tests=100, inds="random"
):
    # data is the data that we want to test privacy for (original data)
    # mspn is the mspn that we want to know of how private it is
    # for this specific data set
    # sens is a range of variable indices that we consider to be sensitive
    # the default is that all variables are considered sensitive

    # this algorithm only considers privacy for all combinations of
    # auxiliary information and assumes all variables can be used
    # as auxiliary information
    # we still use sampling to establish proximity

    # no_tests is the number of individuals for which we evaluate privacy
    # the default is for the first 100 individuals

    # this function tests privacy only for maximum auxiliary information.

    n, d = data.shape
    levels = extract_levels(data)
    privacy = np.zeros((no_tests, d))

    # randomly sample the people under evaluation
    if inds == "random":
        random.seed(2188)
        sampled_ids = random.sample(range(n), k=no_tests)
        test_data = data[sampled_ids]
    else:
        test_data = data[inds[0] : inds[1]]

    if sens == "all":
        sens = range(data.shape[1])

    for j in sens:
        if ordered[j] == 0:
            print(j)
            domain = np.unique(data.T[j])
            vs = np.repeat(domain, no_tests).reshape(domain.shape[0], no_tests).T

            to_sample = np.repeat(test_data, (domain.shape[0]), 0)
            to_sample.T[j] = vs.reshape(-1)
            probs_sens = np.exp(log_likelihood(mspn, to_sample))
            # for each individual we scale the probability of the sensitive value
            # by the probability of having all possible sensitive values combined
            # so for a binary variable we compute the probability of 0 divided by the probability of 0 and 1
            probs_all_vs = np.sum(probs_sens.reshape(-1, domain.shape[0]), axis=1)

            probs_all_vs_rep = probs_sens.reshape(-1) / np.repeat(
                probs_all_vs, domain.shape[0]
            )
            # we count which proportion of values are considered
            # it is assumed here that the true value must be probable.
            privacy[:, j] = (
                np.sum((probs_all_vs_rep > 0.01).reshape(-1, domain.shape[0]), axis=1)
                - 1
            ) / (domain.shape[0] - 1)

        else:
            to_sample_aux = copy.deepcopy(test_data)
            to_sample_aux.T[j] = np.nan
            to_sample = np.repeat(to_sample_aux, p_reps, axis=0)
            # compute proximity
            peers_sensitive = sample_instances(
                mspn,
                np.array([to_sample]).reshape(-1, data.shape[1]),
                RandomState(123 + j),
            )
            diffs_from_sens = peers_sensitive.T[j] - np.repeat(test_data.T[j], p_reps)
            sqrt_diffs = (diffs_from_sens) ** 2

            privacy[:, j] = np.sqrt(np.mean(sqrt_diffs.reshape(-1, p_reps), axis=1))

    return privacy


def PoAC_and_proximity_mspn_no_fer(
    data, mspn, ordered, sens="all", p_reps=500, no_tests=100, no_fer=True
):
    # data is the data that we want to test privacy for (original data)
    # mspn is the mspn that we want to know of how private it is
    # for this specific data set
    # sens is a range of variable indices that we consider to be sensitive
    # the default is that all variables are considered sensitive
    # this algorithm only considers privacy for all combinations of
    # auxiliary information and assumes all variables can be used
    # as auxiliary information
    # we still use sampling to establish proximity
    # no_tests is the number of individuals for which we evaluate privacy
    # the default is for the first 100 individuals
    # this function tests privacy only for maximum auxiliary information.

    n, d = data.shape
    levels = extract_levels(data)
    privacy = np.zeros((no_tests, d))
    # randomly sample the people under evaluation
    random.seed(2188)
    sampled_ids = random.sample(range(n), k=no_tests)
    test_data = data[sampled_ids]

    if sens == "all":
        sens = range(data.shape[1])

    for j in sens:
        if ordered[j] == 0:
            print(j)
            domain = np.unique(data.T[j])
            vs = np.repeat(domain, no_tests).reshape(domain.shape[0], no_tests).T
            to_sample = np.repeat(test_data, (domain.shape[0]), 0)
            to_sample.T[j] = vs.reshape(-1)
            if no_fer:
                to_sample.T[4] = np.nan
            probs_sens = np.exp(log_likelihood(mspn, to_sample))
            # for each individual we scale the probability of the sensitive value
            # by the probability of having all possible sensitive values combined
            # so for a binary variable we compute the probability of 0 divided by the probability of 0 and 1

            # array([0.58861821, 0.41138179, 0.32505015, 0.67494985, 0.95518103,
            #        0.04481897])
            probs_all_vs = np.sum(probs_sens.reshape(-1, domain.shape[0]), axis=1)
            probs_all_vs_rep = probs_sens.reshape(-1) / np.repeat(
                probs_all_vs, domain.shape[0]
            )
            # we count which proportion of values are considered
            # it is assumed here that the true value must be probable.
            privacy[:, j] = (
                np.sum((probs_all_vs_rep > 0.01).reshape(-1, domain.shape[0]), axis=1)
                - 1
            ) / (domain.shape[0] - 1)

        else:
            to_sample_aux = copy.deepcopy(test_data)
            to_sample_aux.T[j] = np.nan
            if no_fer:
                to_sample_aux.T[4] = np.nan
            to_sample = np.repeat(to_sample_aux, p_reps, axis=0)
            # compute proximity
            peers_sensitive = sample_instances(
                mspn,
                np.array([to_sample]).reshape(-1, data.shape[1]),
                RandomState(123 + j),
            )
            diffs_from_sens = peers_sensitive.T[j] - np.repeat(test_data.T[j], p_reps)
            sqrt_diffs = (diffs_from_sens) ** 2
            privacy[:, j] = np.sqrt(np.mean(sqrt_diffs.reshape(-1, p_reps), axis=1))

    return privacy


pd.set_option("display.max_columns", None)

path = "C:/Users/Shannon/Documents/Sanquin/Project 6/Data/"


hyp_male = {
    1: {"C": 0.1, "gamma": 0.1, "kernel": "rbf"},
    2: {"C": 0.1, "gamma": 0.1, "kernel": "rbf"},
    3: {"C": 10, "gamma": 0.01, "kernel": "rbf"},
    4: {"C": 1.0, "gamma": 0.01, "kernel": "rbf"},
    5: {"C": 1.0, "gamma": 0.01, "kernel": "rbf"},
}

hyp_female = {
    1: {"C": 100, "gamma": 0.01, "kernel": "rbf"},
    2: {"C": 10, "gamma": 0.01, "kernel": "rbf"},
    3: {"C": 10, "gamma": 0.01, "kernel": "rbf"},
    4: {"C": 1.0, "gamma": 0.01, "kernel": "rbf"},
    5: {"C": 100, "gamma": 0.001, "kernel": "rbf"},
}

hyperparams = {"F": hyp_female, "M": hyp_male}


def save_object(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def prep_data(data, num_don, sex):
    var = ["KeyID", "Year", "Sex", "Time", "Age", "Month", "Last_Fer", "TimetoFer"]
    for n in range(1, num_don + 1):
        var.extend(["HbPrev" + str(n), "TimetoPrev" + str(n)])
    var.append("HbOK")
    Xy = data[var].copy()
    Xy = Xy.loc[Xy["Sex"] == sex,].dropna()

    Xy_traintest = Xy.loc[Xy["Year"] != 2021,]
    Xy_val = Xy.loc[Xy["Year"] == 2021,]

    Xy_traintest = Xy_traintest[Xy_traintest.columns[3:]]
    Xy_val = Xy_val[Xy_val.columns[3:]]

    cols_to_scale = Xy_traintest.columns[:-1]
    scaler = StandardScaler()
    print(cols_to_scale)
    print(Xy_traintest)
    scaler.fit(Xy_traintest[cols_to_scale])
    Xy_traintest[cols_to_scale] = scaler.transform(Xy_traintest[cols_to_scale])
    Xy_val[cols_to_scale] = scaler.transform(Xy_val[cols_to_scale])

    return (Xy_traintest, Xy_val, scaler)


def train_svm(data, hyperparams):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]

    hyp_c = hyperparams["C"]
    hyp_g = hyperparams["gamma"]
    hyp_k = hyperparams["kernel"]

    clf = SVC(C=hyp_c, gamma=hyp_g, kernel=hyp_k, class_weight="balanced")
    clf.fit(X, y.values.ravel())

    return clf


def calc_accuracy(clf, data):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]

    y_pred = clf.predict(X)

    return classification_report(y, y_pred, output_dict=True)


def prep_data_both_sexes(data, num_don):
    var = ["KeyID", "Year", "Sex", "Time", "Age", "Month", "Last_Fer", "TimetoFer"]
    for n in range(1, num_don + 1):
        var.extend(["HbPrev" + str(n), "TimetoPrev" + str(n)])
    var.append("HbOK")
    Xy = data[var].copy()

    Xy_traintest = Xy.loc[Xy["Year"] != 2021,]
    Xy_val = Xy.loc[Xy["Year"] == 2021,]

    Xy_traintest = Xy_traintest[Xy_traintest.columns[3:]]
    Xy_val = Xy_val[Xy_val.columns[3:]]

    cols_to_scale = Xy_traintest.columns[:-1]
    scaler = StandardScaler()
    scaler.fit(Xy_traintest[cols_to_scale])
    Xy_traintest[cols_to_scale] = scaler.transform(Xy_traintest[cols_to_scale])
    Xy_val[cols_to_scale] = scaler.transform(Xy_val[cols_to_scale])

    return (Xy_traintest, Xy_val, scaler)


def prep_data_no_accuracy(data, num_don, sex):
    var = ["KeyID", "Year", "Sex", "Time", "Age", "Month", "Last_Fer", "TimetoFer"]
    for n in range(1, num_don + 1):
        var.extend(["HbPrev" + str(n), "TimetoPrev" + str(n)])
    var.append("HbOK")
    Xy = data[var].copy()
    Xy = Xy.loc[Xy["Sex"] == sex,].dropna()

    Xy_traintest = Xy.loc[Xy["Year"] != 2021,]

    Xy_traintest = Xy_traintest[Xy_traintest.columns[3:]]

    cols_to_scale = Xy_traintest.columns[:-1]
    scaler = StandardScaler()
    scaler.fit(Xy_traintest[cols_to_scale])
    Xy_traintest[cols_to_scale] = scaler.transform(Xy_traintest[cols_to_scale])

    return (Xy_traintest, scaler)


def do_svm_no_accuracy(data, hyperparam_dict, nback):
    clfs = []
    scalers = []
    for sex in ["M", "F"]:
        print("Sex:", sex)
        print("  Prepping data")
        Xy_traintest, scaler = prep_data_no_accuracy(data, nback, sex)
        print("  Training SVM")

        clf = train_svm(Xy_traintest, hyperparam_dict[sex][nback])
        clfs.append(clf)
        scalers.append(scaler)
    return (clfs, scalers)


"""
for i in range(1, 6):
    print('Starting', i, 'at', datetime.datetime.now())
    res, clf, scaler = do_svm(df, hyperparams, i)
    filename1 = path + 'res_' + str(i) + '.pkl'
    filename2 = path + 'clf_' + str(i) + '.sav'
    filename3 = path + 'scalers_' + str(i) + '.pkl'
    pickle.dump(res, open(filename1, 'wb'))
    pickle.dump(clf, open(filename2, 'wb'))
    pickle.dump(scaler, open(filename3, 'wb'))
"""


def pretty_results(filenames, filepath=""):
    reslist = []
    for index1, filename in enumerate(filenames):
        res = pd.read_pickle(filepath + filename + ".pkl")
        for index2, cr in enumerate(res):
            trainval = ["train", "val"][index2 % 2]
            sex = ["M", "M", "F", "F"][index2 % 4]
            reslist.append(
                [
                    index1 + 1,
                    trainval,
                    sex,
                    cr["0"]["precision"],
                    cr["0"]["recall"],
                    cr["0"]["support"],
                    cr["1"]["precision"],
                    cr["1"]["recall"],
                    cr["1"]["support"],
                ]
            )
    res = pd.DataFrame(reslist).set_axis(
        [
            "nback",
            "trainval",
            "sex",
            "low_precision",
            "low_recall",
            "low_support",
            "ok_precision",
            "ok_recall",
            "ok_support",
        ],
        axis=1,
    )
    return res


def get_scores(res_df):
    res_df["old_defrate"] = res_df["low_support"] / (
        res_df["low_support"] + res_df["ok_support"]
    )
    res_df["new_defrate"] = 1 - res_df["ok_precision"]
    res_df["missed_dons"] = 1 - res_df["ok_recall"]
    res_df["prevented_defs"] = res_df["low_recall"]
    res_df["missed_per_prev"] = (
        res_df["ok_support"] - res_df["ok_recall"] * res_df["ok_support"]
    ) / (res_df["low_support"] - (1 - res_df["ok_precision"]) * res_df["ok_support"])

    res_df["old_def_n"] = res_df["low_support"]
    res_df["new_def_n"] = (1 - res_df["ok_precision"]) * res_df["ok_support"]
    res_df["old_don_n"] = res_df["ok_support"]
    res_df["new_don_n"] = res_df["ok_recall"] * res_df["ok_support"]

    return res_df


def plot_precision_recall(res_df):
    pl_df = res_df.groupby(["sex", "trainval"])

    fig, ax = plt.subplots(5, 2, figsize=(10, 12), sharey="row")

    for name, group in pl_df:
        y = 0 if name[0] == "F" else 1
        off = -0.2 if name[1] == "train" else 0.2
        ax[0, y].bar(
            group.nback + off,
            group.ok_precision,
            label=name[1],
            width=0.4,
            edgecolor="white",
        )
        ax[0, y].set_ylim(0.95, 1)
        ax[1, y].bar(
            group.nback + off,
            group.ok_recall,
            label=name[1],
            width=0.4,
            edgecolor="white",
        )
        ax[1, y].set_ylim(0.6, 0.85)

        ax[2, y].bar(
            group.nback + off,
            group.low_precision,
            label=name[1],
            width=0.4,
            edgecolor="white",
        )
        # ax[2, y].set_ylim(0.95, 1)
        ax[3, y].bar(
            group.nback + off,
            group.low_recall,
            label=name[1],
            width=0.4,
            edgecolor="white",
        )
        # ax[3, y].set_ylim(0.65, 0.85)

        ax[4, y].bar(
            group.nback + off,
            group.missed_per_prev,
            label=name[1],
            width=0.4,
            edgecolor="white",
        )

    ax[0, 1].legend(bbox_to_anchor=(1, 1), loc="upper left", title="Group")

    cols = ["Women", "Men"]
    rows = [
        "Precision - Good Hb",
        "Recall - Good Hb",
        "Precision - Low Hb",
        "Recall - Low Hb",
        "Missed donations per prevented deferral",
    ]
    xlabs = ["Previous donations used"] * 5

    for aks, row in zip(ax[:, 0], rows):
        aks.set_ylabel(row, size="large")

    for aks, col, xlab in zip(ax[0, :], cols, xlabs):
        aks.set_title(col, size="large")
        aks.set_xlabel(xlab)

    for aks, xlab in zip(ax[1, :], xlabs):
        aks.set_xlabel(xlab)

    fig.tight_layout()
    plt.set_cmap("tab20")
    plt.show()


def load_object(filename):
    with open(filename, "rb") as input_file:
        object = pickle.load(input_file)


def prep_data2021(data, num_don, sex):
    var = ["KeyID", "Year", "Sex", "Time", "Age", "Month", "Last_Fer", "TimetoFer"]
    for n in range(1, num_don + 1):
        var.extend(["HbPrev" + str(n), "TimetoPrev" + str(n)])
    var.append("HbOK")
    Xy = data[var].copy()
    Xy = Xy.loc[Xy["Sex"] == sex,].dropna()

    Xy_traintest = Xy.loc[Xy["Year"] == 2021,]
    Xy_val = Xy.loc[Xy["Year"] == 2021,]

    Xy_traintest = Xy_traintest[Xy_traintest.columns[3:]]
    Xy_val = Xy_val[Xy_val.columns[3:]]

    cols_to_scale = Xy_traintest.columns[:-1]
    scaler = StandardScaler()
    scaler.fit(Xy_traintest[cols_to_scale])
    Xy_traintest[cols_to_scale] = scaler.transform(Xy_traintest[cols_to_scale])
    Xy_val[cols_to_scale] = scaler.transform(Xy_val[cols_to_scale])

    return (Xy_traintest, Xy_val, scaler)
