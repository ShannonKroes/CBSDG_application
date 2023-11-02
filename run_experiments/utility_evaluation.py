# -*- coding: utf-8 -*-
"""
In this file we generate a synthetic data set 50 times, train a classifier
and get predictions for the test data with each classifier.
"""
import pickle as pickle

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from spn.algorithms.LearningWrappers import learn_mspn
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from src.required_functions import (
    calc_shap,
    do_svm_no_accuracy,
    prep_data2021,
    save_object,
)

# We use the same hyperparameter settings as used by Vinkenoog et al.
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

with open("data/data_original", "rb") as input_file:
    data_original = pickle.load(input_file)

# We extract the test data.
data2021 = data_original[data_original.Year == 2021]
X_or_F_2021, y, z = prep_data2021(data2021, 1, "F")
X_or_M_2021, y, z = prep_data2021(data2021, 1, "M")
X_or_F_2021 = X_or_F_2021[X_or_F_2021.columns[:-1]]
X_or_M_2021 = X_or_M_2021[X_or_M_2021.columns[:-1]]

# we repeat the analysis 50 times
reps = 50

# Extract training data and preprocess.
data = data_original.drop(["KeyID"], axis=1)
data = data.loc[data["Year"] != 2021,]
data = data.drop(["Year"], axis=1)
data.Sex = pd.factorize(data.Sex)[0]
data_np = data.to_numpy()
n, d = data_np.shape
ns = np.array(np.array([1, 2, 3, 4]) * (n), dtype=int)
synth_predictions_2021 = np.zeros((ns.shape[0], reps, 183867))

# State whether values are discrete or continuous, as is needed to construct an MSPN.
ds_context = Context(
    meta_types=np.array(
        [
            MetaType.DISCRETE,
            MetaType.REAL,
            MetaType.REAL,
            MetaType.DISCRETE,
            MetaType.REAL,
            MetaType.REAL,
            MetaType.REAL,
            MetaType.REAL,
            MetaType.DISCRETE,
        ]
    )
)
ds_context.add_domains(data_np)


# Before we start running the experiment we define a function to go from the
# the generated synthetic data to the same format as the original data
# (dataframe with same column names, etc.).
def pre_process_synth(synth):
    """
    Preprocess the synthetic data.

    :param synth: Numpy array with synthetic data.
    :rtype: Dataframe with preprocessed synthetic data.

    """
    which = lambda lst: list(np.where(lst)[0])

    synth_df = pd.DataFrame(synth)
    synth_df.columns = data.columns

    synth_df.Sex.iloc[which(synth_df.Sex == 0)] = "F"
    synth_df.Sex.iloc[which(synth_df.Sex == 1)] = "M"

    # add the KeyIDs random for now
    synth_df["KeyID"] = 1
    # also round the years, last fer, timetofer and timetoprev  to integers
    synth_df["Year"] = 2017
    synth_df.TimetoPrev1 = synth_df.TimetoPrev1.round()
    synth_df.Last_Fer = synth_df.Last_Fer.round()
    synth_df.TimetoFer = synth_df.TimetoFer.round()
    return synth_df


for n_i, n_ in enumerate(ns):
    for rep in range(reps):
        np.random.seed(int(2206 + rep))
        # Create an MSPN of the data.
        mspn = learn_mspn(
            data_np,
            ds_context,
            min_instances_slice=data_np.shape[0] - 1,
            rand_gen=int(2206 + rep),
            rows="kmeans",
            threshold=-1,
            standardize=True,
            no_clusters=4000,
        )
        # Generate synthetic data.
        synth_current = sample_instances(
            mspn,
            np.array([[np.repeat(np.nan, d)] * int(n_)]).reshape(-1, d),
            RandomState(2206 + rep),
        )
        synth_df_current = pre_process_synth(synth_current)
        if rep == 0:
            save_object(synth_current, "results/synth")
            save_object(mspn, "results/mspn")

        for i in range(1, 2):
            # Train a classifier with the synthetic data
            clf, scaler = do_svm_no_accuracy(synth_df_current, hyperparams, i)

        # Compute the SHAP values for the ith classifier and save these.
        calc_shap(
            data_original,
            clf,
            name="rep" + str(rep) + "_n_" + str(n_i),
            sample_size=100,
        )

        # Make predictions with this classifier on the test/holdout set.
        clfF = clf[1]
        clfM = clf[0]
        predicted_synth_F_2021 = clfF.predict(X_or_F_2021)
        predicted_synth_M_2021 = clfM.predict(X_or_M_2021)

        # Save these predictions.
        synth_predictions_2021[n_i][rep] = np.concatenate(
            [predicted_synth_F_2021, predicted_synth_M_2021]
        )

save_object(synth_predictions_2021, "results/synth_predictions_2021")
