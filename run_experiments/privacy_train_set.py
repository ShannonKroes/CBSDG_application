# -*- coding: utf-8 -*-
"""
In this file we evaluate privacy for the test people, who were 
used to train the model.
"""
import pickle as pickle

import numpy as np
import pandas as pd
from src.required_functions import PoAC_and_proximity_mspn, save_object

# prepare data to run svm going one donation back using all available donations.

with open("data/data_original", "rb") as input_file:
    data_original = pickle.load(input_file)

data_original = data_original.dropna()
data = data_original.drop(["KeyID"], axis=1)
data.Sex = pd.factorize(data.Sex)[0]
data = data.loc[data["Year"] != 2021,]
data = data.drop(["Year"], axis=1)
data_np = data.to_numpy()
n, d = data_np.shape

""" privacy for synthetic data """
ordered = np.array([False, True, True, True, True, True, True, True, False])

with open("results/mspn", "rb") as input_file:
    mspn = pickle.load(input_file)

# In total there are 250729 records, we generate the data in batches so that we save intermittent results.
for i in range(250):
    print("working on batch " + str(i))
    privacy_an = PoAC_and_proximity_mspn(
        data=data_np,
        inds=np.array([int(i * 1000), int((i + 1) * 1000)]),
        mspn=mspn,
        ordered=ordered,
        no_tests=1000,
    )
    save_object(privacy_an, "results/privacy_batch_train_" + str(i))

print("working on last batch")
privacy_an = PoAC_and_proximity_mspn(
    data=data_np,
    inds=np.array([int(250000), int(250729)]),
    mspn=mspn,
    ordered=ordered,
    no_tests=729,
)
save_object(privacy_an, "results/privacy_batch_train_251")
