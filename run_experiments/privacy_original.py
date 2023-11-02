# -*- coding: utf-8 -*-
"""
In this file we evaluate privacy for the test people, who were 
used to train the model.
"""
import pickle
import numpy as np
import pandas as pd
from src.required_functions import PoAC_and_proximity_original, save_object

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

""" privacy for original data """
ordered = np.array([False, True, True, True, True, True, True, True, False])
privacy_or = PoAC_and_proximity_original(data, ordered, no_tests=n)

save_object(privacy_or, "results/privacy_or")
