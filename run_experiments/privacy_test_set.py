# -*- coding: utf-8 -*-
"""
In this file we evaluate privacy for the test people, who were NOT
used to train the model.
"""
from src.required_functions import PoAC_and_proximity_mspn, save_object
import pandas as pd
import numpy as np
import pickle as pickle 

with open("data/data_original", "rb") as input_file:
    data_original= pickle.load(input_file)   

data_original = data_original.dropna()
data2021= data_original.loc[data_original['Year'] == 2021, ]
data2021 = data2021.drop(['KeyID'], axis = 1)
data2021.Sex = pd.factorize(data2021.Sex)[0] 
data2021= data2021.drop(['Year'], axis = 1)
data2021_np = data2021.to_numpy()
n,d = data2021_np.shape 

''' privacy for synthetic data '''
ordered=np.array([False, True,True,True, True, True, True, True, False])
with open("results/mspn", "rb") as input_file:
    mspn= pickle.load(input_file)

# In total there are 183867 records, we generate the data in batches so that we save intermittent results.
for i in range(183):
    print("working on batch " +str(i))
    privacy_an= PoAC_and_proximity_mspn(data=data2021_np, inds=np.array([int(i*1000), int((i+1)*1000)]), mspn=mspn, ordered=ordered, no_tests=1000)
    save_object(privacy_an, 'results/privacy_batch_test_'+str(i))

print("working on last batch")
privacy_an= PoAC_and_proximity_mspn(data=data2021_np, inds=np.array([int(183000), int(183867)]), mspn=mspn, ordered=ordered, no_tests=867)
save_object(privacy_an, 'results/privacy_batch_test_184')
