# -*- coding: utf-8 -*-
"""
Get the privacy results for the training and test individuals.
This script yields the numbers for Table 1. 
"""
import os
import numpy as np
import pickle as pickle

os.chdir(r"\data")
with open("data_original", "rb") as input_file:
    data_original= pickle.load(input_file)   

# We create separate dataframes for the training and holdout/test data.
data_original['index_don'] = np.arange(0, data_original.shape[0])
data_train = data_original.loc[data_original['Year'] != 2021, ]

# We only select people in the holdout data who were not in the training data.
unique_ids_train = data_train.groupby('KeyID').first()
train_inds = unique_ids_train.index_don
data_test = data_original.loc[data_original['Year'] == 2021, ]
unique_ids_test = data_test.groupby('KeyID').first()
unique_ids_test['in_train'] = np.isin(unique_ids_test.index, unique_ids_train.index)
test_inds = unique_ids_test.index_don[unique_ids_test.in_train==False]

os.chdir(r"\results")
reps = 35 
# load privacy results for training data.
all_privacy_train = np.zeros((int(reps*1000), 9))
for rep in range(reps):
    with open("privacy_batch_train_"+str(rep), "rb") as input_file:
        privacy_train= pickle.load(input_file)
    all_privacy_train[int(rep*1000):int((1+rep)*1000)] = privacy_train
    
# load privacy results for test (holdout) data.
all_privacy_test = np.zeros((int(reps*1000), 9))
for rep in range(reps):
    with open("privacy_batch_test_"+str(rep), "rb") as input_file:
        privacy_test= pickle.load(input_file)
    all_privacy_test[int(rep*1000):int((1+rep)*1000)] = privacy_test
   
# Take a selection of values that have been generated.
smaller_than_highest_indtest = test_inds<len(all_privacy_test)
smaller_than_highest_indtrain = train_inds<len(all_privacy_train)
test_inds_sel = test_inds[smaller_than_highest_indtest]
train_inds_sel = train_inds[smaller_than_highest_indtrain]

# Print values for Table 1.
print(test_inds_sel.shape) 
print(train_inds_sel.shape) 
print(np.mean(all_privacy_test[test_inds_sel],0))
print(np.mean(all_privacy_train[train_inds_sel],0))
print(np.min(all_privacy_test[test_inds_sel],0))
print(np.min(all_privacy_train[train_inds_sel],0))
print(np.mean(all_privacy_test[test_inds_sel],0))
print(np.mean(all_privacy_train[train_inds_sel],0))
print(np.mean(all_privacy_test[test_inds_sel]>0,0))
print(np.mean(all_privacy_train[train_inds_sel]>0,0))
