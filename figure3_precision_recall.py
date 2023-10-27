# -*- coding: utf-8 -*-
"""
Analyze the variation of precision and recall for different synthetic data sets
as a measure of utility.
"""
import pickle
import os 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from required_functions import *

with open("data/data_original", "rb") as input_file:
    data_original= pickle.load(input_file)

with open("results/predicted_OR_F_2021", "rb") as input_file:
    predicted_OR_F_2021= pickle.load(input_file)
with open("results/predicted_OR_M_2021", "rb") as input_file:
    predicted_OR_M_2021= pickle.load(input_file)
with open("results/synth_predictions_2021", "rb") as input_file:
    res_total= pickle.load(input_file)

# We select the first two sample sizes.
res_total.shape
res = res_total[0:2]

# lens are 97353, 86514
OR_predictions_2021= np.concatenate([predicted_OR_F_2021,predicted_OR_M_2021])
# we could also look at precision and recall but for now just overlap
res1 = np.zeros(50)
res2 = np.zeros(50)

# we look at the 2021 data to extract the correct labels.
data2021= data_original[data_original.Year==2021]

X_or_F_2021, yf, z= prep_data2021(data2021, 1, "F")
X_or_M_2021, ym, z= prep_data2021(data2021, 1, "M")
# we always do females first and then males.
true_label= np.concatenate([yf['HbOK'],ym['HbOK']])
######################################################################
# Get classification table
true_neg = np.sum((true_label+OR_predictions_2021)==0) # 3338
true_pos = np.sum((true_label+OR_predictions_2021)==2) # 123089
false_neg = np.sum((true_label-OR_predictions_2021)==1) # 3338
false_pos = np.sum((true_label-OR_predictions_2021)==-1) # 123089

total_pos = np.sum(true_label==1) # 179559
total_neg = np.sum(true_label==0) # 4308

3338/4308 # specificity
123089/179559 # sensitivity

true_neg_synth = np.sum((true_label+res[0][1])==0) # 3334
true_pos_synth = np.sum((true_label+res[0][1])==2) # 122957
# so these results are very similar.

######################################################################
true_label_M = ym['HbOK']
true_label_F = yf['HbOK']
  
prec_orig = precision_score(true_label,OR_predictions_2021)
rec_orig = recall_score(true_label,OR_predictions_2021)

prec_orig_M = precision_score(true_label_M,predicted_OR_M_2021)
rec_orig_M = recall_score(true_label_M,predicted_OR_M_2021)
prec_orig_F = precision_score(true_label_F,predicted_OR_F_2021)
rec_orig_F = recall_score(true_label_F,predicted_OR_F_2021)

prec_orig_persex = [prec_orig_F, prec_orig_M]
rec_orig_persex = [rec_orig_F, rec_orig_M]

######################################################################

prec_synth = np.zeros((2, 50))
rec_synth = np.zeros((2,50))
for i in range(2):
    for j  in range (50):
        prec_synth[i][j] = precision_score(true_label,res[i][j])
        rec_synth[i][j] = recall_score(true_label,res[i][j])
        
# also compute precision and recall per sex        
ind1 = np.array([0, len(true_label_F)])
ind2 = np.array([len(true_label_F), len(true_label_F)+len(true_label_M)])

prec_synth_persex = np.zeros((2, 50))
rec_synth_persex = np.zeros((2,50))
for j  in range (50):
    prec_synth_persex[0][j] = precision_score(true_label_F,res[0][j][ind1[0]:ind2[0]])
    prec_synth_persex[1][j] = precision_score(true_label_M,res[0][j][ind1[1]:ind2[1]])    

    rec_synth_persex[0][j] = recall_score(true_label_F,res[0][j][ind1[0]:ind2[0]])
    rec_synth_persex[1][j] = recall_score(true_label_M,res[0][j][ind1[1]:ind2[1]])    


np.mean(prec_synth[0]) # 0.9920430474716895
np.mean(rec_synth[0]) # 0.6876666722358667
np.std(prec_synth[0]) # 0.00019236340860483673
np.std(rec_synth[0]) # 0.00637323353256762

# Compute true negatives, true positives, false negatives and false positives
# only for the first sample size

res_1n = res[0]
true_neg_synth = np.zeros(50)
true_pos_synth = np.zeros(50)
false_neg_synth = np.zeros(50)
false_pos_synth = np.zeros(50)

for j  in range (50):
    true_neg_synth[j] = np.sum((true_label+res_1n[j])==0)
    true_pos_synth[j] = np.sum((true_label+res_1n[j])==2)
    false_neg_synth[j] = np.sum((true_label-res_1n[j])==1)
    false_pos_synth[j] = np.sum((true_label-res_1n[j])==-1)
        
np.mean(true_neg_synth) # 3317.44    
np.mean(true_pos_synth) # 123476.74     
np.mean(false_neg_synth) # 56082.26
np.mean(false_pos_synth) # 990.56

np.std(true_neg_synth) # 31.92313267835724
np.std(true_pos_synth) # 1144.3714398743093
np.std(false_neg_synth) # 1144.3714398743093
np.std(false_pos_synth) # 31.92313

# 0 = females, 1 = males
np.mean(rec_synth_persex[0]) # 0.634249104114163
np.mean(rec_synth_persex[1]) # 0.7464080076708997
np.mean(prec_synth_persex[0]) # 0.986895451591832
np.mean(prec_synth_persex[1]) # 0.9969032224644403

np.std(rec_synth_persex[0]) #0.007374365425976907
np.std(rec_synth_persex[1]) # 0.00813023820673146
np.std(prec_synth_persex[0]) # 0.0003267521340845145
np.std(prec_synth_persex[1]) # 0.00016557739884043117

###############################################################
plt.figure(dpi=600)
sns.set_style('whitegrid')
sns.kdeplot(np.array(rec_synth_persex[1]), bw=0.4, label ="Synthetic data: males")
sns.kdeplot(np.array(rec_synth_persex[0]), bw=0.4, label ="Synthetic data: females")
plt.axvline(rec_orig_M, c="black", label="Original data")
plt.axvline(rec_orig_F, c="black")

plt.legend(loc='upper center')
plt.savefig('figures/Recall_50_reps_per_sex.png', dpi=600)


###############################################################
plt.figure(dpi=600)
sns.set_style('whitegrid')
sns.kdeplot(np.array(prec_synth_persex[1]), bw=0.4, label ="Synthetic data: males")
sns.kdeplot(np.array(prec_synth_persex[0]), bw=0.4, label ="Synthetic data: females")
plt.axvline(prec_orig_M, c="black", label="Original data")
plt.axvline(prec_orig_F, c="black")

plt.legend(loc='upper center')
plt.savefig('figures/Precision_50_reps_per_sex.png', dpi=600)

