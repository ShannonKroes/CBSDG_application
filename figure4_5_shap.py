# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:01:48 2022

@author: Shannon

Get results for accuracy.
We analyze the results that were generated by the server.
"""
import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# prepare data to run svm going one donation back using all available donations.
with open("data/data_original", "rb") as input_file:
    data_original = pickle.load(input_file)
n, d = data_original.shape

with open("run_experiments/results/synth", "rb") as input_file:
    synth_total = pickle.load(input_file)

n, d = data_original.shape
# We remove the KeyIDs and Year, because we will not anonymize this.
data2021 = data_original.loc[data_original["Year"] == 2021,]
data = data_original.drop(["KeyID"], axis=1)
data = data.loc[data["Year"] != 2021,]
data = data.drop(["Year"], axis=1)
data.Sex = pd.factorize(data.Sex)[0]
data_np = data.to_numpy()

reps = 40
ns = np.array(np.array([1]) * (data_np.shape[0]), dtype=int)
sample_size = 100

shapvals_all = np.zeros((2, len(ns), reps, 100, 7))
shapmeans = np.zeros((2, len(ns), reps, 7))
for n_i, n_ in enumerate(ns):
    for r, rep in enumerate(range(10, 50)):
        name = "rep" + str(rep) + "_n_" + str(n_i)
        # print('n'+str(n_i))
        for i, sex in enumerate(["M", "F"]):
            filename1 = "synth_X_" + name + sex + str(sample_size) + ".pkl"
            filename2 = "synth_shap_" + name + sex + str(sample_size) + ".pkl"
            # X_test is altijd hetzelfde, maar anders per geslacht
            X_test = pickle.load(open("results/" + filename1, "rb"))
            # print(X_test.iloc[0])
            shapvals = pickle.load(open("results/" + filename2, "rb"))
            shapvals_all[i][n_i][r] = shapvals
            shapmeans[i][n_i][r] = np.mean(np.absolute(shapvals), axis=0)
            # print(np.mean(np.absolute(shapvals), axis=0))

shapmeans_M = shapmeans[0]
shapmeans_F = shapmeans[1]

path = "results/"
name = "shap_values_rep" + str(0) + "_n_" + str(0)
sex = "F"
n = 100
filename1 = name + sex + str(100) + ".pkl"
filename2 = name + sex + str(100) + ".pkl"
shapvals = pickle.load(open(path + filename2, "rb"))
OR_shap_F = np.mean(np.absolute(shapvals), axis=0)
sex = "M"
filename1 = name + sex + str(100) + ".pkl"
filename2 = name + sex + str(100) + ".pkl"
shapvals = pickle.load(open(path + filename2, "rb"))
OR_shap_M = np.mean(np.absolute(shapvals), axis=0)

var_names = np.array(
    [
        "Time (hours)",
        "Age (years)",
        "Month",
        "Last ferritin",
        "Time to last ferritin (days)",
        "Previous Hb",
        "Time to previous Hb (days)",
    ]
)

# try with broken axis
shapmeans_F0 = shapmeans_F[0]
var = np.repeat(var_names.reshape(-1, 1), reps, axis=-1).T.reshape(1, -1)[0]
df = pd.DataFrame()
df["shap"] = shapmeans_F0.reshape(-1)
df["var"] = var
f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True, dpi=400)
ax = sns.violinplot(data=df, x="shap", y="var", inner=None, ax=ax1, color="darksalmon")
ax1.scatter(OR_shap_F, var_names, c="black", s=25)
plt.ylabel("")
ax = sns.violinplot(data=df, x="shap", y="var", inner=None, ax=ax2, color="darksalmon")
ax2.scatter(OR_shap_F, var_names, c="black", s=25)
ax1.set_xlim(0, 0.15)
ax1.set_xticks(np.array([0, 0.05, 0.1, 0.15]))
ax2.set_xlim(0.3, 0.45)
ax2.tick_params(left=False)
ax1.tick_params(left=False)
plt.subplots_adjust(wspace=0.2, hspace=0)
ax2.spines[["left"]].set_visible(False)
ax1.spines[["right"]].set_visible(False)
ax1.spines[["top"]].set_visible(False)
ax2.spines[["top"]].set_visible(False)
ax2.spines[["right"]].set_visible(False)
ax1.spines[["left"]].set_visible(False)

d = 0.015  
kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
ax1.plot((-0.2 - d, -0.2 + d), (-d, +d), **kwargs)
ax1.set_ylabel(" ")
ax2.set_ylabel(" ")
ax1.set_xlabel(" ")
ax2.set_xlabel(" ")
kwargs.update(transform=ax2.transAxes) 
ax2.plot((-d, +d), (-d, +d), **kwargs)
f.text(0.5, 0.0, "SHAP values", ha="center")
plt.savefig("figures/shap_females.png", dpi=600, bbox_inches="tight")

# try with broken axis
shapmeans_M0 = shapmeans_M[0]
var = np.repeat(var_names.reshape(-1, 1), reps, axis=-1).T.reshape(1, -1)[0]
df = pd.DataFrame()
df["shap"] = shapmeans_M0.reshape(-1)
df["var"] = var
f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=True, dpi=400)
ax = sns.violinplot(
    data=df, x="shap", y="var", inner=None, ax=ax1, color="lightsteelblue"
)
ax1.scatter(OR_shap_M, var_names, c="black", s=25)
plt.ylabel("")
ax = sns.violinplot(
    data=df, x="shap", y="var", inner=None, ax=ax2, color="lightsteelblue"
)
ax2.scatter(OR_shap_M, var_names, c="black", s=25)
ax1.set_xlim(0, 0.15)
ax1.set_xticks(np.array([0, 0.05, 0.1, 0.15]))
ax2.set_xlim(0.3, 0.45)
ax2.tick_params(left=False)
ax1.tick_params(left=False)
plt.subplots_adjust(wspace=0.2, hspace=0)
ax2.spines[["left"]].set_visible(False)
ax1.spines[["right"]].set_visible(False)
ax1.spines[["top"]].set_visible(False)
ax2.spines[["top"]].set_visible(False)
ax2.spines[["right"]].set_visible(False)
ax1.spines[["left"]].set_visible(False)

d = 0.015 
kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
ax1.plot((-0.2 - d, -0.2 + d), (-d, +d), **kwargs)
ax1.set_ylabel(" ")
ax2.set_ylabel(" ")
ax1.set_xlabel(" ")
ax2.set_xlabel(" ")
kwargs.update(transform=ax2.transAxes) 
ax2.plot((-d, +d), (-d, +d), **kwargs)
f.text(0.5, 0.0, "SHAP values", ha="center")
plt.savefig("figures/shap_males.png", dpi=600, bbox_inches="tight")
