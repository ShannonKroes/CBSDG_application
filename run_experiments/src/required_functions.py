# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:09:39 2022

@author: Shannon
"""
import numpy as np
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import shap 
from functools import partial
from multiprocessing import Pool, cpu_count
import copy 
import random
from numpy.random.mtrand import RandomState
from spn.algorithms.Sampling import sample_instances
from spn.algorithms.Inference import log_likelihood

which = lambda lst:list(np.where(lst)[0])

def extract_levels(data):
    levels=np.zeros(data.shape[1])
    
    for i in range(data.shape[1]):
        levels[i]= len(np.unique(data.T[i]))
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
            privacy = (np.unique(peers_sensitive).shape[0] - 1) / (levels[j]-1)
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
    return np.array([
        [x for x in range(n) if x != j]
        for j in range(n)
    ])


def PoAC_and_proximity_original(data, ordered, no_tests=100):
    """With this function we compute privacy for the original data.
    This function also assess privacy for maximum auxiliary information onl,
    i.e. all variables can be used as background information.
    Optimized by Sander van Rijn <s.j.van.rijn@liacs.leidenuniv.nl> ORCID: 0000-0001-6159-041X
    :param data:      Data for which is want to compute privacy. Order is assumed to be random.
    :param sim:       Simulation specification
    :param no_tests:  Number of tests to perform
    """
    n, d = data.shape
    levels = extract_levels(data)
  
    a_indices = column_exclusion_indices(d)
    func = partial(single_PoAC_test, data=data, levels=levels, ordered=ordered, a_indices=a_indices)

    with Pool(cpu_count()) as p:
        privacy = p.map(func, range(no_tests))

    return np.array(privacy)



def PoAC_and_proximity_mspn(data,mspn,ordered, sens= "all", p_reps=500,
                            no_tests=100, inds='random' ):
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
        
    n,d = data.shape
    privacy = np.zeros((no_tests,d))

    # randomly sample the people under evaluation  
    if inds == 'random':       
        random.seed(2188)
        sampled_ids = random.sample(range(n), k=no_tests)
        test_data = data[sampled_ids]
    else:
        test_data = data[inds[0]:inds[1]]
        
    if sens=="all":
        sens = range(data.shape[1])   

    for j in sens:
        
        if ordered[j]==0:
            print(j)
            domain = np.unique(data.T[j])
            vs = np.repeat(domain, no_tests).reshape(domain.shape[0], no_tests).T

            to_sample = np.repeat(test_data,(domain.shape[0]),0)
            to_sample.T[j] = vs.reshape(-1)
            probs_sens=np.exp(log_likelihood(mspn,to_sample))
            # for each individual we scale the probability of the sensitive value
            # by the probability of having all possible sensitive values combined
            # so for a binary variable we compute the probability of 0 divided by the probability of 0 and 1
            probs_all_vs= np.sum(probs_sens.reshape(-1, domain.shape[0]), axis=1)
            
            probs_all_vs_rep= probs_sens.reshape(-1)/np.repeat(probs_all_vs,  domain.shape[0])
            # we count which proportion of values are considered
            # it is assumed here that the true value must be probable. 
            privacy[:,j]= (np.sum((probs_all_vs_rep>.01).reshape(-1, domain.shape[0]), axis=1)-1)/(domain.shape[0]-1)
                         
        else:                   
            to_sample_aux= copy.deepcopy(test_data)
            to_sample_aux.T[j] = np.nan
            to_sample= np.repeat(to_sample_aux, p_reps, axis=0)
            # compute proximity 
            peers_sensitive= sample_instances(mspn, np.array([to_sample]).reshape(-1, data.shape[1]), RandomState(123+j))
            diffs_from_sens= peers_sensitive.T[j]-np.repeat(test_data.T[j],p_reps)
            sqrt_diffs= (diffs_from_sens)**2
            
            privacy[:,j]=np.sqrt(np.mean(sqrt_diffs.reshape(-1, p_reps), axis=1))

    return privacy


pd.set_option('display.max_columns', None)

path = 'C:/Users/Shannon/Documents/Sanquin/Project 6/Data/'


hyp_male = {1: {'C': 0.1, 'gamma': 0.1, 'kernel': 'rbf'},
            2: {'C': 0.1, 'gamma': 0.1, 'kernel': 'rbf'},
            3: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'},
            4: {'C': 1.0, 'gamma': 0.01, 'kernel': 'rbf'}, 
            5: {'C': 1.0, 'gamma': 0.01, 'kernel': 'rbf'}}

hyp_female = {1: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'},
              2: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'},
              3: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'},
              4: {'C': 1.0, 'gamma': 0.01, 'kernel': 'rbf'},
              5: {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}}

hyperparams = {'F': hyp_female,
               'M': hyp_male}

def calc_shap(data, clf, name, sample_size=100):
    for index, sex in enumerate(['M', 'F']):
        train, val, scaler = prep_data(data, 1, sex)
        clf_s = clf[index]
        X_val = val[val.columns[:-1]]
        X_shap = shap.sample(X_val, sample_size)
        explainer = shap.KernelExplainer(clf_s.predict, X_shap)
        shapvals = explainer.shap_values(X_shap)
        
        path_shap = "results/"
        filename1 = 'AN_X_'+ name + sex + str(sample_size) + '.pkl'
        filename2 = 'AN_shap_' + name + sex + str(sample_size) + '.pkl'
        
        pickle.dump(X_shap, open(path_shap+filename1, 'wb'))
        pickle.dump(shapvals, open(path_shap+filename2, 'wb'))

def save_object(obj, filename):
    """
    
    :param obj: DESCRIPTION
    :type obj: TYPE
    :param filename: DESCRIPTION
    :type filename: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    with open(filename, 'wb') as output: 
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
def prep_data(data, num_don, sex):
    var = ['KeyID', 'Year', 'Sex', 'Time', 'Age', 'Month', 'Last_Fer', 'TimetoFer']
    for n in range(1, num_don+1):
        var.extend(['HbPrev'+str(n), 'TimetoPrev'+str(n)])
    var.append('HbOK')
    Xy = data[var].copy()
    Xy = Xy.loc[Xy['Sex'] == sex, ].dropna()
    
    Xy_traintest = Xy.loc[Xy['Year'] != 2021, ]
    Xy_val = Xy.loc[Xy['Year'] == 2021, ]
    
    Xy_traintest = Xy_traintest[Xy_traintest.columns[3:]]
    Xy_val = Xy_val[Xy_val.columns[3:]]
    
    cols_to_scale = Xy_traintest.columns[:-1]
    scaler = StandardScaler()
    print(cols_to_scale)
    print(Xy_traintest)
    scaler.fit(Xy_traintest[cols_to_scale])
    Xy_traintest[cols_to_scale] = scaler.transform(Xy_traintest[cols_to_scale])
    Xy_val[cols_to_scale] = scaler.transform(Xy_val[cols_to_scale])
    
    return(Xy_traintest, Xy_val, scaler)

def train_svm(data, hyperparams):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    
    hyp_c = hyperparams['C']
    hyp_g = hyperparams['gamma']
    hyp_k = hyperparams['kernel']
    
    clf = SVC(C = hyp_c, gamma = hyp_g, kernel = hyp_k, class_weight = 'balanced')
    clf.fit(X, y.values.ravel())
    
    return(clf)

def calc_accuracy(clf, data):
    X = data[data.columns[:-1]]
    y = data[data.columns[-1:]]
    
    y_pred = clf.predict(X)
    
    return(classification_report(y, y_pred, output_dict=True))

def prep_data_both_sexes(data, num_don):
    var = ['KeyID', 'Year', 'Sex', 'Time', 'Age', 'Month', 'Last_Fer', 'TimetoFer']
    for n in range(1, num_don+1):
        var.extend(['HbPrev'+str(n), 'TimetoPrev'+str(n)])
    var.append('HbOK')
    Xy = data[var].copy()
   
    Xy_traintest = Xy.loc[Xy['Year'] != 2021, ]
    Xy_val = Xy.loc[Xy['Year'] == 2021, ]
    
    Xy_traintest = Xy_traintest[Xy_traintest.columns[3:]]
    Xy_val = Xy_val[Xy_val.columns[3:]]
    
    cols_to_scale = Xy_traintest.columns[:-1]
    scaler = StandardScaler()
    scaler.fit(Xy_traintest[cols_to_scale])
    Xy_traintest[cols_to_scale] = scaler.transform(Xy_traintest[cols_to_scale])
    Xy_val[cols_to_scale] = scaler.transform(Xy_val[cols_to_scale])
    
    return(Xy_traintest, Xy_val, scaler)

def prep_data_no_accuracy(data, num_don, sex):
    var = ['KeyID', 'Year', 'Sex', 'Time', 'Age', 'Month', 'Last_Fer', 'TimetoFer']
    for n in range(1, num_don+1):
        var.extend(['HbPrev'+str(n), 'TimetoPrev'+str(n)])
    var.append('HbOK')
    Xy = data[var].copy()
    Xy = Xy.loc[Xy['Sex'] == sex, ].dropna()
    
    Xy_traintest = Xy.loc[Xy['Year'] != 2021, ]
    
    Xy_traintest = Xy_traintest[Xy_traintest.columns[3:]]
    
    cols_to_scale = Xy_traintest.columns[:-1]
    scaler = StandardScaler()
    scaler.fit(Xy_traintest[cols_to_scale])
    Xy_traintest[cols_to_scale] = scaler.transform(Xy_traintest[cols_to_scale])
    
    return(Xy_traintest,  scaler)

def do_svm_no_accuracy(data, hyperparam_dict, nback):
    clfs = []
    scalers = []
    for sex in ['M','F']:
        print('Sex:', sex)
        print('  Prepping data')
        Xy_traintest, scaler = prep_data_no_accuracy(data, nback, sex)
        print('  Training SVM')
        
        clf = train_svm(Xy_traintest, hyperparam_dict[sex][nback])
        clfs.append(clf)
        scalers.append(scaler)
    return( clfs, scalers)


def plot_precision_recall(res_df):
    pl_df = res_df.groupby(['sex', 'trainval'])

    fig, ax = plt.subplots(5, 2, figsize=(10, 12), sharey='row')

    for name, group in pl_df:
        y = 0 if name[0] == 'F' else 1
        off = -0.2 if name[1] == 'train' else 0.2
        ax[0, y].bar(group.nback + off, group.ok_precision, label=name[1], width=0.4, edgecolor='white')
        ax[0, y].set_ylim(0.95, 1)
        ax[1, y].bar(group.nback + off, group.ok_recall, label=name[1], width=0.4, edgecolor='white')
        ax[1, y].set_ylim(0.6, 0.85)
        
        ax[2, y].bar(group.nback + off, group.low_precision, label=name[1], width=0.4, edgecolor='white')
        # ax[2, y].set_ylim(0.95, 1)
        ax[3, y].bar(group.nback + off, group.low_recall, label=name[1], width=0.4, edgecolor='white')
        # ax[3, y].set_ylim(0.65, 0.85)
        
        ax[4, y].bar(group.nback + off, group.missed_per_prev, label=name[1], width=0.4, edgecolor='white')

    ax[0, 1].legend(bbox_to_anchor=(1, 1), loc='upper left', title='Group')

    cols = ['Women', 'Men']
    rows = ['Precision - Good Hb', 'Recall - Good Hb', 'Precision - Low Hb', 'Recall - Low Hb', 'Missed donations per prevented deferral']
    xlabs = ['Previous donations used'] * 5 

    for aks, row in zip(ax[:, 0], rows):
        aks.set_ylabel(row, size='large')

    for aks, col, xlab in zip(ax[0,:], cols, xlabs):
        aks.set_title(col, size='large')
        aks.set_xlabel(xlab)

    for aks, xlab in zip(ax[1,:], xlabs):
        aks.set_xlabel(xlab)

    fig.tight_layout()
    plt.set_cmap('tab20')
    plt.show()  
    

def prep_data2021(data, num_don, sex):
    var = ['KeyID', 'Year', 'Sex', 'Time', 'Age', 'Month', 'Last_Fer', 'TimetoFer']
    for n in range(1, num_don+1):
        var.extend(['HbPrev'+str(n), 'TimetoPrev'+str(n)])
    var.append('HbOK')
    Xy = data[var].copy()
    Xy = Xy.loc[Xy['Sex'] == sex, ].dropna()
    
    Xy_traintest = Xy.loc[Xy['Year'] == 2021, ]
    Xy_val = Xy.loc[Xy['Year'] == 2021, ]
    
    Xy_traintest = Xy_traintest[Xy_traintest.columns[3:]]
    Xy_val = Xy_val[Xy_val.columns[3:]]
    
    cols_to_scale = Xy_traintest.columns[:-1]
    scaler = StandardScaler()
    scaler.fit(Xy_traintest[cols_to_scale])
    Xy_traintest[cols_to_scale] = scaler.transform(Xy_traintest[cols_to_scale])
    Xy_val[cols_to_scale] = scaler.transform(Xy_val[cols_to_scale])
    
    return(Xy_traintest, Xy_val, scaler)