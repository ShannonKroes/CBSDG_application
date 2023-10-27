# -*- coding: utf-8 -*-
"""
Plot the univariate distributions for Figure 2.
"""
import _pickle as cPickle
which = lambda lst:list(np.where(lst)[0])
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
with open("data/data_original", "rb") as input_file:
    data_original = cPickle.load(input_file)
# data_original.Sex= pd.factorize(data_original.Sex)[0] 
n,d= data_original.shape

with open("results/synth", "rb") as input_file:
    synth_total= cPickle.load(input_file)

synth_df= pd.DataFrame(synth_total)
synth_df.columns=data_original.columns[2:11]
data_original = data_original.loc[data_original['Year'] != 2021, ]
data_or=data_original.to_numpy().T[2:11].T
synth_df.Sex.iloc[which(synth_df.Sex==0)]="F"
synth_df.Sex.iloc[which(synth_df.Sex==1)]="M"

synth_total=synth_df.to_numpy()
data_an= synth_total

def get_cols(dat):
    dat.columns=  ['Original data','Synthetic data' ]
    return dat

datsex = get_cols(pd.DataFrame(np.vstack([np.repeat(data_or.T[0],3), synth_total.T[0]]).T))
dattime = get_cols( pd.DataFrame(np.vstack([np.repeat(data_or.T[1],3), synth_total.T[1]]).T))
datage = get_cols( pd.DataFrame(np.vstack([np.repeat(data_or.T[2],3), synth_total.T[2]]).T))
datmonth = get_cols( pd.DataFrame(np.vstack([np.repeat(data_or.T[3],3), synth_total.T[3]]).T))
datlastfer = get_cols( pd.DataFrame(np.vstack([np.repeat(data_or.T[4],3), synth_total.T[4]]).T))
dattimetofer= get_cols( pd.DataFrame(np.vstack([np.repeat(data_or.T[5],3), synth_total.T[5]]).T))
dathbprev1 = get_cols(pd.DataFrame( np.vstack([np.repeat(data_or.T[6],3), synth_total.T[6]]).T))
dattimetoprev1 = get_cols(pd.DataFrame(np.vstack([np.repeat(data_or.T[7],3), synth_total.T[7]]).T))
datHb = get_cols(pd.DataFrame(np.vstack([np.repeat(data_or.T[8],3), synth_total.T[8]]).T))

datsex_counts= np.zeros((2,2))
datsex_counts[:,0]= data_original['Sex'].value_counts().to_numpy()/len(data_original['Sex'])
datsex_counts[:,1]= synth_df['Sex'].value_counts().to_numpy()/len(synth_df['Sex'])
datsex_counts= get_cols(pd.DataFrame(datsex_counts))
datsex_counts = datsex_counts.rename( index={0: 'Female'})
datsex_counts = datsex_counts.rename( index={1: 'Male'})

datmonth_counts= np.zeros((12,2))
datmonth_counts[:,0]= datmonth['Original data'].value_counts().sort_index().to_numpy()
datmonth_counts[:,1]= datmonth['Synthetic data'].value_counts().sort_index().to_numpy()
datmonth_counts=get_cols(pd.DataFrame(datmonth_counts))
datmonth_counts = datmonth_counts.rename( index={0: 'January'})
datmonth_counts = datmonth_counts.rename( index={1: 'February'})
datmonth_counts = datmonth_counts.rename( index={2: 'March'})
datmonth_counts = datmonth_counts.rename( index={3: 'April'})
datmonth_counts = datmonth_counts.rename( index={4: 'May'})
datmonth_counts = datmonth_counts.rename( index={5: 'June'})
datmonth_counts = datmonth_counts.rename( index={6: 'July'})
datmonth_counts = datmonth_counts.rename( index={7: 'August'})
datmonth_counts = datmonth_counts.rename( index={8: 'September'})
datmonth_counts = datmonth_counts.rename( index={9: 'October'})
datmonth_counts = datmonth_counts.rename( index={10: 'November'})
datmonth_counts = datmonth_counts.rename( index={11: 'December'})
totals= np.sum(datmonth_counts.to_numpy(),0)
totals=np.repeat(totals,12 ).reshape(datmonth_counts.shape)
datmonth_counts=datmonth_counts/totals

datHb_counts= np.zeros((2,2))
datHb_counts[:,0]= data_original['HbOK'].value_counts().to_numpy()/len(data_original['HbOK'])
datHb_counts[:,1]= synth_df['HbOK'].value_counts().to_numpy()/len(synth_df['HbOK'])
datHb_counts= get_cols(pd.DataFrame(datHb_counts))
datHb_counts = datHb_counts.rename( index={0: 'Not deferred'})
datHb_counts = datHb_counts.rename( index={1: 'Deferred'})

colors= ['dodgerblue', 'orangered']
fig, axes = plt.subplots(3, 3, figsize=(16.69, 13.27))

axes[2, 0].set_title("Sex")
axes[2, 1].set_title("Month")
axes[2, 2].set_title("Deferral Status")
axes[1, 0].set_title("Time (hours)")
axes[1, 1].set_title("Last ferritin level (ng/mL)")
axes[1, 2].set_title("Time since last ferritin measurement (days)")
axes[0, 0].set_title("Previous Hb (mM/L)")
axes[0, 1].set_title("Time since previous Hb measurement (days)")
axes[0, 2].set_title("Age (years)")

dathbprev1.plot.density(linewidth=1, legend=False, ax=axes[0, 0] , alpha=1,bw_method=0.3, style=['b-', 'r--'])
dattimetoprev1.plot.density(linewidth=1, legend=False, ax=axes[0, 1], alpha=1,  style=['b-', 'r--'])
datage.plot.density(linewidth=1, legend=False, ax=axes[0, 2], alpha=1, style=['b-', 'r--'])
axes[0, 0].set_ylabel("Density")
axes[0, 1].set_ylabel("")
axes[0, 2].set_ylabel("")
axes[0, 1].set_xlim(np.min(dattimetoprev1.min()),np.max(dattimetoprev1.max()))
axes[0, 2].set_xlim(np.min(datage.min()),np.max(datage.max()))
axes[0, 0].set_xlim(np.min(dathbprev1.min()),np.max(dathbprev1.max()))

dattime.plot.density(linewidth=1, legend=False, ax=axes[1, 0] , alpha=1, style=['b-', 'r--'])
datlastfer.plot.density(linewidth=1, legend=False, ax=axes[1, 1], alpha=1,  style=['b-', 'r--'])
dattimetofer.plot.density(linewidth=1, legend=False, ax=axes[1, 2], alpha=1 , style=['b-', 'r--'])

axes[1, 0].set_ylabel("Density")
axes[1, 1].set_ylabel("")
axes[1, 2].set_ylabel("")
axes[1, 1].set_xlim(np.min(datlastfer.min()),np.max(datlastfer.max()))
axes[1, 2].set_xlim(np.min(dattimetofer.min()),np.max(dattimetofer.max()))
axes[1, 0].set_xlim(np.min(dattime.min()),np.max(dattime.max()))


datsex_counts.plot.bar(linewidth=1, legend=False, ax=axes[2, 0], alpha=1, color = colors)
datmonth_counts.plot.bar(linewidth=1, legend=False, ax=axes[2, 1], alpha=1, color = colors)
datHb_counts.plot.bar(linewidth=1, legend=False, ax=axes[2,2], alpha=1, color = colors)
axes[2, 0].set_ylabel("Proportion")
axes[2, 2].set_ylabel("")
axes[2, 1].set_ylabel("")

lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'lower center',  prop={'size': 6}, ncol=2)
plt.savefig('figures/univariate_distributions.png', dpi=600)


