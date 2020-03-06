#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:19:00 2020

@author: podvae01
"""
import HLTP_pupil
from HLTP_pupil import MEG_pro_dir, subjects
import numpy as np
import mne
import scipy
from scipy.optimize import curve_fit
import scipy.stats
from scipy.stats import sem
import pandas as pd

bhv_df = pd.read_pickle(MEG_pro_dir + '/results/all_subj_bhv_df.pkl')
group_percentile = np.arange(0., 100., 20);

p_group = []; pupil_epochs = []; mean_pup = []
for s, subject in enumerate(subjects):
    sub_pro_dir = MEG_pro_dir + '/' + subject
    
    epoch_fname = sub_pro_dir + '/task_prestim_ds-epo.fif'
    epochs = mne.read_epochs(epoch_fname, preload = True )
    mean_pupil = epochs._data[:, HLTP_pupil.pupil_chan, :].mean(axis = 1)
    perc = np.percentile(mean_pupil, group_percentile)
    p_group.append(np.digitize(mean_pupil, perc))
    mean_pup.append(mean_pupil)
    
    epoch_fname = sub_pro_dir + '/task_posstim_ds-epo.fif'
    epochs = mne.read_epochs(epoch_fname, preload = True )
    pupil_epochs.append(epochs._data[:, HLTP_pupil.pupil_chan, :])
    
m = np.zeros( (24, 5, 513) )

fig, ax = plt.subplots(1, 1, figsize = (3.5,2.5))
    
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')       
for grp in range(2):
    for s, subject in enumerate(subjects):
        seen = bhv_df[(bhv_df.subject == subject)].seen
        #trials = np.where(p_group[s] == grp)[0]
        trials = np.where(seen == grp)[0]
        if (subject == 'BJB') : trials = trials[:-1]
        psc = ( pupil_epochs[s][trials, :] - mean_pup[s][trials].mean())
        m[s, grp - 1, :] = psc.mean(axis = 0)

    mm = np.mean(m[:, grp - 1, :], axis = 0)   
    er = sem(m[:, grp - 1, :], axis = 0)   
    plt.plot(epochs.times, mm)
    plt.fill_between(epochs.times, mm- er, mm+ er, 
                         alpha =.3, label = grp)
plt.ylabel('Pupil'); plt.xlabel('Time')
fig.savefig(figures_dir + '/pupil_yes_no_tbd.png', 
                bbox_inches = 'tight', transparent=True)

plt.legend()

