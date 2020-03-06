#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:14:20 2020

@author: podvae01
"""
import HLTP_pupil
from HLTP_pupil import MEG_pro_dir, subjects
import numpy as np
import scipy
import scipy.stats
from scipy.stats import sem
import pandas as pd
import mne
from mne.time_frequency import psd_multitaper
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

bhv_df = pd.read_pickle(MEG_pro_dir + '/results/all_subj_bhv.pkl')

group_percentile = np.arange(0., 100., 20);

p_group = []; pupil_epochs = []; mean_meg = []
for s, subject in enumerate(subjects):
    sub_pro_dir = MEG_pro_dir + '/' + subject
    
    epoch_fname = sub_pro_dir + '/task_prestim_ds-epo.fif'
    epochs = mne.read_epochs(epoch_fname, preload = True )
    mean_pupil = epochs._data[:, HLTP_pupil.pupil_chan, :].mean(axis = 1)
    perc = np.percentile(mean_pupil, group_percentile)
    p_group.append(np.digitize(mean_pupil, perc))
    
    epoch_fname = sub_pro_dir + '/task_posstim_ds-epo.fif'
    epochs = mne.read_epochs(epoch_fname, preload = True )
    #epochs.apply_baseline(baseline = (0, .05))
    # check BLP here
    epochs.resample(20)
    picks = mne.pick_types(epochs.info, meg = True, 
                           ref_meg = False, exclude = 'bads')
    MEG_data = epochs.get_data(picks)
        
    mean_meg.append(MEG_data.copy())
    
n_chan = MEG_data.shape[1]; n_times = MEG_data.shape[2]; n_subj = len(subjects)

# EFFECT OF PUPIL ON REACTION TIMES
allCRT = np.zeros( (5, 24) ); allRRT = np.zeros( (5, 24) );
for s, subject in enumerate(subjects): 
    rim = bhv_df[bhv_df.subject == subject].real_img
    if (subject == 'BJB') : rim = rim[:-1]

    cat_RT = bhv_df[bhv_df.subject == subject].cat_RT
    rec_RT = bhv_df[bhv_df.subject == subject].rec_RT
    for grp in range(1, 6):    
        trials = np.where( (p_group[s] == grp) & rim)[0]
        if (subject == 'BJB') : trials = trials[:-1]

        allCRT[grp - 1, s] = cat_RT[trials].mean()
        allRRT[grp - 1, s] = rec_RT[trials].mean()

fig, ax = plt.subplots(1, 1, figsize = (2.5,1.5))
plt.errorbar(range(5), allCRT.mean(axis = -1), 
             allCRT.std(axis = -1) / np.sqrt(n_subj), color = 'k')
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
plt.xlim([-1, 5]); plt.ylim([1, 1.2])
plt.xlabel('Pupil'); plt.ylabel('RT (s)')
fig.savefig(figures_dir + '/pupil_cat_RT.png',
                dpi = 800,  bbox_inches = 'tight',transparent = True)


fig, ax = plt.subplots(1, 1, figsize = (2.5,1.5))
plt.errorbar(range(5), allRRT.mean(axis = -1), 
             allRRT.std(axis = -1) / np.sqrt(n_subj), color = 'k')
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
plt.xlim([-1, 5]); plt.ylim([.6, .8])
plt.xlabel('Pupil'); plt.ylabel('RT (s)')
fig.savefig(figures_dir + '/pupil_rec_RT.png',
                dpi = 800,  bbox_inches = 'tight',transparent = True)


plt.plot(allCRT.mean(axis = -1))
plt.plot(allRRT.mean(axis = -1))

   
# EFFECT OF PRE-STIM PUPIL ON CATEGORY REPRESENTATION
clf = make_pipeline(StandardScaler(), 
                    LogisticRegression(C = 1, 
                                       solver='newton-cg', 
                                       multi_class ='multinomial'))
time_decod = SlidingEstimator(clf, n_jobs=1, verbose=True)
cv = LeaveOneOut()
scores = np.zeros( (5, n_times, n_subj) )
s_scores = np.zeros( (5, n_times, n_subj) ); s_scores[:] = np.nan
u_scores = np.zeros( (5, n_times, n_subj) ); u_scores[:] = np.nan
scores = np.zeros( (5, n_times, n_subj) ); scores[:] = np.nan

for s, subject in enumerate(subjects): 
    MEG_data = mean_meg[s]# n_trials x n_channels x n_times        
    cat = bhv_df[bhv_df.subject == subject].cat_protocol.values   
    rim = bhv_df[bhv_df.subject == subject].real_img
    seen = bhv_df[bhv_df.subject == subject].seen.values
    unseen = bhv_df[bhv_df.subject == subject].unseen.values
    if (subject == 'BJB'): 
        rim = rim[:-1];seen = seen[:-1];unseen = unseen[:-1];
    for grp in range(1, 6):    
        #trials = np.where( (p_group[s] == grp) & rim)[0]
        #if grp == 1:
        #    trials = np.where(bhv_df[bhv_df.subject == subject].seen.values == 1)[0]
        #else: 
        #    trials = np.where(bhv_df[bhv_df.subject == subject].unseen.values == 1)[0]
        trials = np.where( (p_group[s] == grp) & rim)[0]
        if (subject == 'BJB') : trials = trials[:-1]
        X = MEG_data[trials, :, :]; y = cat[trials]
        if len(np.unique(y)) > 2:
            score = cross_val_multiscore(time_decod, X, y, 
                                         cv = cv, n_jobs = 1)
            scores[grp - 1, :, s] = score.mean(axis = 0)

        
        trials = np.where( (p_group[s] == grp) & rim & seen)[0]
        if (subject == 'BJB') : trials = trials[:-1]
        X = MEG_data[trials, :, :]; y = cat[trials]
        if len(np.unique(y)) > 2:
            score = cross_val_multiscore(time_decod, X, y, cv = cv, n_jobs = 10)
            s_scores[grp - 1, :, s] = score.mean(axis = 0)
        
        trials = np.where( (p_group[s] == grp) & rim & unseen)[0]
        if (subject == 'BJB') : trials = trials[:-1]
        X = MEG_data[trials, :, :]; y = cat[trials]
        if len(np.unique(y)) > 2:
            score = cross_val_multiscore(time_decod, X, y, cv = cv, n_jobs = 10)
            u_scores[grp - 1, :, s] = score.mean(axis = 0)
            
for scores in [s_scores, u_scores]:
    means = np.nanmean(scores, axis = -1).T;
    errors = np.nanstd(scores, axis = -1).T / np.sqrt(n_subj);
    fig, ax = plt.subplots(1, 1, figsize = (2.5,2))
    for grp in range(5):
        #plt.plot(epochs.times, means[:, grp])
        plt.errorbar(epochs.times, np.squeeze(means[:, grp]), 
                     errors[:, grp], label = grp)
    plt.xlabel('Time (s)')
    plt.plot([0, 2], [0.25, 0.25])
    plt.ylim([0.2, 0.4])
    plt.xlim([0., 0.5])
    plt.ylabel('Decoding acc')
    plt.legend()
    
#############################################################################

import matplotlib.pyplot as plt
nt = 10
cmap = cm.plasma( np.linspace(0, 1, nt) )
scores2  = scores[:5].mean(axis = 1)
fig, ax = plt.subplots( 1, 1, figsize = (2.5, 2) )
x = 2
for t in range(0, nt, x):
    means = np.nanmean(scores[:, t:(t+x), :].mean(axis = 1), axis = -1).T;
    errors = np.nanstd(
            scores[:, t:(t+x), :].mean(axis = 1), 
            axis = -1).T / np.sqrt(n_subj);
    tval, pval = scipy.stats.ttest_1samp(
            scores[:, t:(t+x), :].mean(axis = 1), 0.25)
    if sum(pval < 0.05):
        plt.errorbar(range(5), np.squeeze(means),
                     errors, label = epochs.times[t], color = cmap[t] )
plt.xlabel('Pupil size')
plt.plot([0, 4], [0.25, 0.25], 'r--')
plt.ylim([0.2, 0.3])
#plt.xlim([0., 0.5])
plt.ylabel('Decoding acc')
plt.legend()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#fig.savefig(figures_dir + '/seen_unseen_cat_decoding.png',
#            dpi = 800, bbox_inches = 'tight', transparent = True)
for scores in [s_scores, u_scores]:
    mm = np.nanmean(np.nanmean(scores[:, 1:9, :], axis = 1), axis = -1)
    er = np.nanstd(np.nanmean(scores[:, 1:9, :], axis = 1), axis = -1) / np.sqrt(24)
    fig, ax = plt.subplots(1, 1, figsize = (2.5,2))
    plt.bar(range(5), mm, color = 'w', edgecolor = 'k');
    plt.errorbar(range(5), mm, er, color = 'r')
    plt.plot([-.5, 5], [0.25, 0.25], '--k')
    plt.ylim([0.2, 0.4])
    plt.ylabel('Decod. acc')
    plt.xlabel('Pupil')
fig.savefig(figures_dir + '/pupil_cat_decode_acc.png',
                dpi = 800,  bbox_inches = 'tight',transparent = True)


mne.stats.f_mway_rm(scores[:, 1:9, :].mean(axis = 1).T, 
                     [5], effects = effects, return_pvals = True)

scipy.stats.ttest_1samp(scores[:, 1:9, :].mean(axis = 1).T, 0.25)

# EFFECT OF PRE-STIM PUPIL ON RESPONSE AMPLITUDE
    
# average according to pupil groups and in time
X = []
for grp in range(1,6):    
    pupil_meg = np.zeros( (n_subj, n_chan) )
    for s, subject in enumerate(subjects):        
        MEG_data = mean_meg[s]# n_trials x n_channels x n_times
        trials = np.where(p_group[s] == grp)[0]
        if (subject == 'BJB') : trials = trials[:-1]
        pupil_meg[s, :] = np.abs(MEG_data[trials, :, 10:110].mean(axis = 0).T).mean(axis = 0)
        
    X.append(pupil_meg)
   
x = np.array(X)    
plt.errorbar(range(5), x.mean(axis = -1).mean(axis = -1), x.mean(axis = -1).std(axis = -1) / np.sqrt(n_subj))    
# ANOVA spatial clusters
T_obs, clusters, cluster_p_values, H0 = clu = \
            mne.stats.permutation_cluster_test(X, 
                                               connectivity = connectivity.astype('int'), 
                             n_jobs = 24, 
                             threshold = f_thresh, stat_fun=stat_fun, 
                             n_permutations=1000) 
x = np.array(X)    
for c in range(100,150):  
    plt.figure(); plt.title(str(c))
    plt.ylim([0.1* 10**-13, 1.2* 10**-13] )
    mm = x[:, :, c].mean(axis = -1)
    ee = x[:, :, c].std(axis = -1) / np.sqrt(24)
    plt.errorbar(range(5), mm,mm+ee, mm-ee, 'o--')            
            
# average according to pupil groups  
X = []
n_chan = 272; n_times = 20; n_subj = 24
for grp in range(1,6):    
    pupil_meg = np.zeros( (n_subj, n_times, n_chan) )
    for s, subject in enumerate(subjects):        
        MEG_data = mean_meg[s]# n_trials x n_channels x n_times
        trials = np.where(p_group[s] == grp)[0]
        if (subject == 'BJB') : trials = trials[:-1]
        pupil_meg[s, :, :] = MEG_data[trials, :, :].mean(axis = 0).T
    X.append(pupil_meg)


            
# ANOVA time and space
p_accept = 0.05   
effects = 'A'
factor_levels = [5]# one factor : pupil group 
def stat_fun(*args):
    return mne.stats.f_mway_rm(np.swapaxes(np.array(args), 0, 1), 
                     factor_levels, effects = effects, return_pvals = False)[0]

f_thresh = mne.stats.f_threshold_mway_rm(23, factor_levels, ['A'] , 0.01)   

mask_params = dict(marker='.', markerfacecolor='b', markeredgecolor='b',
     linewidth=0, markersize=2)
connectivity, pos = HLTP_pupil.get_connectivity()

T_obs, clusters, cluster_p_values, H0 = clu = \
                mne.stats.spatio_temporal_cluster_test(X, connectivity=connectivity.astype('int'), 
                                 n_jobs=24,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=1000, buffer_size=None) 

mask = np.zeros(T_obs.shape, dtype=bool)
good_cluster_inds = np.where( cluster_p_values < p_accept)[0]
if len(good_cluster_inds) > 0:
    for g in good_cluster_inds:
        mask[clusters[g]] = True
epochs.pick(picks)
info = epochs.info
subj_evoked = mne.EvokedArray(np.transpose(T_obs), info, tmin = 0)
times2plot = subj_evoked.times[subj_evoked.times > 0][np.arange(0, 20, 2)]

fig = subj_evoked.plot_topomap(times = times2plot, cmap = 'hot',
                                           vmin=0*10**15, vmax=10*10**15, 
                                           sensors = False, contours = 0)
fig.savefig(figures_dir + '/pupil_raw_erf_anova.png',
                dpi = 800,  bbox_inches = 'tight',transparent = True)
     
m = np.zeros( (24, 5, 20) )
for chan in range(150, 200): 
    fig, ax = plt.subplots(1, 1, figsize = (2.5,2.5))
    for grp in range(1,6):
        for s, subject in enumerate(subjects):
            MEG_data = mean_meg[s]
            #seen = bhv_df[(bhv_df.subject == subject)].seen
            trials = np.where(p_group[s] == grp)[0]
            #trials = np.where(seen == grp)[0]
            if (subject == 'BJB') : trials = trials[:-1]
            psc = MEG_data[trials, chan, :]
            m[s, grp - 1, :] = psc.mean(axis = 0)
    
        mm = np.mean(m[:, grp - 1, :], axis = 0)   
        er = sem(m[:, grp - 1, :], axis = 0)   
        plt.plot(mm)
        plt.fill_between(range(20), mm- er, mm+ er, 
                             alpha =.3, label = grp)
plt.legend()




