#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:14:20 2020
How object category representation changes with pre-stimulus pupil and time

@author: podvae01
"""
import HLTP_pupil
from HLTP_pupil import MEG_pro_dir, subjects
import numpy as np
from statsmodels.stats.multitest import multipletests

import pandas as pd
import mne

from mne.decoding import (SlidingEstimator,
                          cross_val_multiscore)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from statsmodels.stats.anova import AnovaRM

#bhv_df = pd.read_pickle(MEG_pro_dir + '/results/all_subj_bhv.pkl')
#bhv_df = pd.read_pickle(HLTP_pupil.result_dir +
#                        '/all_subj_bhv_w_pupil_power.pkl')
df_name = 'all_subj_bhv_df_w_pupil_power'# prepare this with 4_DICS_roi_analysis
#df_name = 'all_subj_bhv_df_w_pupil'
bhv_df = pd.read_pickle(HLTP_pupil.result_dir + '/' + df_name + '.pkl')

group_percentile = np.arange(0., 100., 20)

mean_meg = []
for s, subject in enumerate(subjects):
    sub_pro_dir = MEG_pro_dir + '/' + subject
    
    # use here 256Hz ds cleaned after ICA data
    epoch_fname = sub_pro_dir + '/task_posstim_ds-epo.fif'
    epochs = mne.read_epochs(epoch_fname, preload = True)
    #epochs.apply_baseline(baseline = (0, .05))
    epochs.resample(10, n_jobs = 10)
    picks = mne.pick_types(epochs.info, meg = True, 
                           ref_meg = False, exclude = 'bads')
    MEG_data = epochs.get_data(picks)
    mean_meg.append(MEG_data.copy())
    
n_chan = MEG_data.shape[1]
n_times = MEG_data.shape[2]
n_subj = len(subjects)

# EFFECT OF PRE-STIM PUPIL ON CATEGORY REPRESENTATION
clf = make_pipeline(StandardScaler(), 
                    LogisticRegression(C = 1, solver='newton-cg', 
                                       multi_class ='multinomial'))
time_decod = SlidingEstimator(clf, n_jobs=1, verbose=True)
cv = LeaveOneOut()
K = 500
scores = np.zeros( (5, n_times, n_subj) ); scores[:] = np.nan
perm_scores = np.zeros( (5, n_times, n_subj, K) ); perm_scores[:] = np.nan

for s, subject in enumerate(subjects):
    MEG_data = mean_meg[s]# n_trials x n_channels x n_times        
    cat = bhv_df[bhv_df.subject == subject].cat_protocol.values   
    rim = bhv_df[bhv_df.subject == subject].real_img
    
    p_group = bhv_df[bhv_df.subject == subject].pupil.values
    
    for grp in range(1, 6):    
          
        trials = np.where( (p_group == grp) & rim)[0]
        X = MEG_data[trials, :, :]
        y = cat[trials]
        if len(np.unique(y)) > 2:
            score = cross_val_multiscore(time_decod, X, y, 
                                         cv = cv, n_jobs = 10)
            scores[grp - 1, :, s] = score.mean(axis = 0)
            #perm_test
            for k in range(K):
                perm_score = cross_val_multiscore(time_decod, X, np.random.permutation(y),
                                         cv = cv, n_jobs = 10)
                perm_scores[grp - 1, :, s, k] = perm_score.mean(axis = 0)



HLTP_pupil.save([scores, perm_scores], HLTP_pupil.result_dir +
                        '/cat_decod_score.pkl')

plt.plot(perm_scores.mean(axis = 0).mean(axis = -1).mean(axis = -1).T);plt.plot(scores.mean(axis = 0).mean(axis = -1).T)

plt.plot(perm_scores.mean(axis = -1).mean(axis = -1).T);plt.plot(scores.mean(axis = -1).T)
# test whether they exceed chance level for each time point
p_val = np.ones(20)
mean_perm_score = perm_scores.mean(axis = 0).mean(axis = 1)
mean_score = scores.mean(axis = 0).mean(axis = -1)
for t in range(20):
    p_val[t] = (mean_perm_score[t,:] > mean_score[t]).sum() / K
corr_pval = multipletests(p_val, alpha=0.05, method='fdr_bh')

# test whether they exceed chance level for each group
p_val = np.ones(5)
mean_perm_score = perm_scores.mean(axis = 1).mean(axis = 1)
mean_score = scores.mean(axis = 1).mean(axis = -1)
for t in range(5):
    p_val[t] = (mean_perm_score[t,:] > mean_score[t]).sum() / K
corr_pval = multipletests(p_val, alpha=0.05, method='fdr_bh')



# test the effect of prestim pupil and time on devolepemnt of category representations

scores = HLTP_pupil.load(HLTP_pupil.result_dir +
                        '/cat_decod_score.pkl')
sub= []; pup = []; tim = []; scr = []
for s in range(24):
    for p in range(5):
        for t in range(20):
            sub.append(s); pup.append(p); tim.append(t); scr.append(scores[p, t, s])
            
score_df = pd.DataFrame.from_dict(
    {'pupil':pup, 'subject':sub, 'score':np.arcsin(np.sqrt(np.array(scr))), 'time':tim})
                       
mod = AnovaRM(score_df, 'score', 'subject', within = ['pupil', 'time'])
res = mod.fit()
print(res.anova_table)

#############################################################################
# figures to be moved from here

# effects = 'A'

# fval, pval = mne.stats.f_mway_rm(scores.mean(axis = 1).T, 
#                      [5], effects = effects, return_pvals = True)

# effects = 'A'
# sig = np.zeros(20)
# for t in range(20):
#     fval, pval = mne.stats.f_mway_rm(scores[:, t, :].T, 
#                      [5], effects = effects, return_pvals = True)
#     sig[t] = pval 
# fig, ax = plt.subplots(1, 1,figsize = (3, 5))     
# plt.imshow(amp.mean(axis = -1), vmin = 0, vmax = amp.max()/4,
#                    cmap = 'Spectral_r')
# effects = 'A'
# sig = np.zeros(20)
# for t in range(20):
#     fval, pval = mne.stats.f_mway_rm(amp[:, t, :].T, 
#                      [5], effects = effects, return_pvals = True)
#     sig[t] = pval 
# epochs.times[sig < 0.05]

# fig, ax = plt.subplots(1, 1,figsize = (3, 5))     
# plt.imshow(vrb.mean(axis = -1), vmin = 0, vmax = vrb.max()/4,
#                    cmap = 'Spectral_r') 

# for scores in [s_scores, u_scores]:
#     means = np.nanmean(scores, axis = -1).T;
#     errors = np.nanstd(scores, axis = -1).T / np.sqrt(n_subj);
#     fig, ax = plt.subplots(1, 1, figsize = (2.5,2))
#     for grp in range(5):
#         #plt.plot(epochs.times, means[:, grp])
#         plt.errorbar(epochs.times, np.squeeze(means[:, grp]), 
#                      errors[:, grp], label = grp)
#     plt.xlabel('Time (s)')
#     plt.plot([0, 2], [0.25, 0.25])
#     plt.ylim([0.2, 0.4])
#     plt.xlim([0., 0.5])
#     plt.ylabel('Decoding acc')
#     plt.legend()

# nt = 40
# cmap = cm.plasma( np.linspace(0, 1, nt) )
# fig, ax = plt.subplots( 1, 1, figsize = (2.5, 2) )
# for t in range(nt):
#     means = np.nanmean(scores[:, t, :], axis = 1).T;
#     errors = np.nanstd(
#             scores[:, t, :], 
#             axis = -1).T / np.sqrt(n_subj);
#     tval, pval = scipy.stats.ttest_1samp(
#             scores[:, t, :].T, 0.25)
#     if sum((pval < 0.025) & (tval > 0)) :
#         plt.figure(); plt.title(epochs.times[t])
#         plt.errorbar(range(5), np.squeeze(means),
#                      errors, label = epochs.times[t], color = cmap[t] )
#         plt.xlabel('Pupil size')
#         plt.plot([0, 4], [0.25, 0.25], 'r--')
#         plt.ylim([0.2, 0.3])
# #plt.xlim([0., 0.5])
# plt.ylabel('Decoding acc')
# plt.legend()
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# #fig.savefig(figures_dir + '/seen_unseen_cat_decoding.png',
# #            dpi = 800, bbox_inches = 'tight', transparent = True)
# for scores in [s_scores, u_scores]:
#     mm = np.nanmean(np.nanmean(scores[:, 1:9, :], axis = 1), axis = -1)
#     er = np.nanstd(np.nanmean(scores[:, 1:9, :], axis = 1), axis = -1) / np.sqrt(24)
#     fig, ax = plt.subplots(1, 1, figsize = (2.5,2))
#     plt.bar(range(5), mm, color = 'w', edgecolor = 'k');
#     plt.errorbar(range(5), mm, er, color = 'r')
#     plt.plot([-.5, 5], [0.25, 0.25], '--k')
#     plt.ylim([0.2, 0.3])
#     plt.ylabel('Decod. acc')
#     plt.xlabel('Pupil')
# fig.savefig(figures_dir + '/pupil_cat_decode_acc.png',
#                 dpi = 800,  bbox_inches = 'tight',transparent = True)


# mne.stats.f_mway_rm(scores[:, 1:9, :].mean(axis = 1).T, 
#                      [5], effects = effects, return_pvals = True)

# scipy.stats.ttest_1samp(scores[:, 1:9, :].mean(axis = 1).T, 0.25)

# # EFFECT OF PRE-STIM PUPIL ON RESPONSE AMPLITUDE
    
# # average according to pupil groups and in time
# X = []
# for grp in range(1,6):    
#     pupil_meg = np.zeros( (n_subj, n_chan) )
#     for s, subject in enumerate(subjects):        
#         MEG_data = mean_meg[s]# n_trials x n_channels x n_times
#         trials = np.where(p_group[s] == grp)[0]
#         if (subject == 'BJB') : trials = trials[:-1]
#         pupil_meg[s, :] = np.abs(MEG_data[trials, :, 10:110].mean(axis = 0).T).mean(axis = 0)
        
#     X.append(pupil_meg)
   
# x = np.array(X)    
# plt.errorbar(range(5), x.mean(axis = -1).mean(axis = -1), x.mean(axis = -1).std(axis = -1) / np.sqrt(n_subj))    


# # ANOVA spatial clusters
# T_obs, clusters, cluster_p_values, H0 = clu = \
#             mne.stats.permutation_cluster_test(X, 
#                                                connectivity = connectivity.astype('int'), 
#                              n_jobs = 24, 
#                              threshold = f_thresh, stat_fun=stat_fun, 
#                              n_permutations=1000) 
# x = np.array(X)    
# for c in range(100,150):  
#     plt.figure(); plt.title(str(c))
#     plt.ylim([0.1* 10**-13, 1.2* 10**-13] )
#     mm = x[:, :, c].mean(axis = -1)
#     ee = x[:, :, c].std(axis = -1) / np.sqrt(24)
#     plt.errorbar(range(5), mm,mm+ee, mm-ee, 'o--')            
            
# # average according to pupil groups  
# X = []
# n_chan = 272; n_times = 20; n_subj = 24
# for grp in range(1,6):    
#     pupil_meg = np.zeros( (n_subj, n_times, n_chan) )
#     for s, subject in enumerate(subjects):        
#         MEG_data = mean_meg[s]# n_trials x n_channels x n_times
#         trials = np.where(p_group[s] == grp)[0]
#         if (subject == 'BJB') : trials = trials[:-1]
#         pupil_meg[s, :, :] = MEG_data[trials, :, :].mean(axis = 0).T
#     X.append(pupil_meg)


            
# # ANOVA time and space
# p_accept = 0.05   
# effects = 'A'
# factor_levels = [5]# one factor : pupil group 
# def stat_fun(*args):
#     return mne.stats.f_mway_rm(np.swapaxes(np.array(args), 0, 1), 
#                      factor_levels, effects = effects, return_pvals = False)[0]

# f_thresh = mne.stats.f_threshold_mway_rm(23, factor_levels, ['A'] , 0.01)   

# mask_params = dict(marker='.', markerfacecolor='b', markeredgecolor='b',
#      linewidth=0, markersize=2)
# connectivity, pos = HLTP_pupil.get_connectivity()

# T_obs, clusters, cluster_p_values, H0 = clu = \
#                 mne.stats.spatio_temporal_cluster_test(X, connectivity=connectivity.astype('int'), 
#                                  n_jobs=24,
#                                  threshold=f_thresh, stat_fun=stat_fun,
#                                  n_permutations=1000, buffer_size=None) 

# mask = np.zeros(T_obs.shape, dtype=bool)
# good_cluster_inds = np.where( cluster_p_values < p_accept)[0]
# if len(good_cluster_inds) > 0:
#     for g in good_cluster_inds:
#         mask[clusters[g]] = True
# epochs.pick(picks)
# info = epochs.info
# subj_evoked = mne.EvokedArray(np.transpose(T_obs), info, tmin = 0)
# times2plot = subj_evoked.times[subj_evoked.times > 0][np.arange(0, 20, 2)]

# fig = subj_evoked.plot_topomap(times = times2plot, cmap = 'hot',
#                                            vmin=0*10**15, vmax=10*10**15, 
#                                            sensors = False, contours = 0)
# fig.savefig(figures_dir + '/pupil_raw_erf_anova.png',
#                 dpi = 800,  bbox_inches = 'tight',transparent = True)
     
# m = np.zeros( (24, 5, 20) )
# for chan in range(150, 200): 
#     fig, ax = plt.subplots(1, 1, figsize = (2.5,2.5))
#     for grp in range(1,6):
#         for s, subject in enumerate(subjects):
#             MEG_data = mean_meg[s]
#             #seen = bhv_df[(bhv_df.subject == subject)].seen
#             trials = np.where(p_group[s] == grp)[0]
#             #trials = np.where(seen == grp)[0]
#             if (subject == 'BJB') : trials = trials[:-1]
#             psc = MEG_data[trials, chan, :]
#             m[s, grp - 1, :] = psc.mean(axis = 0)
    
#         mm = np.mean(m[:, grp - 1, :], axis = 0)   
#         er = sem(m[:, grp - 1, :], axis = 0)   
#         plt.plot(mm)
#         plt.fill_between(range(20), mm- er, mm+ er, 
#                              alpha =.3, label = grp)
# plt.legend()


# # EFFECT OF PRE-STIM PUPIL ON RESPONSE AMPLITUDE (mean across chann)
# amp = np.zeros( (5, n_times, n_subj) ); amp[:] = np.nan
# vrb = np.zeros( (5, n_times, n_subj) ); vrb[:] = np.nan
# for s, subject in enumerate(subjects): 
#     MEG_data = mean_meg[s]# n_trials x n_channels x n_times        
#     cat = bhv_df[bhv_df.subject == subject].cat_protocol.values   
#     rim = bhv_df[bhv_df.subject == subject].real_img
#     #seen = bhv_df[bhv_df.subject == subject].seen.values
#     #unseen = bhv_df[bhv_df.subject == subject].unseen.values
#     p_group = bhv_df[bhv_df.subject == subject].pupil_size_pre.values
    
#     for grp in range(1, 6):              
#         trials = np.where( (p_group == grp) & rim)[0]
#         X = MEG_data[trials, :, :];
#         amp[grp - 1, :, s] = np.abs((X).mean(axis = 0).mean(axis = 0))
#         vrb[grp - 1, :, s] = ((X.std(axis = 0))**2).mean(axis = 0)
# # EFFECT OF PRE-STIM PUPIL ON RESPONSE AMPLITUDE (each )

