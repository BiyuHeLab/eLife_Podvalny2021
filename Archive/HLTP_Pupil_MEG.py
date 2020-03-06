#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:45:16 2019

@author: podvae01
"""
import HLTP
import pandas as pd 
from mne import viz
import matplotlib.pyplot as plt
import numpy as np
figures_dir = HLTP.MEG_pro_dir  +'/_figures'
excluded = 'NC' # no pupil data due to bad quality eye-tracking. 
group_percentile = np.arange(0., 100.1, 20);#np.arange(0., 100.1, 33); #
groups = np.arange(1, len(group_percentile), 1)
connectivity, pos = HLTP.get_connectivity()
bhv_dataframe = pd.read_pickle(HLTP.MEG_pro_dir + '/results/all_bhv_pupil_df.pkl')

sensors = {}
for s in ['O','F','T','P','C']:
    sensors[s] = HLTP.get_ROI('', s)[1]
sensors['OT'] = np.hstack([HLTP.get_ROI('', 'O')[1], HLTP.get_ROI('', 'T')[1]])
sensors['FP'] = np.hstack([HLTP.get_ROI('', 'F')[1], HLTP.get_ROI('', 'P')[1]])
sensors['AL'] = HLTP.get_ROI('', 'A')[1]

for roi in sensors.keys():
    fig, ax = plt.subplots(1, 1,figsize = (3,3)) 
    all_bands = []
    group_allchan=[]
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        all_data, trial_n = HLTP.load(HLTP.MEG_pro_dir + 
                                      '/results/prestim_data_' + band + '.p')
        group_chan_mean = [];group_chan_ste = [];group_allchan_band=[] ;
        bhv_array = np.zeros((23, 5))

        for group in groups:
            all_subj = []
            for sub_idx, sub in enumerate(HLTP.subjects):
                if sub == excluded: continue   
                group1_df = bhv_dataframe[(bhv_dataframe.subject == sub) & 
                                         (bhv_dataframe.pupil_group == 1)]
                meg1_group = []
                for tr in group1_df.index: 
                    meg1_group.append(all_data[sub_idx][trial_n[sub_idx] == tr, :])
                group_df = bhv_dataframe[(bhv_dataframe.subject == sub) & 
                                         (bhv_dataframe.pupil_group == group)]
                meg_group = []
                for tr in group_df.index: 
                    meg_group.append(all_data[sub_idx][trial_n[sub_idx] == tr, :])
                subj_mean_meg = np.log(np.mean(np.array(meg_group).squeeze(), axis = 0)\
                    /np.mean(np.array(meg1_group).squeeze(), axis = 0))
                all_subj.append(subj_mean_meg)
            group_allchan_band.append(np.array(all_subj).squeeze())
            group_allchan.append(np.array(all_subj).squeeze())
            group_mean = np.mean(np.array(all_subj).squeeze(), axis = 0)
            group_chan_mean.append(np.mean(np.array(all_subj)[:, sensors[roi]].squeeze(), axis = 1))
            bhv_array[:, group - 1] = df[df.group == group].c
            fig, ax = plt.subplots(1, 1,figsize = (3,3))
            viz.plot_topomap(group_mean, pos[:, :2], axes = ax, sensors = False, 
                             outlines = 'skirt',
                         vmin = -0.3, vmax = 0.3, contours = [-300,300],
                         cmap = 'coolwarm')
            fig.savefig(figures_dir + '/MEG_pupil_' + band + str(group) + '.png', 
                        dpi = 800, transparent = True)
        #fig, ax = plt.subplots(1, 1,figsize = (3,3))    
        
        # correlation of power with criterion
        allcorr = np.zeros((272, 23));allcorrp = np.zeros((272))
        for c in range(272): 
            for s in range(23):
                [r,p]=scipy.stats.spearmanr(np.array(group_allchan_band)[:, s, c].T, 
                    bhv_array[s,:])
                allcorr[c, s] = r;
            [r, p] = scipy.stats.wilcoxon(allcorr[c, :]);
            allcorrp[c] = p
        fig, ax = plt.subplots(1, 1,figsize = (3,3))
        viz.plot_topomap(allcorr.mean(axis = 1), pos[:, :2], axes = ax, sensors = False, 
                             outlines = 'skirt', mask = allcorrp<0.05,
                         vmin = -0.3, vmax = 0.3, contours = [-300,300],
                        cmap = 'coolwarm')
        fig.savefig(figures_dir + '/MEG_pupil_corr_c_' + band + '.png', 
                  dpi = 800, transparent = True)
        
        p_accept = 0.025   
        x = np.array(group_allchan_band)   
        factor_levels = [5]# two factors : pupil group x frequency bands
        def stat_fun(*args):
            return mne.stats.f_mway_rm(np.swapaxes(np.array(args), 0, 1), factor_levels, 
                         effects = effects, return_pvals=False)[0]
        f_thresh = mne.stats.f_threshold_mway_rm(23, factor_levels, ['A'] , 0.025)   

        for idx, effects in enumerate(['A']):   
            T_obs, clusters, cluster_p_values, H0 = clu = \
                    mne.stats.permutation_cluster_test(x, 
                                    connectivity = connectivity.astype('int'), 
                                     n_jobs = 24,tail = 0,
                                     threshold = f_thresh, stat_fun=stat_fun, 
                                     n_permutations=1000)
            mask = np.zeros(T_obs.shape, dtype=bool)
            good_cluster_inds = np.where( cluster_p_values < p_accept)[0]
            if len(good_cluster_inds) > 0:
                for g in good_cluster_inds:
                    mask[clusters[g]] = True
            fig,ax = plt.subplots(1, 1, figsize = (2,2))
            mne.viz.plot_topomap(T_obs, pos, mask = mask, sensors = False, 
                                 mask_params = mask_params,
                                 vmin = 0, vmax = 12, contours = [-300,300])
            fig.savefig(figures_dir + '/PupilFreq_2WRMANOVA_' + band + '.png',
                        dpi = 800,  bbox_inches = 'tight',transparent = True)
        
        
        plt.errorbar(range(5), np.mean(np.array(group_chan_mean), axis = 1), 
                     np.std(np.array(group_chan_mean), axis = 1)/np.sqrt(23), label = band)
        plt.xlabel('pupil size (%)'); plt.ylabel('relative power (dB)')
        plt.title(roi)
        plt.ylim([-0.1, 0.3])
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        all_bands.append(np.array(group_chan_mean))
    [fval, pval] = mne.stats.f_mway_rm(np.reshape(np.array(all_bands).T, [23,25]), [5,5], effects='all')
    #print fval,pval<0.01,roi
    
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.savefig(figures_dir + '/MEG_pupil_fbands_' +band + roi+ '.png', 
                        dpi = 800,  bbox_inches = 'tight',transparent = True)
        
    
p_accept = 0.01   
x = np.array(group_allchan)   
factor_levels = [5, 5]# two factors : pupil group x frequency bands
def stat_fun(*args):
        return mne.stats.f_mway_rm(np.swapaxes(np.array(args), 0, 1), factor_levels, 
                         effects = effects, return_pvals=False)[0]
f_thresh = mne.stats.f_threshold_mway_rm(23, factor_levels, ['A:B', 'A', 'B'] , 0.01)   

mask_params = dict(marker='.', markerfacecolor='b', markeredgecolor='b',
     linewidth=0, markersize=2)
for idx, effects in enumerate(['A:B', 'A', 'B']):   
    T_obs, clusters, cluster_p_values, H0 = clu = \
            mne.stats.permutation_cluster_test(x, 
                                               connectivity = connectivity.astype('int'), 
                             n_jobs = 24,tail = 0,
                             threshold = f_thresh[idx], stat_fun=stat_fun, 
                             n_permutations=1000)
    mask = np.zeros(T_obs.shape, dtype=bool)
    good_cluster_inds = np.where( cluster_p_values < p_accept)[0]
    if len(good_cluster_inds) > 0:
        for g in good_cluster_inds:
            mask[clusters[g]] = True
    fig,ax = plt.subplots(1, 1, figsize = (2,2))
    mne.viz.plot_topomap(T_obs, pos, mask = mask, sensors = False, mask_params=mask_params,
                         vmin = 0, vmax = 12, contours = [-300,300])
    fig.savefig(figures_dir + '/PupilFreq_2WRMANOVA_' + str(idx) + '.png',
                dpi = 800,  bbox_inches = 'tight',transparent = True)