#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:23:29 2019

To clean up this script, srsly

@author: podvae01
"""

import mne
import numpy as np
from HLTP_pupil import subjects, freq_bands, FS_dir, MEG_pro_dir
import HLTP_pupil
import time
import copy 
from os import path
from scipy import stats    
from numpy.ma import masked_array    


figures_dir = MEG_pro_dir  +'/_figures'
results_dir = MEG_pro_dir  +'/results'
method = 'dics'

def norm_across_pupil_groups(sub_pro_dir, epoch_name):
    # normalize data by mean across pupil size groups
    stc = {'1':[],'2':[],'3':[],'4':[],'5':[]}
    # load data
    for grp in range(1,6):
        fname = method + '_power_map_' + epoch_name + str(grp)
        stc_fname = sub_pro_dir + '/fsaverage_' + fname
        stc[str(grp)] = mne.read_source_estimate(stc_fname)
    # calculate mean across pupil groups
    stc_group_mean = np.zeros(shape = stc[str(grp)].data.shape)
    for grp in range(1,6):
        stc_group_mean += stc[str(grp)].data
    stc_group_mean /= 5
    #stc_group_mean = np.tile(stc_group_mean.mean(axis = 1), (5,1)).T
    # divide the data by its mean and logtransform to obtain relative power
    norm_stc = copy.deepcopy(stc)
    for grp in range(1,6):
        norm_stc[str(grp)] /= stc_group_mean
        norm_stc[str(grp)]._data = np.log(norm_stc[str(grp)]._data)
    return stc, norm_stc

#------------------------------------------------------------------------------
# Save raw and normalize source estimates for all subjects    
#------------------------------------------------------------------------------

for epoch_name in ['rest01_ds', 'rest02_ds', 'task_prestim_ds']:
    norm_stcs = []; raw_stcs = []
    for s, subject in enumerate(subjects): 
        sub_pro_dir = MEG_pro_dir + '/' + subject; raw = []; norm = []
        if not path.exists(sub_pro_dir + '/' + epoch_name + '-epo.fif'): continue;
        raw, norm = norm_across_pupil_groups(sub_pro_dir, epoch_name)
        norm_stcs.append(norm)
        raw_stcs.append(raw)
    HLTP_pupil.save(norm_stcs, MEG_pro_dir + '/pupil_result/norm_stc_' + epoch_name)
    HLTP_pupil.save(raw_stcs, MEG_pro_dir + '/pupil_result/raw_stc_' + epoch_name)

#------------------------------------------------------------------------------
# calculate mean power across the two rest runs
#------------------------------------------------------------------------------
    
epoch_name = 'rest01_ds'
norm_stcs = []; raw_stcs = []
for s, subject in enumerate(subjects): 
    sub_pro_dir = MEG_pro_dir + '/' + subject; raw = []; norm = []
    raw, norm = norm_across_pupil_groups(sub_pro_dir, epoch_name)
    nblocks = 2; e_name = 'rest02_ds'
    fname = sub_pro_dir + '/fsaverage_dics_power_map_' + e_name + '1-lh.stc'
    if path.exists(fname): 
        raw2, norm2 = norm_across_pupil_groups(sub_pro_dir, e_name) 
        for p in range(1, 6):
            raw[str(p)] += raw2[str(p)]; norm[str(p)] += norm2[str(p)]  
            raw[str(p)] /= nblocks; norm[str(p)] /= nblocks; 
    norm_stcs.append(norm)
    raw_stcs.append(raw)
HLTP_pupil.save(norm_stcs, MEG_pro_dir + '/pupil_result/norm_stc_mean_rest')
HLTP_pupil.save(raw_stcs, MEG_pro_dir + '/pupil_result/raw_stc_mean_rest')



#------------------------------------------------------------------------------
#------1WAY repeated measures ANOVA with pupil group as factor-----------------
#------------------------------------------------------------------------------

mne_dir = '/isilon/LFMI/VMdrive/Ella/mne_data/MNE-sample-data'
src_fname = mne_dir + '/subjects/fsaverage/bem/fsaverage-ico-5-src.fif'
src = mne.read_source_spaces(src_fname)
connectivity = mne.spatial_src_connectivity(src)
factor_levels = [5]# 
effects = ['A']            
crit_alpha = 0.05
f_thresh = mne.stats.f_threshold_mway_rm(23, factor_levels, 
                                         effects , crit_alpha)
def stat_fun(*args):
        return mne.stats.f_mway_rm(np.swapaxes(
                np.array(args), 0, 1), factor_levels, effects = effects, 
            return_pvals=False)[0]

for epoch_name in ['mean_rest', 'task_prestim_ds']:
    raw_stcs = HLTP_pupil.load( MEG_pro_dir + '/pupil_result/raw_stc_' + epoch_name)
        
    stcs = raw_stcs
    n_subjects = len(raw_stcs)
    
    for fband, band in enumerate(freq_bands.keys()):
        data = []
        for grp in range(1,6):
            data_grp = np.zeros((n_subjects, 20484))
            for s in range(n_subjects):
                data_grp[s, :] = stcs[s][str(grp)]._data[:, fband]
            data.append(data_grp)
    
        T_obs, clusters, cluster_p_values, H0 = clu = \
                mne.stats.permutation_cluster_test(data,
                                 connectivity = connectivity.astype('int'),
                                 n_jobs = -1, tail = 0,
                                 threshold = f_thresh, stat_fun=stat_fun,
                                 n_permutations = 5000)        
        
        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]             
        F_val =  np.zeros(T_obs.shape)
        for i in good_cluster_inds:
            F_val[clusters[i]] = T_obs[clusters[i]]
        
        band_stc = stcs[0][str(grp)].copy()
        band_stc._data = np.expand_dims( F_val, axis = 1)
        band_stc.save(results_dir + '/' +  epoch_name + '_src_1wANOVA_'  + band, 
                      verbose = 5)







     











