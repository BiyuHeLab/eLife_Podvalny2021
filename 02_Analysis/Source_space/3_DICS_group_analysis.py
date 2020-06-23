#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:23:29 2019
Whole-brain average and normalize the DICS power for figures and submit this 
to ROI analysis
TODO: clean up this script

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

