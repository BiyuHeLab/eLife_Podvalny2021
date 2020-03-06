#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:19:13 2019

@author: podvae01
"""
import HLTP_pupil
from HLTP_pupil import MEG_pro_dir, subjects
import mne
import numpy as np
from scipy.stats import pearsonr, distributions
from os import path

def meg_cross_corr_sampled(meg_data, pupil_data, lags):
    
    n_channels = meg_data.shape[0]
    n_lags = len(lags)
    
    cc = np.zeros( (n_channels, n_lags) )
    
    for chan in range(n_channels):
        for n, lag in enumerate(lags):
            if lag > 0: 
                cc[chan, n] = pearsonr( meg_data[chan, lag:], 
                  pupil_data[:-lag])[0]
            elif lag < 0: 
                cc[chan, n] = pearsonr( meg_data[chan, :lag], 
                  pupil_data[-lag:])[0]
            else: cc[chan, n] = pearsonr( meg_data[chan, :], 
                  pupil_data)[0]
    return cc

# two sec trials selected lags pearson
# might want to do this for filtered data, low-pass
timelags = np.linspace(-1500, 1500, 31)   
lags = np.round(HLTP_pupil.resamp_fs * timelags / 1000).astype('int')

for b in ['rest01', 'rest02', 'task_prestim']:
    all_cc = []    
    for subject in subjects:
        cc = []
        filename = MEG_pro_dir + '/' + subject + '/' + b + '_ds-epo.fif'
        if not path.exists(filename): 
            print('No such file'); continue
        epochs = mne.read_epochs(filename)
        pdata = epochs._data[:, HLTP_pupil.pupil_chan, :].copy()
        epochs.pick_types(meg = True, ref_meg = False, exclude=[])
        mdata = epochs._data
        for ep in range(len(epochs.events)):
            cc.append(meg_cross_corr_sampled(mdata[ep, :, :], 
                                             pdata[ep, :], lags))
        all_cc.append(np.array(cc))
    HLTP_pupil.save(all_cc, HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                        '/cross_corr_' + b + '.pkl')
# blp cross-corr
for fband, freq in HLTP_pupil.freq_bands.items():      
    for b in ['rest01', 'rest02']:
        all_cc = []    
        for subject in subjects:
            cc = []
            filename = MEG_pro_dir + '/' + subject + '/' + b + '_ds' \
                + fband + '-epo.fif'
            if not path.exists(filename): 
                print('No such file'); continue
            epochs = mne.read_epochs(filename)
            pdata = epochs._data[:, HLTP_pupil.pupil_chan, :].copy()
            epochs.pick_types(meg = True, ref_meg = False, exclude=[])
            mdata = epochs._data
            for ep in range(len(epochs.events)):
                cc.append(meg_cross_corr_sampled(mdata[ep, :, :], 
                                                 pdata[ep, :], lags))
            all_cc.append(np.array(cc))
        HLTP_pupil.save(all_cc, HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                            '/cross_corr_' + b + fband + '.pkl')

# group level analysis
for b in ['rest01', 'rest02', 'task_prestim']:
    for fband, freq in HLTP_pupil.freq_bands.items():      
        # load array of correlation coefficient
        all_cc = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                            '/cross_corr_' + b + fband + '.pkl')
        # fisher transform and average across trials
        x = np.array([np.arctanh(np.array(all_cc[s])).mean(axis = 0) 
                                                for s in range(len(all_cc))])
        # t-test with spatio-temporal cluster correction for multiple comparisons
        x = np.swapaxes(x, 1, 2)
        connectivity, pos = HLTP_pupil.get_connectivity()   
        alpha = 0.05; p_accept = 0.05
        threshold = -distributions.t.ppf( alpha / 2.,  len(all_cc) - 1)
        cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test(x, 
                                  n_permutations = 1000,
                                  threshold = threshold, tail=0,
                                  n_jobs = 1, connectivity=connectivity)
        T_obs, clusters, p_values, _ = cluster_stats   
        
        mask = np.zeros(T_obs.shape, dtype=bool)
        good_cluster_inds = np.where( p_values < p_accept)[0]
        if len(good_cluster_inds) > 0:
            for g in good_cluster_inds:
                sensors =  clusters[g][1]
                times = clusters[g][0]        
                mask[times, sensors.astype(int)] = True
        HLTP_pupil.save(mask, HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                            '/cross_corr_sig_mask_' + b + fband + '.pkl')


    
    
    
    