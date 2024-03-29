#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:07:55 2020

@author: podvae01
"""
import mne
import numpy as np
import HLTP_pupil
from HLTP_pupil import subjects, freq_bands, MEG_pro_dir, FS_dir
from os import path  
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_multitaper
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore 
import os
method = 'dics'
#------------------------------------------------------------------------------

def get_mean_norm_pwr_by_pupil_group():
    """ROI analysis calculate mean normalized power across subjects for each group  """
    atlas_name = 'Yeo2011_7Networks_N1000'
    for ename in ['mean_rest', 'task_prestim_ds']:

        raw_stcs = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result/raw_stc_' + ename)
        # ROI analysis
        labels = {}
        for hs in ['lh', 'rh']:
            labels[hs] = mne.read_labels_from_annot(
                'fsaverage', atlas_name, hs, subjects_dir = HLTP_pupil.FS_dir)

        n_subjects = len(raw_stcs)
        n_roi = len(labels[hs]) - 1
        n_freq = 5
        n_pupil = 5

        stcs = raw_stcs
        data = np.zeros(shape = (n_freq, n_pupil, n_roi, n_subjects))
        for fband, band in enumerate(HLTP_pupil.freq_bands.keys()):
            for i in range(n_roi):
                for grp in range(1,6):
                    data_grp = np.zeros((n_subjects))
                    for s in range(n_subjects):
                        # average across hemispheres
                        lh_data = stcs[s][str(grp)].in_label(labels['lh'][i]
                            ).data[:, fband].mean()
                        rh_data = stcs[s][str(grp)].in_label(labels['rh'][i]
                            ).data[:, fband].mean()
                        data_grp[s] = (lh_data  + rh_data) / 2.
                    data[fband, grp - 1, i, :] = data_grp
                # normalize across pupil sizes for plotting
                for s in range(n_subjects):
                    data[fband, :, i, s] = np.log(
                            data[fband, :, i, s] / data[fband, :, i, s].mean())
        # this file will be used for plotting  fig 2
        HLTP_pupil.save(data, HLTP_pupil.MEG_pro_dir +
                        '/pupil_result/roi_power_map_7nets_' + ename)

def save_subject_fullband_filter(subject, epoch_name):
    fmin = np.arange(1, 100, 1)
    fmax = fmin + 1
    sub_pro_dir = MEG_pro_dir + '/' + subject
    epoch_file = sub_pro_dir + '/' + epoch_name + '-epo.fif'
    if not path.exists(epoch_file): print('No such file'); return

    info = mne.io.read_info(sub_pro_dir + '/' + epoch_name + '-epo.fif')
    fwd = mne.read_forward_solution(sub_pro_dir + '/HLTP_fwd.fif')
    # read the csd for each event and average csds:
    csds = {}
    for evnt in range(1,6):
        csd = mne.time_frequency.read_csd(sub_pro_dir + '/' + method
                       + '_csd_multitaper_' + epoch_name + str(evnt))
        csds[str(evnt)] = csd.copy()
        if evnt == 1: mean_data = csd._data;
        else: mean_data += csd._data
    csd._data = mean_data / 5

    # Compute DICS spatial filter and estimate source power,
    filters = make_dics(info, fwd, csd.mean(fmin, fmax), reg = 0.05,
                            weight_norm = 'nai', verbose='error')
    filters.save(sub_pro_dir + '/' + method + '_filter_fullband_' + epoch_name +
                 '-dics.h5', overwrite = True)

def save_fullband_filters():
    '''  calculate source space for each epoch separately
    and use atlas to save only 7 nets data
    and also for fine res power maps in roi
    '''
    for epoch_name in ['rest01_ds', 'rest02_ds', 'task_prestim_ds']:
        for subject in subjects:
            save_subject_fullband_filter(subject, epoch_name)

def save_subject_epoch_roi_band_pwr(subject, epoch_name, labels, fmin, fmax):

    sub_pro_dir = MEG_pro_dir + '/' + subject
    epoch_file = sub_pro_dir + '/' + epoch_name + '-epo.fif'
    if not path.exists(epoch_file): print('No such file ', subject); return
    n_roi = len(labels['lh']) - 1
    n_freq = len(fmin)
    filters = mne.beamformer.read_beamformer(sub_pro_dir + '/' + method
                        + '_filter_fullband_' + epoch_name + '-dics.h5')
    # load detrended epoch data
    epochs = mne.read_epochs(epoch_file, preload = False )
    epochs.detrend = 1; epochs.load_data()

    roi_data = np.zeros( (n_roi, n_freq, len(epochs.selection)) )

    # calculate CSD for each epoch, apply full band filter,
    # get power in each network
    for epoch in epochs.selection:

        csd_epoch = csd_multitaper(epochs[epoch], fmin = 1, fmax = 100,
                  tmin = epochs[epoch].tmin, tmax = epochs[epoch].tmax,
                  adaptive = True, n_jobs = 9)

        # apply filter to each epoch and calculate power in RSN
        stc, freq = apply_dics_csd(csd_epoch.mean(fmin, fmax), filters)
        stc.subject = subject
        stc_fsaverage = mne.compute_source_morph(stc, subject, 'fsaverage',
                                 subjects_dir = HLTP_pupil.MRI_dir).apply(stc)

        for fband in range(n_freq):
            for i in range(n_roi):
                lh_data = stc_fsaverage.in_label(labels['lh'][i]
                        ).data[:, fband].mean()
                # potentially a problem that I average before log, because of exp dist
                rh_data = stc_fsaverage.in_label(labels['rh'][i]
                        ).data[:, fband].mean()
                roi_data[i, fband, epoch] = (lh_data  + rh_data) / 2.
    HLTP_pupil.save(roi_data, sub_pro_dir + '/roi_single_epoch_dics_power_'
                     + epoch_name)

def save_single_epoch_roi_pwr():
    fmin = np.arange(1, 100, 1)
    fmax = fmin + 1
    labels = {}
    atlas_name = 'Yeo2011_7Networks_N1000'
    for hs in ['lh', 'rh']:
        labels[hs] = mne.read_labels_from_annot(
                'fsaverage', atlas_name, hs, subjects_dir = FS_dir)
    
    for epoch_name in ['task_prestim_ds', 'rest01_ds', 'rest02_ds']:
        for subject in subjects:
            save_subject_epoch_roi_band_pwr(subject, epoch_name, labels, fmin, fmax)
           
def update_df_w_roi_power():
    # update the behavior dataframe to include ROI power, use in future analyses
    bhv_dataframe = pd.read_pickle(HLTP_pupil.result_dir +
                                    '/all_subj_bhv_df_w_pupil.pkl')
    epoch_name = "task_prestim_ds"
    for s, subject in enumerate(subjects):  
        sub_pro_dir = MEG_pro_dir + '/' + subject
   
        roi_data = HLTP_pupil.load(sub_pro_dir + '/roi_single_epoch_dics_power_' + #roi_epoch_dics_power_'
                                    epoch_name)
        for roi in range(7):
            
            pwr_bands = [ zscore(np.log(roi_data[roi, 
                        (frange[0] - 1) : (frange[1] - 1), :]).mean(axis = 0))
                        for fband, frange in HLTP_pupil.freq_bands.items()]
            #if subject == 'BJB':
            #    pwr_bands = [ np.append(pwr_bands[i], 0)
            #        for i in range(len(pwr_bands)) ]
            k = 0
            for fband, frange in HLTP_pupil.freq_bands.items():
                bhv_dataframe.loc[bhv_dataframe.subject == subject, 
                                  fband + str(roi)] = pwr_bands[k]
                k += 1

    bhv_dataframe.to_pickle(HLTP_pupil.result_dir +
                            '/all_subj_bhv_df_w_pupil_power.pkl')

save_single_epoch_roi_pwr()
    
#---- not used now: Prepare power maps with binning according to pupil size------------------
        
# group_percentile = np.arange(0., 100., 20);
# n_roi = 7

# epoch_name = 'task_prestim_ds'
# n_freq = 99
# power_map = np.zeros( (n_roi, len(group_percentile), n_freq, len(subjects)) )

# for s, subject in enumerate(subjects):
#     sub_pro_dir = MEG_pro_dir + '/' + subject
#     epoch_fname = sub_pro_dir + '/' + epoch_name + '-epo.fif'

#     p_group, pupil_size = get_pupil_groups(epoch_fname, group_percentile)
#     pupil = np.unique(p_group)
#     roi_data = HLTP_pupil.load(sub_pro_dir + '/roi_epoch_dics_power_' 
#                      + epoch_name)
    
#     n_roi, n_freq, n_epoch = roi_data.shape
    
#     for roi in range(7):
#         for p in pupil:
#             power_map[roi, p - 1, :, s] = roi_data[roi, :, 
#                      p_group == p].mean(axis = 0)
#         for f in range(n_freq):
#             power_map[roi, :, f, s] = np.log(power_map[roi, :, f, s] / 
#                      np.nanmean(power_map[roi, :, f, s]))
     
# HLTP_pupil.save(power_map, MEG_pro_dir + 
#                     '/pupil_result/group_roi_power_map_7nets_' + epoch_name)        

   