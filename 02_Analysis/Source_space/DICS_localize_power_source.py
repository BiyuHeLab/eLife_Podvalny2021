#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:27:25 2019

In this script I use DICS to localize MEG activity at diffrent frequency 
bands accoridng to pupil size. 

@author: podvae01
"""
import HLTP_pupil
import mne
from mne.time_frequency import csd_multitaper
from mne.beamformer import make_dics, apply_dics_csd
import numpy as np        
from os import path         
from HLTP_pupil import subjects, freq_bands, MEG_pro_dir, FS_dir

# FUNCTIONS
def get_epochs_and_pupil(sub_pro_dir, b):
    epoch_fname = sub_pro_dir + '/' + b + '_ds-epo.fif'
        
    if not path.exists(epoch_fname):
        print('No such file'); return np.NaN;
    
    epochs = mne.read_epochs(epoch_fname, preload = False )
    # this step loads detrended epochs -> better for CSD
    epochs.detrend = 1; epochs.load_data()     
    
    pupil_states = HLTP_pupil.load(HLTP_pupil.result_dir + '/pupil_states_' + 
                            b + subject + '.pkl')
    pupil_group = pupil_states.perc_group.values
   
    epochs.events[:, 2] = pupil_group
    epochs.event_id = {'1':1, '2':2, '3':3, '4':4, '5':5}
    epochs.pick_types(meg = True, ref_meg = False, exclude=[])
    if len(epochs.info['bads']):
        epochs.interpolate_bads(reset_bads = False)
    
    return epochs
# DEFINITIONS
method = 'dics'
#------------------------------------------------------------------------------
# calculate the CSD matrices - this is computationally consuming...
#------------------------------------------------------------------------------
for epoch_name in ['rest01', 'rest02', 'task_prestim']:
    for subject in subjects:
        sub_pro_dir = MEG_pro_dir + '/' + subject

        epochs = get_epochs_and_pupil(sub_pro_dir, epoch_name)
        if not (type(epochs) == mne.epochs.EpochsFIF):  continue

        # localize the sources of each frequency for each pupil group
        for evnt, _ in epochs.event_id.items():
            # we compute the csd matrix for each epoch in the group
            csd = csd_multitaper(epochs[evnt], fmin = 1, fmax = 100,
                      tmin = epochs[evnt].tmin, tmax = epochs[evnt].tmax, 
                      adaptive = True, n_jobs = 10)
            csd.save(sub_pro_dir + '/' + method + '_csd_multitaper_'  + 
                     epoch_name + evnt)

#------------------------------------------------------------------------------
# calculate spatial filters w DICS for each freq range
#------------------------------------------------------------------------------
fmin = [freq_bands[f][0] for f in freq_bands]
fmax = [freq_bands[f][1] for f in freq_bands]

for epoch_name in ['rest01_ds', 'rest02_ds', 'task_prestim_ds']:
    for subject in subjects:
        sub_pro_dir = MEG_pro_dir + '/' + subject
        epoch_file = sub_pro_dir + '/' + epoch_name + '-epo.fif'
        if not path.exists(epoch_file): print('No such file'); continue
    
        info = mne.io.read_info(sub_pro_dir + '/' + epoch_name + '-epo.fif')
        fwd = mne.read_forward_solution(sub_pro_dir + '/HLTP_fwd.fif')
        # read the csd for each event and average csds:
        csds = {}
        for evnt, i in epochs.event_id.items():
            csd = mne.time_frequency.read_csd(sub_pro_dir + '/' + method
                                     + '_csd_multitaper_' + epoch_name + evnt)
            csds[evnt] = csd.copy()
            if i == 1: mean_data = csd._data;
            else: mean_data += csd._data
        csd._data = mean_data / len(epochs.event_id)
        
        # Compute DICS spatial filter and estimate source power, 
        # we use mean csd here
        filters = make_dics(info, fwd, csd.mean(fmin, fmax), reg = 0.05, 
                                weight_norm = 'nai', verbose='error')
        filters.save(sub_pro_dir + '/' + method + '_filter_' + epoch_name + 
                     '-dics.h5', overwrite=True)
    
#------------------------------------------------------------------------------
# calculate source space for each averaged pupil group of trials
#------------------------------------------------------------------------------    
for epoch_name in ['rest01_ds', 'rest02_ds', 'task_prestim_ds']:
    for subject in subjects:    
        sub_pro_dir = MEG_pro_dir + '/' + subject
        epoch_file = sub_pro_dir + '/' + epoch_name + '-epo.fif'
        if not path.exists(epoch_file): print('No such file'); continue
        filters = mne.beamformer.read_beamformer(sub_pro_dir + '/' + method 
                                    + '_filter_' + epoch_name + '-dics.h5')
        stcs = {}
        for evnt in range(1, 6):
            csd = mne.time_frequency.read_csd(sub_pro_dir + '/' + method
                             + '_csd_multitaper_' + epoch_name + str(evnt))
            stcs[evnt], freq = apply_dics_csd(csd.mean(fmin, fmax), filters)
            stcs[evnt].save(sub_pro_dir + '/' + method + '_power_map_' 
                + epoch_name + str(evnt))
        
#------------------------------------------------------------------------------        
# transform the averaged power maps to common space for future group analysis  
#------------------------------------------------------------------------------      
for epoch_name in ['rest01_ds', 'rest02_ds', 'task_prestim_ds']:
    for subject in subjects:
        sub_pro_dir = MEG_pro_dir + '/' + subject
        for grp in range(1,6):
            fname = method + '_power_map_' + epoch_name + str(grp)
            if not path.exists(sub_pro_dir + '/' + fname + '-rh.stc'): 
                print('No such file', subject); continue
    
            stc = mne.read_source_estimate( sub_pro_dir + '/' + fname )
            stc.subject = subject         
            stc_fsaverage = mne.compute_source_morph(stc, subject, 'fsaverage', 
                                subjects_dir = HLTP_pupil.MRI_dir).apply(stc)
            stc_fsaverage.save( sub_pro_dir + '/fsaverage_' + fname )   
            

                


