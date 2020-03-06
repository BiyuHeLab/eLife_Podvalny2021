#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:31:37 2019

@author: podvae01
"""
import HLTP_pupil
import mne
import numpy as np
from scipy import signal    
from HLTP_pupil import clean_pupil
import pandas as pd
from scipy.interpolate import interp1d

group_percentile = np.arange(0., 100.1, 20);

def get_detrended_raw(fname, subject):
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.info['bads'] = HLTP_pupil.bads[subject]    
    picks = mne.pick_types(raw.info, meg = True, 
                           ref_meg = False, exclude = 'bads')
    raw.apply_function(signal.detrend, picks=picks, dtype=None, n_jobs=24)
    return raw, picks
  
def get_pupil_in_win(subject, fname, win_in_s = 2):
    raw = mne.io.read_raw_fif(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                              '/' + fname, preload=True)
    fs = int(raw.info['sfreq'])
    data_len = raw.n_times; 
    win = win_in_s * fs
    samples = np.arange(0, data_len - win, win)
    time = samples / fs
    pupil_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir 
                                 + '/' + subject + '/clean_pupil.pkl')

    pupil = np.zeros( len(samples) )
    for n, samp in enumerate( samples ):
        pdata_sample = pupil_data[samp:(samp + win)]
        # see if generally much samples are not due to blink
        num_nan_samp = sum( np.isnan( pdata_sample ) )
        if num_nan_samp < (win / 2):
            pupil[n] = np.nanmedian( pdata_sample )
        else: pupil[n] = np.nan
    return pupil, samples, time

def prepare_pupil_df():     
    all_df = []
    for subject in HLTP_pupil.subjects:
        pupil, samples, time = get_pupil_in_win(subject, fname, 2)
        bad_trials = np.isnan(pupil)
        # equal number of trials
        p_group = np.digitize(pupil[~bad_trials], 
                              np.percentile(pupil[~bad_trials], 
                                            group_percentile))
        df_dict = {}
        df_dict['subject'] = subject
        df_dict['time'] = time
        df_dict['samples'] = samples
        df_dict['pupil'] = pupil
        df = pd.DataFrame(df_dict)
        df['group'] = np.nan 
        df.loc[~np.isnan(df.pupil.values), 'group'] = p_group 
        all_df.append(df)
    pupil_df = pd.concat(all_df)
    return pupil_df

def prepare_epochs_from_rest(subject, fname):
    
    raw_data, picks = get_detrended_raw(HLTP_pupil.MEG_pro_dir + 
                                        '/' + subject + '/' + fname, subject)    
    fs = int(raw_data.info['sfreq'])
    win = 2*fs
    
    pupil_df = pd.read_pickle(HLTP_pupil.result_dir +'/pupil_rest_df.pkl')
    pupil = pupil_df[pupil_df.subject == subject].pupil.values

    bad_trials = np.isnan(pupil)
    
    pupil =   pupil[~bad_trials]
    p_group = np.digitize(pupil, np.percentile(pupil, group_percentile))
    
    samples = pupil_df[pupil_df.subject == subject].samples.values
    samples = samples[~bad_trials]
    
    events = np.zeros( (len(samples), 3) ).astype('int')
    events[:, 0] = samples; events[:, 2] = p_group
    epochs = mne.Epochs(raw_data, events, event_id = 
                        {'1':1, '2':2, '3':3, '4':4, '5':5},
                        baseline = None, proj = True, detrend = 0,
                        tmin = 0, tmax = win / fs, preload = True)
    return epochs

fname = 'rest01_stage2_rest_raw.fif'
epoch_name = 'rest_pupil2s'
pupil_df = prepare_pupil_df()
pupil_df.to_pickle(HLTP_pupil.result_dir + '/pupil_rest_df.pkl')

for subject in HLTP_pupil.subjects:
    sub_pro_dir = HLTP_pupil.MEG_pro_dir + '/' + subject
    #save_blink_clean_pupil(subject, fname)
    epochs = prepare_epochs_from_rest(subject, fname)
    epochs.save(sub_pro_dir + '/' + epoch_name + '-epo.fif')
    
