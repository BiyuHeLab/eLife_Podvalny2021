#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:29:29 2020

Analyze pupil data to characterize slow states and fast events such as 
dilation and constriction.

@author: podvae01

"""
import sys
sys.path.append('../../')
from os import path
import HLTP_pupil
from scipy.stats import linregress, zscore
from scipy.signal import welch
import numpy as np
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
import pandas as pd
import mne

def get_pupil_events(subject, filt_pupil, fs):
    ''' find local minima maxima within 500 ms vicinity '''
    # minima specify onset of dilation, maxima onset of constriction
    dil = argrelextrema(filt_pupil, np.less, 
                         order = np.int(.5 * fs))[0]
    con = argrelextrema(filt_pupil, np.greater, 
                         order = np.int(.5 * fs))[0]
    pupil_events = np.concatenate([dil, con])    
    event_type = np.concatenate([['dil']*len(dil), ['con']*len(con)]) 
    
    # identify the steepness of constriction and dilation
    slopes = np.zeros(len(pupil_events)); R2 = np.zeros(len(pupil_events))
    for event_id, m  in enumerate(pupil_events):
        # fit 100 ms after the detected event. 
        X = filt_pupil[m:(m + np.int(0.1 * HLTP_pupil.resamp_fs))]
        model = LinearRegression().fit(np.arange(len(X)).reshape(-1, 1), X)
        
        R2[event_id] = model.score(np.arange(len(X)).reshape(-1, 1), X)
        slopes[event_id] = model.coef_

    return pupil_events, event_type, slopes, R2

def save_pupil_events(block_name):
    for subject in HLTP_pupil.subjects:
        filt_pupil = HLTP_pupil.load( HLTP_pupil.MEG_pro_dir + '/' + subject + 
                        '/clean_5hz_lpf_pupil_' + block_name + '.pkl')
    
        pupil_events, event_type, slope, R2 = get_pupil_events(
                subject, filt_pupil, HLTP_pupil.raw_fs)
        
        events = pd.DataFrame({'sample':pupil_events, #bad word to use -fix
                               'event_type':event_type,
                               'slope':slope,
                               'R2':R2})
        # because some data is missing in the beginning for AC rest 01:
        #if (subject == 'AC') & (block_name == 'rest01'):
        #    mini = mini[mini > 8000]; maxi = maxi[maxi > 8000]

        events.to_pickle(HLTP_pupil.result_dir + '/pupil_events_' + 
                            block_name + subject + '.pkl')
    return

def save_pupil_state(epoch_name):
    ''' use task or rest epochs to calculate 2-sec mean pupil'''
    group_percentile = np.arange(0., 100., 20);
    for s, subject in enumerate(HLTP_pupil.subjects):  
        fname = HLTP_pupil.MEG_pro_dir + '/' + subject + '/' + epoch_name + '_ds-epo.fif'
        if not path.exists(fname): 
            print('No such file'); continue;
        epochs = mne.read_epochs(fname, preload = True )
        mean_pupil = epochs._data[:, HLTP_pupil.pupil_chan, :].mean(axis = 1)
        perc = np.percentile(mean_pupil, group_percentile)
        p_group = np.digitize(mean_pupil, perc)
        pupil_states = pd.DataFrame({'mean_pupil':mean_pupil, 
                                     'perc_group':p_group})
        pupil_states.to_pickle(HLTP_pupil.result_dir + '/pupil_states_' + 
                            epoch_name + subject + '.pkl')
    return True

def save_event_related_pupil(block_name):
    '''save pupil size time course of -1 to 1 sec 
    around dilation and constriction '''
    mean_con = []; mean_dil = []
    for subject in HLTP_pupil.subjects:
        filt_pupil = HLTP_pupil.load( 
                HLTP_pupil.MEG_pro_dir + '/' + subject + 
                '/clean_5hz_lpf_pupil_' + block_name + '.pkl')
        filt_pupil = zscore(filt_pupil)
        
        pupil_events = HLTP_pupil.load(
                HLTP_pupil.result_dir + '/pupil_events_' + block_name + 
                subject + '.pkl')

        d = np.int(HLTP_pupil.raw_fs)
        
        pupil_events = pupil_events[
                (pupil_events['sample'] > d) &
                (pupil_events['sample'] < len(filt_pupil) - d)]
        trials = np.stack([filt_pupil[(s - d):(s + d)]
                                for s in pupil_events['sample'].values])
        
        mean_con.append(np.mean(trials[(pupil_events.event_type == 'con'
                                   ).values.astype('bool'), :], axis = 0))   
        mean_dil.append(np.mean(trials[(pupil_events.event_type == 'dil'
                                   ).values.astype('bool'), :], axis = 0))
    HLTP_pupil.save([mean_con, mean_dil], 
                     HLTP_pupil.result_dir + '/ERpupil_' + block_name + '.pkl')                  
        
        
def save_pupil_PSD(block_name):
    '''calculate and save power spectral density of pupil time course (5m)'''
    file_name = '/clean_interp_pupil_'
    
    for subject in HLTP_pupil.subjects:
        data_clean = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                                file_name + block_name + '.pkl')
    
        m, b, r_val, p_val, std_err = linregress(range(len(data_clean)),
                                                            data_clean)
        data_clean = data_clean - (m * range(len(data_clean)) + b)
        f, Pxx_den = welch(zscore(data_clean), HLTP_pupil.raw_fs,
                                        nperseg = 2**15)
        HLTP_pupil.save([f, Pxx_den], HLTP_pupil.result_dir + '/Pupil_PSD_' + 
                        block_name + subject + '.pkl')

def update_bhv_df_w_pupil(block):
    bhv_df = pd.read_pickle(HLTP_pupil.MEG_pro_dir + 
                            '/results/all_subj_bhv_df.pkl')
    for subject in HLTP_pupil.subjects:
        
        pupil_states = HLTP_pupil.load(HLTP_pupil.result_dir + 
                           '/pupil_states_' + block + subject + '.pkl')
        pupil_group = pupil_states.perc_group.values
        if subject == 'BJB':# one last trial is missing from meg
            bhv_df = bhv_df[ ~((bhv_df.index == 288) & 
                               (bhv_df.subject == subject))]
            
        bhv_df.loc[bhv_df.subject == subject, 'pupil_size_pre'] = pupil_group
    bhv_df.to_pickle(HLTP_pupil.MEG_pro_dir +
                     '/results/all_subj_bhv_df_w_pupil.pkl')
   
    return

for block_name in ['rest01', 'rest02']:
    save_pupil_events(block_name)
    save_event_related_pupil(block_name)
    save_pupil_PSD(block_name) 

for epoch_name in ['task_prestim', 'rest01', 'rest02']:#'task_prestim', 
    save_pupil_state(epoch_name)

update_bhv_df_w_pupil('task_prestim')
