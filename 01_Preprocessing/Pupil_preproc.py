#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:01:07 2019

Preprocessing of the contionuous pupil size data.
Files generated:
    - raw_pupil_' + block_name + '.pkl' - this is just raw data
    - clean_pupil_' + block_name + '.pkl' - raw where blinks replaced with NaNs
    - clean_interp_pupil_' + block_name + '.pkl'- blinks interpolated
These files are saved in each subjects processed data directory (MEG_pro_dir)

@author: podvae01
"""
import HLTP_pupil
import mne
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt
from os import path
''' Define threshold of blink detection
Overal, the threshold of -4 was good for most subjects to remove most blinks,
but, upon visual inspection I saw some blinks were not removed 
for some subjects, therefore I lower the threshold
'''   
subj_thr = {}; 
for s in HLTP_pupil.subjects: subj_thr[s] = -4.
for s in ['AC', 'MC', 'JS']: subj_thr[s] = -3.6 
for s in ['AA', 'SL', 'EC']: subj_thr[s] = -3.7
for s in ['AL', 'BJB', 'DJ', 'TK', 'TL']: subj_thr[s] = -3.8
for s in ['FSM', 'JP']: subj_thr[s] = -3.9
for s in ['SF', 'CW', 'AR', 'NM', 'SM']: subj_thr[s] = -3.5 

def get_blinks(pupil_data, subject):
    '''Identify blinks in the timecourse of pupil size recording and return 
    start and end for each blink'''
    thr = subj_thr[subject];  
     
    # setting first and last data point to mean to detect blinks on onset
    pupil_data[0] = np.mean(pupil_data[pupil_data > thr]); 
    pupil_data[-1] = np.mean(pupil_data[pupil_data > thr]) 

    blinks = np.diff((pupil_data < thr).astype(int), n = 1, axis = 0)
    blink_start = (blinks == 1).nonzero()[0]
    blink_end = (blinks == -1).nonzero()[0]
    
    # if there are more ends than starts then 
    if len(blink_start) > len(blink_end):
        print('WARNING: more blink starts than ends')
    #    blink_end = np.append(blink_end, len(pupil_data))
    
    #test
    #plt.plot(pupil_data); plt.plot(blink_start, thr * np.ones(len(blink_start)), 'o');
    return blink_start, blink_end

def clean_pupil(pupil_data, fs, subject):
    '''replace blinks with nans'''
    blink_start, blink_end = get_blinks(pupil_data, subject)
    # remove 100 ms before and after the blink was detected because the data 
    # can be contaminated by movement before/after blink is detected
    safe_int = fs * 0.1 
    for ii in range(len(blink_start)):
        pupil_data[blink_start[ii]: blink_end[ii]] = np.NaN
        s =  int(blink_start[ii] - safe_int) 
        e =  int(blink_end[ii]   + safe_int)
        if ( s >= 0) & ( e < len(pupil_data)):
            pupil_data[ s: e] = np.NaN
        # I start with 1, end with -2 for future interpolation    
        elif ( s < 0): pupil_data[ 1: e] = np.NaN 
        elif ( e >= len(pupil_data)): pupil_data[ s:-2] = np.NaN
    # remove linear trend:
    #time = np.arange(len(pupil_data)); nnan_ind = ~np.isnan(pupil_data)
    #m, b, r_val, p_val, std_err = scipy.stats.linregress(time[nnan_ind], pupil_data[nnan_ind])
    #pupil_data = pupil_data - (m*time + b)
    return pupil_data

def save_blink_clean_pupil(subject, block_name):
    '''saves pupil data with blinks replaced by NaNs'''
    fname = HLTP_pupil.MEG_pro_dir + '/' + subject + \
                              '/' + block_name + '_stage2_raw.fif'
    if not path.exists(fname): print('No such file ' + fname); return False; 
    raw = mne.io.read_raw_fif(fname, preload = True)
    data = raw.get_data()
    fs = int(raw.info['sfreq'])
    
    pupil_data = data[HLTP_pupil.pupil_chan, :]
    HLTP_pupil.save(pupil_data, HLTP_pupil.MEG_pro_dir + '/' + subject + 
                    '/raw_pupil_' + block_name + '.pkl')
    pupil_data = clean_pupil(pupil_data, fs, subject)
    HLTP_pupil.save(pupil_data, HLTP_pupil.MEG_pro_dir + '/' + subject + 
                    '/clean_pupil_' + block_name + '.pkl')
    return True;
    
def save_clean_interp_pupil(subject, block_name):
    data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                                  '/clean_pupil_' + block_name + '.pkl')
    data[0] = np.nanmean(data)
    data[-1] = np.nanmean(data)
    #f = interp1d(np.where(~np.isnan(clean_pupil))[0],
    #                     clean_pupil[~np.isnan(clean_pupil)], 
    #                     kind = 'slinear')
    f = PchipInterpolator(np.where(~np.isnan(data))[0],
                         data[~np.isnan(data)])
    pupil_data = f(np.arange(len(data)))
    HLTP_pupil.save(pupil_data, HLTP_pupil.MEG_pro_dir + '/' + subject + 
                    '/clean_interp_pupil_' + block_name + '.pkl')

def save_resample_pupil(subject, block_name):
    ''' downsample the data '''
    data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                                  '/clean_interp_pupil_' + block_name + '.pkl')
    data = mne.filter.resample(data, down = 
                                      HLTP_pupil.raw_fs/HLTP_pupil.resamp_fs)
    HLTP_pupil.save(data, HLTP_pupil.MEG_pro_dir + '/' + subject + 
                    '/clean_interp_pupil_ds_' + block_name + '.pkl')    

def save_5hz_lpf_pupil(subject, block_name):
    ''' low pass filter the data  '''
    file_name = HLTP_pupil.MEG_pro_dir + '/' + subject + \
                            '/clean_interp_pupil_' + block_name + '.pkl'
    if not path.exists(file_name): 
        print('No such file ' + file_name); return False; 

    data_clean = HLTP_pupil.load(file_name)

    filter_order = 2; frequency_cutoff = 5
    sampling_frequency = HLTP_pupil.raw_fs
    b, a = butter(filter_order, frequency_cutoff, 
                  btype='low', output='ba', fs=sampling_frequency)
    filt_pupil = filtfilt(b, a, data_clean)
    
    HLTP_pupil.save(filt_pupil, HLTP_pupil.MEG_pro_dir + '/' + subject + 
                    '/clean_5hz_lpf_pupil_' + block_name + '.pkl')
    return True

def find_any_blinks_in_prestim():        
    # save blink starts & ends for further analysis:
    for subject in HLTP_pupil.subjects:  
        blinks = {'start'}
        task_blinks = []
        for block_name in HLTP_pupil.block_names[:-2]:
            fname = HLTP_pupil.MEG_pro_dir + '/' + subject + \
                                  '/' + block_name + '_stage2_raw.fif'
            if not path.exists(fname): 
                print('No such file ' + fname); continue
            raw = mne.io.read_raw_fif(fname, preload = True)
            events = mne.find_events(raw, stim_channel=HLTP_pupil.stim_channel, 
                                          mask = HLTP_pupil.event_id['Stimulus'], 
                                          mask_type = 'and')
            file_name = HLTP_pupil.MEG_pro_dir + '/' + subject + \
                        '/raw_pupil_' + block_name + '.pkl'
            if not path.exists(file_name): 
                print('No such file ' + file_name); continue
            pupil_data = HLTP_pupil.load(file_name)
            blink_start, blink_end = get_blinks(pupil_data, subject)
            e_start, e_end = events[:, 0] - 2 * raw.info['sfreq'], events[:, 0]
            blinks = np.zeros( len(e_start) ).astype('bool')
            for epoch_i in range(len(e_start)):
                # blink started during 
                start_during = sum((blink_start > e_start[epoch_i]) & 
                                   (blink_start < e_end[epoch_i]))            
                # blink ended during
                end_during = sum((blink_end > e_start[epoch_i]) & 
                                   (blink_end < e_end[epoch_i])) 
                # started before and ended after: Unlikely since blinks are short,
                # this would be mossing data
                blinks[epoch_i] = start_during | end_during
            task_blinks.append(blinks)     
        HLTP_pupil.save(np.concatenate(task_blinks), HLTP_pupil.MEG_pro_dir 
                       + '/' + subject +  '/blinks_for_prestim_epochs.pkl')   
    
            
for subject in HLTP_pupil.subjects:        

    for block_name in HLTP_pupil.block_names:
        save_5hz_lpf_pupil(subject, block_name)
        r = save_blink_clean_pupil(subject, block_name)
        if not r: continue;
        save_clean_interp_pupil(subject, block_name)  
        save_resample_pupil(subject, block_name)    

    