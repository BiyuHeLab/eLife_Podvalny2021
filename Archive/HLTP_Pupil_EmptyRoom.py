#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:25:55 2019

Calculate noise covariance matrices for source reconstruction. 

@author: podvae01
"""
import HLTP_pupil
import mne
from mne.preprocessing import read_ica
from HLTP_bad_ica_components import bad_comps

fname = "MEG_EmptyRoom_2017"
# Empty room recordings collected during dates close to our data collection
# within few days - provided number of days for each subject:
days_diff = {'AA':1, 'AC':1, 'AL':1, 'AR':0, 'BJB':0, 'CW':6, 'DJ':0,
 'EC':1, 'FSM':1, 'JA':2, 'JP':1, 'JS':2, 'LS':2, 'NA':0, 'NM':6, 
 'MC':2, 'SM':3,  'TL':4, 'TK':1}

dates = {'0104_01':['LS', 'JA', 'JS'],
         '0202_01':['AR', 'AC'],
         '0204_01':['SF', 'NA',  'AL', 'JP', 'MC'],
         '0209_01':['DJ', 'BJB', 'EC', 'AA', 'FSM', 'TK'],
         '0515_01':['NM', 'CW',  'AW', 'JC', 'NC',  'TL', 'SM', 'SL']}

fs_ds = 2**8
for d in list(dates.keys())[1:]:
    for subject in dates[d]:
        raw = mne.io.read_raw_ctf(HLTP_pupil.MEG_raw_dir + '/EmptyRoom/' 
                        + fname + d + '.ds', preload = True)
        fdir = HLTP_pupil.MEG_pro_dir + '/' + subject + '/rest01-ica.fif'
        ica = read_ica(fdir)
        ica.exclude = bad_comps[subject]['rest01']   
        ica.apply(raw)
        raw.resample(fs_ds, n_jobs = 5)
        noise_cov = mne.compute_raw_covariance(raw, method='shrunk')
        noise_cov.save(HLTP_pupil.MEG_pro_dir + '/' 
                       + subject + '/empty_room_for_rest1-cov.fif')
        
        
        
        
        
        
        
        