#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:40:59 2019
This file includes general definitions releted to experimental setup and paradigm design, some defaults too.

@author: podvae01
"""
#import numpy as np
import mne
import socket
import pandas as pd
import pickle
import scipy.io as sio
import scipy
from scipy import signal

if socket.gethostname() == 'bjhlpdcpvm01.nyumc.org': # Virtual machine
    Project_dir = '/isilon/LFMI/VMdrive/Ella/HLTP_MEG'
else: # Gago
    Project_dir = '/data/disk3/Ella/HLTP_MEG' 
 
MRI_dir = Project_dir + '/proc_data/freesurfer'
MEG_pro_dir = Project_dir + '/proc_data'
MEG_raw_dir = Project_dir + '/raw_data'
result_dir = MEG_pro_dir + '/pupil_result'
FS_dir = MEG_pro_dir + '/freesurfer'

subjects = ['AA', 'AC', 'AL', 'AR', 'AW', 'BJB', 'CW', 'DJ', 
            'EC', 'FSM', 'JA', 'JC', 'JP', 'JS', 'LS', 'MC',
            'NA', 'NC', 'NM', 'SF', 'SL', 'SM', 'TL', 'TK']

mri_subj = {'good':['AA', 'AC', 'AL', 'AR', 'BJB', 'CW', 'DJ', 'EC', 'FSM','JA',
                    'JP', 'JS', 'LS', 'NA', 'NC', 'NM', 'MC', 'SM', 'TL', 'TK'],
            'bad':['SL', 'AW'],# these subjects mri cannot be used, bad bem 
            'no':['JC', 'SF']} # mri not available 
            # I'm using now AL's mri for the "bad" and "no" MRI subjects
            # because it looks good/standard
  
freq_bands = dict(
    delta=(1, 4), theta=(4, 8), alpha=(8, 13), beta=(13, 30), gamma=(30, 90))    

pupil_chan = 304
raw_fs = 1200.
resamp_fs = 256.

bads= {'AW':['MLO52-1609','MLO53-1609'],
       'DJ':['MLT16-1609'],
       'EC':['MRT31-1609', 'MRT41-1609'],
       'MC':['MLT16-1609'],
       'NA':['MLT16-1609'],
       'SM':['MLF67-1609'],
       'TL':['MLC63-1609'],
       'AA':[],'AC':[],'AL':[],'AR':[], 'BJB':[], 'CW':[],'FSM':[], 'JA':[],
       'JP':[],'JS':[],'LS':[],'NC':[], 'NM':[], 'SL':[], 'TK':[], 'JC':[], 
       'SF':[]}

#----- ICA parameters ---------------------------------------------------------

ica_decim = 3
ica_lo_freq, ica_hi_freq = 1, 45
ica_phase='zero-double' # this part makes difference in ICA components
ica_n_com = 50
ica_random_state = 23
ica_method = 'fastica'

#-----  PreProcessing parameters ----------------------------------------------
filter_phase = 'zero-double'
filter_length = '200ms'
filter_method = 'fir'

# Experiment and triggers
block_type = ['thresh',     # main experiment run - usually we have 10 blocks
              'localizer',  # visual localizer, images from 4 categories
              'rest',       # eyes open, fixation on a grey screen
              'quest'       # image threshold intensity estimation
              ]

# Bad channels 
bad_ch = [32, 172, 191] # these channels are missing from original CTF275 layout

stim_channel = 'UPPT002'       
PD_delay = 44 # samples, measures with photodiode, minimal delay is corrected
event_id = {"Stimulus": 1, "Trial_onset": 8,
            "Question1_category": 2, "Question2_experience": 4}

category_id = { "face": 1, "house": 2, "object": 3, "animal": 4, }

block_names = ['thresh' + str(n + 1).zfill(2) for n in range(10)] + ['rest01', 'rest02']
 
#-----  Helper functions ------------------------------------------------------
def get_detrended_raw(fname, subject):
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.info['bads'] = bads[subject]    
    picks = mne.pick_types(raw.info, meg = True, 
                           ref_meg = False, exclude = 'bads')
    raw.apply_function(signal.detrend, picks=picks, dtype=None, n_jobs=24)
    return raw, picks


def get_raw_epochs(s, epoch_name):
    '''
    Read epochs, remove reference channels, interpolate bad channels, 
    remove bad trials
    '''
    epochs = mne.read_epochs(MEG_pro_dir + '/' + s + '/' + epoch_name + 
                             '-epo.fif')
    epochs.pick_types(meg = True, ref_meg = False, exclude=[])
    
    if len(epochs.info['bads']):
        epochs.interpolate_bads(reset_bads = False)
    #epochs.drop_bad(reject=dict(mag = 4e-12))
    return epochs

# These are mostly remained from matlab, all should be fixed 
# Read details matlab struct, used historicaly, TODO: fix this separately
def get_experimental_details(subj_raw_dir):
    import os.path
    details = sio.loadmat(subj_raw_dir  + 'details.mat')['details']    
    filenames = details['block_file'][0,0]
    date = details['date'][0,0][0]
    subj_code = details['subj_code'][0,0][0]     
    subj_meg_dir = subj_raw_dir + 'MEG/' + subj_code  
     
    raw_filenames = []
    for b in range(0, filenames.shape[1]):   
        raw_filenames.append(subj_meg_dir + filenames[0, b][0] + '.ds')        
    rest_run1 = subj_meg_dir + '_rest_' + date + '_01.ds'
    raw_filenames.append(rest_run1)
    rest_run2 = subj_meg_dir + '_rest_' + date + '_02.ds' 
    if os.path.exists(rest_run2):     
        raw_filenames.append(rest_run2)
    n_blocks = details['block_n'][0,0][0][0]
    return raw_filenames, subj_code, n_blocks, date

# needed for cluster tests, adjacency matrix
def get_connectivity():
    info = load(MEG_pro_dir + 'info.p')
    connectivity, ch_names = mne.channels.read_ch_connectivity('ctf275')
    connectivity = connectivity.toarray()      
    pos = mne.find_layout(info).pos
    
    # remove bad channels:
    chn2remove = [];
    for bad_name in info['bads']:
        chn2remove.append(ch_names.index([s for s in ch_names if 
                                          bad_name[0:5] in s][0]))
    chn2remove= chn2remove + bad_ch
    chn2remove.sort()
    for k in chn2remove:
        pos = scipy.delete(pos, k, 0)
        connectivity = scipy.delete(connectivity, k, 0)
        connectivity = scipy.delete(connectivity, k, 1)
    connectivity = scipy.sparse.csr_matrix(connectivity)
    return connectivity, pos

# Get file name of meg raw data (i.e. *.ds) and return the block type
def get_block_type(filename):
    blockn = ''
    for b in block_type:
        idx = filename.find(b)
        if idx > 0:
            btype = b
            break
    if btype == 'thresh' or btype == 'rest':
        blockn = filename[-5:-3]
            
    btype =  btype + blockn
    return btype

def save(var, file_name):
    outfile = open(file_name, 'wb')          
    pickle.dump(var, outfile)
    outfile.close()
            
def load(file_name):
    return pd.read_pickle(file_name)
