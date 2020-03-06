#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:48:45 2017

HLTP main paradigm automatic preprocessing

@author: podvalnye
"""
import sys
sys.path.append('../../')
import mne
from mne.preprocessing import ICA, read_ica
import HLTP_pupil
from HLTP_pupil import subjects, MEG_pro_dir, MEG_raw_dir
from HLTP_bad_ica_components import bad_comps
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
from os import path
from scipy.stats import zscore
# this line is needed to fix strange error while reading ctf files: 
import locale
locale.setlocale(locale.LC_ALL, "en_US")

def raw_block_preproc_autostage1(filename, subject):
    '''automatic preprocessing stage 1:
        filter the data and fit ICA (later applied on unfiltered data)
    '''
    raw = mne.io.read_raw_ctf(filename, preload=True)     
    raw.info['bads'] = HLTP_pupil.bads[subject]
    raw.pick_types(meg=True, ref_meg=False, exclude='bads')    
    # TODO: remove and repair jumps if needed?
        
    # ICA
    # filter the data, only for ICA purposes
    raw.filter(l_freq=HLTP_pupil.ica_lo_freq, h_freq=HLTP_pupil.ica_hi_freq, 
               phase=HLTP_pupil.ica_phase)
        
    ica = ICA(n_components=HLTP_pupil.ica_n_com, method=HLTP_pupil.ica_method,
              random_state=HLTP_pupil.ica_random_state)
    
    ica.fit(raw, decim=HLTP_pupil.ica_decim, reject=dict(mag=4.5e-12))
    
    block_name = HLTP_pupil.get_block_type(filename)
    fdir = MEG_pro_dir + '/' + subject + '/'+ block_name    
    ica.save(fdir + '-ica.fif')
        
    # plot_ica components
    fdir = MEG_pro_dir + '/' + subject + '/figures/'+ block_name
    plot_ica_timecourse(ica, raw, subject, fdir)
    plot_ica_components(ica, subject, fdir)
  
def plot_ica_timecourse(ica, raw, subject, fdir):
    ica_range=range(0,49,10)
    n_courses2plot=10
    ica_sources=ica.get_sources(raw)
    source_data= ica_sources.get_data()
    for i in ica_range:
        plt.figure(figsize=(20,20))
        plt.title(subject + "Components %s through %s" %(i,i+n_courses2plot))
        for courses in range(i , i+n_courses2plot):
            plt.plot(10*courses+source_data[courses][:], linewidth=0.2)
            plt.text(-15000, courses*10, '%s' %(courses))
        plt.savefig(fdir + "_timecourses_%s.png" %(courses),dpi=300)        
        plt.close()
        
def plot_ica_components(ica, subject, fdir):
    ica_range=range(0,49,10)
    n_components2plot=10
    for i in ica_range:
        picks= range(i,i+n_components2plot)
        comps=ica.plot_components(picks)
        picks_min= str(picks[0])
        picks_max= str(picks[-1])
        comps.savefig(fdir + '_ica_components_' + picks_min + '_through_'
                    +picks_max+ '.png') 
        plt.close(comps)
     
def raw_block_preproc_autostage2(filename, subject):
    '''
    reject bad ICA components and save the clean data
    '''
    block_name = HLTP_pupil.get_block_type(filename)
    fdir = MEG_pro_dir + '/' + subject + '/'+ block_name
    raw = mne.io.read_raw_ctf(filename, preload=True)       

    ica = read_ica(fdir + '-ica.fif')
    ica.exclude = bad_comps[subject][block_name]   
    ica.apply(raw)
    
    raw.save(fdir + '_stage2_raw.fif', overwrite=True)   
    
def raw_block_detrend(block_name, subject):
    fdir = MEG_pro_dir + '/' + subject + '/'+ block_name
    if not path.exists(fdir + '_stage2_raw.fif'): 
        print('No such file'); return False;
    raw = mne.io.read_raw_fif(fdir + '_stage2_raw.fif', preload=True)
    raw.info['bads'] = HLTP_pupil.bads[subject]    
    picks = mne.pick_types(raw.info, meg = True, 
                           ref_meg = False, exclude = 'bads')
    raw.apply_function(signal.detrend, picks=picks, dtype=None, n_jobs=24)
    raw.save(fdir + '_detrended_raw.fif', overwrite=True)
    return True

def raw_block_downsample(block_name, subject):
    fdir = MEG_pro_dir + '/' + subject + '/'+ block_name
    if not path.exists(fdir + '_detrended_raw.fif'): 
        print('No such file'); return False;
    raw = mne.io.read_raw_fif(fdir + '_stage2_raw.fif', preload=True)
    raw.resample(HLTP_pupil.resamp_fs)
    raw.save(fdir + '_ds_raw.fif', overwrite=True)
    return True

def prep_clean_rest_epochs(block_name, subject, fband = ''):
    '''results in downsampled (256Hz), ICA-cleaned, detrended 2 sec epochs
    with clean interpolated pupil'''
    fdir = MEG_pro_dir + '/' + subject + '/'+ block_name
    
    if not path.exists(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                    '/clean_interp_pupil_ds_' + block_name + '.pkl'): 
        print('No such file'); return False;
        
    raw = mne.io.read_raw_fif(fdir + '_ds_raw.fif', preload=True)
    if len(fband) > 1:
        freq = HLTP_pupil.freq_bands[fband]
        raw = raw.filter(freq[0], freq[1])
        raw = raw.apply_hilbert(envelope = True)
  
    clean_pupil = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                    '/clean_interp_pupil_ds_' + block_name + '.pkl')
    
    clean_pupil = zscore(clean_pupil)
    raw._data[HLTP_pupil.pupil_chan, :] = clean_pupil
    
    win = 2. * raw.info['sfreq'] # epoch in two sec intervals
    
    samples = np.arange(0, raw.n_times, win)
    events = np.zeros( (len(samples), 3) ).astype('int')
    events[:, 0] = samples; 
    events[:, 2] = 1
    epochs = mne.Epochs(raw, events, baseline = None, proj = True, 
                        detrend = 0, tmin = 0, tmax = 2., preload = True)
    epochs.save(fdir + '_ds' + fband + '-epo.fif')
    return True
    
def prep_clean_task_epochs(subject, tmin, tmax, epoch_name):
    '''results in downsampled (256Hz), ICA-cleaned, detrended 2 sec epochs
    with clean interpolated pupil'''
    
    block_epochs = [];
    blocks = HLTP_pupil.block_names.copy();
    blocks.remove('rest01'); blocks.remove('rest02')
    for b in blocks:
        fdir = HLTP_pupil.MEG_pro_dir + '/' + subject + '/'+ b
        if not path.exists(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                    '/clean_interp_pupil_ds_' + b + '.pkl'): 
            print('No such file'); continue;
        raw = mne.io.read_raw_fif(fdir + '_ds_raw.fif', preload=True)
        clean_pupil = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                    '/clean_interp_pupil_ds_' + b + '.pkl')
        
        clean_pupil = zscore(clean_pupil)
        
        raw._data[HLTP_pupil.pupil_chan, :] = clean_pupil
                
        events = mne.find_events(raw, stim_channel=HLTP_pupil.stim_channel, 
                                      mask = HLTP_pupil.event_id['Stimulus'], 
                                      mask_type = 'and')
        # Correct for photo-diode delay:
        events[:, 0] = events[:, 0] + round(HLTP_pupil.PD_delay * 
              HLTP_pupil.resamp_fs / HLTP_pupil.raw_fs)

        epochs = mne.Epochs(raw, events, {'Stimulus': 1}, tmin=tmin, tmax=tmax, 
                        proj=True, baseline=None, preload=True, detrend=0, 
                        verbose=False)
        block_epochs.append(epochs)
    #Concatenate    
    #lie about head position, otherwise concatenation doesn't work:
    for b in range(len(block_epochs)):
        block_epochs[b].info['dev_head_t'] = block_epochs[0].info['dev_head_t']
    
    all_epochs = mne.concatenate_epochs(block_epochs)
    
    #Do not interpolate channels, otherwise mne can't read missing channels   
    all_epochs.save(HLTP_pupil.MEG_pro_dir + '/' + subject + '/' + epoch_name +
                    '_ds-epo.fif')
 
# run automatic stage 1 for all subjects   
for s in subjects:      
    subj_raw_dir = MEG_raw_dir + '/' + s + '/'
    filenames, _, _, _ = \
        HLTP_pupil.get_experimental_details(subj_raw_dir)    
    for f in filenames:
        raw_block_preproc_autostage1(f, s)

# run automatic stage 2 for all subjects, after exploring the ICA figures    
for s in subjects:
    subj_raw_dir = MEG_raw_dir + '/' + s + '/'
    filenames, _, _, _ = \
        HLTP_pupil.get_experimental_details(subj_raw_dir) 
    for f in filenames:
        raw_block_preproc_autostage2(f, s)

for s in subjects:
    for b in HLTP_pupil.block_names:
        raw_block_detrend(b, s)
        
for s in subjects:
    for b in HLTP_pupil.block_names:
        raw_block_downsample(b, s)        
        
for s in subjects:
    for b in ['rest01', 'rest02']:
        prep_clean_rest_epochs(b, s, '')

for fband, freq in HLTP_pupil.freq_bands.items():      
    for s in subjects:
        for b in ['rest01', 'rest02']:
            prep_clean_rest_epochs(b, s, fband)
            
for s in subjects:   
    prep_clean_task_epochs(subject = s, tmin = -2, tmax = 0, 
                           epoch_name = 'task_prestim')
    #prep_clean_task_epochs(subject = s, tmin = 0, tmax = 2, 
    #                       epoch_name = 'task_posstim')