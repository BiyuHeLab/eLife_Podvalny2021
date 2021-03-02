#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:49:19 2020

Describe MEG activity time-locked to pupillary events - constriction 
and dilation - in rest

@author: podvae01
"""
import scipy
import HLTP_pupil
import numpy as np
from scipy.signal import butter, filtfilt, argrelextrema
import mne 
from sklearn.linear_model import LinearRegression
from scipy.stats import distributions
import pandas as pd
import os

def analyse_events(subject, block, filt_pupil, fs):
    # find local minima maxima within 500 ms vicinity
    mini = argrelextrema(filt_pupil, np.less, 
                         order = np.int(.5 * fs))[0]
    maxi = argrelextrema(filt_pupil, np.greater, 
                         order = np.int(.5 * fs))[0]
    # We forgot to turn on the eye-tracker for first minutes in one block only
    if (subject == 'AC') & (block == 'rest01'): 
        mini = mini[mini > 8000]; maxi = maxi[maxi > 8000]
    # identify the steepness of constriction and dilation
    pupil_events = np.concatenate([mini, maxi])    
    n_mini = len(mini)

    slopes = np.zeros(len(pupil_events))
    for event_id, m  in enumerate(pupil_events):
        # fit 100 ms after the detected event. 
        X = filt_pupil[m:(m + np.int(0.1 * HLTP_pupil.resamp_fs))]
        model = LinearRegression().fit(np.arange(len(X)).reshape(-1, 1), X)
        #R2 = model.score(np.arange(len(X)).reshape(-1, 1), X)
        slopes[event_id] = model.coef_
    
    # here I split by slope steepness  "2" = fast "1" = slow  
    dil_code = np.digitize(slopes[:n_mini], 
                           np.percentile(slopes[:n_mini], [0., 50.]))
    con_code = -np.digitize(-slopes[n_mini:], 
                            np.percentile(-slopes[n_mini:], [0., 50.]))
    event_code = np.concatenate([con_code, dil_code])
    return pupil_events, event_code, slopes

def get_pupil_events(block, subject):
    # Create the filter for pupil
    filter_order = 2; 
    frequency_cutoff = 5
    sampling_frequency = HLTP_pupil.raw_fs
    b, a = butter(filter_order, frequency_cutoff, 
              btype='low', output='ba', fs=sampling_frequency)
    pupil_fname = 'clean_interp_pupil_' + block + '.pkl'
    pupil_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir 
                                     + '/' + subject + '/' + pupil_fname)
    # Apply the filter & resample
    filt_pupil = scipy.stats.zscore(filtfilt(b, a, pupil_data))
    filt_pupil = mne.filter.resample(filt_pupil, 
                down = HLTP_pupil.raw_fs / HLTP_pupil.resamp_fs )
    pupil_events, event_code, slopes = analyse_events(subject, block, 
                                      filt_pupil, HLTP_pupil.resamp_fs)
    return pupil_events, event_code, slopes

def save_rest_pupil_event_related_MEG():
    freq = [0, 5]
    # Note: this analysis would not be informative for task because events 
    # caused by stimuli
    for block in  ['rest01', 'rest02']:
        meg_fname = block + '_ds_raw.fif'
        evo = {}
        for subject in HLTP_pupil.subjects:
            
            subj_data_file = HLTP_pupil.MEG_pro_dir + '/' + subject + '/' + meg_fname
            if not os.path.exists(subj_data_file):   
                continue

            pupil_events, event_code, slope = get_pupil_events(block, subject)
            events = np.zeros( (len(pupil_events), 3) ).astype('int')
            events[:, 0] = pupil_events
            events[:, 2] = event_code
            # calculate event related MEG around these events
            raw_data, picks = HLTP_pupil.get_detrended_raw(subj_data_file, 
                subject)  
            raw_data = raw_data.filter(freq[0], freq[1])
            #raw_data = raw_data.apply_hilbert(envelope = True)
            epochs = mne.Epochs(raw_data, events, 
                                    event_id = {'slow_con':1, 'fast_con':2, 
                                                'slow_dil':-1, 'fast_dil':-2 }, 
                                        baseline = None, proj = True, detrend = 0,
                                        tmin = -1, tmax = 1, preload = True)
            epochs.pick_types(meg = True, ref_meg = False, exclude=[])
                
            for event_id in epochs.event_id.keys():
                    evo[event_id + subject] = []
                    evo[event_id + subject] = epochs[event_id].average()
        
        HLTP_pupil.save(evo, HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                                '/evoked_by_pupil_event2_' + block + '.pkl')
    
def combine_rest_blocks_evo():
     evo1 = HLTP_pupil.load( HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                                '/evoked_by_pupil_event2_rest01.pkl')
     evo2 = HLTP_pupil.load( HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                                '/evoked_by_pupil_event2_rest02.pkl')
     for subject in HLTP_pupil.subjects:
         for event_id in ['slow_con', 'fast_con', 'slow_dil', 'fast_dil']:
             if (event_id + subject) in evo2.keys():
                 evo1[event_id + subject].data = (evo1[event_id + subject].data +
                                              evo2[event_id + subject].data)/2
     HLTP_pupil.save(evo1, HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                                '/evoked_by_pupil_event_rest.pkl')

def plot_evoked_example():
    evo = HLTP_pupil.load( HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                                '/evoked_by_pupil_event_rest.pkl')
    evo = pd.read_pickle( '/isilon/LFMI/VMdrive/Ella/HLTP_MEG/proc_data/pupil_result' +
                                '/evoked_by_pupil_event_rest.pkl')
    eid = 'slow_dil'
    times = evo[eid + 'AA'].times

    for chn in [189, 59]:#range(35, 70):##range(30):    

        fig, ax = plt.subplots(1, 1, figsize = (2., 2.8))
        plt.plot([0, 0], [-30, 30], 'k--')
        plt.plot([-1, 1], [0, 0], 'k--')
        clr = {'dil':'r', 'con':'c'}
        for evnt in ['dil', 'con']:
            data_to_plot = [(evo['slow_' + evnt + s].data
                             + evo['fast_' + evnt + s].data) / 2. / 1e-15
                        for s in subjects]
            m_dil = np.mean(data_to_plot, axis = 0)[chn, :]
            e_dil = np.std(data_to_plot, axis = 0)[chn, :] / np.sqrt(24)
            ax.fill_between(times, m_dil + e_dil, m_dil - e_dil, color = clr[evnt],
                            alpha = 0.5, edgecolor='none', linewidth=0)

        plt.xlabel('Time (s)');plt.ylabel('MEG (fT)')
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        plt.xlim([-1, 1]);plt.ylim([-30, 30])
        fig.savefig(figures_dir + '/DvsC_chan_example_' + str(chn)+ 
                    '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
     
        for eid in ['slow_con', 'fast_con', 'slow_dil', 'fast_dil']:
            m = np.mean([evo[eid + s].data for s in subjects], axis = 0)[chn, :]
            plt.plot(evo[eid + 'AA'].times, m, '--', linewidth = 2., 
                         alpha = .5)
        plt.plot([0, 0], [-3e-14, 3e-14], ':', color = 'gray')

def plot_peak_time_distriution():
'''plot distribution of time lags'''
        for evnt in ['dil', 'con']:
            data_to_plot = [(evo['slow_' + evnt + s].data
                             + evo['fast_' + evnt + s].data) / 2. 
                        for s in subjects]
            m_dil = np.mean(data_to_plot, axis = 0)
            peak_lag = np.argmax(np.abs(m_dil), axis = 1)
            time_lag = np.array([times[pl] for pl in peak_lag] )
            fig, ax = plt.subplots(1, 1, figsize = (1.5,1.5))

            plt.hist(time_lag, 20)

def plot_topos_time(times, samp_evo, mask, savetag):
    mparam = dict(marker = '.', markerfacecolor = 'k', markeredgecolor = 'k',
                    linewidth = 0, markersize = 3)  
    
    fig, axes = plt.subplots(1, len(times))
    fig.set_size_inches(12, 6)
    samp_evo.plot_topomap(times, contours = 0, vmin = -1e16, vmax = 1e16,
                          mask_params = mparam, cmap = 'RdYlBu_r', axes = axes,
                          colorbar = False, outlines = 'head', mask = mask.T, 
                          sensors = False)
    plt.subplots_adjust(wspace = 0, hspace =0)
    fig.show()
    fig.savefig(figures_dir + '/pupil_event_related_topo' + savetag + 
                    '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
     
def spatiotemp_perm_test(dd, samp_evo):
    connectivity, pos = HLTP_pupil.get_connectivity()   
    alpha = 0.05; p_accept = 0.05
    threshold = -distributions.t.ppf( alpha / 2., len(HLTP_pupil.subjects) - 1)
    cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test(
        np.swapaxes(dd, 1,2), n_permutations = 1000,
                                  threshold = threshold, tail = 0,
                                  n_jobs = 1, connectivity = connectivity)
    T_obs, clusters, p_values, _ = cluster_stats 
    
    samp_evo.data = T_obs.T
    #samp_evo.data = dd.mean(axis = 0)
    
    mask = np.zeros(T_obs.shape, dtype=bool)
    good_cluster_inds = np.where( p_values < p_accept)[0]
    if len(good_cluster_inds) > 0:
        for g in good_cluster_inds:
            sensors =  clusters[g][1]
            times = clusters[g][0]        
            mask[times, sensors.astype(int)] = True
           
    return samp_evo, mask
    
def plot_topos(): 
        
    for  evnt in ['dil',  'con']:
        dd = np.array([(evo['slow_' + evnt + s].data + 
                        evo['fast_' + evnt + s].data) / 2. 
                            for s in HLTP_pupil.subjects])
    
        template_evo = evo['slow_conAA'].copy()
        samp_evo, mask = spatiotemp_perm_test(dd, template_evo)
        
        times = np.arange(-0.8, 1, 0.2); savetag = 'all' + evnt
        plot_topos_time(times, samp_evo, mask, savetag)
        times = np.arange(0.1, 1., 0.1); savetag = 'post' + evnt
        plot_topos_time(times, samp_evo, mask, savetag)
        
    for  evnt in ['dil',  'con']:
        for speed in ['slow', 'fast']:
            dd = np.array([evo[speed + '_' + evnt + s].data  
                                for s in HLTP_pupil.subjects])
        
            template_evo = evo['slow_conAA'].copy()
            samp_evo, mask = spatiotemp_perm_test(dd, template_evo)
            
            times = np.arange(-1, .01, 0.1); savetag = 'pre' + speed + evnt
            plot_topos_time(times, samp_evo, mask, savetag)
            times = np.arange(0.1, 1., 0.1); savetag = 'post' + speed + evnt
            plot_topos_time(times, samp_evo, mask, savetag)
    

    
    
    
    
    
    
    
    
    

        
        
    
    


# import  matplotlib.pyplot as plt
# subject = 'AC'
# pupil_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir 
#                                      + '/' + subject + '/' + pupil_fname)
# # Apply the filter & resample
# filt_pupil = scipy.stats.zscore(filtfilt(b, a, pupil_data))
# filt_pupil = mne.filter.resample(filt_pupil, 
#              down = HLTP_pupil.raw_fs / HLTP_pupil.resamp_fs )
# pupil_events, event_code = get_pupil_events(subject, 
#                                                     filt_pupil, HLTP_pupil.resamp_fs)
    
# time = np.arange(0, len(filt_pupil)) / HLTP_pupil.raw_fs

# fig, ax = plt.subplots(figsize = [2., 2.]);      

# plt.plot(time-30, filt_pupil, color = 'gray')
# plt.plot(time[pupil_events[event_code > 0]]-30, 
#          filt_pupil[pupil_events[event_code > 0]], 'co', alpha = 0.5)
# plt.plot(time[pupil_events[event_code < 0]]-30, 
#          filt_pupil[pupil_events[event_code < 0]], 'ro', alpha = 0.5)
# plt.ylim([-3, 3])
# ax.spines['left'].set_position(('outward', 10))
# ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('outward', 15))
# ax.xaxis.set_ticks_position('bottom')
# plt.xlabel('time (s)')
# plt.xlim([0, 5])
# plt.ylabel('Pupil size (s.d.)')
# fig.savefig(figures_dir + '/example_pupil_min_max.png', 
#                 bbox_inches = 'tight', transparent=True)