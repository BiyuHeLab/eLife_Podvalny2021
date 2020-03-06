#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:49:19 2020

@author: podvae01
"""
import scipy
import HLTP_pupil
import numpy as np
from scipy.signal import butter, filtfilt, argrelextrema
import mne 
from sklearn.linear_model import LinearRegression
from scipy.stats import  distributions

block = 'rest01'
pupil_fname = 'clean_interp_pupil_' + block + '.pkl'
meg_fname = block + '_ds_raw.fif'

def get_pupil_events(subject, filt_pupil, fs):
    # find local minima maxima within 500 ms vicinity
    mini = argrelextrema(filt_pupil, np.less, 
                         order = np.int(.5 * fs))[0]
    maxi = argrelextrema(filt_pupil, np.greater, 
                         order = np.int(.5 * fs))[0]
    if (subject == 'AC'): mini = mini[mini > 8000]; maxi = maxi[maxi > 8000]
    # identify the steepness of constriction and dilation
    pupil_events = np.concatenate([mini, maxi])    
    n_mini = len(mini);  

    slopes = np.zeros(len(pupil_events))
    for event_id, m  in enumerate(pupil_events):
        # fit 100 ms after the detected event. 
        X = filt_pupil[m:(m + np.int(0.1 * HLTP_pupil.resamp_fs))]
        model = LinearRegression().fit(np.arange(len(X)).reshape(-1, 1), X)
        #R2 = model.score(np.arange(len(X)).reshape(-1, 1), X)
        slopes[event_id] = model.coef_
        
    con_code = np.digitize(slopes[:n_mini], 
                           np.percentile(slopes[:n_mini], [0., 50.]))
    dil_code = -np.digitize(-slopes[n_mini:], 
                            np.percentile(-slopes[n_mini:], [0., 50.]))
    event_code = np.concatenate([con_code, dil_code])
    return pupil_events, event_code


# Create the filter for pupil
filter_order = 2; frequency_cutoff = 5
sampling_frequency = HLTP_pupil.raw_fs
b, a = butter(filter_order, frequency_cutoff, 
              btype='low', output='ba', fs=sampling_frequency)
for fband, freq in HLTP_pupil.freq_bands.items():
    evo = {'slow_con':[], 'fast_con':[], 'slow_dil':[], 'fast_dil':[]}
    
    for subject in HLTP_pupil.subjects:
    
        pupil_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir 
                                     + '/' + subject + '/' + pupil_fname)
        # Apply the filter & resample
        filt_pupil = scipy.stats.zscore(filtfilt(b, a, pupil_data))
        filt_pupil = mne.filter.resample(filt_pupil, 
                                         down = HLTP_pupil.raw_fs / HLTP_pupil.resamp_fs )
        pupil_events, event_code = get_pupil_events(subject, 
                                                    filt_pupil, HLTP_pupil.resamp_fs)
        events = np.zeros( (len(pupil_events), 3) ).astype('int')
        events[:, 0] = pupil_events
        events[:, 2] = event_code
        
        # calculate event related MEG around these events
        raw_data, picks = HLTP_pupil.get_detrended_raw(HLTP_pupil.MEG_pro_dir + 
                                           '/' + subject + '/' + meg_fname, subject)  
        raw_data = raw_data.filter(freq[0], freq[1])
        raw_data = raw_data.apply_hilbert(envelope = True)
        epochs = mne.Epochs(raw_data, events, 
                            event_id = {'slow_con':1, 'fast_con':2, 
                                        'slow_dil':-1, 'fast_dil':-2 }, 
                                baseline = None, proj = True, detrend = 0,
                                tmin = -1, tmax = 1, preload = True)
        epochs.pick_types(meg = True, ref_meg = False, exclude=[])
        
        for eid in epochs.event_id.keys():
            evo[eid].append(epochs[eid].average())

    HLTP_pupil.save(evo, HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                        '/evoked_by_pupil_event_' + fband + block + '.pkl')
#compare constriction/dilation onsets - group-level analysis
for fband, freq in HLTP_pupil.freq_bands.items():
    
    evo = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                            '/evoked_by_pupil_event_' + fband + block + '.pkl')
    c = {'slow_con':'b', 'fast_con':'c', 'slow_dil':'r', 'fast_dil':'m'}
    times = evo['slow_con'][0].times
    #fig, ax = plt.subplots(1, 1, figsize = (2.,2))    
    for chn in [189, 59]:#range(35, 70):##range(30):
    
        fig, ax = plt.subplots(1, 1, figsize = (1.5,1.5))    
        
        plt.title(str(chn))
        
        #for k in list(evo.keys()):
        #    m = np.mean([evo[k][s].data for s in range(24)], axis = 0)[chn, :]
        #    plt.plot(evo[k][s].times, m, ':', color = c[k],  linewidth=2., alpha = .5)
        
        m_dil = np.mean([(evo['slow_dil'][s].data + evo['fast_dil'][s].data) / 2. 
                        for s in range(24)], axis = 0)[chn, :]
        e_dil = np.std([(evo['slow_dil'][s].data + evo['fast_dil'][s].data) / 2.
                        for s in range(24)], axis = 0)[chn, :] / np.sqrt(24)
        ax.fill_between(times, m_dil + e_dil, m_dil-e_dil, color = 'r',
                        alpha = 0.5, edgecolor='none', linewidth=0)
        
        m_con = np.mean([(evo['slow_con'][s].data + evo['fast_con'][s].data) / 2. 
                        for s in range(24)], axis = 0)[chn, :]
        e_con = np.std([(evo['slow_con'][s].data + evo['fast_con'][s].data) / 2.
                        for s in range(24)], axis = 0)[chn, :] / np.sqrt(24)
        ax.fill_between(times, m_con + e_con, m_con-e_con, color = 'c',
                        alpha = 0.5, edgecolor='none', linewidth=0)
        
        plt.xlim([-1, 1]);plt.ylim([-.5e-14, .5e-14])
        
        plt.xlabel('Time (s)');plt.ylabel('MEG')
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        plt.plot([0, 0], [-3e-14, 3e-14], ':', color = 'gray')
        fig.savefig(figures_dir + '/DvsC_m_' + fband + '_' + block + str(chn)+
                    '.png', dpi = 800, 
                    bbox_inches = 'tight', transparent = True)
    
    
for fband, freq in HLTP_pupil.freq_bands.items():
    
    evo = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                            '/evoked_by_pupil_event_' + fband + block + '.pkl')

    dd = np.array([(evo['slow_con'][s].data + evo['fast_con'][s].data
           - evo['fast_dil'][s].data - evo['slow_dil'][s].data) for s in range(24)])
    
    connectivity, pos = HLTP_pupil.get_connectivity()   
    alpha = 0.05; p_accept = 0.05
    threshold = -distributions.t.ppf( alpha / 2., 24 - 1)
    cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test(np.swapaxes(dd, 1,2), 
                                  n_permutations = 1000,
                                  threshold = threshold, tail=0,
                                  n_jobs = 1, connectivity=connectivity)
    T_obs, clusters, p_values, _ = cluster_stats 
    
    samp_evo = evo['slow_con'][0].copy()
    samp_evo.data = T_obs.T
    #samp_evo.save(HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
    #                        '/DvsC_Tval_' + block + '-ave.fif')
    #dd = np.mean([e.data for e in evo_dilate], axis = 0) - np.mean([e.data for e in evo_constrict], axis = 0)
    samp_evo.data = dd.mean(axis = 0)
    
    mask = np.zeros(T_obs.shape, dtype=bool)
    good_cluster_inds = np.where( p_values < p_accept)[0]
    if len(good_cluster_inds) > 0:
        for g in good_cluster_inds:
            sensors =  clusters[g][1]
            times = clusters[g][0]        
            mask[times, sensors.astype(int)] = True    
    #HLTP_pupil.save(mask, HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
    #                        '/DvsC_mask' + block + '.pkl')

    # plot the results - to be moved to a plotting script
    mparam = dict(marker = '.', markerfacecolor = 'k', markeredgecolor = 'k',
                    linewidth = 0, markersize = 3)  
    
    times = np.arange(-1, .01, 0.1)
    fig, axes = plt.subplots(1, len(times))
    fig.set_size_inches(12, 6)
    samp_evo.plot_topomap(times, contours = 0, vmin = -30, vmax = 30,
                          mask_params = mparam, cmap = 'RdYlBu_r', axes = axes,
                          colorbar = False, outlines = 'head', mask = mask.T, sensors = False)
    plt.subplots_adjust(wspace = 0, hspace =0)
    fig.show()
    fig.savefig(figures_dir + '/DvsC_topo_neg_' + fband + block + 
                    '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
      
    times = np.arange(0., 1.01, 0.1)
    fig, axes = plt.subplots(1, len(times))
    fig.set_size_inches(12, 6)
    samp_evo.plot_topomap(times, contours = 0, vmin = -30, vmax = 30,
                          mask_params = mparam, cmap = 'RdYlBu_r', axes = axes,
                          colorbar = False, outlines = 'head', mask = mask.T, sensors = False)
    plt.subplots_adjust(wspace = 0, hspace =0)
    fig.show()
    fig.savefig(figures_dir + '/DvsC_topo_pos_' + fband + block + 
                    '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
     
fig, axes = plt.subplots(figsize = (11, 11))
times = -.6
samp_evo.plot_topomap(times, contours = 0, vmin = -6e15, vmax = 6e15, 
                     cmap = 'RdYlBu_r', show_names = True,
                      outlines = 'head', size=6,  colorbar = False, sensors = False)
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88)

import  matplotlib.pyplot as plt
subject = 'AC'
pupil_data = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir 
                                     + '/' + subject + '/' + pupil_fname)
# Apply the filter & resample
filt_pupil = scipy.stats.zscore(filtfilt(b, a, pupil_data))
filt_pupil = mne.filter.resample(filt_pupil, 
             down = HLTP_pupil.raw_fs / HLTP_pupil.resamp_fs )
pupil_events, event_code = get_pupil_events(subject, 
                                                    filt_pupil, HLTP_pupil.resamp_fs)
    
time = np.arange(0, len(filt_pupil)) / HLTP_pupil.raw_fs

fig, ax = plt.subplots(figsize = [2., 2.]);      

plt.plot(time-30, filt_pupil, color = 'gray')
plt.plot(time[pupil_events[event_code > 0]]-30, 
         filt_pupil[pupil_events[event_code > 0]], 'co', alpha = 0.5)
plt.plot(time[pupil_events[event_code < 0]]-30, 
         filt_pupil[pupil_events[event_code < 0]], 'ro', alpha = 0.5)
plt.ylim([-3, 3])
ax.spines['left'].set_position(('outward', 10))
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('outward', 15))
ax.xaxis.set_ticks_position('bottom')
plt.xlabel('time (s)')
plt.xlim([0, 5])
plt.ylabel('Pupil size (s.d.)')
fig.savefig(figures_dir + '/example_pupil_min_max.png', 
                bbox_inches = 'tight', transparent=True)