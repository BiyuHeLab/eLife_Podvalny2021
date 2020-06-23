#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:10:11 2020
Fast time scale analysis -s plot results
@author: podvae01
"""
import fig_params
from fig_params import *
import HLTP_pupil
import numpy as np
#import mne
from matplotlib import pyplot as plt
import pandas as pd
figures_dir = HLTP_pupil.MEG_pro_dir  +'/_figures'
con, pos = HLTP_pupil.get_connectivity()   

for b in ['rest01', 'rest02', 'task_prestim', 'task_posstim']:
    for fband, freq in HLTP_pupil.freq_bands.items():

        all_cc = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                            '/cross_corr_' + b + fband + '.pkl')
        mask = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                            '/cross_corr_sig_mask_' + b + fband + '.pkl')
        
        subj_all_cc = [all_cc[s].mean(axis = 0) for s in range(len(all_cc))]
        mean_lag_cc = np.arctanh(np.array(subj_all_cc)).mean(axis = 0)
    
        mparam = dict(marker = '.', markerfacecolor = 'k', markeredgecolor = 'k',
                    linewidth = 0, markersize = 1)  
        #times = np.arange(-1500, 1501, 100)
        times = np.arange(-1500, 1501, 100)
        fig, axes = plt.subplots(1, 11)
        fig.set_size_inches(11, 2)
        for n,l in enumerate(range(10, 21, 1)):
            c = mne.viz.plot_topomap(mean_lag_cc[:, l], pos, axes = axes[n],
                                 show  =False, vmin = -0.025, vmax = 0.025,
                                 contours = 0, mask = mask[l, :],
                                 mask_params = mparam, cmap = 'RdYlBu_r',
                                 outlines = 'head', sensors = False,  
                                 extrapolate = 'none')
            axes[n].set_title(str(times[l]))
        plt.subplots_adjust(wspace = 0, hspace =0)
        cax = plt.axes([0.91, 0.1, 0.01, 0.8])
        plt.colorbar(c[0], cax = cax, ax = axes[n], label = 'Pearson r')
        fig.show()
        fig.savefig(figures_dir + '/ttcorr_pupil_sensor_hr_' + b + fband +
                    '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
        
    fig, ax = plt.subplots(1, 1, figsize = (4.,3.))
    ind = np.unravel_index(np.where(mask), mask.shape)
    sig_chan = np.unique(ind[1])
    non_sig_chan = np.setdiff1d(range(272), sig_chan)
    
    for c in non_sig_chan:
        plt.plot(times/1000, mean_lag_cc[c, :], color='k', alpha = .1)
    for c in sig_chan:
        plt.plot(times/1000, mean_lag_cc[c, :], color='g', alpha = .1)
        
    plt.xlim([-1.5, 1.5]);plt.ylim([-0.06, 0.06])
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('Time (s)'); plt.ylabel('Pearson r')
    fig.savefig(figures_dir + '/corr_pupil_sensor_tc_hr_' + b + 
                '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
    
    ind = np.unravel_index( np.argmax(np.abs(mean_lag_cc), axis = 1), 
                           mean_lag_cc.shape)
    
    sig_t = np.array([times[i] for i in ind[1]])
    fig, ax = plt.subplots(1, 1, figsize = (2.7,1.4))
    plt.hist(sig_t[sig_chan] / 1000., bins = times/ 1000., 
             alpha = 0.5, color = 'k')
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    plt.xlim([-1., 1.]);plt.ylim([0, 150])
    plt.xlabel('Time (s)'); plt.ylabel('# of sensors')
    fig.savefig(figures_dir + '/corr_pupil_sensor_timedist_hr_' + b + '.png', 
                dpi = 800, bbox_inches = 'tight', transparent = True)