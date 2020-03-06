#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:09:12 2020

Prints plots for figure 1

@author: podvae01
"""
import fig_params
from fig_params import *
import HLTP_pupil
import numpy as np
#from matplotlib import cm
import matplotlib.pyplot as plt
#import pandas as pd

def plot_fig_1C():
    
    subject = 'AC'
    block_name = 'rest01'
    
    file_name = '/raw_pupil_'
    data_raw = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                        file_name + block_name + '.pkl')         
    file_name = '/clean_interp_pupil_'
    data_clean = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/' + subject + 
                        file_name + block_name + '.pkl')
    time = np.arange(0, len(data_raw)) / HLTP_pupil.raw_fs

    fig, ax = plt.subplots(1, 1, figsize = (3.,1.5))
    plt.plot(time - 100, data_raw, color = [.8, .8, .8])
    plt.plot(time - 100, data_clean, color = [.5, .5, .5])
    plt.xlim([0, 60])
    plt.ylim([-3.3, -2.7])
    plt.xlabel('Time (s)');plt.ylabel('Pupil size (a.u.)')
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    fig.savefig(fig_params.figures_dir + '/example1_pupil_' + subject + '.png', 
                    bbox_inches = 'tight', transparent=True)

    # zoomin/inset
    filt_pupil = HLTP_pupil.load( HLTP_pupil.MEG_pro_dir + '/' + subject + 
                        '/clean_5hz_lpf_pupil_' + block_name + '.pkl')
    
    pupil_events = HLTP_pupil.load(HLTP_pupil.result_dir + '/pupil_events_' + 
                            block_name + subject + '.pkl')
    fig, ax = plt.subplots(1, 1, figsize = (2.,1.5))
    plt.plot(time - 100, data_raw, color = [.8, .8, .8])
    #plt.plot(time, data_clean, color = [.8, .8, .8], alpha = 0.5)
    plt.plot(time - 100, filt_pupil, 'k', linewidth = 2)
    dil = pupil_events[pupil_events.event_type == 'dil']['sample']
    con = pupil_events[pupil_events.event_type == 'con']['sample']
    plt.plot(time[dil] - 100, filt_pupil[dil], 'co', alpha = .7)
    plt.plot(time[con] - 100, filt_pupil[con], 'ro', alpha = .7)
    plt.xlim([5, 15])
    plt.ylim([-3.2, -2.8])
    plt.xlabel('Time (s)');plt.ylabel('Pupil size (a.u.)')
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    fig.savefig(fig_params.figures_dir + '/example2_pupil_' + subject + '.png', 
                    bbox_inches = 'tight', transparent=True)
    return;
    
    
def plot_fig_1D():
    ''' plot mean power spectrum of all subjects '''
    all_subj_pxx = []
    fig, ax = plt.subplots(1, 1, figsize = (2.,2.))
    for subject in HLTP_pupil.subjects:
        [f, Pxx_den] = HLTP_pupil.load(HLTP_pupil.result_dir + '/Pupil_PSD_' + 
                    subject + '.pkl')
        plt.plot(f, Pxx_den, color = 'k', alpha = .1)
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        all_subj_pxx.append(Pxx_den)
    
    m = np.mean(all_subj_pxx, axis = 0);
    e = np.std(all_subj_pxx, axis = 0) / np.sqrt(24);
    ax.fill_between(f, m + e, m-e, color = 'k',
                        alpha = 0.5, edgecolor='none', linewidth=0) 
    plt.plot(f, m, color = 'k')
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-2, 100])
    plt.ylim([1e-5, 10])
    plt.xlabel('Frequency (Hz)');plt.ylabel('Pupil Power (a.u.)')
    
    fig.savefig(fig_params.figures_dir + '/pupil_PSD.png', 
                bbox_inches = 'tight', transparent = True)  
    return

def plot_fig_1E():
    epoch_name = 'task_prestim'
    subject = 'SF'
    pupil_states = HLTP_pupil.load(HLTP_pupil.result_dir + '/pupil_states_' + 
                            epoch_name + subject + '.pkl')
    group_percentile = np.arange(0., 100., 20);
    perc = np.percentile(pupil_states.mean_pupil, group_percentile)

    fig, ax = plt.subplots(1, 1, figsize = (3.5,2.5))
    
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')

    plt.hist(pupil_states.mean_pupil, bins = np.arange(-3, 3.01, .15), 
             normed = True, color = [.7,.7,.7])
    for p in perc[1:]:
        plt.axvline(p, color='k', linestyle=':')
    plt.ylim([0, .6]); plt.xlim([-3, 3])
    plt.locator_params(axis='x', nbins=7)

    plt.ylabel('% of samples')
    plt.xlabel('pupil size (zscore)')
    
    fig.savefig(fig_params.figures_dir + '/pupil_state_size_hist.png', 
                bbox_inches = 'tight', transparent = True)
    return

def plot_fig_1X():
    block_name = 'rest01'
    [mean_con, mean_dil] = HLTP_pupil.load(
                     HLTP_pupil.result_dir + '/ERpupil_' + block_name + '.pkl')                  
    times = np.arange(-1, 1, 2/2400)
    m1 = np.mean(np.array(mean_con), axis = 0)
    e1 = np.std(np.array(mean_con), axis = 0) / np.sqrt(24)
    m2 = np.mean(np.array(mean_dil), axis = 0)
    e2 = np.std(np.array(mean_dil), axis = 0) / np.sqrt(24)
    
    fig, ax = plt.subplots(1, 1, figsize = (2.,2.))
    ax.fill_between(times, m1 + e1, m1 - e1, color = 'c',
                        alpha = 0.5, edgecolor='none', linewidth=0) 
    ax.fill_between(times, m2 + e2, m2 - e2, color = 'r',
                        alpha = 0.5, edgecolor='none', linewidth=0)
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    plt.xlim([-1, 1])
    plt.ylim([-.7, .7])
    plt.xlabel('Dilation onset');plt.ylabel('Pupil size (a.u.)')

    #plt.plot(np.array(mean_con).mean(axis = 0));
    #plt.plot(np.array(mean_dil).mean(axis = 0));  
#------------------------------------------------------------------------------

plot_fig_1C()    
plot_fig_1D() 
plot_fig_1E()
    
    
    
    
    
    