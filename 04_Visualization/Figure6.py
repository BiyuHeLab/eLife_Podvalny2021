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
from HLTP_pupil import subjects
import numpy as np
import scipy
#import mne
from matplotlib import pyplot as plt
import pandas as pd
import mne 
figures_dir = HLTP_pupil.MEG_pro_dir  +'/_figures'
con, pos = HLTP_pupil.get_connectivity()


def plot_evoked_example():
    evo = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' +
                          '/evoked_by_pupil_event_rest.pkl')
    evo = pd.read_pickle('/isilon/LFMI/VMdrive/Ella/HLTP_MEG/proc_data/pupil_result' +
                         '/evoked_by_pupil_event_rest.pkl')
    eid = 'slow_dil'
    times = evo[eid + 'AA'].times

    for chn in [189, 59]:  # range(35, 70):##range(30):

        fig, ax = plt.subplots(1, 1, figsize=(2., 2.8))
        plt.plot([0, 0], [-30, 30], 'k--')
        plt.plot([-1, 1], [0, 0], 'k--')
        clr = {'dil': 'r', 'con': 'c'}
        for evnt in ['dil', 'con']:
            data_to_plot = [(evo['slow_' + evnt + s].data
                             + evo['fast_' + evnt + s].data) / 2. / 1e-15
                            for s in subjects]
            m_dil = np.mean(data_to_plot, axis=0)[chn, :]
            e_dil = np.std(data_to_plot, axis=0)[chn, :] / np.sqrt(24)
            ax.fill_between(times, m_dil + e_dil, m_dil - e_dil, color=clr[evnt],
                            alpha=0.5, edgecolor='none', linewidth=0)

        plt.xlabel('Time (s)')
        plt.ylabel('MEG (fT)')
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        plt.xlim([-1, 1])
        plt.ylim([-30, 30])
        fig.savefig(figures_dir + '/DvsC_chan_example_' + str(chn) +
                    '.png', dpi=800, bbox_inches='tight', transparent=True)

        for eid in ['slow_con', 'fast_con', 'slow_dil', 'fast_dil']:
            m = np.mean([evo[eid + s].data for s in subjects], axis=0)[chn, :]
            plt.plot(evo[eid + 'AA'].times, m, '--', linewidth=2.,
                     alpha=.5)
        plt.plot([0, 0], [-3e-14, 3e-14], ':', color='gray')

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
    mparam = dict(marker='.', markerfacecolor='k', markeredgecolor='k',
                  linewidth=0, markersize=3)

    fig, axes = plt.subplots(1, len(times))
    fig.set_size_inches(12, 6)
    samp_evo.plot_topomap(times, contours=0, vmin=-1e16, vmax=1e16,
                          mask_params=mparam, cmap='RdYlBu_r', axes=axes,
                          colorbar=False, outlines='head', mask=mask.T,
                          sensors=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.show()
    fig.savefig(figures_dir + '/pupil_event_related_topo' + savetag +
                '.png', dpi=800, bbox_inches='tight', transparent=True)


def plot_topos():
    for evnt in ['dil', 'con']:
        dd = np.array([(evo['slow_' + evnt + s].data +
                        evo['fast_' + evnt + s].data) / 2.
                       for s in HLTP_pupil.subjects])

        template_evo = evo['slow_conAA'].copy()
        samp_evo, mask = spatiotemp_perm_test(dd, template_evo)

        times = np.arange(-0.8, 1, 0.2)
        savetag = 'all' + evnt
        plot_topos_time(times, samp_evo, mask, savetag)
        times = np.arange(0.1, 1., 0.1)
        savetag = 'post' + evnt
        plot_topos_time(times, samp_evo, mask, savetag)

    for evnt in ['dil', 'con']:
        for speed in ['slow', 'fast']:
            dd = np.array([evo[speed + '_' + evnt + s].data
                           for s in HLTP_pupil.subjects])

            template_evo = evo['slow_conAA'].copy()
            samp_evo, mask = spatiotemp_perm_test(dd, template_evo)

            times = np.arange(-1, .01, 0.1);
            savetag = 'pre' + speed + evnt
            plot_topos_time(times, samp_evo, mask, savetag)
            times = np.arange(0.1, 1., 0.1);
            savetag = 'post' + speed + evnt
            plot_topos_time(times, samp_evo, mask, savetag)

bhv_result = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' +
                                '/fast_event_bhv.pkl')
bhv_vars = ['HR', 'FAR', 'c', 'd', 'p_correct', 'catRT', 'm_pupil']

y_lims = {'HR': [-.2, 1], 'FAR': [-.2, 1], 'c': [-.5, 2.], 'd': [-.5, 2.5],
          'p_correct': [0.2, 0.8], 'catRT': [0.6, 1.6]}
for v in bhv_vars[:-1]:
    data = [np.array(bhv_result[v + 'con']),
            np.array(bhv_result[v + 'dil'])]
    print(scipy.stats.wilcoxon(data[0], data[1]))
    fig, ax = plt.subplots(1, 1, figsize=(0.8, 1.5))

    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    box1 = plt.boxplot(data, positions=[0, 1], patch_artist=True,
                       widths=0.8, showfliers=False,
                       boxprops=None, showbox=None, whis=0, showcaps=False)

    box1['boxes'][0].set(facecolor='c', lw=0, zorder=0, alpha=0.1)
    box1['boxes'][1].set(facecolor='r', lw=0, zorder=0, alpha=0.1)

    box1['medians'][0].set(color='c', lw=2, zorder=20)
    box1['medians'][1].set(color='r', lw=2, zorder=20)
    plt.plot([0, 1], data,
             color=[.5, .5, .5], lw=0.5)
    plt.plot([0], [data[0]], 'o',
             markerfacecolor=[.9, .9, .9], color='c',
             alpha=1.)
    plt.plot([1], [data[1]], 'o',
             markerfacecolor=[.9, .9, .9], color='r', alpha=1.)
    plt.locator_params(axis='y', nbins=6)
    plt.ylim(y_lims[v])
    plt.xlim([-.4, 1.4])

    plt.ylabel(v)
    fig.savefig(figures_dir + '/bhv_convsdil_' + v +
                '.png', dpi=800, bbox_inches='tight', transparent=True)

# ----------------------------------------------------------------------------
# FIGURE A and B: pupil-meg cross-correlation plots
# ----------------------------------------------------------------------------

for b in ['rest', 'task_prestim']: #'task_posstim'

    all_cc = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                        '/cross_corr_' + b   + '.pkl')
    mask = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
                        '/cross_corr_sig_mask_' + b  + '.pkl')
    
    subj_all_cc = [all_cc[s].mean(axis = 0) for s in subjects]
    mean_lag_cc = np.arctanh(np.array(subj_all_cc)).mean(axis = 0)

    mparam = dict(marker = '.', markerfacecolor = 'k', markeredgecolor = 'k',
                linewidth = 0, markersize = 1)  
    times = np.arange(-1500, 1501, 100)
    samp_to_plot = np.arange(3, 28, 3)
    fig, axes = plt.subplots(1, len(samp_to_plot))
    fig.set_size_inches(11, 2)
    for n,l in enumerate(samp_to_plot):
        c = mne.viz.plot_topomap(mean_lag_cc[:, l], pos, axes = axes[n],
                             show  =False, vmin = -0.05, vmax = 0.05,
                             contours = 0, mask = mask[l, :],
                             mask_params = mparam, cmap = 'RdYlBu_r',
                             outlines = 'head', sensors = False,  
                             extrapolate = 'none')
        axes[n].set_title(str(times[l]))
    plt.subplots_adjust(wspace = 0, hspace =0)
    cax = plt.axes([0.91, 0.1, 0.01, 0.8])
    plt.colorbar(c[0], cax = cax, ax = axes[n], label = 'Pearson r')
    fig.show()
    fig.savefig(figures_dir + '/corr_pupil_sensor_topo_' + b +
                '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
    plt.close(fig) 
    
    fig, ax = plt.subplots(1, 1, figsize = (4,3.1))
    ind = np.unravel_index(np.where(mask), mask.shape)
    sig_chan = np.unique(ind[1])
    non_sig_chan = np.setdiff1d(range(272), sig_chan)
    
    for c in non_sig_chan:
        plt.plot(times/1000, mean_lag_cc[c, :], color='k', alpha = .05)
    for c in sig_chan:
        plt.plot(times/1000, mean_lag_cc[c, :], color='k', alpha = .05)
        
    plt.xlim([-1.5, 1.5]);plt.ylim([-0.06, 0.06])
    ax.spines['left'].set_position(('outward', 10))
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('outward', 15))
    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('Time (s)'); plt.ylabel('Pearson r')
    fig.savefig(figures_dir + '/corr_pupil_sensor_timecourse_' + b + 
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
    fig.savefig(figures_dir + '/corr_pupil_sensor_timelag_' + b + '.png', 
                dpi = 800, bbox_inches = 'tight', transparent = True)

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
    fig.savefig(figures_dir + '/corr_pupil_sensor_timedist_' + b + '.png', 
                dpi = 800, bbox_inches = 'tight', transparent = True)

# ----------------------------------------------------------------------------
# Figure C: 
# ----------------------------------------------------------------------------

l = mne.channels.read_layout('CTF275')
l.plot(picks = [189, 59])
# # not sure this is of interest, but reviewers may ask to plot for freq. bands:
# for b in ['rest01', 'rest02', 'task_prestim', 'task_posstim']:
#     for fband, freq in HLTP_pupil.freq_bands.items():

#         all_cc = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
#                             '/cross_corr_' + b + fband + '.pkl')
#         mask = HLTP_pupil.load(HLTP_pupil.MEG_pro_dir + '/pupil_result' + 
#                             '/cross_corr_sig_mask_' + b + fband + '.pkl')
        
#         subj_all_cc = [all_cc[s].mean(axis = 0) for s in range(len(all_cc))]
#         mean_lag_cc = np.arctanh(np.array(subj_all_cc)).mean(axis = 0)
    
#         mparam = dict(marker = '.', markerfacecolor = 'k', markeredgecolor = 'k',
#                     linewidth = 0, markersize = 1)  
#         #times = np.arange(-1500, 1501, 100)
#         times = np.arange(-1500, 1501, 100)
#         fig, axes = plt.subplots(1, 11)
#         fig.set_size_inches(11, 2)
#         for n,l in enumerate(range(10, 21, 1)):
#             c = mne.viz.plot_topomap(mean_lag_cc[:, l], pos, axes = axes[n],
#                                  show  =False, vmin = -0.025, vmax = 0.025,
#                                  contours = 0, mask = mask[l, :],
#                                  mask_params = mparam, cmap = 'RdYlBu_r',
#                                  outlines = 'head', sensors = False,  
#                                  extrapolate = 'none')
#             axes[n].set_title(str(times[l]))
#         plt.subplots_adjust(wspace = 0, hspace =0)
#         cax = plt.axes([0.91, 0.1, 0.01, 0.8])
#         plt.colorbar(c[0], cax = cax, ax = axes[n], label = 'Pearson r')
#         fig.show()
#         fig.savefig(figures_dir + '/ttcorr_pupil_sensor_hr_' + b + fband +
#                     '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
        
#     fig, ax = plt.subplots(1, 1, figsize = (4.,3.))
#     ind = np.unravel_index(np.where(mask), mask.shape)
#     sig_chan = np.unique(ind[1])
#     non_sig_chan = np.setdiff1d(range(272), sig_chan)
    
#     for c in non_sig_chan:
#         plt.plot(times/1000, mean_lag_cc[c, :], color='k', alpha = .1)
#     for c in sig_chan:
#         plt.plot(times/1000, mean_lag_cc[c, :], color='g', alpha = .1)
        
#     plt.xlim([-1.5, 1.5]);plt.ylim([-0.06, 0.06])
#     ax.spines['left'].set_position(('outward', 10))
#     ax.yaxis.set_ticks_position('left')
#     ax.spines['bottom'].set_position(('outward', 15))
#     ax.xaxis.set_ticks_position('bottom')
#     plt.xlabel('Time (s)'); plt.ylabel('Pearson r')
#     fig.savefig(figures_dir + '/corr_pupil_sensor_tc_hr_' + b + 
#                 '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
    
#     ind = np.unravel_index( np.argmax(np.abs(mean_lag_cc), axis = 1), 
#                            mean_lag_cc.shape)
    
#     sig_t = np.array([times[i] for i in ind[1]])
#     fig, ax = plt.subplots(1, 1, figsize = (2.7,1.4))
#     plt.hist(sig_t[sig_chan] / 1000., bins = times/ 1000., 
#              alpha = 0.5, color = 'k')
#     ax.spines['left'].set_position(('outward', 10))
#     ax.yaxis.set_ticks_position('left')
#     ax.spines['bottom'].set_position(('outward', 15))
#     ax.xaxis.set_ticks_position('bottom')
#     plt.xlim([-1., 1.]);plt.ylim([0, 150])
#     plt.xlabel('Time (s)'); plt.ylabel('# of sensors')
#     fig.savefig(figures_dir + '/corr_pupil_sensor_timedist_hr_' + b + '.png', 
#                 dpi = 800, bbox_inches = 'tight', transparent = True)