#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:16:59 2020

@author: podvae01
"""
import fig_params
from fig_params import *
import HLTP_pupil
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats
from scipy.stats import sem
import pandas as pd

def plot_fig_1A(b):
    # plot sensor space parameter estimates (supplementary)
    # the betas in this figures are from analysis in Pupil_PSD_sensor_level
    res = pd.read_pickle(
                HLTP_pupil.result_dir + '/LM_betas_sensor_' + b + '.pkl')  #blink_control_
    con, pos = HLTP_pupil.get_connectivity()   
    mparam = dict(marker = '.', markerfacecolor = 'k', markeredgecolor = 'k',
                        linewidth = 0, markersize = 1)  
    for term in ['L', 'Q']:#, 'Blink'
        if term == 'L':
            v = 0.1; 
        else: v = 0.05
        for fband, _ in HLTP_pupil.freq_bands.items():
            fig,ax = plt.subplots(1, 1, figsize = (3,2))
            c = mne.viz.plot_topomap(res[fband + term], pos, 
                                 mask = res[fband + term + 'pval_corrected'  ] < 0.05, 
                                 sensors = False, 
                                 mask_params = mparam, axes = ax, 
                                 vmin = -v, vmax = v,   
                                 extrapolate = 'none',
                                 contours = [-300,300], cmap = 'Spectral_r')
            plt.subplots_adjust(wspace = 0, hspace =0)
            #cax = plt.axes([0.91, 0.1, 0.03, 0.8])
            #plt.colorbar(c[0], cax = cax, ax = ax, label = 't-value')
            fig.show()
        
            fig.savefig(fig_params.figures_dir + 
                        '/test_spectral_sensor_topo_'+ term + b + fband +
                        '.png', dpi = 800, bbox_inches = 'tight', transparent = True)
def pwr_data_to_plot():
    for s, subject in enumerate(HLTP_pupil.subjects):
        fname = HLTP_pupil.result_dir + '/PSD' + b + subject +'.pkl'
        if not path.exists(fname):
            print('No such file'); continue
        psd, freq, pupil = HLTP_pupil.load(fname) 
        
        band_pwr = []
        for band, frange in HLTP_pupil.freq_bands.items():                          
            band_pwr.append(psd[:, :, (freq > frange[0]) & (freq <= frange[1])
                    ].mean(axis = -1))

def plot_fig_1B(resolution = 10):
    freq = np.arange(0, 128.01, .5)
    pupil = np.arange(resolution/2, 105, resolution)
    #color = {'rest':'c', 'task_prestim':'r'}
    for band, freq_range in HLTP_pupil.freq_bands.items():
        fig, ax = plt.subplots(figsize = [1.2, 2.]);  
    
        for b in ['rest', 'task_prestim']:
            mean_meg, freq = HLTP_pupil.load(HLTP_pupil.result_dir + 
                            '/mean_meg_by_pupil_state' + b + 
                            str(resolution) + '.pkl')                    
            band_mean = np.nanmean((mean_meg[:, :, :, 
                        (freq > freq_range[0]) & (freq <= freq_range[1])]), 
                        axis = -1)
            res = pd.read_pickle(
                HLTP_pupil.result_dir + '/LM_betas_' + b + '.pkl')  
            mask = (res[band + 'Qpval_corrected'] < 0.05
                    ) | (res[band + 'Lpval_corrected'] < 0.05)
            
            m1 = band_mean.mean(axis = 0)[:, mask].mean(axis = 1)
            e1 = sem(band_mean[:, :, mask].mean(axis = 2), axis = 0)

            plt.fill_between(pupil, m1- e1, m1+ e1, alpha =.4 )#, color = color[b]
            #plt.scatter(pupil, (res[band + 'inter'][mask]).mean() + (res[band + 'L'][mask]).mean() * pupil + 
            #         (res[band + 'Q'][mask]).mean() * pupil ** 2 +  )
            plt.ylabel('relative power (dB)')
            plt.xlabel('pupil size (%)')
            plt.xlim([0, 100]); plt.ylim([-.2,.2]); 
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        fig.savefig(fig_params.figures_dir + 
                    '/mean_sensor_power_band_by_pupil_size' + band + 
                    str(resolution) + '.png', 
                    bbox_inches = 'tight', 
                    dpi = 800, transparent = True)

def plot_fig_1B_archive():
    freq = np.arange(0, 128.01, .5)
    for b in ['task_prestim', 'rest']:
        mean_meg = HLTP_pupil.load(HLTP_pupil.result_dir + 
                        '/mean_meg_by_pupil_state' + b + '.pkl')
         
        fig, ax = plt.subplots(figsize = [1.5, 2.]);  
        data = mean_meg[:,:,:, (freq > 1) & (freq < 100)].mean(axis = 2)
        tstt, pval=stats.ttest_1samp(data, 
                                     popmean = 0, axis= 0 )    
       
        plt.imshow(tstt.T, interpolation = 'bilinear',#tstt.T
                       vmin = -3, vmax = 3, cmap = 'RdYlGn_r', alpha = 0.9,
                      extent = [0, 100, 100, 1]);     
       
        plt.colorbar()      
        plt.yscale('log'); #plt.xscale('log'); 
        plt.xlabel('Pupil size (%)')
        plt.ylabel('Frequency (Hz)')
        fig.savefig(fig_params.figures_dir + 
                    '/mean_sensor_power_map_by_pupil_size' + b + '.png', 
                    bbox_inches = 'tight', 
                    dpi = 800, transparent = True)

def plot_ANOVA_result(b):

    con, pos = HLTP_pupil.get_connectivity()   
    mparam = dict(marker = '.', markerfacecolor = 'k', markeredgecolor = 'k',
                        linewidth = 0, markersize = 1)  
    for band, freq_range in HLTP_pupil.freq_bands.items():
        [T_obs, mask] = HLTP_pupil.load(HLTP_pupil.result_dir + 
                        '/pupil_states_ANOVA_' + band + b + '.pkl')
        plt.figure()
  
        fig,ax = plt.subplots(1, 1, figsize = (3,2))
        mne.viz.plot_topomap(T_obs, pos, mask = mask, sensors = False, 
                             mask_params = mparam, axes = ax,show  =False, 
                             vmin = 0, vmax = 10,   
                             extrapolate = 'none',
                             contours = [-300,300], cmap = 'YlGnBu')
        plt.subplots_adjust(wspace = 0, hspace =0)
        #cax = plt.axes([0.91, 0.1, 0.03, 0.8])
        #plt.colorbar(c[0], cax = cax, ax = ax, label = 't-value')
        fig.show()
    
        fig.savefig(fig_params.figures_dir + '/spectral_sensor_topo_'+ b + band +
                        '.png', dpi = 800, bbox_inches = 'tight', transparent = True)

def plot_betas(block_name):
    for term in [1, 2]:
        errs = []; betas = []

        for fband, frange in HLTP_pupil.freq_bands.items():
            mdf_Q = pd.read_pickle(
                    HLTP_pupil.result_dir +
                    '/mixedlmQ_full_pupil_' + block_name
                           + fband +'.pkl')
            betas.append(mdf_Q.params[term])
            errs.append(mdf_Q.bse[term])
            print(fband, term, mdf_Q.pvalues[term])
        fig, ax = plt.subplots(1, 1,figsize = (1.5,2))     
        plt.bar([r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$'], 
                betas, yerr = errs, facecolor = 'w', edgecolor = 'k', capsize = 4)
        plt.ylim([-.1, .1])
        ax.spines["top"].set_visible(False); 
        ax.spines["right"].set_visible(False) 
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        plt.ylabel('', fontsize = 14)
        plt.xlabel('Freq. band', fontsize = 14)
        fig.savefig(fig_params.figures_dir + 
                    '/betas_mean_sensor_power_band_by_pupil_size' + block_name + str(term) + '.png', 
                    bbox_inches = 'tight', 
                    dpi = 800, transparent = True) 

def plot_lin_betas(block_name):
    # plot sensor level betas
        errs = []; betas = []

        for fband, frange in HLTP_pupil.freq_bands.items():
            mdf_L = pd.read_pickle(
                    HLTP_pupil.result_dir + '/mixedlmL_full_pupil_' + block_name
                           + fband +'.pkl')
            betas.append(mdf_L.params[1])
            errs.append(mdf_L.bse[1])
            print(fband, 1, mdf_L.pvalues[1])
        fig, ax = plt.subplots(1, 1,figsize = (1.5,2))     
        plt.bar([r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$'], 
                betas, yerr = errs, facecolor = 'w', edgecolor = 'k', capsize = 4)
        plt.ylim([-.1, .1])
        ax.spines["top"].set_visible(False); 
        ax.spines["right"].set_visible(False) 
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('outward', 15))
        ax.xaxis.set_ticks_position('bottom')
        plt.ylabel('', fontsize = 14)
        plt.xlabel('Freq. band', fontsize = 14)
        fig.savefig(fig_params.figures_dir + 
                    '/betas_mean_sensor_power_band_by_pupil_size_lin' + block_name + '.png', 
                    bbox_inches = 'tight', 
                    dpi = 800, transparent = True)   


def plot_blink_dependent_bhv():
    adf = HLTP_pupil.load(HLTP_pupil.result_dir + '/blink_dependent_SDT_bhv.pkl')

    y_lims = {'HR': [0., 1], 'FAR': [-.2, 1.2], 'c': [-1., 2.], 'd': [-1., 2.5],
              'p_correct': [0.2, 0.8], 'catRT': [0.6, 1.6]}
    bhv_vars = ['HR', 'FAR', 'c', 'd', 'p_correct', 'catRT']

    for bhv_var in bhv_vars:
        data = [adf[~adf.blink][bhv_var].values, adf[adf.blink][bhv_var].values]
        nansubj = np.concatenate([np.where(np.isnan(data[0]))[0], np.where(np.isnan(data[1]))[0]])
        if nansubj.shape[0] > 0:
            data[0] = np.delete(data[0], nansubj);
            data[1] = np.delete(data[1], nansubj)
        print(bhv_var, scipy.stats.wilcoxon(data[0], data[1]))
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
        plt.ylim(y_lims[bhv_var])
        plt.xlim([-.4, 1.4])

        plt.ylabel(bhv_var)
        fig.savefig(figures_dir + '/bhv_blink_' + bhv_var +
                    '.png', dpi=800, bbox_inches='tight', transparent=True)


plot_fig_1A('task_prestim')
plot_fig_1A('rest')    
plot_betas('rest_')  
plot_betas('bhv_')    
    
